import { describe, it, expect } from "vitest";
import {
  recommendThreads,
  recommendBatch,
  recommendContext,
  computeOptimization,
} from "../../src/analysis/optimizer.js";
import { CONTEXT_TIERS } from "../../src/analysis/memory-model.js";
import { getModelById } from "../../src/models/database.js";
import { getAvailableVram } from "../../src/analysis/scorer.js";
import { memoryNeedMb } from "../../src/analysis/context-capacity.js";
import type {
  CpuInfo,
  HardwareProfile,
  ModelEntry,
  QuantizationVariant,
} from "../../src/core/types.js";
import highEnd from "../fixtures/hardware-profiles/high-end-nvidia.json";
import cpuOnly from "../fixtures/hardware-profiles/cpu-only.json";
import appleM2 from "../fixtures/hardware-profiles/apple-m2.json";

const HIGH_END = highEnd as HardwareProfile;
const CPU_ONLY = cpuOnly as HardwareProfile;
const APPLE_M2 = appleM2 as HardwareProfile;

const llama8b = getModelById("llama-3.1-8b")!;
const llama70b = getModelById("llama-3.1-70b")!;

function quantOf(model: ModelEntry, name: string): QuantizationVariant {
  const q = model.quantizations.find((x) => x.name === name);
  if (!q) throw new Error(`fixture model missing quant ${name}`);
  return q;
}

// Clone a discrete-GPU profile with a different VRAM size (stays non-metal).
function withVram(profile: HardwareProfile, vramMb: number): HardwareProfile {
  return {
    ...profile,
    gpus: [{ ...profile.gpus[0], vramMb }],
    primaryGpu: profile.primaryGpu ? { ...profile.primaryGpu, vramMb } : null,
  };
}

function withCpu(profile: HardwareProfile, cpu: Partial<CpuInfo>): HardwareProfile {
  return { ...profile, cpu: { ...profile.cpu, ...cpu } };
}

describe("recommendThreads", () => {
  it("uses physical performance cores on a hybrid CPU", () => {
    const hw = withCpu(HIGH_END, {
      cores: 16,
      threads: 24,
      performanceCores: 8,
      efficiencyCores: 8,
    });
    expect(recommendThreads(hw.cpu).value).toBe(8);
  });

  it("falls back to physical cores — never logical SMT threads — when no P/E split", () => {
    // HIGH_END fixture: 16 cores / 32 threads, no performanceCores field.
    expect(recommendThreads(HIGH_END.cpu).value).toBe(16);
    expect(recommendThreads(HIGH_END.cpu).value).not.toBe(HIGH_END.cpu.threads);
  });
});

describe("recommendBatch", () => {
  // autoBatch: ctx <= 4096 -> 512, <= 32768 -> 1024, above -> 2048 (O server/sched.go:900-909).
  it("uses Ollama's automatic batch for the context when the quant fits comfortably", () => {
    expect(recommendBatch("excellent", 4096).value).toBe(512);
    expect(recommendBatch("comfortable", 8192).value).toBe(1024);
    expect(recommendBatch("comfortable", 65536).value).toBe(2048);
    expect(recommendBatch("comfortable", 4096).note).toContain("automatic");
  });

  it("pins 256 on tight or barely fits", () => {
    expect(recommendBatch("tight", 8192).value).toBe(256);
    expect(recommendBatch("barely", 4096).value).toBe(256);
  });

  // The budget is VRAM minus reserves, the Metal working set, or RAM on a CPU-only machine: never plain VRAM.
  it("names usable memory, not VRAM, in its notes", () => {
    for (const level of ["excellent", "tight"] as const) {
      expect(recommendBatch(level, 4096).note).not.toContain("VRAM");
    }
    expect(recommendBatch("tight", 4096).note).toContain("memory");
  });
});

describe("recommendContext", () => {
  it("never exceeds the model's native context window", () => {
    const smallCtxModel: ModelEntry = { ...llama8b, contextWindow: 8192 };
    const q4 = quantOf(llama8b, "Q4_K_M");
    expect(recommendContext(smallCtxModel, q4, HIGH_END).value).toBe(8192);
  });

  it("floors at 2048 when the weights exceed the memory budget", () => {
    const q70 = quantOf(llama70b, "Q4_K_M"); // 40 GB of weights
    const ctx = recommendContext(llama70b, q70, CPU_ONLY);
    expect(ctx.value).toBe(2048);
    expect(ctx.note).toBe("KV cache won't fit at 2048 — expect CPU offload");
  });

  it("recommends a larger context on a larger memory budget", () => {
    const q4 = quantOf(llama8b, "Q4_K_M");
    const big = recommendContext(llama8b, q4, HIGH_END).value as number;
    const small = recommendContext(llama8b, q4, CPU_ONLY).value as number;
    expect(big).toBeGreaterThan(small);
  });

  it("keeps phi-3-mini's recommended context within budget at its real MHA KV size", () => {
    // Real fp16 KV for phi-3-mini: 32 layers x 32 KV heads x 96 head_dim x 4 bytes = 375 MiB / 1k.
    const realKvMbPer1k = 375;
    const phi3 = getModelById("phi-3-mini")!;
    const budgetMb = HIGH_END.primaryGpu!.vramMb;
    for (const quant of phi3.quantizations) {
      const ctx = recommendContext(phi3, quant, HIGH_END).value as number;
      expect((ctx / 1000) * realKvMbPer1k + quant.vramMb).toBeLessThanOrEqual(budgetMb);
    }
  });

  it("snaps to a context tier rather than an arbitrary number", () => {
    const q4 = quantOf(llama8b, "Q4_K_M");
    const ctx = recommendContext(llama8b, q4, HIGH_END).value as number;
    expect(CONTEXT_TIERS).toContain(ctx);
    expect(ctx).toBeGreaterThanOrEqual(16384); // 24 GB gives an 8B-Q4 model plenty of room
  });
});

describe("recommendContext with an estimated window", () => {
  const q4 = quantOf(llama8b, "Q4_K_M");

  it("keeps the estimate as the ceiling and says it is unconfirmed", () => {
    const guessed = { ...llama8b, contextWindow: 32768, contextWindowEstimated: true };
    const result = recommendContext(guessed, q4, HIGH_END);
    expect(result.value).toBe(32768);
    expect(result.note).toContain("est. window");
  });

  it("keeps the native-window note for a confirmed window", () => {
    const confirmed = { ...llama8b, contextWindow: 32768 };
    expect(recommendContext(confirmed, q4, HIGH_END).note).toBe("capped at native context");
  });

  it("does not mention the estimate when VRAM is what holds the context down", () => {
    const guessed = { ...llama8b, contextWindow: 131072, contextWindowEstimated: true };
    expect(recommendContext(guessed, quantOf(llama70b, "Q4_K_M"), HIGH_END).note).not.toContain("est. window");
  });
});

describe("computeOptimization", () => {
  it("recommends all layers on GPU for a full discrete-GPU fit", () => {
    const p = computeOptimization(llama8b, HIGH_END)!;
    expect(p).not.toBeNull();
    expect(p.numGpu?.value).toBe("all");
    expect(p.quantForced).toBe(false);
    expect(p.numThread.value).toBe(16);
    // 8B llama context is 131072 on this card: above 32768 the automatic batch is 2048.
    expect(p.numBatch.value).toBe(2048);
    expect(typeof p.numCtx.value).toBe("number");
  });

  it("omits num_gpu on Apple Silicon (unified memory)", () => {
    const p = computeOptimization(llama8b, APPLE_M2)!;
    expect(p.numGpu).toBeNull();
  });

  it("recommends num_gpu 0 on a CPU-only machine", () => {
    const p = computeOptimization(llama8b, CPU_ONLY)!;
    expect(p.numGpu?.value).toBe(0);
  });

  it("halves num_batch and warns of offload when the budget cannot hold the 2048 floor", () => {
    // Rule: fit is judged at the default context (4096 here), so "comfortable" implies 4096 tokens fit and this state is "barely".
    // 6 GB GPU: budget 6144 - 457 - 1024 = 4663; Q5_K_M weights 7e9*5.7036/8/2^20 = 4759 > budget, so no context fits.
    // need(4096) = 4759 + 614.4 + 563.2 = 5937; 0.75 x 5937 = 4453 <= 4663 -> barely.
    const qwen7b = getModelById("qwen-2.5-7b")!;
    const profile = computeOptimization(qwen7b, withVram(HIGH_END, 6144), quantOf(qwen7b, "Q5_K_M"))!;
    expect(profile.fitLevel).toBe("barely");
    expect(profile.numCtx.value).toBe(2048);
    expect(profile.numCtx.note).toContain("CPU offload");
    expect(profile.numBatch.value).toBe(256);
  });

  it("returns null when no quantization fits at all", () => {
    expect(computeOptimization(llama70b, CPU_ONLY)).toBeNull();
  });

  it("recommends a numeric layer count for a partial offload (forced F16 on 12 GB)", () => {
    const gpu12 = withVram(HIGH_END, 12288);
    const f16 = quantOf(llama8b, "F16");
    const p = computeOptimization(llama8b, gpu12, f16)!;
    expect(p.quant.name).toBe("F16");
    expect(p.quantForced).toBe(true);
    expect(typeof p.numGpu?.value).toBe("number");
    expect(p.numGpu!.value).toBeGreaterThan(0);
    expect(p.numGpu!.value).toBeLessThan(32);
    // The even per-layer split is an estimate (the real one keeps token_embd on the CPU), so the count is marked.
    expect(p.numGpu!.note).toBe(`≈${p.numGpu!.value} of 32 on GPU, rest on CPU`);
  });
});

// FROM <tag>, `ollama run <tag>` and OLLAMA_CONTEXT_LENGTH all load the file the tag installs, so that is the file to tune.
describe("computeOptimization quant choice", () => {
  it("tunes the file the tag installs when it fits whole, not the sweet spot", () => {
    // Apple M2 Pro: budget 10977 - 512 - 1024 = 9441 MiB, default context 4096; need(4096) = weights + KV
    // 32 x 8 x 256 x 2 B x 4096 = 512 MiB + compute 128256 x 512 x 4 x 1.1 / 2^20 = 275.55 MiB.
    // Q4_K_M (the tag's file, 4693): 5480.6 <= 0.8 x 9441 -> excellent. Q8_0 (8e9 x 8.5008 / 8 / 2^20 = 8107):
    // 8894.6 <= 9441 -> comfortable, and the sweet spot (higher retention).
    const p = computeOptimization(llama8b, APPLE_M2)!;
    expect(p.quant).toMatchObject({ name: "Q4_K_M", exact: true });
    expect(p.quantForced).toBe(false);
  });

  it("falls back to the sweet spot when the tag's file would not fit whole", () => {
    // 2781 MiB GPU: budget 2781 - 457 - 1024 = 1300 MiB, default context 4096; llama3.2:1b need(4096) = weights + KV
    // 16 x 8 x 128 x 2 B x 4096 = 128 MiB + compute 275.55 MiB. Q8_0 (the tag's file, 1260): 1663.6 > 1300 >= 0.75 x 1663.6
    // -> barely. Q4_K_M (1e9 x 4.8944 / 8 / 2^20 = 583): 986.6 <= 0.8 x 1300 -> excellent.
    const llama1b = getModelById("llama-3.2-1b")!;
    const p = computeOptimization(llama1b, withVram(HIGH_END, 2781))!;
    expect(p.quant.name).toBe("Q4_K_M");
    expect(p.fitLevel).toBe("excellent");
  });
});

describe("computeOptimization num_gpu at the recommended num_ctx", () => {
  it("is 'all' whenever the weights + KV + compute at num_ctx fit the budget (4090, 32B models)", () => {
    for (const id of ["qwen-2.5-32b", "gemma-3-27b", "deepseek-r1-32b"]) {
      const model = getModelById(id)!;
      const profile = computeOptimization(model, HIGH_END)!;
      if (profile.fitLevel === "cannot_run") continue;
      const need = memoryNeedMb(model, profile.quant, profile.numCtx.value as number);
      if (need <= getAvailableVram(HIGH_END)) expect(profile.numGpu?.value, id).toBe("all");
    }
  });
});
