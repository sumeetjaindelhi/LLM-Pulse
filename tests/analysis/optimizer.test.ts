import { describe, it, expect } from "vitest";
import {
  recommendThreads,
  recommendBatch,
  recommendContext,
  computeOptimization,
  CONTEXT_TIERS,
} from "../../src/analysis/optimizer.js";
import { getModelById } from "../../src/models/database.js";
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
  it("uses the full 512 batch when the quant fits comfortably", () => {
    expect(recommendBatch("excellent").value).toBe(512);
    expect(recommendBatch("comfortable").value).toBe(512);
  });

  it("shrinks the batch to 256 on tight or barely fits", () => {
    expect(recommendBatch("tight").value).toBe(256);
    expect(recommendBatch("barely").value).toBe(256);
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
    expect(recommendContext(llama70b, q70, CPU_ONLY).value).toBe(2048);
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

describe("computeOptimization", () => {
  it("recommends all layers on GPU for a full discrete-GPU fit", () => {
    const p = computeOptimization(llama8b, HIGH_END)!;
    expect(p).not.toBeNull();
    expect(p.numGpu?.value).toBe("all");
    expect(p.quantForced).toBe(false);
    expect(p.numThread.value).toBe(16);
    expect(p.numBatch.value).toBe(512);
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

  it("halves num_batch and warns of offload when the KV budget cannot hold the 2048 floor", () => {
    // 6 GB GPU: Q5_K_M weights (5300 MB) fit comfortably, but only ~500 tokens of KV remain.
    const qwen7b = getModelById("qwen-2.5-7b")!;
    const profile = computeOptimization(qwen7b, withVram(HIGH_END, 6144), quantOf(qwen7b, "Q5_K_M"))!;
    expect(profile.fitLevel).toBe("comfortable");
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
  });
});
