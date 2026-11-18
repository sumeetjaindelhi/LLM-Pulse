import { describe, it, expect } from "vitest";
import {
  computeMb,
  kvCacheMb,
  kvCacheMbPer1kTokens,
  maxContextTokensForVram,
  memoryBreakdown,
  memoryNeedMb,
} from "../../src/analysis/context-capacity.js";
import { getModelById } from "../../src/models/database.js";
import type { HardwareProfile, ModelEntry, QuantizationVariant } from "../../src/core/types.js";
import highEnd from "../fixtures/hardware-profiles/high-end-nvidia.json";
import cpuOnly from "../fixtures/hardware-profiles/cpu-only.json";

const HIGH_END = highEnd as HardwareProfile;
const CPU_ONLY = cpuOnly as HardwareProfile;

const llama8b = getModelById("llama-3.1-8b")!;
const llama70b = getModelById("llama-3.1-70b")!;

function quantOf(model: ModelEntry, name: string): QuantizationVariant {
  const q = model.quantizations.find((x) => x.name === name);
  if (!q) throw new Error(`fixture model missing quant ${name}`);
  return q;
}

describe("kvCacheMbPer1kTokens", () => {
  const withParams = (parametersBillion: number): ModelEntry => ({ ...llama8b, parametersBillion });

  it("maps param-count buckets to fp16 KV-cache MB per 1k tokens", () => {
    expect(kvCacheMbPer1kTokens(withParams(1.5))).toBe(40);
    expect(kvCacheMbPer1kTokens(withParams(4))).toBe(80);
    expect(kvCacheMbPer1kTokens(withParams(8))).toBe(150);
    expect(kvCacheMbPer1kTokens(withParams(15))).toBe(210);
    expect(kvCacheMbPer1kTokens(withParams(22))).toBe(280);
    expect(kvCacheMbPer1kTokens(withParams(35))).toBe(360);
    expect(kvCacheMbPer1kTokens(withParams(80))).toBe(440);
    expect(kvCacheMbPer1kTokens(withParams(81))).toBe(600);
  });

  it("uses the model's architecture-derived kvMbPer1kTokens over the bucket", () => {
    expect(kvCacheMbPer1kTokens({ ...withParams(3.8), kvMbPer1kTokens: 375 })).toBe(375);
  });

  it("sizes curated non-GQA models from their architecture, which the bucket underestimates", () => {
    // phi-3-mini (registry GGUF): 32 layers x 32 KV heads x (96 + 96) x 2 B x 1024 tokens = 384 MiB
    expect(kvCacheMb(getModelById("phi-3-mini")!, 1024)).toBe(384);
    // codellama-7b: 32 x 32 x (128 + 128) x 2 B x 1024 tokens = 512 MiB
    expect(kvCacheMb(getModelById("codellama-7b")!, 1024)).toBe(512);
  });
});

describe("maxContextTokensForVram", () => {
  // llama-3.1-8b carries its registry facts: Q4_K_M file 4693 MiB, KV 32 x 8 x 256 x 2 B = 0.125 MiB/token,
  // vocab 128256 (compute 128256 x batch x 4 x 1.1 B: 1102.2 MiB at batch 2048), training context 131072.
  it("stops at the training context on a discrete GPU", () => {
    // Rule: maxGpuContext over need = weights + KV + compute, against the memory-model budget, up to the training context.
    // budget 24576 - 457 - 1024 = 23095; need(131072) = 4693 + 16384 + 1102.2 = 22179.2 fits.
    const q4 = quantOf(llama8b, "Q4_K_M");
    expect(maxContextTokensForVram(llama8b, q4, HIGH_END)).toBe(131072);
  });

  it("uses available system RAM on a CPU-only box", () => {
    // CPU-only budget = available RAM 10000 (no reserve). Above 32768 tokens the batch is 2048, and each context
    // cell costs KV 0.125 + mask 2 B x 2048 (0.0039) MiB: c <= (10000 - 4693 - 1102.2) / 0.12890625 = 32620 < 32768, so none fits.
    // At 32768 (batch 1024): 4693 + 551.1 + 32768 x (0.125 + 2 x 1024 / 2^20 = 0.126953125) = 9404.1 <= 10000.
    const q4 = quantOf(llama8b, "Q4_K_M");
    expect(maxContextTokensForVram(llama8b, q4, CPU_ONLY)).toBe(32768);
  });

  it("returns 0 when the weights alone exceed the memory pool", () => {
    const q70 = quantOf(llama70b, "Q4_K_M"); // 40551 MiB on a 10000 MiB box
    expect(maxContextTokensForVram(llama70b, q70, CPU_ONLY)).toBe(0);
  });

  it("affords more context for a smaller quant", () => {
    const q4 = quantOf(llama8b, "Q4_K_M");
    const f16 = quantOf(llama8b, "F16");
    expect(maxContextTokensForVram(llama8b, q4, HIGH_END)).toBeGreaterThan(
      maxContextTokensForVram(llama8b, f16, HIGH_END),
    );
  });

  it("affords more context on a larger memory budget", () => {
    const q4 = quantOf(llama8b, "Q4_K_M");
    expect(maxContextTokensForVram(llama8b, q4, HIGH_END)).toBeGreaterThan(
      maxContextTokensForVram(llama8b, q4, CPU_ONLY),
    );
  });
});

describe("KV cache type", () => {
  it("scales the fp16 estimate by the ggml block ratios", () => {
    expect(kvCacheMbPer1kTokens(llama8b, "f16")).toBe(kvCacheMbPer1kTokens(llama8b));
    expect(kvCacheMbPer1kTokens(llama8b, "q8_0")).toBeCloseTo(150 * 0.53125, 6);
    expect(kvCacheMbPer1kTokens(llama8b, "q4_0")).toBeCloseTo(150 * 0.28125, 6);
  });

  it("affords more tokens with a quantized KV cache, q4_0 most", () => {
    // CPU box, budget 10000, batch 2048 (compute 1102.2, mask 0.00390625 MiB per context cell):
    // q8_0 KV 0.125 x 34/64 = 0.06640625 -> c <= 4204.8 / 0.0703125 = 59801 -> 59648;
    // q4_0 KV 0.125 x 18/64 = 0.03515625 -> c <= 4204.8 / 0.0390625 = 107642 -> 107520; f16 is 32768 (above).
    const quant = quantOf(llama8b, "Q4_K_M");
    expect(maxContextTokensForVram(llama8b, quant, CPU_ONLY)).toBe(32768);
    expect(maxContextTokensForVram(llama8b, quant, CPU_ONLY, "q8_0")).toBe(59648);
    expect(maxContextTokensForVram(llama8b, quant, CPU_ONLY, "q4_0")).toBe(107520);
  });
});

describe("memoryNeedMb", () => {
  it("is weights + KV (bucket x padded context) + compute buffer", () => {
    // No architecture facts: 4096 tokens, 8B bucket 150 MiB/1k: 4668 + 150*4.096 + 262144*512*4*1.1/2^20 (563.2)
    // + KQ mask 2 B x 4096 x 512 (4) = 5849.6
    expect(memoryNeedMb({ ...llama8b, arch: undefined }, { vramMb: 4668 }, 4096)).toBeCloseTo(5849.6, 1);
  });

  it("uses exact architecture facts when the model has them", () => {
    // 32 layers x 8 KV heads x (128 + 128) x 2 B = 131072 B/token; 4096 tokens = 512 MiB; vocab 128256 -> compute 128256*512*4.4/2^20 = 275.55 + KQ mask 4
    const arch = { architecture: "llama", layers: 32, headCountKv: 8, keyLength: 128, valueLength: 128, vocabSize: 128256 };
    expect(memoryNeedMb({ ...llama8b, arch }, { vramMb: 4668 }, 4096)).toBeCloseTo(4668 + 512 + 275.55 + 4, 1);
  });
});

describe("computeMb: activations stand-in + KQ mask copies", () => {
  // llama-3.1-8b: vocab 128256, no sliding window. ctx 4096 -> batch 512: vocab term 128256 x 512 x 4.4 / 2^20 = 275.55;
  // mask 2 B x 4096 x 512 = 4 MiB per copy.
  it("one GPU or CPU: one mask copy", () => {
    expect(computeMb(llama8b, 4096)).toBeCloseTo(275.55 + 4, 2);
  });

  it("two GPUs with pipeline parallelism: 4 copies on each = 8 x 4 MiB (L ggml-backend.cpp:2326, L llama-context.cpp:575-580)", () => {
    expect(computeMb(llama8b, 4096, undefined, 2)).toBeCloseTo(275.55 + 32, 2);
  });

  it("memoryNeedMb takes the GPU count", () => {
    const q4 = quantOf(llama8b, "Q4_K_M");
    expect(memoryNeedMb(llama8b, q4, 4096, "f16", undefined, 2) - memoryNeedMb(llama8b, q4, 4096)).toBeCloseTo(28, 6);
  });

  it("memoryBreakdown counts the GPUs of the budget group", () => {
    const two: HardwareProfile = { ...HIGH_END, gpus: [HIGH_END.primaryGpu!, HIGH_END.primaryGpu!] };
    expect(memoryBreakdown(llama8b, quantOf(llama8b, "Q4_K_M"), two, 4096).computeMb).toBeCloseTo(307.55, 0);
  });
});

describe("memoryBreakdown rounding", () => {
  // qwen2.5-coder:14b, 4096 tokens, one 24 GiB GPU. KV 48 x 8 x 256 x 2 B x 4096 = 768 MiB; activations
  // 152064 x 512 x 4 x 1.1 / 2^20 = 326.7 (a float: 326.70000000000005); KQ mask 2 B x 4096 x 512 = 4 MiB -> compute 330.7;
  // total 8572 + 768 + 330.7 = 9670.7. Both are rounded to 0.1 MiB, so they are exactly these numbers.
  it("rounds compute and total to 0.1 MiB", () => {
    const model = getModelById("qwen-2.5-coder-14b")!;
    const gpu = { ...HIGH_END.primaryGpu!, vramMb: 24564 };
    const memory = memoryBreakdown(model, quantOf(model, "Q4_K_M"), { ...HIGH_END, gpus: [gpu], primaryGpu: gpu }, 4096);
    expect(memory.computeMb).toBe(330.7);
    expect(memory.totalMb).toBe(9670.7);
  });
});

describe("memoryBreakdown", () => {
  const gpu = { ...HIGH_END.primaryGpu!, vramMb: 24564 };
  const gpu24564: HardwareProfile = { ...HIGH_END, gpus: [gpu], primaryGpu: gpu };

  it("splits the need at Ollama's default context and says which parts are exact", () => {
    // 24564 MiB GPU: budget 24564 - 457 - 1024 = 23083; >= 23 GiB, so the default context is 32768.
    // Q8_0 is not the file `ollama pull llama3.1:8b` installs: weights 8e9 x 8.5008 / 8 / 2^20 = 8107 (estimate).
    // KV from arch: 32 x 8 x 256 x 2 B x 32768 = 4096 MiB; compute 128256 x 1024 x 4 x 1.1 / 2^20 = 551.1
    // + KQ mask 2 B x 32768 x 1024 = 64 MiB = 615.1. Total 8107 + 4096 + 615.1 = 12818.1.
    const q8 = quantOf(llama8b, "Q8_0");
    expect(memoryBreakdown(llama8b, q8, gpu24564)).toEqual({
      contextTokens: 32768,
      weightsMb: 8107,
      weightsExact: false,
      kvCacheMb: 4096,
      kvExact: true,
      computeMb: expect.closeTo(615.1, 6),
      totalMb: expect.closeTo(12818.1, 6),
      budgetMb: 23083,
      budgetExact: false,
    });
  });

  it("marks the weights of the installed file exact", () => {
    expect(memoryBreakdown(llama8b, quantOf(llama8b, "Q4_K_M"), HIGH_END).weightsExact).toBe(true);
  });

  it("marks the KV cache estimated without architecture facts or when they do not resolve", () => {
    const q4 = quantOf(llama8b, "Q4_K_M");
    expect(memoryBreakdown({ ...llama8b, arch: undefined }, q4, HIGH_END).kvExact).toBe(false);
    // A per-layer KV head array shorter than the 32 layers is not exact (kvCacheBytes).
    const mismatched = { ...llama8b.arch!, headCountKv: [8, 8] };
    expect(memoryBreakdown({ ...llama8b, arch: mismatched }, q4, HIGH_END).kvExact).toBe(false);
  });

  it("totals memoryNeedMb at a given context and KV cache type", () => {
    const q4 = quantOf(llama8b, "Q4_K_M");
    const memory = memoryBreakdown(llama8b, q4, CPU_ONLY, 50000, "q8_0");
    expect(memory.contextTokens).toBe(50000);
    expect(memory.totalMb).toBeCloseTo(memoryNeedMb(llama8b, q4, 50000, "q8_0"), 1);
    expect(memory.budgetMb).toBe(10000);
  });

  it("rounds KV, compute and total to 0.1 MiB so JSON and CSV carry no float noise", () => {
    const memory = memoryBreakdown(llama8b, quantOf(llama8b, "Q4_K_M"), CPU_ONLY, 5000);
    for (const value of [memory.kvCacheMb, memory.computeMb, memory.totalMb]) expect(value).toBe(Math.round(value * 10) / 10);
  });
});
