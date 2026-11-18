import { describe, it, expect } from "vitest";
import { defaultContextFor, getAvailableVram, scoreModel, suggestGpuOffload } from "../../src/analysis/scorer.js";
import { maxContextTokensForVram, memoryNeedMb } from "../../src/analysis/context-capacity.js";
import { kqMaskMb } from "../../src/analysis/memory-model.js";
import { getModelById } from "../../src/models/database.js";
import type { GpuInfo, HardwareProfile, ModelEntry, QuantizationVariant } from "../../src/core/types.js";
import cpuOnly from "../fixtures/hardware-profiles/cpu-only.json";

const gpuBase: GpuInfo = {
  vendor: "NVIDIA",
  model: "GPU",
  vramMb: 0,
  driverVersion: "",
  acceleratorVersion: null,
  acceleratorType: "cuda",
  utilizationPercent: null,
  temperatureCelsius: null,
  vramUsedMb: null,
};

function withGpus(gpus: GpuInfo[], availableMb = 10000): HardwareProfile {
  const base = cpuOnly as HardwareProfile;
  return { ...base, memory: { ...base.memory, availableMb }, gpus, primaryGpu: gpus[0] ?? null };
}

const rtx4060 = withGpus([{ ...gpuBase, vramMb: 8188 }]);
const rtx4090 = withGpus([{ ...gpuBase, vramMb: 24564 }]);
const apple24 = withGpus([{ ...gpuBase, vendor: "Apple", vramMb: 18186, acceleratorType: "metal" }]);
const apple64 = withGpus([{ ...gpuBase, vendor: "Apple", vramMb: 49152, acceleratorType: "metal" }]);

const curated = (id: string): { model: ModelEntry; q4: QuantizationVariant } => {
  const model = getModelById(id)!;
  return { model, q4: model.quantizations.find((q) => q.name === "Q4_K_M")! };
};

// Library row as library-entries builds it: 14B, 5 bpw weights only, estimated window.
const library14b: ModelEntry = {
  id: "lib:14b",
  name: "lib 14B",
  provider: "Ollama library",
  parametersBillion: 14,
  contextWindow: 32768,
  contextWindowEstimated: true,
  categories: ["general"],
  qualityTier: "lightweight",
  qualityScore: 0,
  quantizations: [{ name: "default", bitsPerWeight: 5, vramMb: 8345, qualityRetention: 0.97 }],
  ollamaTag: "lib:14b",
  releaseDate: "2025-06",
  source: "library",
};

describe("getAvailableVram is the memory-model budget", () => {
  it("discrete GPU: vramMb - 457 - 1024", () => {
    expect(getAvailableVram(rtx4060)).toBe(8188 - 457 - 1024);
  });
  it("Apple Metal: vramMb - 1024", () => {
    expect(getAvailableVram(apple24)).toBe(18186 - 1024);
  });
  it("GB10 unified: available RAM - 457 - 1024", () => {
    const gb10 = withGpus([{ ...gpuBase, vramMb: 100_000, unifiedMemory: true }], 90_000);
    expect(getAvailableVram(gb10)).toBe(90_000 - 457 - 1024);
  });
  it("AMD APU: min(GTT, available RAM) - 457 - 1024", () => {
    const apu = withGpus([{ ...gpuBase, vendor: "AMD", vramMb: 32768, acceleratorType: "rocm", unifiedMemory: true, gfxTarget: "gfx1151" }], 40_000);
    expect(getAvailableVram(apu)).toBe(32768 - 457 - 1024);
  });
  it("CPU only and 0-VRAM iGPU: available RAM", () => {
    expect(getAvailableVram(withGpus([], 7000))).toBe(7000);
    expect(getAvailableVram(withGpus([{ ...gpuBase, vendor: "Intel", vramMb: 0 }], 7000))).toBe(7000);
  });
});

describe("scoreModel scenarios", () => {
  // Curated rows carry registry facts (data/models.json): exact default-file size and GGUF architecture.
  it("RTX 4060 8 GB + llama-3.1-8b Q4_K_M: comfortable", () => {
    // budget 8188-1481 = 6707; default ctx 4096 (8188 MiB < 23 GiB)
    // need = weights 4693 (registry file) + KV 32x8x256x2 B x 4096 = 512 + compute 128256*512*4*1.1/2^20 = 275.55 = 5480.55
    // 0.8*6707 = 5365.6 < 5480.55 <= 6707
    const { model, q4 } = curated("llama-3.1-8b");
    expect(scoreModel(model, q4, rtx4060).fitLevel).toBe("comfortable");
  });

  it("Apple 24 GB + 14B library row: excellent", () => {
    // budget 18186-1024 = 17162; default ctx 4096; need = 8345 + 210*4.096 (860.2) + 563.2 + mask 4 = 9772.4 <= 0.8*17162 = 13729.6
    expect(scoreModel(library14b, library14b.quantizations[0], apple24).fitLevel).toBe("excellent");
  });

  it("Apple 24 GB + gemma-3-27b Q4_K_M: barely", () => {
    // weights 16607 (model 15789 + vision projector 818, registry) + KV 944 + compute 262145*512*4.4/2^20 (563.2) + masks (2 x 4096 x 512 = 4, SWA 2 x 1536 x 512 = 1.5) = 18119.7 > 17162.
    // KV at 4096: 10 global layers x 16 x 256 x 2 B x 4096 = 320 MiB; 52 sliding layers hold 1024 + 512 slots = 624 MiB
    // (L src/llama-kv-cache-iswa.cpp:73). 0.75 * 18119.7 = 13589.8 <= 17162 -> partial offload
    const { model, q4 } = curated("gemma-3-27b");
    expect(scoreModel(model, q4, apple24).fitLevel).toBe("barely");
  });

  it("RTX 4090: default context 32768 and qwen-2.5-14b Q4_K_M excellent", () => {
    // 24564 MiB >= 23 GiB (23552) -> 32768 (O server/routes.go:2571-2589), equal to the 32768 training context
    // need = 8572 + KV 48x8x256x2 B x 32768 (6144) + compute 152064*1024*4*1.1/2^20 (653.4) + KQ mask 2 B x 32768 x 1024 (64) = 15433.4
    // budget 24564-1481 = 23083; 0.8*23083 = 18466.4 >= need
    const { model, q4 } = curated("qwen-2.5-14b");
    expect(memoryNeedMb(model, q4, 32768)).toBeCloseTo(15433.4, 0);
    expect(scoreModel(model, q4, rtx4090).fitLevel).toBe("excellent");
  });

  it("Apple 64 GB-class + llama-3.3-70b Q4_K_M: tight, max context between 4096 and 32768", () => {
    // budget 49152-1024 = 48128; weights 40551 (registry); KV 80x8x256x2 B = 0.3125 MiB/token; vocab 128256
    // need(4096) = 40551 + 1280 + 275.55 + mask 4 = 42110.6 <= budget < need(131072) (default 262144 clamped to training context)
    // c <= 32768 (batch 1024: compute 551.1, mask 2 x 1024 B per cell = 1/512 MiB):
    // 40551 + 551.1 + (0.3125 + 0.001953125)*c <= 48128 -> c <= 22342 -> 22272
    const { model, q4 } = curated("llama-3.3-70b");
    expect(scoreModel(model, q4, apple64).fitLevel).toBe("tight");
    expect(maxContextTokensForVram(model, q4, apple64)).toBe(22272);
  });

  it("a 0-VRAM iGPU is not counted as a GPU for the speed estimate", () => {
    // 8B on RAM only: active bytes 4693 MiB > 2 GiB CPU limit -> slow (a GPU would say fast)
    const { model, q4 } = curated("llama-3.1-8b");
    const hw = withGpus([{ ...gpuBase, vendor: "Intel", vramMb: 0 }], 16000);
    expect(scoreModel(model, q4, hw).speedEstimate).toBe("slow");
  });
});

describe("defaultContextFor", () => {
  const { model } = curated("llama-3.1-8b"); // train context 131072

  it("sums VRAM over all GPUs (O server/routes.go:2571-2589)", () => {
    // 2 x 12288 = 24576 MiB >= 23 GiB -> 32768
    expect(defaultContextFor(model, withGpus([{ ...gpuBase, vramMb: 12288 }, { ...gpuBase, vramMb: 12288 }]))).toBe(32768);
    expect(defaultContextFor(model, rtx4060)).toBe(4096);
    expect(defaultContextFor(model, withGpus([]))).toBe(4096);
  });

  it("clamps to the training context, but not to an estimated window", () => {
    expect(defaultContextFor({ ...model, arch: { ...model.arch!, contextLength: 8192 } }, apple64)).toBe(8192);
    expect(defaultContextFor({ ...model, arch: undefined, contextWindow: 8192 }, apple64)).toBe(8192);
    expect(defaultContextFor({ ...model, arch: undefined, contextWindow: 8192, contextWindowEstimated: true }, apple64)).toBe(262144);
  });
});

describe("suggestGpuOffload", () => {
  it("counts layers against budget minus compute, each layer carrying its share of KV (default context, exact layer count)", () => {
    // RTX 4060 budget 6707; default ctx 4096. Arch: 80 layers x 8 KV heads x (128+128) x 2 B = 327680 B/token
    // -> 4096 tokens = 1280 MiB; compute 128256*512*4*1.1/2^20 = 275.55. Weights 40842 + KV 1280 over 80 layers = 526.5 MiB/layer.
    // floor((6707 - 275.55) / 526.5) = floor(12.2) = 12 (KV of CPU layers lives in RAM: L src/llama-kv-cache.cpp:216-221)
    const model = {
      ...curated("llama-3.3-70b").model,
      arch: { architecture: "llama", layers: 80, headCountKv: 8, keyLength: 128, valueLength: 128, vocabSize: 128256 },
    };
    const quant = { name: "Q4_K_M", bitsPerWeight: 4.8944, vramMb: 40842, qualityRetention: 0.97 };
    const offload = suggestGpuOffload(model, quant, rtx4060)!;
    expect(offload).toMatchObject({ gpuLayers: 12, totalLayers: 80, reason: "partial_offload" });
  });

  it("reports a full fit when every layer fits", () => {
    const { model, q4 } = curated("llama-3.1-8b");
    expect(suggestGpuOffload(model, q4, rtx4090)).toMatchObject({ gpuLayers: "all", reason: "full_fit" });
  });
});

describe("suggestGpuOffload on several GPUs", () => {
  // Pipeline parallelism needs every layer on the GPUs (L src/llama-context.cpp:575-578), so a partial offload keeps one KQ
  // mask copy per GPU, not 4 per GPU.
  it("counts one mask copy per GPU", () => {
    // 2 x (7946 - 457 - 1024) = 12930 MiB. ctx 4096, ubatch 512: KV 1280, activations 275.55, mask 2 B x 4096 x 512 = 4 MiB a copy.
    // Weights 40842 + KV 1280 over 80 layers = 526.525 MiB/layer. One copy per GPU: 2 x 4 = 8 MiB ->
    // floor((12930 - 275.55 - 8) / 526.525) = floor(24.02) = 24. With 4 copies per GPU (32 MiB): floor(23.97) = 23.
    const model = {
      ...curated("llama-3.3-70b").model,
      arch: { architecture: "llama", layers: 80, headCountKv: 8, keyLength: 128, valueLength: 128, vocabSize: 128256 },
    };
    const quant = { name: "Q4_K_M", bitsPerWeight: 4.8944, vramMb: 40842, qualityRetention: 0.97 };
    const gpu = { ...gpuBase, vramMb: 7946 };
    expect(suggestGpuOffload(model, quant, withGpus([gpu, gpu]))).toMatchObject({ gpuLayers: 24, reason: "partial_offload" });
  });

  it("keeps one mask copy per GPU, not one in total", () => {
    // Two copies are 8 MiB (one in total would be 4). 2 x (7940 - 1481) = 12918:
    // floor((12918 - 275.55 - 8) / 526.525) = floor(23.99) = 23; with 4 MiB it would be floor(24.003) = 24.
    const arch = { architecture: "llama", layers: 80, headCountKv: 8, keyLength: 128, valueLength: 128, vocabSize: 128256 };
    expect(kqMaskMb(arch, 4096, 512, 2, false)).toBe(8);
    const model = { ...curated("llama-3.3-70b").model, arch };
    const quant = { name: "Q4_K_M", bitsPerWeight: 4.8944, vramMb: 40842, qualityRetention: 0.97 };
    const gpu = { ...gpuBase, vramMb: 7940 };
    expect(suggestGpuOffload(model, quant, withGpus([gpu, gpu]))).toMatchObject({ gpuLayers: 23 });
  });
});

describe("GPU count reaches the fit level and the context limit", () => {
  // Full offload on 2 GPUs keeps 4 mask copies each (8 in all), one GPU keeps one: the need is not the single-GPU need.
  const { model, q4 } = curated("llama-3.1-8b");
  const small = { ...gpuBase, vramMb: 8188 };
  const hw2 = withGpus([small, small]); // budget 2 x 6707 = 13414

  it("maxContextTokensForVram stops at the last context whose 2-GPU need fits", () => {
    const ctx = maxContextTokensForVram(model, q4, hw2);
    expect(ctx).toBeGreaterThan(0);
    expect(ctx).toBeLessThan(131072);
    expect(memoryNeedMb(model, q4, ctx, "f16", undefined, 2)).toBeLessThanOrEqual(13414);
    expect(memoryNeedMb(model, q4, ctx + 256, "f16", undefined, 2)).toBeGreaterThan(13414);
  });

  it("assessModelFit rates against the 2-GPU need", () => {
    const ctx = defaultContextFor(model, hw2);
    const need = memoryNeedMb(model, q4, ctx, "f16", undefined, 2);
    expect(memoryNeedMb(model, q4, ctx, "f16", undefined, 1)).toBeLessThan(need);
    expect(scoreModel(model, q4, hw2).fitRatio).toBeCloseTo(13414 / need, 9);
  });
});

describe("GPUs Ollama schedules on", () => {
  const { model, q4 } = curated("llama-3.1-8b");
  const rtx3090 = { ...gpuBase, vramMb: 24564 };

  it("budgets every discrete GPU: 2 x 24 GB = 2 x (24564 - 457 - 1024)", () => {
    expect(getAvailableVram(withGpus([rtx3090, rtx3090]))).toBe(2 * (24564 - 457 - 1024));
  });

  it("rates an 8B model excellent on 2 x 24 GB (default context 131072 fits the pooled budget)", () => {
    // need(131072) = 4693 weights + 16384 KV (32x8x256x2 B x 131072) + 1102.2 compute (128256 x 2048 x 4.4 / 2^20) = 22179.2 <= 0.8 x 46166 = 36932.8
    expect(scoreModel(model, q4, withGpus([rtx3090, rtx3090])).fitLevel).toBe("excellent");
  });

  it("ignores an integrated GPU beside a discrete one", () => {
    const apu = { ...gpuBase, vendor: "AMD", vramMb: 16384, acceleratorType: "rocm" as const, unifiedMemory: true };
    const hw = withGpus([{ ...gpuBase, vramMb: 8188 }, apu]);
    expect(defaultContextFor(model, hw)).toBe(4096);
    expect(getAvailableVram(hw)).toBe(8188 - 457 - 1024);
  });
});

describe("suggestGpuOffload at an explicit context", () => {
  it("keeps all layers on GPU when the chosen context fits (qwen-2.5-32b on a 4090)", () => {
    const { model, q4 } = curated("qwen-2.5-32b");
    const budget = getAvailableVram(rtx4090);
    expect(memoryNeedMb(model, q4, 8192)).toBeLessThanOrEqual(budget);
    expect(suggestGpuOffload(model, q4, rtx4090, 8192)).toMatchObject({ gpuLayers: "all" });
  });

  it("counts the KV of CPU-resident layers in RAM, not VRAM", () => {
    // budget 6707, ctx 4096, KV 1280, compute 275.55, weights 40842 over 80 layers:
    // per-layer weights+KV = (40842 + 1280) / 80 = 526.5 -> floor((6707 - 275.55) / 526.5) = 12
    const model = {
      ...curated("llama-3.3-70b").model,
      arch: { architecture: "llama", layers: 80, headCountKv: 8, keyLength: 128, valueLength: 128, vocabSize: 128256 },
    };
    const quant = { name: "Q4_K_M", bitsPerWeight: 4.8944, vramMb: 40842, qualityRetention: 0.97 };
    expect(suggestGpuOffload(model, quant, rtx4060)).toMatchObject({ gpuLayers: 12 });
  });
});

describe("maxContextTokensForVram ceiling", () => {
  it("uses a known training window beyond the 262144 display tier", () => {
    const { model, q4 } = curated("llama-3.1-8b");
    const huge = { ...model, arch: { ...model.arch!, contextLength: 1_048_576 } };
    const cpu = withGpus([], 400_000);
    expect(maxContextTokensForVram(huge, q4, cpu)).toBeGreaterThan(262144);
  });
});
