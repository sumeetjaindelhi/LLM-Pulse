import { describe, it, expect } from "vitest";
import { checkContextFit } from "../../src/analysis/context-fit.js";
import type { HardwareProfile, ModelEntry, QuantizationVariant } from "../../src/core/types.js";

function makeQuant(over: Partial<QuantizationVariant> = {}): QuantizationVariant {
  return { name: "Q4_K_M", bitsPerWeight: 4.83, vramMb: 4096, qualityRetention: 0.9, ...over };
}

function makeModel(over: Partial<ModelEntry> = {}): ModelEntry {
  return {
    id: "test-model",
    name: "Test Model",
    provider: "Test",
    parametersBillion: 8, // KV = 150 MB / 1k tokens
    contextWindow: 131072,
    categories: ["general"],
    qualityTier: "strong",
    qualityScore: 80,
    quantizations: [makeQuant()],
    ollamaTag: "test:8b",
    releaseDate: "2024-01",
    ...over,
  };
}

function gpuProfile(vramMb: number): HardwareProfile {
  const gpu = {
    vendor: "NVIDIA",
    model: "Test GPU",
    vramMb,
    driverVersion: "1",
    acceleratorVersion: "12",
    acceleratorType: "cuda" as const,
    utilizationPercent: null,
    temperatureCelsius: null,
    vramUsedMb: null,
  };
  return {
    cpu: {
      brand: "Test", manufacturer: "Test", cores: 8, threads: 16, speed: 3, speedMax: 4,
      architecture: "x64", flags: [], hasAvx2: true, performanceCores: null, efficiencyCores: null,
    },
    gpus: [gpu],
    memory: { totalMb: 65536, availableMb: 60000, usedMb: 5536, usedPercent: 8, type: "DDR5", speedMhz: 5600 },
    disk: { type: "NVMe", freeGb: 500, totalGb: 1000 },
    primaryGpu: gpu,
  };
}

describe("checkContextFit", () => {
  it("verdict yes, limited by hardware, with correct ceilings", () => {
    // Rule: need = weights + KV + compute against budget = vramMb - 457 - 1024 (memory-model).
    // budget 10240 - 1481 = 8759; weights 4096; KV 0.15 MiB/token; batch 1024 up to 32768 tokens -> compute 1126.4;
    // plus the KQ mask, 2 B x 1024 tokens = 0.001953125 MiB per context cell:
    // c <= (8759 - 4096 - 1126.4)/(0.15 + 0.001953125) = 23275 -> 23040 (x256)
    const r = checkContextFit(makeModel(), gpuProfile(10240), { promptTokens: 1000, responseTokens: 512 });
    expect(r.affordedByVramTokens).toBe(23040);
    expect(r.effectiveMaxTokens).toBe(23040);
    expect(r.limitedBy).toBe("hardware");
    expect(r.verdict).toBe("yes");
    expect(r.neededTokens).toBe(1512);
    expect(r.headroomTokens).toBe(21528);
    expect(r.remedy).toBeNull();
  });

  it("defaults responseTokens to 512", () => {
    const r = checkContextFit(makeModel(), gpuProfile(10240), { promptTokens: 1000 });
    expect(r.responseTokens).toBe(512);
    expect(r.neededTokens).toBe(1512);
  });

  it("verdict no, limited by hardware, recommends a smaller quant that fits", () => {
    // Two quants; on a 24576 MB GPU (budget 23095) both are excellent, so the sweet-spot pick is the
    // higher-retention Q8_0. Above 32768 tokens compute is 2252.8 (batch 2048) and each context cell adds a 2 x 2048 B mask
    // (0.00390625 MiB). Q8_0 (8000): c <= (23095-8000-2252.8)/0.15390625 = 83443 -> 83200; needed 108000 -> no.
    // Q4_K_M (4096): c <= (23095-4096-2252.8)/0.15390625 = 108806 -> 108800 >= 108000, so the remedy names it.
    const model = makeModel({
      quantizations: [
        makeQuant({ name: "Q4_K_M", bitsPerWeight: 4.83, vramMb: 4096, qualityRetention: 0.9 }),
        makeQuant({ name: "Q8_0", bitsPerWeight: 8, vramMb: 8000, qualityRetention: 0.99 }),
      ],
    });
    const r = checkContextFit(model, gpuProfile(24576), { promptTokens: 107500, responseTokens: 500 });
    expect(r.quant.name).toBe("Q8_0");
    expect(r.quantForced).toBe(false);
    expect(r.affordedByVramTokens).toBe(83200);
    expect(r.verdict).toBe("no");
    expect(r.limitedBy).toBe("hardware");
    expect(r.remedy).toContain("Q4_K_M");
  });

  it("verdict no, limited by model when the native window is the wall", () => {
    const model = makeModel({ contextWindow: 4096 });
    const r = checkContextFit(model, gpuProfile(24576), { promptTokens: 5000, responseTokens: 0 });
    expect(r.effectiveMaxTokens).toBe(4096);
    expect(r.limitedBy).toBe("model");
    expect(r.verdict).toBe("no");
    expect(r.remedy).toContain("native window");
  });

  it("verdict tight just under the effective ceiling", () => {
    const model = makeModel({ contextWindow: 4096 }); // effective = 4096
    const r = checkContextFit(model, gpuProfile(24576), { promptTokens: 3800, responseTokens: 0 });
    expect(r.verdict).toBe("tight"); // 3800 > 0.9*4096 (3686.4) and <= 4096
  });

  it("the response reserve can flip the verdict", () => {
    const model = makeModel({ contextWindow: 4096 });
    expect(checkContextFit(model, gpuProfile(24576), { promptTokens: 3000, responseTokens: 0 }).verdict).toBe("yes");
    expect(checkContextFit(model, gpuProfile(24576), { promptTokens: 3000, responseTokens: 1500 }).verdict).toBe("no");
  });

  it("honors a forced quant over the sweet-spot pick", () => {
    const small = makeQuant({ name: "Q4_K_M", bitsPerWeight: 4.83, vramMb: 4096, qualityRetention: 0.9 });
    const big = makeQuant({ name: "Q8_0", bitsPerWeight: 8, vramMb: 8000, qualityRetention: 0.99 });
    const model = makeModel({ quantizations: [small, big] });
    const r = checkContextFit(model, gpuProfile(24576), { promptTokens: 1000, quant: small });
    expect(r.quant.name).toBe("Q4_K_M");
    expect(r.quantForced).toBe(true);
  });

  it("reports 0 afforded and verdict no when the weights exceed VRAM", () => {
    const model = makeModel({ quantizations: [makeQuant({ vramMb: 16000 })] });
    const r = checkContextFit(model, gpuProfile(8192), { promptTokens: 1000, responseTokens: 512 });
    expect(r.affordedByVramTokens).toBe(0);
    expect(r.verdict).toBe("no");
    expect(r.limitedBy).toBe("hardware");
    expect(r.remedy).toBe("Q4_K_M weights alone exceed the memory pool — use a smaller quant or model.");
  });

  it("tells the caller to lower the response reserve when it alone fills the ceiling", () => {
    // afforded 23,040 tokens (see first test); a 40,000-token reserve leaves no room for any prompt
    const r = checkContextFit(makeModel(), gpuProfile(10240), { promptTokens: 100, responseTokens: 40000 });
    expect(r.verdict).toBe("no");
    expect(r.remedy).toBe("Response reserve (40,000 tokens) leaves no room in the 23,040-token ceiling — lower the response reserve.");
  });

  it("clamps the smaller-quant remedy to the native window, not raw VRAM capacity", () => {
    const model = makeModel({
      contextWindow: 32000,
      quantizations: [
        makeQuant({ name: "Q4_K_M", bitsPerWeight: 4.83, vramMb: 4000, qualityRetention: 0.9 }),
        makeQuant({ name: "Q8_0", bitsPerWeight: 8, vramMb: 8000, qualityRetention: 0.99 }),
      ],
    });
    // 12000 MB: budget 10519. Q8_0 (8000): c <= (10519-8000-1126.4)/0.151953125 = 9164 -> 8960 (sweet-spot pick, comfortable).
    // Q4_K_M (4000) fits 32768 (need 4000 + 4915.2 + 1126.4 + mask 64 = 10105.6) but not 33024 (4000 + 4953.6 + 2252.8 + 129
    // = 11335.4), so raw capacity 32768 clamps to the 32000 window.
    const r = checkContextFit(model, gpuProfile(12000), { promptTokens: 19500, responseTokens: 500 });
    expect(r.quant.name).toBe("Q8_0");        // sweet-spot pick
    expect(r.affordedByVramTokens).toBe(8960);
    expect(r.verdict).toBe("no");
    expect(r.limitedBy).toBe("hardware");
    expect(r.remedy).toContain("Q4_K_M");
    expect(r.remedy).toContain("32,000");     // window-clamped, not raw 32,768
    expect(r.remedy).not.toContain("32,768");
  });

  it("does not promise a smaller quant when the workload exceeds the native window", () => {
    const model = makeModel({
      contextWindow: 32000,
      quantizations: [
        makeQuant({ name: "Q4_K_M", bitsPerWeight: 4.83, vramMb: 4000, qualityRetention: 0.9 }),
        makeQuant({ name: "Q8_0", bitsPerWeight: 8, vramMb: 8000, qualityRetention: 0.99 }),
      ],
    });
    const r = checkContextFit(model, gpuProfile(12000), { promptTokens: 40000, responseTokens: 0 });
    expect(r.verdict).toBe("no");
    expect(r.limitedBy).toBe("hardware");
    expect(r.remedy).toContain("Trim");       // no quant can clear a 32k window for 40k tokens
    expect(r.remedy).not.toContain("Q4_K_M");
  });
});

describe("checkContextFit quantized KV remedy", () => {
  // 10240 MB GPU (budget 8759), 4096 MB weights, 8B model (150 MB/1k): f16 affords 23,040 (first test).
  // q8_0 (0.0797 MiB/token) affords 32,768: need(32768) = 4096 + 2611.2 + 1126.4 + mask 64 = 7897.6, need(33024) = 4096 + 2631.9 + 2252.8 + 129 = 9109.7 > budget.
  // q4_0 (0.0422 MiB/token + 0.0039 mask), compute 2252.8 above 32768: c <= (8759-4096-2252.8)/0.04609375 = 52284 -> 52224.
  const hw = gpuProfile(10240);

  it("names q8_0 with its ceiling when it is the smallest change that fits", () => {
    const r = checkContextFit(makeModel(), hw, { promptTokens: 30000, responseTokens: 500 });
    expect(r.verdict).toBe("no");
    expect(r.remedy).toContain("OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0");
    expect(r.remedy).toContain("32,768");
  });

  it("names q8_0 for a 'no' verdict it fits, even inside the tight band", () => {
    const r = checkContextFit(makeModel(), hw, { promptTokens: 32000, responseTokens: 500 });
    expect(r.verdict).toBe("no");
    expect(r.remedy).toContain("OLLAMA_KV_CACHE_TYPE=q8_0");
  });

  it("names q4_0 only when q8_0 is not enough", () => {
    const r = checkContextFit(makeModel(), hw, { promptTokens: 50000, responseTokens: 500 });
    expect(r.remedy).toContain("OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q4_0");
    expect(r.remedy).not.toContain("q8_0");
  });

  it("offers it for a tight verdict too", () => {
    const r = checkContextFit(makeModel(), hw, { promptTokens: 22000, responseTokens: 500 });
    expect(r.verdict).toBe("tight");
    expect(r.remedy).toContain("OLLAMA_KV_CACHE_TYPE=q8_0");
  });

  it("keeps the trim remedy when no KV type fits", () => {
    const r = checkContextFit(makeModel(), hw, { promptTokens: 130000, responseTokens: 500 });
    expect(r.remedy).not.toContain("OLLAMA_KV_CACHE_TYPE");
    expect(r.remedy).toContain("Trim the prompt");
  });
});

describe("estimated native window", () => {
  // budget 24576 - 1481 = 23095, quant 4096; an estimated window sets no ceiling below 262144 tokens:
  // c <= (23095 - 4096 - 2252.8)/(0.15 + 0.00390625) = 108806 -> 108800 at 150 MB / 1k
  const estimated = makeModel({ contextWindow: 32768, contextWindowEstimated: true });

  it("does not bind on the guessed window: the VRAM-afforded ceiling decides", () => {
    const r = checkContextFit(estimated, gpuProfile(24576), { promptTokens: 50000, responseTokens: 512 });
    expect(r.affordedByVramTokens).toBe(108800);
    expect(r.effectiveMaxTokens).toBe(108800);
    expect(r.limitedBy).toBe("hardware");
    expect(r.verdict).toBe("yes");
    expect(r.nativeWindowEstimated).toBe(true);
  });

  it("treats the same window as binding once it is not an estimate", () => {
    const r = checkContextFit(makeModel({ contextWindow: 32768 }), gpuProfile(24576), { promptTokens: 50000, responseTokens: 512 });
    expect(r.effectiveMaxTokens).toBe(32768);
    expect(r.limitedBy).toBe("model");
    expect(r.verdict).toBe("no");
    expect(r).not.toHaveProperty("nativeWindowEstimated");
  });

  it("names a KV cache type by the VRAM ceiling, not the guessed window", () => {
    // 10 GB: afforded 23040 at f16, 32768 at q8_0 (see the quantized KV block above).
    const r = checkContextFit(estimated, gpuProfile(10240), { promptTokens: 30000, responseTokens: 512 });
    expect(r.verdict).toBe("no");
    expect(r.remedy).toContain("OLLAMA_KV_CACHE_TYPE=q8_0");
    expect(r.remedy).toContain("32,768");
  });
});
