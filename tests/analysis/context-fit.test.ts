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
    // budget 10240, quant 4096, reserve max(768,512)=768 -> kvBudget 5376 -> afforded 35840
    const r = checkContextFit(makeModel(), gpuProfile(10240), { promptTokens: 1000, responseTokens: 512 });
    expect(r.affordedByVramTokens).toBe(35840);
    expect(r.effectiveMaxTokens).toBe(35840);
    expect(r.limitedBy).toBe("hardware");
    expect(r.verdict).toBe("yes");
    expect(r.neededTokens).toBe(1512);
    expect(r.headroomTokens).toBe(34328);
    expect(r.remedy).toBeNull();
  });

  it("defaults responseTokens to 512", () => {
    const r = checkContextFit(makeModel(), gpuProfile(10240), { promptTokens: 1000 });
    expect(r.responseTokens).toBe(512);
    expect(r.neededTokens).toBe(1512);
  });

  it("verdict no, limited by hardware, recommends a smaller quant that fits", () => {
    // Two quants; on a 24576 MB GPU both are excellent, so the sweet-spot pick is
    // the higher-retention Q8_0. Q8_0 affords 102314; needed 110000 -> no. Q4_K_M
    // affords 128341 >= 110000, so the remedy names it.
    const model = makeModel({
      quantizations: [
        makeQuant({ name: "Q4_K_M", bitsPerWeight: 4.83, vramMb: 4096, qualityRetention: 0.9 }),
        makeQuant({ name: "Q8_0", bitsPerWeight: 8, vramMb: 8000, qualityRetention: 0.99 }),
      ],
    });
    const r = checkContextFit(model, gpuProfile(24576), { promptTokens: 109500, responseTokens: 500 });
    expect(r.quant.name).toBe("Q8_0");
    expect(r.quantForced).toBe(false);
    expect(r.affordedByVramTokens).toBe(102314);
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
  });

  it("clamps the smaller-quant remedy to the native window, not raw VRAM capacity", () => {
    const model = makeModel({
      contextWindow: 32000,
      quantizations: [
        makeQuant({ name: "Q4_K_M", bitsPerWeight: 4.83, vramMb: 4000, qualityRetention: 0.9 }),
        makeQuant({ name: "Q8_0", bitsPerWeight: 8, vramMb: 8000, qualityRetention: 0.99 }),
      ],
    });
    const r = checkContextFit(model, gpuProfile(11000), { promptTokens: 19500, responseTokens: 500 });
    expect(r.quant.name).toBe("Q8_0");        // sweet-spot pick
    expect(r.affordedByVramTokens).toBe(14880);
    expect(r.verdict).toBe("no");
    expect(r.limitedBy).toBe("hardware");
    expect(r.remedy).toContain("Q4_K_M");
    expect(r.remedy).toContain("32,000");     // window-clamped, not raw 41,546
    expect(r.remedy).not.toContain("41,546");
  });

  it("does not promise a smaller quant when the workload exceeds the native window", () => {
    const model = makeModel({
      contextWindow: 32000,
      quantizations: [
        makeQuant({ name: "Q4_K_M", bitsPerWeight: 4.83, vramMb: 4000, qualityRetention: 0.9 }),
        makeQuant({ name: "Q8_0", bitsPerWeight: 8, vramMb: 8000, qualityRetention: 0.99 }),
      ],
    });
    const r = checkContextFit(model, gpuProfile(11000), { promptTokens: 40000, responseTokens: 0 });
    expect(r.verdict).toBe("no");
    expect(r.limitedBy).toBe("hardware");
    expect(r.remedy).toContain("Trim");       // no quant can clear a 32k window for 40k tokens
    expect(r.remedy).not.toContain("Q4_K_M");
  });
});
