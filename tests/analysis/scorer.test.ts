import { describe, it, expect } from "vitest";
import { classifyFit, scoreModel, getAvailableVram, suggestGpuOffload, unifiedMemoryNote } from "../../src/analysis/scorer.js";
import { getRecommendations } from "../../src/analysis/recommender.js";
import type { GpuInfo, HardwareProfile, ModelEntry, QuantizationVariant } from "../../src/core/types.js";
import highEnd from "../fixtures/hardware-profiles/high-end-nvidia.json";
import cpuOnly from "../fixtures/hardware-profiles/cpu-only.json";
import appleM2 from "../fixtures/hardware-profiles/apple-m2.json";
import windowsNvidia from "../fixtures/hardware-profiles/windows-nvidia.json";

const dummyModel: ModelEntry = {
  id: "test-7b",
  name: "Test 7B",
  provider: "Test",
  parametersBillion: 7,
  contextWindow: 4096,
  categories: ["general"],
  qualityTier: "strong",
  qualityScore: 80,
  quantizations: [
    { name: "Q4_K_M", bitsPerWeight: 4.83, vramMb: 4500, qualityRetention: 0.92 },
    { name: "Q8_0", bitsPerWeight: 8.5, vramMb: 7800, qualityRetention: 0.99 },
  ],
  ollamaTag: "test:7b",
  releaseDate: "2024-01",
};

describe("classifyFit", () => {
  it("returns excellent when VRAM is 1.5x+ the requirement", () => {
    expect(classifyFit(9000, 4500)).toBe("excellent");
  });

  it("returns comfortable when VRAM is 1.15x-1.5x", () => {
    expect(classifyFit(5500, 4500)).toBe("comfortable");
  });

  it("returns tight when VRAM is 1.0x-1.15x", () => {
    expect(classifyFit(4600, 4500)).toBe("tight");
  });

  it("returns barely when VRAM is 0.75x-1.0x", () => {
    expect(classifyFit(3500, 4500)).toBe("barely");
  });

  it("returns cannot_run when VRAM is below 0.75x", () => {
    expect(classifyFit(2000, 4500)).toBe("cannot_run");
  });
});

describe("scoreModel", () => {
  it("scores higher for better fitting hardware", () => {
    const q4 = dummyModel.quantizations[0];
    const scoreHigh = scoreModel(dummyModel, q4, highEnd as HardwareProfile);
    const scoreCpu = scoreModel(dummyModel, q4, cpuOnly as HardwareProfile);

    expect(scoreHigh.fitLevel).toBe("excellent");
    expect(scoreHigh.compositeScore).toBeGreaterThan(0);
    // CPU-only uses RAM as VRAM proxy — 10000 MB available vs 4500 needed
    expect(scoreCpu.fitLevel).not.toBe("cannot_run");
  });

  it("returns cannot_run for models too large", () => {
    const tinyHardware: HardwareProfile = {
      ...(cpuOnly as HardwareProfile),
      memory: { ...cpuOnly.memory, availableMb: 1000 },
      primaryGpu: null,
    };
    const q8 = dummyModel.quantizations[1]; // 7800 MB needed
    const score = scoreModel(dummyModel, q8, tinyHardware);
    expect(score.fitLevel).toBe("cannot_run");
  });

  it("estimates speed based on params and fit", () => {
    const q4 = dummyModel.quantizations[0];
    const score = scoreModel(dummyModel, q4, highEnd as HardwareProfile);
    expect(score.speedEstimate).toBe("fast");
  });
});

describe("scoreModel — fixture profiles", () => {
  it("scores windows-nvidia RTX 4090 as excellent fit for 7B Q4", () => {
    const q4 = dummyModel.quantizations[0]; // 4500 MB needed
    const score = scoreModel(dummyModel, q4, windowsNvidia as HardwareProfile);
    // 24576 MB VRAM / 4500 MB needed = 5.46 ratio → excellent
    expect(score.fitLevel).toBe("excellent");
    expect(score.speedEstimate).toBe("fast");
  });

  it("budgets the Metal working set minus llama.cpp's fit margin on Apple Silicon", () => {
    // Rule: Apple budget = vramMb - 1024 (L common/common.h:509 fit margin; Metal free memory ~ working set before load),
    // no extra unified headroom. Fixture working set 10977 MiB -> budget 9953. Dummy model: window 4096 -> default context 4096;
    // need = 4500 + KV 150*4.096 (614.4) + compute 262144*512*4*1.1/2^20 (563.2) + KQ mask 2 B x 4096 x 512 (4) = 5681.6 <= 0.8 x 9953 = 7962.4.
    const q4 = dummyModel.quantizations[0];
    const score = scoreModel(dummyModel, q4, appleM2 as HardwareProfile);
    expect(score.fitLevel).toBe("excellent");
    expect(score.fitRatio).toBeCloseTo(9953 / 5681.6, 3);

    // 13000 MiB of weights: need 14181.6; 0.75 x 14181.6 = 10636 > 9953 -> cannot_run.
    const bigQuant: QuantizationVariant = {
      name: "Q8_0",
      bitsPerWeight: 8.5,
      vramMb: 13000,
      qualityRetention: 0.99,
    };
    const bigScore = scoreModel(dummyModel, bigQuant, appleM2 as HardwareProfile);
    expect(bigScore.fitLevel).toBe("cannot_run");
  });

  it("has no budget floor: a working set under the fit margin leaves nothing", () => {
    const withMetalVram = (vramMb: number): HardwareProfile => ({
      ...(appleM2 as HardwareProfile),
      primaryGpu: { ...(appleM2 as HardwareProfile).primaryGpu!, vramMb },
    });
    const tinyQuant: QuantizationVariant = {
      name: "Q4_K_M",
      bitsPerWeight: 4.83,
      vramMb: 1500, // a 2B-class model
      qualityRetention: 0.9,
    };
    // 5000 - 1024 = 3976; need 1500 + 614.4 + 563.2 + 4 = 2681.6 <= 0.8 x 3976 = 3180.8 -> excellent
    expect(scoreModel(dummyModel, tinyQuant, withMetalVram(5000)).fitLevel).toBe("excellent");
    // 1000 < 1024 fit margin -> budget max(0, -24) = 0 -> cannot_run
    expect(getAvailableVram(withMetalVram(1000))).toBe(0);
    expect(scoreModel(dummyModel, tinyQuant, withMetalVram(1000)).fitLevel).toBe("cannot_run");
  });
});

describe("getRecommendations", () => {
  it("returns top N recommendations sorted by score", () => {
    const recs = getRecommendations(highEnd as HardwareProfile, { top: 5 });
    expect(recs).toHaveLength(5);
    expect(recs[0].rank).toBe(1);
    expect(recs[4].rank).toBe(5);

    // Scores should be descending
    for (let i = 1; i < recs.length; i++) {
      expect(recs[i - 1].score.compositeScore).toBeGreaterThanOrEqual(
        recs[i].score.compositeScore,
      );
    }
  });

  it("filters by category", () => {
    const recs = getRecommendations(highEnd as HardwareProfile, {
      category: "coding",
      top: 10,
    });
    for (const rec of recs) {
      expect(rec.score.model.categories).toContain("coding");
    }
  });

  it("excludes models that cannot run when onlyFitting is true", () => {
    const recs = getRecommendations(cpuOnly as HardwareProfile, {
      onlyFitting: true,
      top: 50,
    });
    for (const rec of recs) {
      expect(rec.score.fitLevel).not.toBe("cannot_run");
    }
  });

  it("includes pull commands for models with ollama tags", () => {
    const recs = getRecommendations(highEnd as HardwareProfile, { top: 3 });
    for (const rec of recs) {
      if (rec.score.model.ollamaTag) {
        expect(rec.pullCommand).toContain("ollama pull");
      }
    }
  });
});

const unifiedBaseGpu: GpuInfo = {
  vendor: "NVIDIA",
  model: "NVIDIA GB10",
  vramMb: 100_000,
  driverVersion: "580.95.05",
  acceleratorVersion: "13.0",
  acceleratorType: "cuda",
  utilizationPercent: null,
  temperatureCelsius: null,
  vramUsedMb: null,
};

function profileWith(gpu: GpuInfo): HardwareProfile {
  return { ...(appleM2 as HardwareProfile), gpus: [gpu], primaryGpu: gpu };
}

describe("unified memory on a non-Metal GPU", () => {
  it("budgets available RAM, not the reported pool, on GB10-style unified memory", () => {
    // Fixture RAM available 10000 MiB: 10000 - 457 - 1024 = 8519 (L ggml-cuda.cu:6137-6145: UMA free = MemAvailable).
    const hw = profileWith({ ...unifiedBaseGpu, unifiedMemory: true });
    expect(getAvailableVram(hw)).toBe(10_000 - 457 - 1024);
    expect(getAvailableVram(profileWith(unifiedBaseGpu))).toBe(100_000 - 457 - 1024);
  });

  it("skips the CPU/GPU layer-offload suggestion", () => {
    const quant = { name: "Q4_K_M", bitsPerWeight: 4.8, vramMb: 40_000, qualityRetention: 0.9 };
    const model = {
      parametersBillion: 70,
      ollamaTag: "x:70b",
      quantizations: [quant],
    } as ModelEntry;
    expect(suggestGpuOffload(model, quant, profileWith({ ...unifiedBaseGpu, unifiedMemory: true }))).toBeNull();
    expect(suggestGpuOffload(model, quant, profileWith({ ...unifiedBaseGpu, vramMb: 8000 }))).not.toBeNull();
  });
});

describe("unifiedMemoryNote", () => {
  const note = (gpu: Partial<GpuInfo>) => unifiedMemoryNote(profileWith({ ...unifiedBaseGpu, unifiedMemory: true, ...gpu }));
  it("names the budget for each unified-memory kind and is null for discrete GPUs", () => {
    expect(note({ acceleratorType: "metal" })).toContain("Metal working set");
    expect(note({ vendor: "AMD", acceleratorType: "rocm", gfxTarget: "gfx1151" })).toBe("GPU pool = GTT");
    // An AMD APU Ollama drops is CPU-only: no unified-memory budget to explain.
    expect(note({ vendor: "AMD", acceleratorType: "rocm", gfxTarget: "gfx1103" })).toBeNull();
    expect(note({ vendor: "NVIDIA", acceleratorType: "cuda" })).toBe("unified memory — budget is available RAM");
    expect(note({ unifiedMemory: false, vendor: "NVIDIA", acceleratorType: "cuda" })).toBeNull();
  });
});
