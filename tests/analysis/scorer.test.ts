import { describe, it, expect } from "vitest";
import { classifyFit, scoreModel } from "../../src/analysis/scorer.js";
import { getRecommendations } from "../../src/analysis/recommender.js";
import type { HardwareProfile, ModelEntry, QuantizationVariant } from "../../src/core/types.js";
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

  it("applies unified-memory headroom on Apple Silicon", () => {
    // As of v0.9.1, the scorer subtracts UNIFIED_MEMORY_HEADROOM_MB (6144)
    // from the wired-limit cap on Metal GPUs, because that cap is a kernel
    // ceiling and doesn't reserve anything for the OS/editor/Node sharing
    // the same RAM pool. Fixture wiredCap = 10977 MB (16 GB M2 Pro) →
    // practical VRAM = 10977 - 6144 = 4833 MB.
    const q4 = dummyModel.quantizations[0]; // 4500 MB needed
    const score = scoreModel(dummyModel, q4, appleM2 as HardwareProfile);

    // 4833 / 4500 = 1.07 → tight (previously excellent at 2.44x, which was
    // unrealistic for a machine actively being used for dev work).
    expect(score.fitLevel).toBe("tight");

    // A quantization that exceeded the raw cap already fails hard under
    // the practical cap: 4833 / 13000 = 0.37 → cannot_run.
    const bigQuant: QuantizationVariant = {
      name: "Q8_0",
      bitsPerWeight: 8.5,
      vramMb: 13000,
      qualityRetention: 0.99,
    };
    const bigScore = scoreModel(dummyModel, bigQuant, appleM2 as HardwareProfile);
    expect(bigScore.fitLevel).toBe("cannot_run");
  });

  it("leaves a 2 GB floor when headroom would exceed total VRAM", () => {
    // On a small Apple system (e.g. 8 GB M1 with wiredCap ≈ 5 GB), the
    // 6 GB headroom would go negative. The floor should keep it at 2 GB so
    // tiny models still register as runnable rather than universal
    // cannot_run.
    const tinyApple: HardwareProfile = {
      ...(appleM2 as HardwareProfile),
      primaryGpu: {
        ...(appleM2 as HardwareProfile).primaryGpu!,
        vramMb: 5000, // below the 6144 headroom
      },
    };
    const tinyQuant: QuantizationVariant = {
      name: "Q4_K_M",
      bitsPerWeight: 4.83,
      vramMb: 1500, // a 2B-class model
      qualityRetention: 0.9,
    };
    const score = scoreModel(dummyModel, tinyQuant, tinyApple);
    // floor is 2048 MB → 2048 / 1500 = 1.37 → comfortable
    expect(score.fitLevel).toBe("comfortable");
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
