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

  it("scores windows-nvidia RTX 4090 as excellent fit for 7B Q4", () => {
    const q4 = dummyModel.quantizations[0]; // 4500 MB needed
    const score = scoreModel(dummyModel, q4, windowsNvidia as HardwareProfile);
    // 24576 MB VRAM / 4500 MB needed = 5.46 ratio → excellent
    expect(score.fitLevel).toBe("excellent");
    expect(score.speedEstimate).toBe("fast");
  });

  it("treats Apple Silicon vramMb as the resolved wired-memory cap", () => {
    // As of v0.9.0, `primaryGpu.vramMb` is the already-resolved usable VRAM
    // (sysctl iogpu.wired_limit_mb, or 67% of RAM as fallback). No factor is
    // applied by the scorer — it trusts the detector. Fixture sets 10977 MB
    // (67% of a 16 GB M2 Pro, matching macOS's default wired limit).
    const q4 = dummyModel.quantizations[0]; // 4500 MB needed
    const score = scoreModel(dummyModel, q4, appleM2 as HardwareProfile);

    // 10977 / 4500 = 2.44 → excellent (>= 1.5x headroom)
    expect(score.fitLevel).toBe("excellent");

    // A quantization that barely fits after cap
    const bigQuant: QuantizationVariant = {
      name: "Q8_0",
      bitsPerWeight: 8.5,
      vramMb: 13000, // 10977 / 13000 = 0.844 → barely (>= 0.75, < 1.0)
      qualityRetention: 0.99,
    };
    const bigScore = scoreModel(dummyModel, bigQuant, appleM2 as HardwareProfile);
    expect(bigScore.fitLevel).toBe("barely");
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
