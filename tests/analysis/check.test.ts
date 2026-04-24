import { describe, it, expect } from "vitest";
import { scoreModel, deriveVerdict, suggestGpuOffload } from "../../src/analysis/scorer.js";
import { resolveModel, getModelById, searchModels } from "../../src/models/database.js";
import type { HardwareProfile, FitLevel } from "../../src/core/types.js";
import highEnd from "../fixtures/hardware-profiles/high-end-nvidia.json";
import cpuOnly from "../fixtures/hardware-profiles/cpu-only.json";
import appleM2 from "../fixtures/hardware-profiles/apple-m2.json";

describe("deriveVerdict", () => {
  it("returns 'yes' for excellent fit", () => {
    expect(deriveVerdict("excellent")).toBe("yes");
  });

  it("returns 'yes' for comfortable fit", () => {
    expect(deriveVerdict("comfortable")).toBe("yes");
  });

  it("returns 'maybe' for tight fit", () => {
    expect(deriveVerdict("tight")).toBe("maybe");
  });

  it("returns 'maybe' for barely fit", () => {
    expect(deriveVerdict("barely")).toBe("maybe");
  });

  it("returns 'no' for cannot_run", () => {
    expect(deriveVerdict("cannot_run")).toBe("no");
  });
});

describe("check command logic", () => {
  it("all quants fit on high-end hardware for 8B model", () => {
    const model = getModelById("llama-3.1-8b");
    expect(model).toBeDefined();

    const scores = model!.quantizations.map((q) =>
      scoreModel(model!, q, highEnd as HardwareProfile),
    );

    for (const s of scores) {
      expect(s.fitLevel).not.toBe("cannot_run");
      expect(deriveVerdict(s.fitLevel)).not.toBe("no");
    }
  });

  it("large model cannot run on cpu-only hardware", () => {
    const model = getModelById("llama-3.1-70b");
    expect(model).toBeDefined();

    // Even the smallest quant of 70B needs ~40GB — cpu-only has 10GB available RAM
    const scores = model!.quantizations.map((q) =>
      scoreModel(model!, q, cpuOnly as HardwareProfile),
    );

    const best = scores.sort((a, b) => b.compositeScore - a.compositeScore)[0];
    expect(deriveVerdict(best.fitLevel)).toBe("no");
  });

  it("selects best quantization by composite score", () => {
    const model = getModelById("llama-3.1-8b")!;
    const scores = model.quantizations
      .map((q) => scoreModel(model, q, highEnd as HardwareProfile))
      .sort((a, b) => b.compositeScore - a.compositeScore);

    // Best score should be first after sorting
    for (let i = 1; i < scores.length; i++) {
      expect(scores[0].compositeScore).toBeGreaterThanOrEqual(scores[i].compositeScore);
    }
  });

  it("generates pull command for models with ollama tag", () => {
    const model = getModelById("llama-3.1-8b");
    expect(model).toBeDefined();
    expect(model!.ollamaTag).toBeTruthy();

    const pullCommand = model!.ollamaTag ? `ollama pull ${model!.ollamaTag}` : null;
    expect(pullCommand).toContain("ollama pull");
  });

  it("returns null pull command for models without ollama tag", () => {
    const model = getModelById("gpt-4o");
    if (model) {
      // GPT-4o is not an Ollama model — should have no tag
      expect(model.ollamaTag).toBeNull();
    }
  });
});

describe("suggestGpuOffload", () => {
  it("returns full_fit for a model that fits entirely on a high-end GPU", () => {
    const model = getModelById("llama-3.1-8b")!;
    const quant = model.quantizations[0]; // Q4_K_M, 5000 MB
    const suggestion = suggestGpuOffload(model, quant, highEnd as HardwareProfile);

    expect(suggestion).not.toBeNull();
    expect(suggestion!.reason).toBe("full_fit");
    expect(suggestion!.gpuLayers).toBe("all");
    expect(suggestion!.ollamaCommand).toBe("ollama run llama3.1:8b");
  });

  it("returns partial_offload when a large model overflows a mid-VRAM GPU", () => {
    const model = getModelById("llama-3.1-70b")!;
    const q4 = model.quantizations[0]; // Q4_K_M, 40000 MB — larger than 24 GB RTX 4090
    const suggestion = suggestGpuOffload(model, q4, highEnd as HardwareProfile);

    expect(suggestion).not.toBeNull();
    expect(suggestion!.reason).toBe("partial_offload");
    expect(typeof suggestion!.gpuLayers).toBe("number");
    expect(suggestion!.gpuLayers).toBeGreaterThan(0);
    expect(suggestion!.gpuLayers).toBeLessThan(suggestion!.totalLayers);
    // Recommendation must leave headroom — estimated VRAM used stays below the 24 GB ceiling.
    expect(suggestion!.estimatedVramUsedMb).toBeLessThanOrEqual(24576);
    expect(suggestion!.ollamaCommand).toContain("/set parameter num_gpu");
  });

  it("returns null on Apple Silicon (unified memory — no offload split)", () => {
    const model = getModelById("llama-3.1-8b")!;
    const quant = model.quantizations[0];
    const suggestion = suggestGpuOffload(model, quant, appleM2 as HardwareProfile);
    expect(suggestion).toBeNull();
  });

  it("returns null on CPU-only systems (no GPU to offload to)", () => {
    const model = getModelById("llama-3.1-8b")!;
    const quant = model.quantizations[0];
    const suggestion = suggestGpuOffload(model, quant, cpuOnly as HardwareProfile);
    expect(suggestion).toBeNull();
  });

  it("estimates more layers for a smaller model at the same VRAM budget", () => {
    const big = getModelById("llama-3.1-70b")!;
    const small = getModelById("llama-3.2-3b")!;
    const bigQ = big.quantizations[0];
    const smallQ = small.quantizations[0];

    const bigSuggestion = suggestGpuOffload(big, bigQ, highEnd as HardwareProfile);
    const smallSuggestion = suggestGpuOffload(small, smallQ, highEnd as HardwareProfile);

    // 70B Q4 overflows and gets a partial count; 3B fits entirely.
    expect(bigSuggestion!.reason).toBe("partial_offload");
    expect(smallSuggestion!.reason).toBe("full_fit");
  });
});

describe("model resolution", () => {
  it("resolves by exact id", () => {
    const model = resolveModel("llama-3.1-8b");
    expect(model).toBeDefined();
    expect(model!.id).toBe("llama-3.1-8b");
  });

  it("resolves by ollama tag", () => {
    const model = resolveModel("llama3.1:8b");
    expect(model).toBeDefined();
    expect(model!.ollamaTag).toBe("llama3.1:8b");
  });

  it("returns null for nonexistent model", () => {
    const model = resolveModel("nonexistent-model-xyz-123");
    expect(model).toBeNull();
  });

  it("provides suggestions for fuzzy matches", () => {
    const results = searchModels("llama");
    expect(results.length).toBeGreaterThan(0);
    for (const r of results) {
      expect(r.name.toLowerCase() + r.id.toLowerCase() + r.provider.toLowerCase()).toContain("llama");
    }
  });
});
