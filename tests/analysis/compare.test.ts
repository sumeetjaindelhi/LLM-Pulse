import { describe, it, expect } from "vitest";
import { resolveModel } from "../../src/cli/commands/compare.js";
import { scoreModel } from "../../src/analysis/scorer.js";
import { getRecommendations } from "../../src/analysis/recommender.js";
import type { HardwareProfile, ModelEntry, ModelScore } from "../../src/core/types.js";
import highEnd from "../fixtures/hardware-profiles/high-end-nvidia.json";

const hw = highEnd as HardwareProfile;

describe("resolveModel", () => {
  it("resolves by exact ID", () => {
    const model = resolveModel("llama-3.1-8b");
    expect(model).not.toBeNull();
    expect(model!.id).toBe("llama-3.1-8b");
  });

  it("resolves by Ollama tag", () => {
    const model = resolveModel("llama3.1:8b");
    expect(model).not.toBeNull();
    expect(model!.ollamaTag).toBe("llama3.1:8b");
  });

  it("resolves by search when exactly 1 result", () => {
    // "deepseek-coder-v2-lite" is a unique enough name
    const model = resolveModel("deepseek-coder-v2-lite");
    expect(model).not.toBeNull();
    expect(model!.id).toBe("deepseek-coder-v2-lite");
  });

  it("returns null for unknown model", () => {
    const model = resolveModel("nonexistent-model-xyz");
    expect(model).toBeNull();
  });

  it("returns null when search yields multiple results", () => {
    // "llama" matches multiple models
    const model = resolveModel("llama");
    expect(model).toBeNull();
  });
});

describe("compare scoring", () => {
  it("picks correct winner between two models on high-end hardware", () => {
    const model1 = resolveModel("llama-3.1-8b")!;
    const model2 = resolveModel("deepseek-r1-7b")!;
    expect(model1).not.toBeNull();
    expect(model2).not.toBeNull();

    // Score best quant for each
    const score1 = bestScore(model1);
    const score2 = bestScore(model2);

    // Both should fit on 24 GB GPU
    expect(score1.fitLevel).not.toBe("cannot_run");
    expect(score2.fitLevel).not.toBe("cannot_run");

    // Winner has higher compositeScore
    const winner = score1.compositeScore >= score2.compositeScore ? score1 : score2;
    expect(winner.compositeScore).toBeGreaterThan(0);
  });

  it("handles category mode — picks top N from category", () => {
    const recs = getRecommendations(hw, {
      category: "coding",
      top: 3,
      onlyFitting: true,
    });
    expect(recs.length).toBeGreaterThanOrEqual(2);
    for (const rec of recs) {
      expect(rec.score.model.categories).toContain("coding");
      expect(rec.score.fitLevel).not.toBe("cannot_run");
    }
  });

  it("all models cannot_run produces no winner", () => {
    const tinyHw: HardwareProfile = {
      ...(highEnd as HardwareProfile),
      memory: { ...highEnd.memory, availableMb: 500 },
      primaryGpu: null,
    };
    const model1 = resolveModel("llama-3.1-70b")!;
    const model2 = resolveModel("deepseek-r1-32b")!;

    const s1 = scoreModel(model1, model1.quantizations[0], tinyHw);
    const s2 = scoreModel(model2, model2.quantizations[0], tinyHw);

    expect(s1.fitLevel).toBe("cannot_run");
    expect(s2.fitLevel).toBe("cannot_run");

    // Winner determination: no runnable model
    const scores = [s1, s2];
    const runnable = scores.filter((s) => s.fitLevel !== "cannot_run");
    expect(runnable).toHaveLength(0);
  });
});

function bestScore(model: ModelEntry): ModelScore {
  return model.quantizations
    .map((q) => scoreModel(model, q, hw))
    .sort((a, b) => b.compositeScore - a.compositeScore)[0];
}
