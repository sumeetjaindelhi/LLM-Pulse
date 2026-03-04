import { getAllModels, filterByCategory } from "../models/database.js";
import { scoreModel } from "./scorer.js";
import type {
  HardwareProfile,
  ModelCategory,
  ModelScore,
  Recommendation,
} from "../core/types.js";

export function getRecommendations(
  hardware: HardwareProfile,
  options: {
    category?: ModelCategory | "all";
    top?: number;
    onlyFitting?: boolean;
  } = {},
): Recommendation[] {
  const { category = "all", top = 5, onlyFitting = false } = options;

  const models = category === "all" ? getAllModels() : filterByCategory(category);

  // Score every model+quantization combination
  const allScores: ModelScore[] = [];
  for (const model of models) {
    // Pick the best quantization that fits (prefer higher quality)
    const quantScores = model.quantizations
      .map((q) => scoreModel(model, q, hardware, category))
      .sort((a, b) => b.compositeScore - a.compositeScore);

    // Take only the best quantization per model
    if (quantScores.length > 0) {
      allScores.push(quantScores[0]);
    }
  }

  // Filter out models that can't run if requested
  const filtered = onlyFitting
    ? allScores.filter((s) => s.fitLevel !== "cannot_run")
    : allScores;

  // Sort by composite score descending, then by fit ratio
  const sorted = filtered.sort((a, b) => {
    if (a.fitLevel === "cannot_run" && b.fitLevel !== "cannot_run") return 1;
    if (b.fitLevel === "cannot_run" && a.fitLevel !== "cannot_run") return -1;
    return b.compositeScore - a.compositeScore;
  });

  return sorted.slice(0, top).map((score, i) => ({
    rank: i + 1,
    score,
    pullCommand: score.model.ollamaTag ? `ollama pull ${score.model.ollamaTag}` : null,
  }));
}
