import { filterByCategory } from "../models/database.js";
import { CODER_SLUG_RE, TERNARY_QUANT_NAME } from "../models/library-entries.js";
import { scoreModel, isComfortable, assessModelFit } from "./scorer.js";
import { isUnifiedMemory } from "../core/unified-memory.js";
import type {
  HardwareProfile,
  ModelCategory,
  ModelEntry,
  ModelScore,
  QuantizationVariant,
  Recommendation,
} from "../core/types.js";

// Embedding models return vectors, not chat replies.
const EMBEDDING_RE = /embed/i;

// Natively ternary weights have no independent per-parameter quality measurement, so the quality formula cannot place them.
function isRecommendable(model: ModelEntry): boolean {
  if (EMBEDDING_RE.test(model.id) || EMBEDDING_RE.test(model.ollamaTag ?? "")) return false;
  return !model.quantizations.every((q) => q.name === TERNARY_QUANT_NAME);
}

// The quant that `ollama pull <tag>` fetches: the exact one (see QuantizationVariant.exact), else Q4_K_M,
// else the only ("default") quant of a library row.
function defaultQuant(model: ModelEntry): QuantizationVariant | undefined {
  return model.quantizations.find((q) => q.exact) ?? model.quantizations.find((q) => q.name === "Q4_K_M") ?? model.quantizations[0];
}

// Estimate: 32768 tokens is Ollama's default context for 23-47 GiB of VRAM (O server/routes.go:2580-2586: >= 47 GiB
// gives 262144, < 23 GiB gives 4096). Large models read "tight" only because of the bigger default; room for this much is usable.
const PRACTICAL_CONTEXT_TOKENS = 32768;

// Fit levels already encode context room, so rank by them: excellent/comfortable 0, tight 1, barely 2, cannot_run 3.
// A tight row that still holds PRACTICAL_CONTEXT_TOKENS ranks with the comfortable ones.
function rankOf(score: ModelScore, hardware: HardwareProfile): number {
  if (isComfortable(score.fitLevel)) return 0;
  switch (score.fitLevel) {
    case "tight": return assessModelFit(score.model, score.quantization, hardware).maxContext >= PRACTICAL_CONTEXT_TOKENS ? 0 : 1;
    case "barely": return 2;
    default: return 3;
  }
}

// One pick per model line: "qwen3" and "qwen3.8" share a family, a coder variant has its own.
function familyOf(model: ModelEntry): string {
  const slug = model.ollamaTag?.split(":")[0];
  const prefix = slug?.match(/^[a-z]+/)?.[0];
  if (!slug || !prefix) return model.id;
  return CODER_SLUG_RE.test(slug) ? `${prefix}-code` : prefix;
}

export function getRecommendations(
  hardware: HardwareProfile,
  options: {
    category?: ModelCategory | "all";
    top?: number;
    onlyFitting?: boolean;
  } = {},
): Recommendation[] {
  const { category = "all", top = 5, onlyFitting = false } = options;

  // CPU offload only helps when a discrete GPU can take part of the model.
  const canOffload = hardware.primaryGpu !== null && hardware.primaryGpu.vramMb > 0 && !isUnifiedMemory(hardware.primaryGpu);

  const ranked: { score: ModelScore; rank: number }[] = [];
  for (const model of filterByCategory(category)) {
    if (!isRecommendable(model)) continue;
    const quant = defaultQuant(model);
    if (!quant) continue;
    const score = scoreModel(model, quant, hardware, category);
    const rank = rankOf(score, hardware);
    if (rank === 3 && onlyFitting) continue;
    if (score.fitLevel === "barely" && !canOffload) continue;
    ranked.push({ score, rank });
  }

  ranked.sort(
    (a, b) =>
      a.rank - b.rank ||
      b.score.compositeScore - a.score.compositeScore ||
      b.score.model.qualityScore - a.score.model.qualityScore ||
      a.score.model.id.localeCompare(b.score.model.id),
  );

  const families = new Set<string>();
  const picks: ModelScore[] = [];
  for (const { score } of ranked) {
    const family = familyOf(score.model);
    if (families.has(family)) continue;
    families.add(family);
    picks.push(score);
  }

  return picks.slice(0, top).map((score, i) => ({
    rank: i + 1,
    score,
    pullCommand: score.model.ollamaTag ? `ollama pull ${score.model.ollamaTag}` : null,
  }));
}
