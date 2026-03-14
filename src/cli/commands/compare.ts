import ora from "ora";
import { detectHardware } from "../../hardware/index.js";
import { scoreModel } from "../../analysis/scorer.js";
import { getRecommendations } from "../../analysis/recommender.js";
import { getModelById, getModelByTag, searchModels } from "../../models/database.js";
import { comparisonTable } from "../ui/tables.js";
import { sectionHeader, titleBox } from "../ui/boxes.js";
import { theme } from "../ui/colors.js";
import { APPLE_UNIFIED_MEMORY_FACTOR } from "../../core/constants.js";
import type {
  ModelEntry,
  ModelScore,
  ModelCategory,
  HardwareProfile,
  OutputFormat,
} from "../../core/types.js";

export interface CompareOptions {
  format: OutputFormat;
  category: ModelCategory | "all";
  top: number;
  quant?: string;
}

export function resolveModel(query: string): ModelEntry | null {
  // 1. Exact ID match
  const byId = getModelById(query);
  if (byId) return byId;

  // 2. Exact Ollama tag match
  const byTag = getModelByTag(query);
  if (byTag) return byTag;

  // 3. Search — only if exactly 1 result
  const results = searchModels(query);
  if (results.length === 1) return results[0];

  return null;
}

function pickBestScore(
  model: ModelEntry,
  hardware: HardwareProfile,
  category: ModelCategory | "all",
  forcedQuant?: string,
): ModelScore {
  if (forcedQuant) {
    const match = model.quantizations.find((q) => q.name === forcedQuant);
    if (match) return scoreModel(model, match, hardware, category);
    // Fall through to best if forced quant not found
  }

  // Score all quantizations, pick highest compositeScore
  const scores = model.quantizations
    .map((q) => scoreModel(model, q, hardware, category))
    .sort((a, b) => b.compositeScore - a.compositeScore);

  return scores[0];
}

function getAvailableVram(hardware: HardwareProfile): number {
  if (hardware.primaryGpu) {
    return hardware.primaryGpu.vendor === "Apple"
      ? Math.round(hardware.primaryGpu.vramMb * APPLE_UNIFIED_MEMORY_FACTOR)
      : hardware.primaryGpu.vramMb;
  }
  return hardware.memory.availableMb;
}

function findWinner(scores: ModelScore[]): number {
  const runnable = scores.filter((s) => s.fitLevel !== "cannot_run");
  if (runnable.length === 0) return -1;

  let bestIdx = -1;
  let bestScore = -1;
  let bestFit = -1;
  for (let i = 0; i < scores.length; i++) {
    if (scores[i].fitLevel === "cannot_run") continue;
    const cs = scores[i].compositeScore;
    const fr = scores[i].fitRatio;
    if (cs > bestScore || (cs === bestScore && fr > bestFit)) {
      bestIdx = i;
      bestScore = cs;
      bestFit = fr;
    }
  }
  return bestIdx;
}

export async function compareCommand(
  modelArgs: string[],
  options: CompareOptions,
): Promise<void> {
  const isJson = options.format === "json";
  const spinner = isJson ? null : ora({ text: "Detecting hardware...", color: "cyan" }).start();

  const hardware = await detectHardware();
  spinner?.succeed("Hardware detected");

  let scores: ModelScore[];

  if (modelArgs.length > 0) {
    // Explicit mode: resolve each model arg
    scores = resolveExplicitModels(modelArgs, hardware, options, spinner);
  } else {
    // Category mode: auto-pick top N
    scores = resolveCategoryModels(hardware, options, spinner);
  }

  if (scores.length < 2) {
    if (!isJson) {
      console.log(`\n  ${theme.warning("Need at least 2 models to compare.")}`);
      console.log(`  ${theme.muted("Usage: llm-pulse compare <model1> <model2> [model3...]")}`);
    } else {
      console.log(JSON.stringify({ error: "Need at least 2 models to compare" }));
    }
    return;
  }

  const winnerIdx = findWinner(scores);
  const availableVram = getAvailableVram(hardware);

  if (isJson) {
    outputJson(scores, winnerIdx, availableVram);
  } else {
    outputTable(scores, winnerIdx, availableVram);
  }
}

function resolveExplicitModels(
  args: string[],
  hardware: HardwareProfile,
  options: CompareOptions,
  spinner: ReturnType<typeof ora> | null,
): ModelScore[] {
  const scores: ModelScore[] = [];

  for (const arg of args) {
    const model = resolveModel(arg);
    if (!model) {
      spinner?.stop();
      console.log(`\n  ${theme.fail("✗")} Model not found: ${theme.value(arg)}`);
      const suggestions = searchModels(arg).slice(0, 3);
      if (suggestions.length > 0) {
        console.log(`  ${theme.muted("Did you mean:")}`);
        for (const s of suggestions) {
          console.log(`    ${theme.muted("•")} ${s.name} ${theme.muted(`(${s.id})`)}`);
        }
      }
      continue;
    }

    const forcedQuant = options.quant;
    if (forcedQuant && !model.quantizations.some((q) => q.name === forcedQuant)) {
      spinner?.stop();
      console.log(`  ${theme.warning("⚠")} ${model.name}: quant ${forcedQuant} not available, using best fit`);
    }

    scores.push(pickBestScore(model, hardware, options.category, options.quant));
  }

  return scores;
}

function resolveCategoryModels(
  hardware: HardwareProfile,
  options: CompareOptions,
  spinner: ReturnType<typeof ora> | null,
): ModelScore[] {
  const recs = getRecommendations(hardware, {
    category: options.category,
    top: options.top,
    onlyFitting: true,
  });

  if (recs.length < 2) {
    spinner?.stop();
    console.log(`\n  ${theme.warning("Not enough models fit your hardware for this category.")}`);
    return [];
  }

  return recs.map((r) => r.score);
}

function outputTable(scores: ModelScore[], winnerIdx: number, availableVram: number): void {
  const lines: string[] = [];
  lines.push(sectionHeader("Model Comparison"));
  lines.push("");
  lines.push(comparisonTable(scores, availableVram, winnerIdx));

  if (winnerIdx >= 0) {
    const winner = scores[winnerIdx];
    lines.push("");
    lines.push(`  ${theme.pass("★")} Winner: ${theme.pass(winner.model.name)} — score ${theme.number(String(winner.compositeScore))}`);
    if (winner.model.ollamaTag) {
      lines.push(`  Run: ${theme.command(`ollama pull ${winner.model.ollamaTag}`)}`);
    }
  } else {
    lines.push("");
    lines.push(`  ${theme.warning("No models can run on your hardware.")}`);
  }

  console.log(titleBox(lines.join("\n")));
}

function outputJson(scores: ModelScore[], winnerIdx: number, availableVram: number): void {
  const output = {
    models: scores.map((s) => ({
      id: s.model.id,
      name: s.model.name,
      provider: s.model.provider,
      parametersBillion: s.model.parametersBillion,
      contextWindow: s.model.contextWindow,
      qualityTier: s.model.qualityTier,
      qualityScore: s.model.qualityScore,
      quantization: s.quantization.name,
      vramMb: s.quantization.vramMb,
      availableVramMb: availableVram,
      fitLevel: s.fitLevel,
      compositeScore: s.compositeScore,
      speedEstimate: s.speedEstimate,
    })),
    winner: winnerIdx >= 0 ? scores[winnerIdx].model.id : null,
  };

  console.log(JSON.stringify(output, null, 2));
}
