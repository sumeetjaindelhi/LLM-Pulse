import ora from "ora";
import { detectHardware } from "../../hardware/index.js";
import { scoreModel, getAvailableVram, headlineScore, isFitting } from "../../analysis/scorer.js";
import { getRecommendations } from "../../analysis/recommender.js";
import { findQuant, resolveModel } from "../../models/database.js";
import { withExactFacts } from "../../runtimes/ollama.js";
import { resolveOllamaHost } from "../../core/config.js";
import { comparisonTable } from "../ui/tables.js";
import { sectionHeader, titleBox } from "../ui/boxes.js";
import { pullLines } from "../ui/progress.js";
import { theme } from "../ui/colors.js";
import { renderModelNotFound } from "../ui/errors.js";
import { MEMORY_CSV_HEADERS, memoryCsvCells, toCsv } from "../ui/csv.js";
import { memoryBreakdown } from "../../analysis/context-capacity.js";
import type {
  ModelEntry,
  ModelScore,
  ModelCategory,
  HardwareProfile,
  CompareOptions,
  QuantizationVariant,
} from "../../core/types.js";

function pickBestScore(
  model: ModelEntry,
  hardware: HardwareProfile,
  category: ModelCategory | "all",
  forcedQuant?: QuantizationVariant,
): ModelScore {
  if (forcedQuant) return scoreModel(model, forcedQuant, hardware, category);
  return headlineScore(model.quantizations.map((q) => scoreModel(model, q, hardware, category)));
}

function findWinner(scores: ModelScore[]): number {
  let bestIdx = -1;
  let bestScore = -1;
  let bestFit = -1;
  for (let i = 0; i < scores.length; i++) {
    if (!isFitting(scores[i].fitLevel)) continue;
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
  const isCsv = options.format === "csv";
  const silent = isJson || isCsv;
  const spinner = silent ? null : ora({ text: "Detecting hardware...", color: "cyan" }).start();

  const hardware = await detectHardware();
  spinner?.succeed("Hardware detected");

  let scores: ModelScore[];

  if (modelArgs.length > 0) {
    // Explicit mode: resolve each model arg
    scores = await resolveExplicitModels(modelArgs, hardware, options, spinner, silent);
  } else {
    // Category mode: auto-pick top N
    scores = resolveCategoryModels(hardware, options, spinner, silent);
  }

  if (scores.length < 2) {
    process.exitCode = 1;
    if (!silent) {
      console.log(`\n  ${theme.warning("Need at least 2 models to compare.")}`);
      console.log(`  ${theme.muted("Usage: llm-pulse compare <model1> <model2> [model3...]")}`);
    } else if (isJson) {
      console.log(JSON.stringify({ error: "Need at least 2 models to compare" }));
    } else {
      console.error("Need at least 2 models to compare");
    }
    return;
  }

  const winnerIdx = findWinner(scores);
  const availableVram = getAvailableVram(hardware);

  if (isJson) {
    outputJson(scores, winnerIdx, availableVram, hardware);
  } else if (isCsv) {
    outputCsv(scores, winnerIdx, availableVram, hardware);
  } else {
    outputTable(scores, winnerIdx, hardware);
  }
}

async function resolveExplicitModels(
  args: string[],
  hardware: HardwareProfile,
  options: CompareOptions,
  spinner: ReturnType<typeof ora> | null,
  silent: boolean,
): Promise<ModelScore[]> {
  const scores: ModelScore[] = [];
  // json/csv stdout must carry only the payload, so diagnostics go to stderr there.
  const log = silent ? console.error : console.log;
  const host = resolveOllamaHost(options.host);

  spinner?.start("Reading model facts...");
  const models = await Promise.all(args.map(async (arg) => {
    const model = resolveModel(arg);
    return model && withExactFacts(host, model);
  }));
  spinner?.stop();

  for (const [i, arg] of args.entries()) {
    const model = models[i];
    if (!model) {
      if (silent) {
        process.exitCode = 1;
        log(`Model not found: ${arg}`);
      } else {
        renderModelNotFound(arg, { suggestionsLimit: 3, identityFormat: "id", showBrowseHint: false });
      }
      continue;
    }

    const forcedQuant = options.quant ? findQuant(model, options.quant) : undefined;
    if (options.quant && !forcedQuant) {
      log(`  ${theme.warning("⚠")} ${model.name}: quant ${options.quant} not available, using best fit`);
    }

    scores.push(pickBestScore(model, hardware, options.category, forcedQuant));
  }

  return scores;
}

function resolveCategoryModels(
  hardware: HardwareProfile,
  options: CompareOptions,
  spinner: ReturnType<typeof ora> | null,
  silent: boolean,
): ModelScore[] {
  const recs = getRecommendations(hardware, {
    category: options.category,
    top: options.top,
    onlyFitting: true,
  });

  if (recs.length < 2) {
    spinner?.stop();
    (silent ? console.error : console.log)(`\n  ${theme.warning("Not enough models fit your hardware for this category.")}`);
    return [];
  }

  return recs.map((r) => r.score);
}

function outputTable(scores: ModelScore[], winnerIdx: number, hardware: HardwareProfile): void {
  const lines: string[] = [];
  lines.push(sectionHeader("Model Comparison"));
  lines.push("");
  lines.push(comparisonTable(scores, hardware, winnerIdx));

  if (winnerIdx >= 0) {
    const winner = scores[winnerIdx];
    lines.push("");
    lines.push(`  ${theme.pass("★")} Winner: ${theme.pass(winner.model.name)} — score ${theme.number(String(winner.compositeScore))}`);
    lines.push(...pullLines(winner.model, winner.quantization));
  } else {
    lines.push("");
    lines.push(`  ${theme.warning("No models can run on your hardware.")}`);
  }

  console.log(titleBox(lines.join("\n")));
}

function outputJson(scores: ModelScore[], winnerIdx: number, availableVram: number, hardware: HardwareProfile): void {
  const output = {
    models: scores.map((s) => ({
      id: s.model.id,
      name: s.model.name,
      provider: s.model.provider,
      parametersBillion: s.model.parametersBillion,
      contextWindow: s.model.contextWindow,
      contextWindowEstimated: s.model.contextWindowEstimated === true,
      qualityTier: s.model.qualityTier,
      qualityScore: s.model.qualityScore,
      quantization: s.quantization.name,
      vramMb: s.quantization.vramMb,
      availableVramMb: availableVram,
      fitLevel: s.fitLevel,
      compositeScore: s.compositeScore,
      speedEstimate: s.speedEstimate,
      memory: memoryBreakdown(s.model, s.quantization, hardware),
    })),
    winner: winnerIdx >= 0 ? scores[winnerIdx].model.id : null,
  };

  console.log(JSON.stringify(output, null, 2));
}

function outputCsv(scores: ModelScore[], winnerIdx: number, availableVram: number, hardware: HardwareProfile): void {
  const headers = [
    "id", "name", "provider", "parametersBillion", "contextWindow",
    "qualityTier", "quantization", "vramMb", "availableVramMb",
    "fitLevel", "compositeScore", "speedEstimate", "isWinner",
    ...MEMORY_CSV_HEADERS, "contextWindowEstimated",
  ];
  const rows = scores.map((s, i) => [
    s.model.id, s.model.name, s.model.provider, s.model.parametersBillion,
    s.model.contextWindow, s.model.qualityTier, s.quantization.name,
    s.quantization.vramMb, availableVram, s.fitLevel, s.compositeScore,
    s.speedEstimate, i === winnerIdx,
    ...memoryCsvCells(memoryBreakdown(s.model, s.quantization, hardware)),
    s.model.contextWindowEstimated === true,
  ]);
  console.log(toCsv(headers, rows));
}
