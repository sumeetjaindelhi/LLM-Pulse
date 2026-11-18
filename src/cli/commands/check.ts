import ora from "ora";
import Table from "cli-table3";
import { detectHardware } from "../../hardware/index.js";
import { budgetExact, budgetGpus, scoreModel, deriveVerdict, getAvailableVram, headlineScore, suggestGpuOffload, isFitting, defaultContextFor } from "../../analysis/scorer.js";
import { memoryBreakdown } from "../../analysis/context-capacity.js";
import { findQuant, resolveModel } from "../../models/database.js";
import { withExactFacts } from "../../runtimes/ollama.js";
import { resolveOllamaHost } from "../../core/config.js";
import { titleBox, sectionHeader, keyValue, subLine } from "../ui/boxes.js";
import { theme } from "../ui/colors.js";
import { fitBadge } from "../ui/badges.js";
import { formatBreakdown, gpuRows, formatMb, formatNeed, formatNeedAt, formatUsable, pullLines } from "../ui/progress.js";
import { borderlessTableChars, speedLabel, NEED_NOTE, SPEED_NOTE } from "../ui/tables.js";
import { renderModelNotFound } from "../ui/errors.js";
import { MEMORY_CSV_HEADERS, memoryCsvCells, toCsv } from "../ui/csv.js";
import type {
  ModelEntry,
  ModelScore,
  HardwareProfile,
  Verdict,
  CheckOptions,
  GpuOffloadSuggestion,
  QuantizationVariant,
} from "../../core/types.js";
import { isUnifiedMemory } from "../../core/unified-memory.js";

function formatContext(ctx: number): string {
  if (ctx >= 1_000_000) return `${(ctx / 1_000_000).toFixed(1)}M tokens`;
  if (ctx >= 1_000) return `${Math.round(ctx / 1_000)}K tokens`;
  return `${ctx} tokens`;
}

function verdictLine(verdict: Verdict, modelName: string): string {
  switch (verdict) {
    case "yes":
      return `${theme.pass("✓ YES")} ${theme.pass(`— You can run ${modelName}`)}`;
    case "maybe":
      return `${theme.warning("⚠ MAYBE")} ${theme.warning(`— ${modelName} might run with limitations`)}`;
    case "no":
      return `${theme.fail("✗ NO")} ${theme.fail(`— ${modelName} won't fit on your hardware`)}`;
  }
}

function quantTable(model: ModelEntry, hardware: HardwareProfile, scores: ModelScore[], bestIdx: number): string {
  const table = new Table({
    chars: borderlessTableChars,
    style: { "padding-left": 0, "padding-right": 0, head: [], border: [] },
  });

  table.push(["Quant", `Need at ${defaultContextFor(model, hardware).toLocaleString()} tokens`, `Usable${budgetExact(hardware) ? "" : " est."}`, "Fit", "Speed", "Score"]);

  for (let i = 0; i < scores.length; i++) {
    const s = scores[i];
    const memory = memoryBreakdown(model, s.quantization, hardware);
    const marker = i === bestIdx ? theme.pass("★ ") : "  ";
    const score = isFitting(s.fitLevel)
      ? theme.number(String(s.compositeScore))
      : theme.fail("—");

    table.push([
      `${marker}${s.quantization.name}`,
      formatNeed(memory),
      formatMb(memory.budgetMb),
      fitBadge(s.fitLevel),
      speedLabel(s.speedEstimate),
      score,
    ]);
  }

  return `${table.toString()}\n  ${theme.muted(NEED_NOTE)}\n  ${theme.muted(SPEED_NOTE)}`;
}

/** Every quant scored on this machine, smallest file first (not by bits: estimated bits can misorder files). */
export function scoreQuants(model: ModelEntry, quants: QuantizationVariant[], hardware: HardwareProfile): ModelScore[] {
  return quants
    .map((q) => scoreModel(model, q, hardware))
    .sort((a, b) => a.quantization.vramMb - b.quantization.vramMb);
}

/** The `check --format json` payload; the MCP `check` tool returns the same object. */
export function checkResult(model: ModelEntry, hardware: HardwareProfile, scores: ModelScore[]) {
  const bestScore = headlineScore(scores);
  const availableVramMb = getAvailableVram(hardware);
  const memory = (s: ModelScore) => memoryBreakdown(model, s.quantization, hardware);
  return {
    model: {
      id: model.id,
      name: model.name,
      provider: model.provider,
      parametersBillion: model.parametersBillion,
      contextWindow: model.contextWindow,
      contextWindowEstimated: model.contextWindowEstimated === true,
      qualityTier: model.qualityTier,
      ollamaTag: model.ollamaTag,
    },
    hardware: {
      gpu: hardware.primaryGpu?.model ?? null,
      gpuCount: budgetGpus(hardware).length,
      vramMb: availableVramMb,
      ramMb: hardware.memory.totalMb,
    },
    verdict: deriveVerdict(bestScore.fitLevel),
    bestQuantization: {
      name: bestScore.quantization.name,
      bitsPerWeight: bestScore.quantization.bitsPerWeight,
      vramMb: bestScore.quantization.vramMb,
      fitLevel: bestScore.fitLevel,
      fitRatio: Math.round(bestScore.fitRatio * 100) / 100,
      compositeScore: bestScore.compositeScore,
      speedEstimate: bestScore.speedEstimate,
      memory: memory(bestScore),
    },
    allQuantizations: scores.map((s) => ({
      name: s.quantization.name,
      vramNeeded: s.quantization.vramMb,
      vramAvailable: availableVramMb,
      fitLevel: s.fitLevel,
      compositeScore: s.compositeScore,
      speedEstimate: s.speedEstimate,
      memory: memory(s),
    })),
    pullCommand: model.ollamaTag ? `ollama pull ${model.ollamaTag}` : null,
    pullQuantization: model.quantizations.find((q) => q.exact)?.name ?? null,
    gpuOffload: suggestGpuOffload(model, bestScore.quantization, hardware),
  };
}

export async function checkCommand(
  modelArg: string,
  options: CheckOptions,
): Promise<void> {
  const isJson = options.format === "json";
  const isCsv = options.format === "csv";
  const silent = isJson || isCsv;

  // 1. Detect hardware
  const spinner = silent ? null : ora({ text: "Detecting hardware...", color: "cyan" }).start();
  const hardware = await detectHardware();
  spinner?.succeed("Hardware detected");

  // 2. Resolve model, with the exact facts of its installed or registry file
  const resolved = resolveModel(modelArg);

  if (!resolved) {
    renderModelNotFound(modelArg, { silent });
    return;
  }
  spinner?.start("Reading model facts...");
  const model = await withExactFacts(resolveOllamaHost(options.host), resolved);
  spinner?.stop();

  // 3. Filter quantizations
  let quants = model.quantizations;
  let quantWarning: string | null = null;
  if (options.quant) {
    const match = findQuant(model, options.quant);
    if (match) {
      quants = [match];
    } else {
      quantWarning = `Quantization ${options.quant} not available for ${model.name}, showing all`;
    }
  }

  // 4. Score all quantizations
  const scores = scoreQuants(model, quants, hardware);
  if (isJson) {
    console.log(JSON.stringify(checkResult(model, hardware, scores), null, 2));
    return;
  }

  // 5. Best quantization & verdict
  const bestScore = headlineScore(scores);
  const verdict = deriveVerdict(bestScore.fitLevel);
  const bestIdx = verdict !== "no" ? scores.indexOf(bestScore) : -1; // No star marker when nothing fits
  const availableVramMb = getAvailableVram(hardware);

  // 6. Output
  if (isCsv) {
    outputCsv(model, hardware, scores, availableVramMb, bestIdx);
  } else {
    // GPU layer-offload suggestion for the headline quant. Useful when a
    // model is close to fitting (partial offload) or clearly over the line but
    // the user still wants to squeeze it on with CPU help.
    const offload = suggestGpuOffload(model, bestScore.quantization, hardware);
    outputTable(model, hardware, scores, bestScore, bestIdx, verdict, availableVramMb, quantWarning, options.verbose, offload);
  }
}

function outputTable(
  model: ModelEntry,
  hardware: HardwareProfile,
  scores: ModelScore[],
  bestScore: ModelScore,
  bestIdx: number,
  verdict: Verdict,
  availableVramMb: number,
  quantWarning: string | null,
  verbose: boolean,
  offload: GpuOffloadSuggestion | null,
): void {
  const lines: string[] = [];

  // Verdict headline
  lines.push("");
  lines.push(`  ${verdictLine(verdict, model.name)}`);

  // Model info
  lines.push(sectionHeader("Model"));
  lines.push(keyValue("Name", model.name));
  lines.push(keyValue("By", model.provider));
  lines.push(keyValue("Param", `${model.parametersBillion}B`));
  lines.push(keyValue("Ctx", `${formatContext(model.contextWindow)}${model.contextWindowEstimated ? " est." : ""}`));
  if (verbose) {
    lines.push(keyValue("Tier", model.qualityTier));
    lines.push(keyValue("Score", String(model.qualityScore)));
  }

  // Hardware summary
  lines.push(sectionHeader("Your Hardware"));
  lines.push(...gpuRows(hardware, availableVramMb, keyValue("GPU", theme.muted("None — CPU inference (using RAM)"))));
  const [budgetGpu] = budgetGpus(hardware);
  if (budgetGpu && isUnifiedMemory(budgetGpu)) {
    lines.push(subLine("(unified memory — shared with system RAM)"));
  }
  lines.push(keyValue("RAM", `${formatMb(hardware.memory.totalMb)} ${hardware.memory.type}`));
  if (verbose) {
    lines.push(keyValue("CPU", hardware.cpu.brand));
    // AVX2 is x86-only. On ARM, surface NEON instead of a bogus "✗".
    const simd = hardware.cpu.hasAvx2
      ? "AVX2 ✓"
      : hardware.cpu.architecture === "arm64"
        ? "NEON"
        : "AVX2 ✗";
    lines.push(subLine(`${hardware.cpu.threads} threads · ${simd}`));
  }

  // Quant warning
  if (quantWarning) {
    lines.push("");
    lines.push(`  ${theme.warning("⚠")} ${theme.warning(quantWarning)}`);
  }

  // Quantization breakdown
  lines.push(sectionHeader("Quantizations"));
  lines.push(quantTable(model, hardware, scores, bestIdx));

  // Recommendation: its need at Ollama's default context, then the parts, on their own line to stay within 72 columns
  lines.push("");
  if (verdict !== "no") {
    const memory = memoryBreakdown(model, bestScore.quantization, hardware);
    lines.push(`  ${theme.pass("★")} Recommended: ${theme.pass(bestScore.quantization.name)} — ${formatNeedAt(memory)}`);
    lines.push(subLine(formatBreakdown(memory)));
    lines.push(subLine(formatUsable(memory)));
    lines.push(...pullLines(model, bestScore.quantization));
  } else {
    lines.push(`  ${theme.fail("✗")} No quantization fits your hardware.`);
    lines.push(`  ${theme.muted("Try a smaller model:")} ${theme.command("llm-pulse models --fits")}`);
  }

  // GPU offload tip — only surface when the suggestion adds information beyond
  // the default. `full_fit` on a well-fitting quant is the happy path (Ollama
  // already offloads everything); showing "all layers on GPU" there is noise.
  // For `barely` and `cannot_run` the partial split is the actionable hint.
  if (offload && offload.reason === "partial_offload") {
    lines.push(sectionHeader("GPU Layer Offload"));
    lines.push(`  ${theme.pass("⚡")} Put ≈${theme.value(String(offload.gpuLayers))} of ${theme.value(String(offload.totalLayers))} layers on GPU (~${formatMb(offload.estimatedVramUsedMb)}), rest on CPU`);
    if (offload.ollamaCommand) {
      lines.push(`  ${theme.muted("After")} ${theme.command(`ollama run ${model.ollamaTag}`)}${theme.muted(", type:")}`);
      lines.push(`${" ".repeat(8)}${theme.command(offload.ollamaCommand)}`);
    }
  } else if (offload && offload.reason === "too_small" && verdict === "no") {
    lines.push(sectionHeader("GPU Layer Offload"));
    lines.push(`  ${theme.fail("✗")} VRAM too small for meaningful offload — consider a lighter quant or CPU-only inference`);
  }

  console.log(titleBox(lines.join("\n")));
}

function outputCsv(model: ModelEntry, hardware: HardwareProfile, scores: ModelScore[], availableVramMb: number, bestIdx: number): void {
  const headers = [
    "quantization", "bitsPerWeight", "vramNeeded", "vramAvailable",
    "fitLevel", "compositeScore", "speedEstimate", "isRecommended",
    ...MEMORY_CSV_HEADERS,
  ];
  const rows = scores.map((s, i) => [
    s.quantization.name, s.quantization.bitsPerWeight, s.quantization.vramMb,
    availableVramMb, s.fitLevel, s.compositeScore, s.speedEstimate, i === bestIdx,
    ...memoryCsvCells(memoryBreakdown(model, s.quantization, hardware)),
  ]);
  console.log(toCsv(headers, rows));
}
