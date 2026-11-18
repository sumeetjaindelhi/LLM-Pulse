import ora from "ora";
import Table from "cli-table3";
import { detectHardware } from "../../hardware/index.js";
import { budgetExact, budgetGpus, getAvailableVram, pickSweetSpot, isFitting } from "../../analysis/scorer.js";
import { resolveModel } from "../../models/database.js";
import { withExactFacts } from "../../runtimes/ollama.js";
import { resolveOllamaHost } from "../../core/config.js";
import { titleBox, sectionHeader, keyValue, subLine } from "../ui/boxes.js";
import { theme } from "../ui/colors.js";
import { fitBadge } from "../ui/badges.js";
import { formatBits, formatMb, formatUsableMb, formatQualityPct, formatWeights, pullLines } from "../ui/progress.js";
import { borderlessTableChars, speedLabel, SPEED_NOTE } from "../ui/tables.js";
import { scoreQuants } from "./check.js";
import { renderModelNotFound } from "../ui/errors.js";
import { MEMORY_CSV_HEADERS, memoryCsvCells, toCsv } from "../ui/csv.js";
import { memoryBreakdown } from "../../analysis/context-capacity.js";
import type {
  ModelEntry,
  ModelScore,
  HardwareProfile,
  QuantAdviceOptions,
} from "../../core/types.js";
import { quantRetention } from "../../models/quant-facts.js";
import { isUnifiedMemory } from "../../core/unified-memory.js";

// Retention is exp(-meanKLD) vs FP16 (llama.cpp tools/perplexity/README.md:38-125). These are definitions, not measurements:
// below exp(-0.1) = 0.905 (mean KLD 0.1, Q3_K_M and below in that table) a quant is called lossy.
const LOSSY_RETENTION_THRESHOLD = Math.exp(-0.1);

// Definition: a bigger quant gaining under 0.005 retention is not worth the VRAM. 0.005 sits between the
// Q6_K->Q8_0 step (0.0041) and Q5_K_M->Q6_K (0.0053); README.md:52-54.
const NEGLIGIBLE_RETENTION_GAIN = 0.005;

// Definition: a smaller quant losing more than 0.02 retention is a real drop; the Q5_K_M->Q4_K_M step is 0.0201 (README.md:54,59).
const NOTABLE_RETENTION_DROP = 0.02;

interface QuantNote {
  text: string;
  tone: "pass" | "warning" | "fail" | "muted";
}

interface RecommendedRef {
  weightsMb: number;
  retention: number;
}

// Bundles every value the three output paths (table/JSON/CSV) share so each
// render function takes one parameter instead of six. New fields land here
// rather than threading through three signatures.
interface AdviceContext {
  model: ModelEntry;
  hardware: HardwareProfile;
  scores: ModelScore[];
  recommendedIdx: number;
  availableVramMb: number;
  pullCommand: string | null;
}

export function describeQuant(
  score: ModelScore,
  isRecommended: boolean,
  recommended: RecommendedRef | null,
): QuantNote {
  if (!isFitting(score.fitLevel)) {
    return { text: "Too big — overflows usable memory", tone: "fail" };
  }
  if (!quantRetention(score.quantization.name).published) {
    return isRecommended
      ? { text: "★ Sweet spot — largest that fits (quality n/a)", tone: "pass" }
      : { text: "No published quality data", tone: "muted" };
  }
  if (isRecommended) {
    return { text: "★ Sweet spot — best quality you can fit", tone: "pass" };
  }
  if (score.quantization.qualityRetention < LOSSY_RETENTION_THRESHOLD) {
    return { text: "Noticeably lossy — only if VRAM-starved", tone: "warning" };
  }
  if (recommended === null) {
    return { text: "Fits", tone: "pass" };
  }

  // Weights, not bits: an estimated bits figure (nominal + 0.4) can rank a smaller file as the larger quant.
  const sizeDelta = score.quantization.vramMb - recommended.weightsMb;
  const qualityDelta = score.quantization.qualityRetention - recommended.retention;

  if (sizeDelta > 0) {
    return qualityDelta < NEGLIGIBLE_RETENTION_GAIN
      ? { text: "Overkill — negligible quality gain over sweet spot", tone: "muted" }
      : { text: "Bigger — slightly better quality but tighter fit", tone: "warning" };
  }
  if (sizeDelta < 0) {
    return -qualityDelta > NOTABLE_RETENTION_DROP
      ? { text: "Much smaller — faster but real quality drop", tone: "muted" }
      : { text: "Smaller — faster, slight quality drop", tone: "muted" };
  }
  return { text: "Fits comfortably", tone: "pass" };
}

function renderNote(note: QuantNote): string {
  switch (note.tone) {
    case "pass": return theme.pass(note.text);
    case "warning": return theme.warning(note.text);
    case "fail": return theme.fail(note.text);
    case "muted": return theme.muted(note.text);
  }
}

// Quants outside the published table carry a placeholder retention, so show no number for them.
function qualityCell(score: ModelScore): string {
  return quantRetention(score.quantization.name).published ? formatQualityPct(score.quantization.qualityRetention) : "n/a";
}

function adviceTable(
  scores: ModelScore[],
  recommendedIdx: number,
  availableVramMb: number,
  usableExact: boolean,
): string {
  const table = new Table({
    chars: borderlessTableChars,
    style: { "padding-left": 0, "padding-right": 0, head: [], border: [] },
  });

  table.push(["Quant", "Bits", "Weights", "Quality", "Fit", "Speed"]);

  for (let i = 0; i < scores.length; i++) {
    const s = scores[i];
    const marker = i === recommendedIdx ? theme.pass("★ ") : "  ";
    table.push([
      `${marker}${s.quantization.name}`,
      formatBits(s.quantization),
      formatWeights(s.quantization),
      qualityCell(s),
      fitBadge(s.fitLevel),
      speedLabel(s.speedEstimate),
    ]);
  }

  return `${table.toString()}\n  ${theme.muted(`Usable memory: ${formatUsableMb(availableVramMb, usableExact)}`)}\n  ${theme.muted(SPEED_NOTE)}`;
}

// Renders per-quant commentary as labeled lines below the table. Done outside
// cli-table3 because (a) boxen's `padding: 1` is actually 3 cols horizontally,
// shrinking the content area to ~72 cols at 80-col terminals, and (b) when a
// cli-table3 row wraps, boxen strips leading whitespace on continuation lines —
// so the wrapped Notes column would mis-align under the box's left margin
// instead of staying in its column. Free-flowing labeled lines avoid both.
function adviceNotes(scores: ModelScore[], recommendedIdx: number): string {
  const recommended: RecommendedRef | null = recommendedIdx >= 0 && quantRetention(scores[recommendedIdx].quantization.name).published
    ? {
        weightsMb: scores[recommendedIdx].quantization.vramMb,
        retention: scores[recommendedIdx].quantization.qualityRetention,
      }
    : null;

  // Pad to the widest quant name so the notes line up. Markers are 2 visible
  // cols ("★ " or "  ") and prepended outside the pad to keep alignment.
  const nameWidth = Math.max(...scores.map((s) => s.quantization.name.length));

  return scores
    .map((s, i) => {
      const marker = i === recommendedIdx ? theme.pass("★ ") : "  ";
      const name = theme.value(s.quantization.name.padEnd(nameWidth));
      const note = renderNote(describeQuant(s, i === recommendedIdx, recommended));
      return `  ${marker}${name}  ${note}`;
    })
    .join("\n");
}

export async function quantAdviceCommand(
  modelArg: string,
  options: QuantAdviceOptions,
): Promise<void> {
  const isJson = options.format === "json";
  const isCsv = options.format === "csv";
  const silent = isJson || isCsv;

  const spinner = silent ? null : ora({ text: "Detecting hardware...", color: "cyan" }).start();
  const hardware = await detectHardware();
  spinner?.succeed("Hardware detected");

  const resolved = resolveModel(modelArg);

  if (!resolved) {
    renderModelNotFound(modelArg, { silent });
    return;
  }
  spinner?.start("Reading model facts...");
  const model = await withExactFacts(resolveOllamaHost(), resolved);
  spinner?.stop();

  const scores = scoreQuants(model, model.quantizations, hardware);

  const ctx: AdviceContext = {
    model,
    hardware,
    scores,
    recommendedIdx: pickSweetSpot(scores),
    availableVramMb: getAvailableVram(hardware),
    pullCommand: model.ollamaTag ? `ollama pull ${model.ollamaTag}` : null,
  };

  if (isJson) {
    outputJson(ctx);
  } else if (isCsv) {
    outputCsv(ctx);
  } else {
    outputTable(ctx, options.verbose);
  }
}

function outputTable(ctx: AdviceContext, verbose: boolean): void {
  const { model, hardware, scores, recommendedIdx, availableVramMb } = ctx;
  const lines: string[] = [];

  lines.push("");
  lines.push(`  ${theme.header("Quantization Advice")} — ${theme.value(model.name)}`);

  lines.push(sectionHeader("Model"));
  lines.push(keyValue("By", model.provider));
  lines.push(keyValue("Param", `${model.parametersBillion}B`));
  if (verbose) {
    lines.push(keyValue("Tier", model.qualityTier));
    lines.push(keyValue("Score", String(model.qualityScore)));
  }

  lines.push(sectionHeader("Quantization Tradeoff"));
  lines.push(adviceTable(scores, recommendedIdx, availableVramMb, budgetExact(hardware)));

  lines.push(sectionHeader("Notes"));
  lines.push(adviceNotes(scores, recommendedIdx));

  lines.push("");
  if (recommendedIdx >= 0) {
    const pick = scores[recommendedIdx];
    // Two short lines instead of one long one — at 80-col terminals, any
    // single line over ~72 visible chars trips boxen's wrap mode, which
    // strips leading whitespace from every line in the box and wrecks
    // table/notes alignment. Splitting keeps every line within budget.
    lines.push(
      `  ${theme.pass("★ Recommended:")} ${theme.pass(pick.quantization.name)} — ` +
      (quantRetention(pick.quantization.name).published
        ? `${theme.value(formatQualityPct(pick.quantization.qualityRetention))} quality, `
        : "") +
      `weights ${theme.value(formatWeights(pick.quantization))}`,
    );
    lines.push(subLine(`(${pick.fitLevel.replace("_", " ")} fit, ${pick.speedEstimate} speed)`));
    lines.push(...pullLines(model, pick.quantization));
  } else {
    lines.push(`  ${theme.fail("✗")} No quantization fits your ${formatUsableMb(availableVramMb, budgetExact(hardware))} of usable memory.`);
    lines.push(`  ${theme.muted("Try:")} ${theme.command(`llm-pulse check ${model.ollamaTag ?? model.id}`)} ${theme.muted("for layer-offload tips")}`);
    lines.push(`  ${theme.muted("Or browse smaller models:")} ${theme.command("llm-pulse models --fits")}`);
  }

  // Unified-memory nudge so the usable number makes sense on shared-memory machines, where users
  // see e.g. "10 GiB" but the system reports 16+ GiB total. Trimmed phrasing
  // keeps it under boxen's 72-col content budget at 80-col terminals.
  const [budgetGpu] = budgetGpus(hardware);
  if (budgetGpu && isUnifiedMemory(budgetGpu)) {
    lines.push(subLine("(unified memory — shared with system RAM)"));
  }

  console.log(titleBox(lines.join("\n")));
}

function outputJson(ctx: AdviceContext): void {
  const { model, hardware, scores, recommendedIdx, availableVramMb, pullCommand } = ctx;
  const pick = recommendedIdx >= 0 ? scores[recommendedIdx] : null;
  const output = {
    model: {
      id: model.id,
      name: model.name,
      provider: model.provider,
      parametersBillion: model.parametersBillion,
      qualityTier: model.qualityTier,
      ollamaTag: model.ollamaTag,
    },
    hardware: {
      gpu: hardware.primaryGpu?.model ?? null,
      gpuCount: budgetGpus(hardware).length,
      vramMb: availableVramMb,
    },
    recommended: pick ? {
      name: pick.quantization.name,
      bitsPerWeight: pick.quantization.bitsPerWeight,
      vramMb: pick.quantization.vramMb,
      qualityRetention: pick.quantization.qualityRetention,
      fitLevel: pick.fitLevel,
      speedEstimate: pick.speedEstimate,
      qualityMeasured: quantRetention(pick.quantization.name).published,
      memory: memoryBreakdown(model, pick.quantization, hardware),
    } : null,
    quantizations: scores.map((s, i) => ({
      name: s.quantization.name,
      bitsPerWeight: s.quantization.bitsPerWeight,
      vramMb: s.quantization.vramMb,
      qualityRetention: s.quantization.qualityRetention,
      fitLevel: s.fitLevel,
      speedEstimate: s.speedEstimate,
      isRecommended: i === recommendedIdx,
      qualityMeasured: quantRetention(s.quantization.name).published,
      memory: memoryBreakdown(model, s.quantization, hardware),
    })),
    pullCommand,
    pullQuantization: model.quantizations.find((q) => q.exact)?.name ?? null,
  };

  console.log(JSON.stringify(output, null, 2));
}

function outputCsv(ctx: AdviceContext): void {
  const { model, hardware, scores, availableVramMb, recommendedIdx } = ctx;
  const headers = [
    "quantization", "bitsPerWeight", "vramMb", "qualityRetention",
    "vramAvailable", "fitLevel", "speedEstimate", "isRecommended",
    "qualityMeasured", ...MEMORY_CSV_HEADERS,
  ];
  const rows = scores.map((s, i) => [
    s.quantization.name,
    s.quantization.bitsPerWeight,
    s.quantization.vramMb,
    s.quantization.qualityRetention,
    availableVramMb,
    s.fitLevel,
    s.speedEstimate,
    i === recommendedIdx,
    quantRetention(s.quantization.name).published,
    ...memoryCsvCells(memoryBreakdown(model, s.quantization, hardware)),
  ]);
  console.log(toCsv(headers, rows));
}
