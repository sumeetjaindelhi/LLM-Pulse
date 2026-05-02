import ora from "ora";
import Table from "cli-table3";
import { detectHardware } from "../../hardware/index.js";
import { scoreModel, getAvailableVram, pickSweetSpot, isFitting } from "../../analysis/scorer.js";
import { resolveModel } from "../../models/database.js";
import { titleBox, sectionHeader, keyValue, subLine } from "../ui/boxes.js";
import { theme } from "../ui/colors.js";
import { fitBadge } from "../ui/badges.js";
import { formatMb, formatQualityPct } from "../ui/progress.js";
import { borderlessTableChars, speedLabel } from "../ui/tables.js";
import { renderModelNotFound } from "../ui/errors.js";
import { toCsv } from "../ui/csv.js";
import type {
  ModelEntry,
  ModelScore,
  HardwareProfile,
  QuantAdviceOptions,
} from "../../core/types.js";

// Q2 / Q3 families sit below this — usable on VRAM-starved rigs but users
// should know they're paying a real quality tax.
const LOSSY_QUALITY_THRESHOLD = 0.85;

// Within this delta, a bigger quant is just wasted VRAM (e.g. F16 over Q8_0).
const NEGLIGIBLE_QUALITY_GAIN = 0.02;

// Below this delta, a smaller quant is "slightly faster, slight quality drop".
// Above it, the drop is real and worth flagging differently.
const NOTABLE_QUALITY_DROP = 0.05;

interface QuantNote {
  text: string;
  tone: "pass" | "warning" | "fail" | "muted";
}

interface RecommendedRef {
  bits: number;
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
    return { text: "Too big — overflows VRAM", tone: "fail" };
  }
  if (isRecommended) {
    return { text: "★ Sweet spot — best quality you can fit", tone: "pass" };
  }
  if (score.quantization.qualityRetention < LOSSY_QUALITY_THRESHOLD) {
    return { text: "Noticeably lossy — only if VRAM-starved", tone: "warning" };
  }
  if (recommended === null) {
    return { text: "Fits", tone: "pass" };
  }

  const bitsDelta = score.quantization.bitsPerWeight - recommended.bits;
  const qualityDelta = score.quantization.qualityRetention - recommended.retention;

  if (bitsDelta > 0) {
    return qualityDelta < NEGLIGIBLE_QUALITY_GAIN
      ? { text: "Overkill — negligible quality gain over sweet spot", tone: "muted" }
      : { text: "Bigger — slightly better quality but tighter fit", tone: "warning" };
  }
  if (bitsDelta < 0) {
    return -qualityDelta >= NOTABLE_QUALITY_DROP
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

function adviceTable(
  scores: ModelScore[],
  recommendedIdx: number,
  availableVramMb: number,
): string {
  const table = new Table({
    chars: borderlessTableChars,
    style: { "padding-left": 0, "padding-right": 0, head: ["gray"] },
  });

  table.push(["Quant", "Bits", "Size", "Quality", "Fit", "Speed"]);

  for (let i = 0; i < scores.length; i++) {
    const s = scores[i];
    const marker = i === recommendedIdx ? theme.pass("★ ") : "  ";
    table.push([
      `${marker}${s.quantization.name}`,
      s.quantization.bitsPerWeight.toFixed(1),
      formatMb(s.quantization.vramMb),
      formatQualityPct(s.quantization.qualityRetention),
      fitBadge(s.fitLevel),
      speedLabel(s.speedEstimate),
    ]);
  }

  return `${table.toString()}\n  ${theme.muted(`Your available VRAM: ${formatMb(availableVramMb)}`)}`;
}

// Renders per-quant commentary as labeled lines below the table. Done outside
// cli-table3 because (a) boxen's `padding: 1` is actually 3 cols horizontally,
// shrinking the content area to ~72 cols at 80-col terminals, and (b) when a
// cli-table3 row wraps, boxen strips leading whitespace on continuation lines —
// so the wrapped Notes column would mis-align under the box's left margin
// instead of staying in its column. Free-flowing labeled lines avoid both.
function adviceNotes(scores: ModelScore[], recommendedIdx: number): string {
  const recommended: RecommendedRef | null = recommendedIdx >= 0
    ? {
        bits: scores[recommendedIdx].quantization.bitsPerWeight,
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

  const model = resolveModel(modelArg);

  if (!model) {
    renderModelNotFound(modelArg, { silent });
    return;
  }

  // Ascending bitsPerWeight matches how users scan tradeoff tables — cheapest first.
  const scores = model.quantizations
    .map((q) => scoreModel(model, q, hardware))
    .sort((a, b) => a.quantization.bitsPerWeight - b.quantization.bitsPerWeight);

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
  const { model, hardware, scores, recommendedIdx, availableVramMb, pullCommand } = ctx;
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
  lines.push(adviceTable(scores, recommendedIdx, availableVramMb));

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
      `${theme.value(formatQualityPct(pick.quantization.qualityRetention))} quality, ` +
      `${theme.value(formatMb(pick.quantization.vramMb))}`,
    );
    lines.push(subLine(`(${pick.fitLevel.replace("_", " ")} fit, ${pick.speedEstimate} speed)`));
    if (pullCommand) {
      lines.push(`  ${theme.muted("→")} ${theme.command(pullCommand)}`);
    }
  } else {
    lines.push(`  ${theme.fail("✗")} No quantization fits your ${formatMb(availableVramMb)} of VRAM.`);
    lines.push(`  ${theme.muted("Try:")} ${theme.command(`llm-pulse check ${model.ollamaTag ?? model.id}`)} ${theme.muted("for layer-offload tips")}`);
    lines.push(`  ${theme.muted("Or browse smaller models:")} ${theme.command("llm-pulse models --fits")}`);
  }

  // Unified-memory nudge so the VRAM number makes sense on Macs, where users
  // see e.g. "10 GB" but the system reports 16+ GB total. Trimmed phrasing
  // keeps it under boxen's 72-col content budget at 80-col terminals.
  if (hardware.primaryGpu?.acceleratorType === "metal") {
    lines.push(subLine("(Apple Silicon: unified memory, sysctl-capped)"));
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
      vramMb: availableVramMb,
    },
    recommended: pick ? {
      name: pick.quantization.name,
      bitsPerWeight: pick.quantization.bitsPerWeight,
      vramMb: pick.quantization.vramMb,
      qualityRetention: pick.quantization.qualityRetention,
      fitLevel: pick.fitLevel,
      speedEstimate: pick.speedEstimate,
    } : null,
    quantizations: scores.map((s, i) => ({
      name: s.quantization.name,
      bitsPerWeight: s.quantization.bitsPerWeight,
      vramMb: s.quantization.vramMb,
      qualityRetention: s.quantization.qualityRetention,
      fitLevel: s.fitLevel,
      speedEstimate: s.speedEstimate,
      isRecommended: i === recommendedIdx,
    })),
    pullCommand,
  };

  console.log(JSON.stringify(output, null, 2));
}

function outputCsv(ctx: AdviceContext): void {
  const { scores, availableVramMb, recommendedIdx } = ctx;
  const headers = [
    "quantization", "bitsPerWeight", "vramMb", "qualityRetention",
    "vramAvailable", "fitLevel", "speedEstimate", "isRecommended",
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
  ]);
  console.log(toCsv(headers, rows));
}
