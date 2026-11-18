import Table from "cli-table3";
import { theme } from "./colors.js";
import { fitBadge } from "./badges.js";
import { formatNeed, formatUsableMb } from "./progress.js";
import { budgetExact, getAvailableVram, isCriticalFit } from "../../analysis/scorer.js";
import { BOX_CONTENT_COLUMNS } from "./boxes.js";
import { memoryBreakdown } from "../../analysis/context-capacity.js";
import type { HardwareProfile, Recommendation, ModelScore } from "../../core/types.js";

// Shared cli-table3 character config for the project's borderless / compact
// table style — empty borders, two-space gutter on the left and between cells.
// Imported by every table-rendering site (check, quant-advice, models, etc.)
// so that all tables share the same shape.
export const borderlessTableChars = {
  top: "", "top-mid": "", "top-left": "", "top-right": "",
  bottom: "", "bottom-mid": "", "bottom-left": "", "bottom-right": "",
  left: "  ", "left-mid": "",
  mid: "", "mid-mid": "",
  right: "", "right-mid": "",
  middle: "  ",
} as const;

// Legend for need columns with no single context to name: each model's default context is its own.
export const NEED_NOTE = "Need: weights + KV cache + compute (≈) at Ollama's default context";
// Legend for speed columns: the class comes from weight bytes alone (memory-model.ts speedClass).
export const SPEED_NOTE = "Speed: rough (decode is bandwidth-bound); measure: llm-pulse benchmark";

const LABEL_COLUMN_WIDTH = 13;

// SGR colour codes only; the cells hold nothing else that styles.
const visibleLength = (text: string): number => [...text.replace(/\x1b\[[0-9;]*m/g, "")].length;

/** `text` with every word longer than `width` cut into `width`-sized chunks; cli-table3 would otherwise end it with "…". */
function breakLongWords(text: string, width: number): string {
  return text
    .split(" ")
    .flatMap((word) => (word.length <= width ? [word] : word.match(new RegExp(`.{1,${width}}`, "gu"))!))
    .join(" ");
}

/** The shared table's left margin plus the two-space gutter between each pair of the six columns. */
const RECOMMENDATION_TABLE_CHROME = 2 + 2 * 5;

export function recommendationTable(
  recommendations: Recommendation[],
  hardware: HardwareProfile,
  installedTags?: Set<string>,
): string {
  const head = ["#", "Model", "Quant", "Fit", "Need", "Speed"].map((h) => theme.muted(h));
  const installedRows = recommendations.map((rec) => {
    const tag = rec.score.model.ollamaTag;
    return tag ? (installedTags?.has(tag) ?? false) : false;
  });
  const modelCell = (index: number, name: string) =>
    installedRows[index] ? `${theme.pass("●")} ${theme.value(name)}` : theme.value(name);
  const rows = recommendations.map((rec, index) => {
    const s = rec.score;
    return [
      theme.muted(String(rec.rank)),
      modelCell(index, s.model.name),
      theme.muted(s.quantization.name),
      fitBadge(s.fitLevel),
      theme.number(formatNeed(memoryBreakdown(s.model, s.quantization, hardware))),
      speedLabel(s.speedEstimate),
    ];
  });

  // Every column but Model takes its widest cell plus padding; Model gets what the box leaves, and long names wrap in it.
  const widths = head.map((_, col) => Math.max(...[head, ...rows].map((r) => visibleLength(r[col]))) + 2);
  const others = widths.reduce((sum, w, col) => (col === 1 ? sum : sum + w), 0);
  widths[1] = Math.min(widths[1], BOX_CONTENT_COLUMNS - RECOMMENDATION_TABLE_CHROME - others);
  // Cell padding takes 2 columns, and an installed row's "● " marker 2 more.
  recommendations.forEach((rec, index) => {
    rows[index][1] = modelCell(index, breakLongWords(rec.score.model.name, widths[1] - 2 - (installedRows[index] ? 2 : 0)));
  });

  const table = new Table({
    head,
    style: { head: [], border: [], compact: true },
    chars: borderlessTableChars,
    colWidths: widths,
    wordWrap: true,
  });
  table.push(...rows);

  const trimmed = table.toString().split("\n").map((line) => line.trimEnd()).join("\n");
  return `${trimmed}\n  ${theme.muted(NEED_NOTE)}\n  ${theme.muted(SPEED_NOTE)}`;
}

export function speedLabel(speed: "fast" | "moderate" | "slow"): string {
  switch (speed) {
    case "fast": return theme.pass("fast");
    case "moderate": return theme.warning("moderate");
    case "slow": return theme.fail("slow");
  }
}

export function comparisonTable(
  scores: ModelScore[],
  hardware: HardwareProfile,
  winnerIndex: number,
): string {
  // Transposed layout: attributes as rows, models as columns
  const headerRow = [""];
  for (let i = 0; i < scores.length; i++) {
    const name = scores[i].model.name;
    headerRow.push(i === winnerIndex ? theme.pass(`${name} ★`) : theme.value(name));
  }

  // The box holds 72 content columns; a wider table puts boxen in wrap mode and strips every indent. Then the
  // model columns share what the 13-wide label column leaves, and long cells wrap inside them.
  const modelWidth = Math.floor((BOX_CONTENT_COLUMNS - 2 - 2 * scores.length - LABEL_COLUMN_WIDTH) / scores.length);
  const leading = [
    row("Provider", scores.map((s) => theme.muted(s.model.provider))),
    row("Parameters", scores.map((s) => theme.value(`${s.model.parametersBillion}B`))),
    row("Context", scores.map((s) => theme.value(`${formatContext(s.model.contextWindow)}${s.model.contextWindowEstimated ? " est." : ""}`))),
    row("Quality", scores.map((s) => theme.value(`${s.model.qualityTier} (${s.model.qualityScore})`))),
  ];
  const trailing = [
    row("Quantization", scores.map((s) => theme.muted(s.quantization.name))),
    row("Need", scores.map((s) => {
      const color = isCriticalFit(s.fitLevel) ? theme.fail
        : s.fitLevel === "tight" ? theme.warning
        : theme.pass;
      return color(formatNeed(memoryBreakdown(s.model, s.quantization, hardware)));
    })),
    row("Fit", scores.map((s) => fitBadge(s.fitLevel))),
    row("Score", scores.map((s) => theme.number(String(s.compositeScore)))),
    row("Speed", scores.map((s) => speedLabel(s.speedEstimate))),
  ];
  const render = (tight: boolean): string => {
    const table = new Table({
      head: headerRow,
      style: { head: [], border: [], compact: true, ...(tight && { "padding-left": 0, "padding-right": 0 }) },
      chars: borderlessTableChars,
      ...(tight && { colWidths: [LABEL_COLUMN_WIDTH, ...scores.map(() => modelWidth)], wordWrap: true }),
    });
    const ruleWidth = tight ? modelWidth : LABEL_COLUMN_WIDTH;
    table.push(...leading, [theme.muted("─".repeat(LABEL_COLUMN_WIDTH)), ...scores.map(() => theme.muted("─".repeat(ruleWidth)))], ...trailing);
    return table.toString();
  };
  const natural = render(false);
  const rendered = natural.split("\n").some((line) => visibleLength(line) > BOX_CONTENT_COLUMNS) ? render(true) : natural;

  const usable = `Usable memory: ${formatUsableMb(getAvailableVram(hardware), budgetExact(hardware))}`;
  return `${rendered}\n  ${theme.muted(usable)}\n  ${theme.muted(NEED_NOTE)}\n  ${theme.muted(SPEED_NOTE)}`;
}

function row(label: string, values: string[]): string[] {
  return [theme.label(label), ...values];
}

function formatContext(ctx: number): string {
  return ctx >= 1024 ? `${Math.round(ctx / 1024)}K` : String(ctx);
}
