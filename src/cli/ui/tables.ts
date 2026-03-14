import Table from "cli-table3";
import { theme } from "./colors.js";
import { fitBadge } from "./badges.js";
import type { Recommendation, ModelScore } from "../../core/types.js";

export function recommendationTable(
  recommendations: Recommendation[],
  installedTags?: Set<string>,
): string {
  const showInstalled = installedTags !== undefined;
  const headCols = [
    theme.muted("#"),
    theme.muted("Model"),
    theme.muted("Quant"),
    theme.muted("Fit"),
    theme.muted("VRAM"),
    theme.muted("Speed"),
  ];
  if (showInstalled) headCols.push(theme.muted(""));

  const table = new Table({
    head: headCols,
    style: { head: [], border: ["gray"], compact: true },
    chars: {
      top: "", "top-mid": "", "top-left": "", "top-right": "",
      bottom: "", "bottom-mid": "", "bottom-left": "", "bottom-right": "",
      left: "  ", "left-mid": "",
      mid: "", "mid-mid": "",
      right: "", "right-mid": "",
      middle: "  ",
    },
  });

  for (const rec of recommendations) {
    const s = rec.score;
    const vramGb = (s.quantization.vramMb / 1024).toFixed(0);
    const row = [
      theme.muted(String(rec.rank)),
      theme.value(s.model.name),
      theme.muted(s.quantization.name),
      fitBadge(s.fitLevel),
      theme.number(`${vramGb} GB`),
      speedLabel(s.speedEstimate),
    ];
    if (showInstalled) {
      const tag = s.model.ollamaTag;
      const isInstalled = tag ? (installedTags?.has(tag) ?? false) : false;
      row.push(isInstalled ? theme.pass("● installed") : "");
    }
    table.push(row);
  }

  return table.toString();
}

function speedLabel(speed: "fast" | "moderate" | "slow"): string {
  switch (speed) {
    case "fast": return theme.pass("fast");
    case "moderate": return theme.warning("moderate");
    case "slow": return theme.fail("slow");
  }
}

export function comparisonTable(
  scores: ModelScore[],
  availableVramMb: number,
  winnerIndex: number,
): string {
  // Transposed layout: attributes as rows, models as columns
  const headerRow = [""];
  for (let i = 0; i < scores.length; i++) {
    const name = scores[i].model.name;
    headerRow.push(i === winnerIndex ? theme.pass(`${name} ★`) : theme.value(name));
  }

  const table = new Table({
    head: headerRow,
    style: { head: [], border: ["gray"], compact: true },
    chars: {
      top: "", "top-mid": "", "top-left": "", "top-right": "",
      bottom: "", "bottom-mid": "", "bottom-left": "", "bottom-right": "",
      left: "  ", "left-mid": "",
      mid: "", "mid-mid": "",
      right: "", "right-mid": "",
      middle: "  ",
    },
  });

  // Model info rows
  table.push(row("Provider", scores.map((s) => theme.muted(s.model.provider))));
  table.push(row("Parameters", scores.map((s) => theme.value(`${s.model.parametersBillion}B`))));
  table.push(row("Context", scores.map((s) => theme.value(formatContext(s.model.contextWindow)))));
  table.push(row("Quality", scores.map((s) =>
    theme.value(`${s.model.qualityTier} (${s.model.qualityScore})`),
  )));

  // Separator row
  const sep = Array(scores.length + 1).fill(theme.muted("─────────────"));
  sep[0] = theme.muted("─────────────");
  table.push(sep);

  // Hardware-specific rows
  table.push(row("Quantization", scores.map((s) => theme.muted(s.quantization.name))));
  table.push(row("VRAM", scores.map((s) => {
    const reqGb = (s.quantization.vramMb / 1024).toFixed(0);
    const availGb = (availableVramMb / 1024).toFixed(0);
    const color = s.fitLevel === "cannot_run" || s.fitLevel === "barely" ? theme.fail
      : s.fitLevel === "tight" ? theme.warning
      : theme.pass;
    return color(`${reqGb} GB / ${availGb} GB`);
  })));
  table.push(row("Fit", scores.map((s) => fitBadge(s.fitLevel))));
  table.push(row("Score", scores.map((s) => theme.number(String(s.compositeScore)))));
  table.push(row("Speed", scores.map((s) => speedLabel(s.speedEstimate))));

  return table.toString();
}

function row(label: string, values: string[]): string[] {
  return [theme.label(label), ...values];
}

function formatContext(ctx: number): string {
  return ctx >= 1024 ? `${Math.round(ctx / 1024)}K` : String(ctx);
}
