import Table from "cli-table3";
import { theme } from "./colors.js";
import { fitBadge } from "./badges.js";
import type { Recommendation } from "../../core/types.js";

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
