import Table from "cli-table3";
import { theme } from "./colors.js";
import { fitBadge } from "./badges.js";
import type { Recommendation } from "../../core/types.js";

export function recommendationTable(recommendations: Recommendation[]): string {
  const table = new Table({
    head: [
      theme.muted("#"),
      theme.muted("Model"),
      theme.muted("Quant"),
      theme.muted("Fit"),
      theme.muted("VRAM"),
      theme.muted("Speed"),
    ],
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
    table.push([
      theme.muted(String(rec.rank)),
      theme.value(s.model.name),
      theme.muted(s.quantization.name),
      fitBadge(s.fitLevel),
      theme.number(`${vramGb} GB`),
      speedLabel(s.speedEstimate),
    ]);
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
