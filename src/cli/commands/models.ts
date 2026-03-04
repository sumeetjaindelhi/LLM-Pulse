import ora from "ora";
import Table from "cli-table3";
import { getAllModels, searchModels, filterByCategory } from "../../models/database.js";
import { detectHardware } from "../../hardware/index.js";
import { scoreModel } from "../../analysis/scorer.js";
import { theme } from "../ui/colors.js";
import { fitBadge } from "../ui/badges.js";
import { sectionHeader } from "../ui/boxes.js";
import type { ModelEntry, ModelCategory, HardwareProfile } from "../../core/types.js";

interface ModelsOptions {
  search?: string;
  category: ModelCategory | "all";
  fits: boolean;
  format: string;
}

export async function modelsCommand(options: ModelsOptions): Promise<void> {
  let models: ModelEntry[];

  // Filter by search or category
  if (options.search) {
    models = searchModels(options.search);
  } else if (options.category !== "all") {
    models = filterByCategory(options.category);
  } else {
    models = getAllModels();
  }

  // If --fits, we need hardware info
  let hardware: HardwareProfile | null = null;
  if (options.fits) {
    const spinner = ora({ text: "Detecting hardware...", color: "cyan" }).start();
    hardware = await detectHardware();
    spinner.succeed("Hardware detected");
  }

  // Score and filter if needed
  type ScoredModel = { model: ModelEntry; fitLabel: string; bestQuant: string; vramMb: number };
  const scored: ScoredModel[] = [];

  for (const model of models) {
    if (hardware) {
      // Pick best fitting quantization
      const scores = model.quantizations
        .map((q) => scoreModel(model, q, hardware!, options.category))
        .filter((s) => !options.fits || s.fitLevel !== "cannot_run")
        .sort((a, b) => b.compositeScore - a.compositeScore);

      if (scores.length > 0) {
        const best = scores[0];
        scored.push({
          model,
          fitLabel: fitBadge(best.fitLevel),
          bestQuant: best.quantization.name,
          vramMb: best.quantization.vramMb,
        });
      }
    } else {
      // No hardware filter — show smallest quant
      const smallest = model.quantizations[0];
      scored.push({
        model,
        fitLabel: theme.muted("—"),
        bestQuant: smallest.name,
        vramMb: smallest.vramMb,
      });
    }
  }

  if (options.format === "json") {
    const output = scored.map((s) => ({
      id: s.model.id,
      name: s.model.name,
      provider: s.model.provider,
      parametersBillion: s.model.parametersBillion,
      categories: s.model.categories,
      quantization: s.bestQuant,
      vramMb: s.vramMb,
      ollamaTag: s.model.ollamaTag,
    }));
    console.log(JSON.stringify(output, null, 2));
    return;
  }

  console.log(sectionHeader(`Models${options.search ? ` matching "${options.search}"` : ""} (${scored.length} results)`));
  console.log();

  if (scored.length === 0) {
    console.log(`  ${theme.warning("No models found matching your criteria.")}`);
    console.log();
    return;
  }

  const table = new Table({
    head: [
      theme.muted("Model"),
      theme.muted("Provider"),
      theme.muted("Params"),
      theme.muted("Context"),
      theme.muted("Quant"),
      theme.muted("VRAM"),
      theme.muted("Fit"),
      theme.muted("Categories"),
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

  for (const s of scored) {
    const vramGb = (s.vramMb / 1024).toFixed(0);
    const ctx = s.model.contextWindow >= 1024
      ? `${(s.model.contextWindow / 1024).toFixed(0)}K`
      : String(s.model.contextWindow);

    table.push([
      theme.value(s.model.name),
      theme.muted(s.model.provider),
      theme.number(`${s.model.parametersBillion}B`),
      theme.muted(ctx),
      theme.muted(s.bestQuant),
      theme.number(`${vramGb} GB`),
      s.fitLabel,
      theme.muted(s.model.categories.join(", ")),
    ]);
  }

  console.log(table.toString());
  console.log();
}
