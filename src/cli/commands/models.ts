import ora from "ora";
import Table from "cli-table3";
import { getAllModels, searchModels, filterByCategory } from "../../models/database.js";
import { getMergedModels } from "../../models/merged-models.js";
import { detectHardware } from "../../hardware/index.js";
import { scoreModel, isFitting } from "../../analysis/scorer.js";
import { theme } from "../ui/colors.js";
import { fitBadge } from "../ui/badges.js";
import { sectionHeader } from "../ui/boxes.js";
import { borderlessTableChars } from "../ui/tables.js";
import { toCsv } from "../ui/csv.js";
import { resolveOllamaHost } from "../../core/config.js";
import type { ModelEntry, ModelCategory, HardwareProfile, MergedModel } from "../../core/types.js";

interface ModelsOptions {
  search?: string;
  category: ModelCategory | "all";
  fits: boolean;
  live: boolean;
  installed: boolean;
  library?: boolean;
  refresh?: boolean;
  format: string;
  host?: string;
}

export async function modelsCommand(options: ModelsOptions): Promise<void> {
  const ollamaHost = resolveOllamaHost(options.host);
  if (options.live) {
    return liveModelsCommand(options, ollamaHost);
  }

  let models: ModelEntry[];

  // Filter by search or category
  models = options.search ? searchModels(options.search) : filterByCategory(options.category);

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
      const hw = hardware;
      // Pick best fitting quantization
      const scores = model.quantizations
        .map((q) => scoreModel(model, q, hw, options.category))
        .filter((s) => !options.fits || isFitting(s.fitLevel))
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

  if (options.format === "csv") {
    const headers = ["id", "name", "provider", "parametersBillion", "categories", "quantization", "vramMb", "ollamaTag"];
    const rows = scored.map((s) => [
      s.model.id, s.model.name, s.model.provider, s.model.parametersBillion,
      s.model.categories.join(";"), s.bestQuant, s.vramMb, s.model.ollamaTag,
    ]);
    console.log(toCsv(headers, rows));
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
    chars: borderlessTableChars,
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

async function liveModelsCommand(options: ModelsOptions, ollamaHost: string): Promise<void> {
  const includeLibrary = options.library ?? false;
  const refresh = options.refresh ?? false;
  const fetchLabel = includeLibrary
    ? refresh
      ? "Refreshing Ollama library catalog..."
      : "Fetching Ollama library catalog..."
    : "Fetching models from Ollama...";

  const spinner = ora({ text: fetchLabel, color: "cyan" }).start();

  const mergeOpts = { ollamaHost, includeLibrary, refresh };

  let merged: MergedModel[];
  let hardware: HardwareProfile | null = null;

  try {
    if (options.fits) {
      [merged, hardware] = await Promise.all([getMergedModels(mergeOpts), detectHardware()]);
    } else {
      merged = await getMergedModels(mergeOpts);
    }
  } catch (err) {
    spinner.fail("Model fetch failed");
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`\n  ${theme.fail("Error:")} ${msg}\n`);
    process.exitCode = 1;
    return;
  }

  spinner.succeed(
    includeLibrary
      ? `Catalog loaded (${merged.length} models)`
      : "Models fetched",
  );

  // Apply filters
  if (options.installed) {
    merged = merged.filter((m) => m.installed);
  }
  if (options.search) {
    const q = options.search.toLowerCase();
    merged = merged.filter((m) => {
      const name = m.entry?.name ?? m.libraryModel?.slug ?? m.ollamaModel?.name ?? "";
      const provider = m.entry?.provider ?? "";
      const description = m.libraryModel?.description ?? "";
      const capabilities = m.libraryModel?.capabilities.join(" ") ?? "";
      return (
        name.toLowerCase().includes(q) ||
        provider.toLowerCase().includes(q) ||
        description.toLowerCase().includes(q) ||
        capabilities.toLowerCase().includes(q)
      );
    });
  }
  if (options.category !== "all") {
    merged = merged.filter((m) => m.entry?.categories.includes(options.category as ModelCategory));
  }

  if (options.format === "json") {
    const output = merged.map((m) => ({
      name: m.entry?.name ?? m.libraryModel?.slug ?? m.ollamaModel?.name ?? "",
      slug: m.libraryModel?.slug ?? null,
      ollamaTag: m.ollamaTag,
      installed: m.installed,
      sources: m.sources,
      provider: m.entry?.provider ?? null,
      parametersBillion: m.entry?.parametersBillion ?? null,
      parameterSize: m.ollamaModel?.parameterSize ?? null,
      parameterSizes: m.libraryModel?.parameterSizes ?? null,
      capabilities: m.libraryModel?.capabilities ?? null,
      description: m.libraryModel?.description ?? null,
      quantization: m.ollamaModel?.quantization ?? null,
      family: m.ollamaModel?.family ?? null,
      sizeBytes: m.ollamaModel?.size ?? null,
    }));
    console.log(JSON.stringify(output, null, 2));
    return;
  }

  if (options.format === "csv") {
    const headers = [
      "name", "slug", "ollamaTag", "installed", "sources", "provider",
      "parametersBillion", "parameterSizes", "capabilities", "quantization", "family", "sizeBytes",
    ];
    const rows = merged.map((m) => [
      m.entry?.name ?? m.libraryModel?.slug ?? m.ollamaModel?.name ?? "",
      m.libraryModel?.slug ?? null,
      m.ollamaTag,
      m.installed,
      m.sources.join(";"),
      m.entry?.provider ?? null,
      m.entry?.parametersBillion ?? null,
      m.libraryModel?.parameterSizes.join(";") ?? null,
      m.libraryModel?.capabilities.join(";") ?? null,
      m.ollamaModel?.quantization ?? null,
      m.ollamaModel?.family ?? null,
      m.ollamaModel?.size ?? null,
    ]);
    console.log(toCsv(headers, rows));
    return;
  }

  const label = options.installed
    ? "Installed Models"
    : includeLibrary
      ? "All Models (Library + Local)"
      : "All Models (Live)";
  console.log(sectionHeader(`${label} (${merged.length} results)`));
  console.log();

  if (merged.length === 0) {
    console.log(`  ${theme.warning("No models found.")}`);
    console.log();
    return;
  }

  const table = new Table({
    head: [
      theme.muted("Model"),
      theme.muted("Provider"),
      theme.muted("Params"),
      theme.muted("Quant"),
      theme.muted("Size"),
      theme.muted("Fit"),
      theme.muted("Source"),
    ],
    style: { head: [], border: ["gray"], compact: true },
    chars: borderlessTableChars,
  });

  for (const m of merged) {
    const name = m.entry?.name ?? m.libraryModel?.slug ?? m.ollamaModel?.name ?? "—";
    const provider = m.entry?.provider ?? m.ollamaModel?.family ?? (m.libraryModel ? "ollama.com" : "—");
    const params = m.entry
      ? `${m.entry.parametersBillion}B`
      : m.ollamaModel?.parameterSize
        ?? (m.libraryModel?.parameterSizes.length ? m.libraryModel.parameterSizes.join("/") : "—");
    const quant = m.ollamaModel?.quantization
      ?? m.entry?.quantizations[0]?.name
      ?? "—";
    const sizeGb = m.ollamaModel
      ? `${(m.ollamaModel.size / 1024 / 1024 / 1024).toFixed(1)} GB`
      : "—";

    // Fit label: score if DB entry + hardware, "unrated" if Ollama-only
    let fitLabel: string;
    if (m.entry && hardware) {
      const entry = m.entry;
      const hw = hardware;
      const scores = entry.quantizations
        .map((q) => scoreModel(entry, q, hw, options.category))
        .filter((s) => !options.fits || isFitting(s.fitLevel))
        .sort((a, b) => b.compositeScore - a.compositeScore);
      fitLabel = scores.length > 0 ? fitBadge(scores[0].fitLevel) : theme.fail("✗");
    } else if (!m.entry) {
      fitLabel = theme.muted("unrated");
    } else {
      fitLabel = theme.muted("—");
    }

    const sourceLabel = formatSources(m.sources);

    table.push([
      theme.value(name),
      theme.muted(provider),
      theme.number(params),
      theme.muted(quant),
      theme.number(sizeGb),
      fitLabel,
      sourceLabel,
    ]);
  }

  console.log(table.toString());
  console.log();
}

function formatSources(sources: MergedModel["sources"]): string {
  const parts: string[] = [];
  if (sources.includes("installed")) parts.push(theme.pass("installed"));
  if (sources.includes("curated")) parts.push(theme.value("curated"));
  if (sources.includes("library")) parts.push(theme.muted("library"));
  return parts.length > 0 ? parts.join(theme.muted("+")) : theme.muted("—");
}
