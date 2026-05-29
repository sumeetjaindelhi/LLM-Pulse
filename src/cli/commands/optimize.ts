import ora from "ora";
import { detectHardware } from "../../hardware/index.js";
import { computeOptimization } from "../../analysis/optimizer.js";
import { getAvailableVram } from "../../analysis/scorer.js";
import { resolveModel } from "../../models/database.js";
import { titleBox, sectionHeader, keyValue, subLine } from "../ui/boxes.js";
import { theme } from "../ui/colors.js";
import { fitBadge } from "../ui/badges.js";
import { formatMb } from "../ui/progress.js";
import { renderModelNotFound } from "../ui/errors.js";
import { toCsv } from "../ui/csv.js";
import type {
  ModelEntry,
  HardwareProfile,
  OptimizationProfile,
  OptimizedParameter,
  OptimizeOptions,
  QuantizationVariant,
} from "../../core/types.js";

// Bundles everything the three output paths (table/JSON/CSV) share, so each
// render function takes one parameter. New fields land here rather than threading
// through three signatures — same pattern as quant-advice's AdviceContext.
interface OptimizeContext {
  model: ModelEntry;
  hardware: HardwareProfile;
  profile: OptimizationProfile;
  availableVramMb: number;
  quantWarning: string | null;
}

// "llama3.1:8b" -> "llama3.1-8b-tuned": a valid Ollama model name for `create`.
function tunedName(tag: string): string {
  return `${tag.replace(/[:/]/g, "-")}-tuned`;
}

// The PARAMETER lines worth pasting: num_ctx and num_thread always; num_gpu only
// for a partial offload (a specific layer count — "all" and 0 are Ollama's own
// defaults); num_batch only when it deviates from the 512 default. Keeping the
// list to what actually changes behaviour avoids a Modelfile full of no-ops.
function modelfileParams(profile: OptimizationProfile): Array<[string, number]> {
  const params: Array<[string, number]> = [["num_ctx", profile.numCtx.value as number]];
  if (typeof profile.numGpu?.value === "number" && profile.numGpu.value > 0) {
    params.push(["num_gpu", profile.numGpu.value]);
  }
  params.push(["num_thread", profile.numThread.value as number]);
  if (profile.numBatch.value !== 512) {
    params.push(["num_batch", profile.numBatch.value as number]);
  }
  return params;
}

// The FROM + PARAMETER block, or null when the model isn't on Ollama (no tag to
// build FROM against).
function buildModelfile(profile: OptimizationProfile, model: ModelEntry): string | null {
  if (!model.ollamaTag) return null;
  const lines = [`FROM ${model.ollamaTag}`];
  for (const [name, value] of modelfileParams(profile)) {
    lines.push(`PARAMETER ${name} ${value}`);
  }
  return lines.join("\n");
}

export async function optimizeCommand(
  modelArg: string,
  options: OptimizeOptions,
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

  // Resolve --quant if given; fall back to the sweet spot (with a warning) when
  // the requested name isn't available for this model.
  let forcedQuant: QuantizationVariant | undefined;
  let quantWarning: string | null = null;
  if (options.quant) {
    const match = model.quantizations.find(
      (q) => q.name.toLowerCase() === options.quant!.toLowerCase(),
    );
    if (match) {
      forcedQuant = match;
    } else {
      quantWarning = `Quant ${options.quant} not available for ${model.name}; using the sweet-spot pick`;
    }
  }

  const profile = computeOptimization(model, hardware, forcedQuant);
  if (!profile) {
    renderNoFit(model, hardware, silent);
    return;
  }

  const ctx: OptimizeContext = {
    model,
    hardware,
    profile,
    availableVramMb: getAvailableVram(hardware),
    quantWarning,
  };

  if (isJson) {
    outputJson(ctx);
  } else if (isCsv) {
    outputCsv(ctx);
  } else {
    outputTable(ctx, options.verbose);
  }
}

// Aligned "name  value  why" lines. Built outside cli-table3 for the same reason
// quant-advice renders its notes this way: at 80-col terminals boxen's padding
// shrinks the content area to ~72 cols, and a wrapped table row loses its leading
// whitespace and mis-aligns. Free-flowing padded lines avoid both.
function paramLines(profile: OptimizationProfile): string {
  const rows: Array<{ name: string; p: OptimizedParameter }> = [
    { name: "num_ctx", p: profile.numCtx },
  ];
  if (profile.numGpu) rows.push({ name: "num_gpu", p: profile.numGpu });
  rows.push({ name: "num_thread", p: profile.numThread });
  rows.push({ name: "num_batch", p: profile.numBatch });

  const nameWidth = Math.max(...rows.map((r) => r.name.length));
  const valWidth = Math.max(...rows.map((r) => String(r.p.value).length));
  return rows
    .map((r) => {
      const name = theme.command(r.name.padEnd(nameWidth));
      const val = theme.number(String(r.p.value).padEnd(valWidth));
      return `  ${name}  ${val}  ${theme.muted(r.p.note)}`;
    })
    .join("\n");
}

function outputTable(ctx: OptimizeContext, verbose: boolean): void {
  const { model, hardware, profile, availableVramMb, quantWarning } = ctx;
  const lines: string[] = [];

  lines.push("");
  lines.push(`  ${theme.header("Runtime Optimization")} — ${theme.value(model.name)}`);

  lines.push(sectionHeader("Model"));
  lines.push(keyValue("By", model.provider));
  lines.push(keyValue("Param", `${model.parametersBillion}B`));
  const quantTag = profile.quantForced ? "(forced)" : "(sweet spot — picked for you)";
  lines.push(`  ${theme.label("Quant".padEnd(6))}${theme.pass(`★ ${profile.quant.name}`)} ${theme.muted(quantTag)}`);
  if (verbose) {
    lines.push(`  ${theme.label("Fit".padEnd(6))}${fitBadge(profile.fitLevel)}`);
  }

  if (quantWarning) {
    lines.push("");
    lines.push(`  ${theme.warning("⚠")} ${theme.warning(quantWarning)}`);
  }

  lines.push(sectionHeader("Your Hardware"));
  if (hardware.primaryGpu) {
    lines.push(keyValue("GPU", `${hardware.primaryGpu.model} · ${formatMb(availableVramMb)}`));
  } else {
    lines.push(keyValue("GPU", "None — CPU inference"));
  }
  lines.push(keyValue("CPU", `${hardware.cpu.cores} cores`));

  lines.push(sectionHeader("Recommended Parameters"));
  lines.push(paramLines(profile));

  const modelfile = buildModelfile(profile, model);
  if (modelfile) {
    lines.push(sectionHeader("Modelfile (persistent)"));
    for (const line of modelfile.split("\n")) {
      lines.push(`    ${theme.command(line)}`);
    }
    lines.push(`  ${theme.muted("→")} ${theme.command(`ollama create ${tunedName(model.ollamaTag!)} -f Modelfile`)}`);

    lines.push(sectionHeader("Quick (this session)"));
    lines.push(`  ${theme.muted("→")} ${theme.command(`ollama run ${model.ollamaTag}`)}`);
    for (const [name, value] of modelfileParams(profile)) {
      lines.push(`    ${theme.command(`/set parameter ${name} ${value}`)}`);
    }
  } else {
    lines.push("");
    lines.push(`  ${theme.muted("Not on Ollama — these apply to any llama.cpp-based runtime.")}`);
  }

  if (hardware.primaryGpu?.acceleratorType === "metal") {
    lines.push(subLine("(Apple Silicon: unified memory — Ollama offloads all layers)"));
  }

  console.log(titleBox(lines.join("\n")));
}

function outputJson(ctx: OptimizeContext): void {
  const { model, hardware, profile, availableVramMb } = ctx;
  const param = (p: OptimizedParameter | null) => (p ? { value: p.value, note: p.note } : null);

  const output = {
    model: {
      id: model.id,
      name: model.name,
      provider: model.provider,
      parametersBillion: model.parametersBillion,
      contextWindow: model.contextWindow,
      ollamaTag: model.ollamaTag,
    },
    hardware: {
      gpu: hardware.primaryGpu?.model ?? null,
      vramMb: availableVramMb,
      cpuCores: hardware.cpu.cores,
    },
    quant: {
      name: profile.quant.name,
      bitsPerWeight: profile.quant.bitsPerWeight,
      vramMb: profile.quant.vramMb,
      qualityRetention: profile.quant.qualityRetention,
      forced: profile.quantForced,
    },
    fitLevel: profile.fitLevel,
    parameters: {
      num_ctx: param(profile.numCtx),
      num_gpu: param(profile.numGpu),
      num_thread: param(profile.numThread),
      num_batch: param(profile.numBatch),
    },
    modelfile: buildModelfile(profile, model),
  };

  console.log(JSON.stringify(output, null, 2));
}

function outputCsv(ctx: OptimizeContext): void {
  const { model, profile } = ctx;
  const headers = ["model", "quant", "parameter", "value", "note"];
  const rows: (string | number)[][] = [];
  const add = (name: string, p: OptimizedParameter | null) => {
    if (p) rows.push([model.id, profile.quant.name, name, p.value, p.note]);
  };
  add("num_ctx", profile.numCtx);
  add("num_gpu", profile.numGpu);
  add("num_thread", profile.numThread);
  add("num_batch", profile.numBatch);
  console.log(toCsv(headers, rows));
}

function renderNoFit(model: ModelEntry, hardware: HardwareProfile, silent: boolean): void {
  const availableVramMb = getAvailableVram(hardware);
  if (silent) {
    console.log(JSON.stringify({ error: "No quantization fits", model: model.id, vramMb: availableVramMb }, null, 2));
    return;
  }
  const lines: string[] = [];
  lines.push("");
  lines.push(`  ${theme.fail("✗")} No quantization of ${theme.value(model.name)} fits your ${formatMb(availableVramMb)}.`);
  lines.push(`  ${theme.muted("Try:")} ${theme.command(`llm-pulse check ${model.ollamaTag ?? model.id}`)} ${theme.muted("for layer-offload tips")}`);
  lines.push(`  ${theme.muted("Or browse smaller models:")} ${theme.command("llm-pulse models --fits")}`);
  console.log(titleBox(lines.join("\n")));
}
