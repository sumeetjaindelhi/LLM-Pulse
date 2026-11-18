import ora from "ora";
import { detectHardware } from "../../hardware/index.js";
import { computeOptimization, kvQ8MaxContext } from "../../analysis/optimizer.js";
import { budgetExact, budgetGpus, defaultContextFor, getAvailableVram, isTagFile } from "../../analysis/scorer.js";
import { memoryBreakdown } from "../../analysis/context-capacity.js";
import { autoBatch } from "../../analysis/memory-model.js";
import { findQuant, resolveModel } from "../../models/database.js";
import { withExactFacts } from "../../runtimes/ollama.js";
import { resolveOllamaHost } from "../../core/config.js";
import { titleBox, sectionHeader, keyValue, subLine } from "../ui/boxes.js";
import { theme } from "../ui/colors.js";
import { fitBadge } from "../ui/badges.js";
import { formatBreakdown, gpuRows, formatMb, formatNeedAt, formatUsable, formatUsableMb, pullLines } from "../ui/progress.js";
import { renderModelNotFound } from "../ui/errors.js";
import { toCsv } from "../ui/csv.js";
import type {
  ModelEntry,
  HardwareProfile,
  MemoryBreakdown,
  OptimizationProfile,
  OptimizedParameter,
  OptimizeOptions,
  QuantizationVariant,
} from "../../core/types.js";
import { isUnifiedMemory } from "../../core/unified-memory.js";
import { quantRetention } from "../../models/quant-facts.js";

// Bundles everything the three output paths (table/JSON/CSV) share, so each
// render function takes one parameter. New fields land here rather than threading
// through three signatures — same pattern as quant-advice's AdviceContext.
interface OptimizeContext {
  model: ModelEntry;
  hardware: HardwareProfile;
  profile: OptimizationProfile;
  availableVramMb: number;
  quantWarning: string | null;
  defaultContext: number;
  kvQ8Max: number | null;
  memory: MemoryBreakdown; // need at num_ctx
}

// "llama3.1:8b" -> "llama3.1-8b-tuned": a valid Ollama model name for `create`.
function tunedName(tag: string): string {
  return `${tag.replace(/[:/]/g, "-")}-tuned`;
}

// The PARAMETER lines worth pasting: num_ctx and num_thread always; num_gpu only
// for a partial offload (a specific layer count — "all" and 0 are Ollama's own
// defaults); num_batch only when pinned below the automatic batch. Keeping the
// list to what actually changes behaviour avoids a Modelfile full of no-ops.
function modelfileParams(profile: OptimizationProfile): Array<[string, number]> {
  const params: Array<[string, number]> = [["num_ctx", profile.numCtx.value as number]];
  if (typeof profile.numGpu?.value === "number" && profile.numGpu.value > 0) {
    params.push(["num_gpu", profile.numGpu.value]);
  }
  params.push(["num_thread", profile.numThread.value as number]);
  if (profile.numBatch.value !== autoBatch(profile.numCtx.value as number)) {
    params.push(["num_batch", profile.numBatch.value as number]);
  }
  return params;
}

// The model name FROM and `ollama run` take: the tag when it installs the tuned
// quant, else a placeholder, so a pasted line fails instead of loading another
// file (pullLines says where that quant's tag is listed).
function runName(model: ModelEntry, quant: QuantizationVariant): string {
  return isTagFile(model, quant) ? model.ollamaTag! : `<${quant.name} tag>`;
}

// The FROM + PARAMETER block, or null when the model isn't on Ollama (no tag to
// build FROM against).
function buildModelfile(profile: OptimizationProfile, model: ModelEntry): string | null {
  if (!model.ollamaTag) return null;
  const lines = [`FROM ${runName(model, profile.quant)}`];
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

  const resolved = resolveModel(modelArg);
  if (!resolved) {
    renderModelNotFound(modelArg, { silent });
    return;
  }
  spinner?.start("Reading model facts...");
  const model = await withExactFacts(resolveOllamaHost(), resolved);
  spinner?.stop();

  // Resolve --quant if given; fall back to the sweet spot (with a warning) when
  // the requested name isn't available for this model.
  let forcedQuant: QuantizationVariant | undefined;
  let quantWarning: string | null = null;
  if (options.quant) {
    const match = findQuant(model, options.quant);
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

  const numCtx = profile.numCtx.value as number;
  // A num_batch the Modelfile pins also runs as the ubatch (O llm/llama_server.go:630-631).
  const pinnedBatch = modelfileParams(profile).find(([name]) => name === "num_batch")?.[1];
  const ctx: OptimizeContext = {
    model,
    hardware,
    profile,
    availableVramMb: getAvailableVram(hardware),
    quantWarning,
    defaultContext: defaultContextFor(model, hardware),
    kvQ8Max: kvQ8MaxContext(model, profile.quant, hardware, numCtx),
    memory: memoryBreakdown(model, profile.quant, hardware, numCtx, "f16", pinnedBatch),
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
  const { model, hardware, profile, availableVramMb, quantWarning, defaultContext, kvQ8Max, memory } = ctx;
  const lines: string[] = [];

  lines.push("");
  lines.push(`  ${theme.header("Runtime Optimization")} — ${theme.value(model.name)}`);

  lines.push(sectionHeader("Model"));
  lines.push(keyValue("By", model.provider));
  lines.push(keyValue("Param", `${model.parametersBillion}B`));
  const quantTag = profile.quantForced
    ? "(forced)"
    : profile.quant.exact ? "(the file the tag installs)" : "(sweet spot — picked for you)";
  lines.push(`  ${theme.label("Quant".padEnd(6))}${theme.pass(`★ ${profile.quant.name}`)} ${theme.muted(quantTag)}`);
  if (verbose) {
    lines.push(`  ${theme.label("Fit".padEnd(6))}${fitBadge(profile.fitLevel)}`);
  }
  // FROM and `ollama run` below load the tag's file, which a forced quant or a sweet spot may not be.
  lines.push(...pullLines(model, profile.quant));

  if (quantWarning) {
    lines.push("");
    lines.push(`  ${theme.warning("⚠")} ${theme.warning(quantWarning)}`);
  }

  lines.push(sectionHeader("Your Hardware"));
  lines.push(...gpuRows(hardware, availableVramMb, keyValue("GPU", "None — CPU inference")));
  lines.push(keyValue("CPU", `${hardware.cpu.cores} cores`));

  lines.push(sectionHeader("Recommended Parameters"));
  lines.push(paramLines(profile));

  lines.push(sectionHeader("Memory need"));
  lines.push(`  ${formatNeedAt(memory)}`);
  lines.push(subLine(formatBreakdown(memory)));
  lines.push(subLine(formatUsable(memory)));

  const modelfile = buildModelfile(profile, model);
  if (modelfile) {
    lines.push(sectionHeader("Modelfile (persistent)"));
    for (const line of modelfile.split("\n")) {
      lines.push(`    ${theme.command(line)}`);
    }
    lines.push(`  ${theme.muted("→")} ${theme.command(`ollama create ${tunedName(model.ollamaTag!)} -f Modelfile`)}`);

    lines.push(sectionHeader("Quick (this session)"));
    lines.push(`  ${theme.muted("→")} ${theme.command(`ollama run ${runName(model, profile.quant)}`)}`);
    for (const [name, value] of modelfileParams(profile)) {
      lines.push(`    ${theme.command(`/set parameter ${name} ${value}`)}`);
    }

    lines.push(sectionHeader("Server-wide (all models)"));
    lines.push(`  ${theme.muted("→")} ${theme.command(`OLLAMA_CONTEXT_LENGTH=${profile.numCtx.value} ollama serve`)}`);
    lines.push(subLine(`Ollama's default context here: ${defaultContext.toLocaleString()} tokens`));
  } else {
    lines.push("");
    lines.push(`  ${theme.muted("Not on Ollama — these apply to any llama.cpp-based runtime.")}`);
  }

  // Two lines: one over ~72 visible columns puts boxen in wrap mode, which strips every line's indent.
  if (kvQ8Max !== null) {
    lines.push(subLine("Usable memory limits num_ctx; with OLLAMA_FLASH_ATTENTION=1"));
    lines.push(subLine(`OLLAMA_KV_CACHE_TYPE=q8_0 it affords up to ${kvQ8Max.toLocaleString()} tokens`));
  }

  const [budgetGpu] = budgetGpus(hardware);
  if (budgetGpu && isUnifiedMemory(budgetGpu)) {
    lines.push(subLine("(unified memory — Ollama offloads all layers)"));
  }

  console.log(titleBox(lines.join("\n")));
}

function outputJson(ctx: OptimizeContext): void {
  const { model, hardware, profile, availableVramMb, defaultContext, kvQ8Max, memory } = ctx;
  const param = (p: OptimizedParameter | null) => (p ? { value: p.value, note: p.note } : null);

  const output = {
    model: {
      id: model.id,
      name: model.name,
      provider: model.provider,
      parametersBillion: model.parametersBillion,
      contextWindow: model.contextWindow,
      contextWindowEstimated: model.contextWindowEstimated === true,
      ollamaTag: model.ollamaTag,
    },
    hardware: {
      gpu: hardware.primaryGpu?.model ?? null,
      gpuCount: budgetGpus(hardware).length,
      vramMb: availableVramMb,
      cpuCores: hardware.cpu.cores,
    },
    quant: {
      name: profile.quant.name,
      bitsPerWeight: profile.quant.bitsPerWeight,
      vramMb: profile.quant.vramMb,
      qualityRetention: profile.quant.qualityRetention,
      qualityMeasured: quantRetention(profile.quant.name).published,
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
    ollamaDefaultContext: defaultContext,
    kvQ8MaxContext: kvQ8Max,
    memory,
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
  process.exitCode = 1;
  const availableVramMb = getAvailableVram(hardware);
  if (silent) {
    console.log(JSON.stringify({ error: "No quantization fits", model: model.id, vramMb: availableVramMb }, null, 2));
    return;
  }
  const lines: string[] = [];
  lines.push("");
  lines.push(`  ${theme.fail("✗")} No quantization of ${theme.value(model.name)} fits`);
  lines.push(subLine(`your ${formatUsableMb(availableVramMb, budgetExact(hardware))} of usable memory.`));
  lines.push(`  ${theme.muted("Try:")} ${theme.command(`llm-pulse check ${model.ollamaTag ?? model.id}`)} ${theme.muted("for layer-offload tips")}`);
  lines.push(`  ${theme.muted("Or browse smaller models:")} ${theme.command("llm-pulse models --fits")}`);
  console.log(titleBox(lines.join("\n")));
}
