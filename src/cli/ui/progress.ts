import { theme } from "./colors.js";
import { BOX_CONTENT_COLUMNS, keyValue, subLine, wrappedSubLines } from "./boxes.js";
import { budgetExact, budgetGpus, ignoredGpu, ignoredGpuLabel, isTagFile } from "../../analysis/scorer.js";
import { quantFileBpw } from "../../models/quant-facts.js";
import type { GpuInfo, HardwareProfile, MemoryBreakdown, ModelEntry, QuantizationVariant } from "../../core/types.js";

export function progressBar(
  percent: number,
  width = 20,
  label?: string,
): string {
  const clamped = Math.max(0, Math.min(100, percent));
  const filled = Math.round((clamped / 100) * width);
  const empty = width - filled;

  const bar = "█".repeat(filled) + "░".repeat(empty);

  const coloredBar =
    clamped >= 90 ? theme.fail(bar)
    : clamped >= 70 ? theme.warning(bar)
    : theme.pass(bar);

  const suffix = label ? ` ${label}` : ` ${clamped}%`;
  return `${coloredBar}${theme.muted(suffix)}`;
}

export function formatMb(mb: number): string {
  if (mb >= 1024) {
    return `${(mb / 1024).toFixed(1)} GiB`;
  }
  return `${Math.round(mb).toLocaleString()} MiB`;
}

const estimated = (exact: boolean): string => (exact ? "" : " est.");

/** Total need, " est." when its weights or KV cache are estimated; the compute buffer, always an estimate, does not decide it. */
export function formatNeed(memory: MemoryBreakdown): string {
  return `${formatMb(memory.totalMb)}${estimated(memory.weightsExact && memory.kvExact)}`;
}

/** "12.5 GiB est. at 32,768 tokens"; pair it with formatBreakdown, which marks each estimated part. */
export function formatNeedAt(memory: MemoryBreakdown): string {
  return `${formatNeed(memory)} at ${memory.contextTokens.toLocaleString()} tokens`;
}

/** "22.5 GiB est.": a usable-memory figure, marked when it includes llm-pulse's own reserve estimate. */
export function formatUsableMb(mb: number, exact: boolean): string {
  return `${formatMb(mb)}${estimated(exact)}`;
}

/** "of 22.5 GiB usable est.": its own line, since it does not fit beside formatNeedAt within 72 columns. */
export function formatUsable(memory: MemoryBreakdown): string {
  return `of ${formatMb(memory.budgetMb)} usable${estimated(memory.budgetExact)}`;
}

/** "weights 7.9 GiB est. · KV 4.0 GiB · compute ≈551 MiB" */
export function formatBreakdown(memory: MemoryBreakdown): string {
  return `weights ${formatMb(memory.weightsMb)}${estimated(memory.weightsExact)} · KV ${formatMb(memory.kvCacheMb)}${estimated(memory.kvExact)} · compute ≈${formatMb(memory.computeMb)}`;
}

/** Weights of one quant, " est." unless they are the real file size. */
export function formatWeights(quant: QuantizationVariant): string {
  return `${formatMb(quant.vramMb)}${estimated(quant.exact === true)}`;
}

/** "→ ollama pull <tag>  installs Q4_K_M (4.6 GiB)", naming the file when known, then where `pick`'s tag is when it is
 *  another quant: every command naming the tag (pull, run, FROM) loads that one file. None for a model not on Ollama. */
export function pullLines(model: ModelEntry, pick: QuantizationVariant): string[] {
  if (!model.ollamaTag) return [];
  const pulled = model.quantizations.find((q) => q.exact);
  const installs = pulled ? theme.muted(`  installs ${pulled.name} (${formatMb(pulled.vramMb)})`) : "";
  const lines = [`  ${theme.muted("→")} ${theme.command(`ollama pull ${model.ollamaTag}`)}${installs}`];
  if (!isTagFile(model, pick)) {
    lines.push(subLine(`For ${pick.name}, pick its tag at`));
    lines.push(subLine(`https://ollama.com/library/${model.ollamaTag.split(":")[0]}/tags`));
  }
  return lines;
}

/** One decimal, floored: Q8_0's 99.865% must not print as the F16 reference's 100%. */
export function formatQualityPct(retention: number): string {
  return `${(Math.floor(retention * 1000 + 1e-9) / 10).toFixed(1)}%`;
}

/** "4.9" for a measured file figure, "≈4.9" when derived from nominal bits or a library placeholder ("default", "ternary"). */
export function formatBits(quant: QuantizationVariant): string {
  const approximate = quantFileBpw(quant.name)?.estimated ?? true;
  return `${approximate ? "≈" : ""}${quant.bitsPerWeight.toFixed(1)}`;
}

/** The model with its vendor in front unless the driver string already starts with it ("NVIDIA GeForce RTX 4090", "Apple M2 Pro"). */
export function gpuLabel(gpu: GpuInfo): string {
  return gpu.model.toLowerCase().startsWith(gpu.vendor.toLowerCase()) ? gpu.model : `${gpu.vendor} ${gpu.model}`;
}

/** The GPUs the usable budget sums, "2× NVIDIA GeForce RTX 3090"; "2× A" when every GPU is the same model, else "A + B" ("2× A + B" for repeats), in first-seen order. */
export function gpuNames(gpus: GpuInfo[], label: (gpu: GpuInfo) => string = (gpu) => gpu.model): string {
  const counts = new Map<string, number>();
  for (const gpu of gpus) counts.set(label(gpu), (counts.get(label(gpu)) ?? 0) + 1);
  return [...counts].map(([name, count]) => (count > 1 ? `${count}× ${name}` : name)).join(" + ");
}

// keyValue: two spaces, then the label padded to six.
const GPU_ROW_INDENT = 8;

/** The GPU row for `names` ("A + B + C"); one model per line when the row would pass the box. */
export function gpuNameRows(names: string): string[] {
  if (GPU_ROW_INDENT + names.length <= BOX_CONTENT_COLUMNS) return [keyValue("GPU", names)];
  const [first, ...rest] = names.split(" + ");
  return [keyValue("GPU", first), ...rest.map((name) => subLine(`+ ${name}`))];
}

/**
 * The Hardware section's GPU row: the budget GPUs with the usable memory, the usable figure on a sub-line when one
 * row would pass the box (a longer line puts boxen in wrap mode). Without a budget GPU, `cpuRow` and, when
 * a GPU was detected but Ollama ignores it, why.
 */
export function gpuRows(hardware: HardwareProfile, usableMb: number, cpuRow: string): string[] {
  const gpus = budgetGpus(hardware);
  if (gpus.length > 0) {
    const name = gpuNames(gpus);
    const usable = `${formatMb(usableMb)} usable${estimated(budgetExact(hardware))}`;
    const row = `${name} · ${usable}`;
    return GPU_ROW_INDENT + row.length > BOX_CONTENT_COLUMNS ? [...gpuNameRows(name), subLine(usable)] : [keyValue("GPU", row)];
  }
  const ignored = ignoredGpu(hardware);
  return ignored ? [cpuRow, ...wrappedSubLines(ignoredGpuLabel(ignored))] : [cpuRow];
}
