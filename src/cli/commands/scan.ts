import ora from "ora";
import { detectHardware } from "../../hardware/index.js";
import { detectAllRuntimes } from "../../runtimes/index.js";
import { fetchOllamaModels } from "../../models/ollama-models.js";
import { getRecommendations } from "../../analysis/recommender.js";
import { getCatalogHealth } from "../../models/database.js";
import { titleBox, sectionHeader, keyValue, subLine, wrappedSubLines } from "../ui/boxes.js";
import { theme } from "../ui/colors.js";
import { gpuLabel, gpuNameRows, gpuNames, progressBar, formatMb } from "../ui/progress.js";
import { recommendationTable } from "../ui/tables.js";
import { MEMORY_CSV_HEADERS, memoryCsvCells, toCsv } from "../ui/csv.js";
import { resolveOllamaHost } from "../../core/config.js";
import type { ScanOptions, HardwareProfile, RuntimeInfo } from "../../core/types.js";
import { budgetGpus, getAvailableVram, ignoredGpu, ignoredGpuLabel, unifiedMemoryNote } from "../../analysis/scorer.js";
import { memoryBreakdown, memoryNeedMb } from "../../analysis/context-capacity.js";
import { quantFileBpw } from "../../models/quant-facts.js";

export async function scanCommand(options: ScanOptions): Promise<void> {
  const ollamaHost = resolveOllamaHost(options.host);

  // JSON/CSV mode: no spinners
  if (options.format === "json") {
    return scanJson(options, ollamaHost);
  }
  if (options.format === "csv") {
    return scanCsv(options, ollamaHost);
  }

  const spinner = ora({ text: "Detecting hardware (CPU, GPU, RAM, disk)...", color: "cyan" }).start();

  const [hardware, runtimes, ollamaModels] = await Promise.all([
    detectHardware(),
    detectAllRuntimes(ollamaHost),
    fetchOllamaModels(ollamaHost),
  ]);

  spinner.succeed("Scan complete");

  const installedTags = new Set(ollamaModels.map((m) => m.name));

  const recommendations = getRecommendations(hardware, {
    category: options.category,
    top: options.top,
    onlyFitting: true,
  });

  // Build output
  const lines: string[] = [];

  // ── Hardware Section ──
  lines.push(sectionHeader("Hardware"));
  lines.push("");
  lines.push(formatCpu(hardware));
  lines.push(formatGpu(hardware));
  lines.push(formatMemory(hardware));
  lines.push(formatDisk(hardware));

  // ── Runtimes Section ──
  lines.push(sectionHeader("Runtimes Detected"));
  lines.push("");
  for (const rt of runtimes) {
    lines.push(formatRuntime(rt));
  }

  // ── Recommendations Section ──
  lines.push(sectionHeader("Recommended Models for Your Hardware"));
  lines.push("");
  if (recommendations.length > 0) {
    lines.push(recommendationTable(recommendations, hardware, installedTags));
    lines.push("");
    const top = recommendations[0];
    const topTag = top.score.model.ollamaTag;
    if (topTag && installedTags.has(topTag)) {
      lines.push(`  ${theme.pass("●")} Top pick already installed: ${theme.command(`ollama run ${topTag}`)}`);
    } else if (top.pullCommand) {
      lines.push(`  Run: ${theme.command(top.pullCommand)}`);
    }
  } else {
    lines.push(`  ${theme.warning("No models fit your hardware with current filters.")}`);
  }

  // ── Health Summary ──
  const maxParams = estimateMaxParams(hardware);
  lines.push("");
  lines.push(`  ${theme.muted("Tip:")} You can comfortably run up to ${theme.highlight(`${maxParams}B`)} parameter models with Q4`);
  lines.push(subLine("quantization at a 4K context (est.)."));
  const memoryNote = unifiedMemoryNote(hardware);
  if (memoryNote) lines.push(`  ${theme.muted("Note:")} ${memoryNote}.`);

  console.log(titleBox(lines.join("\n")));
}

async function scanJson(options: ScanOptions, ollamaHost: string): Promise<void> {
  const [hardware, runtimes] = await Promise.all([
    detectHardware(),
    detectAllRuntimes(ollamaHost),
  ]);

  const recommendations = getRecommendations(hardware, {
    category: options.category,
    top: options.top,
    onlyFitting: true,
  });

  const catalog = getCatalogHealth();
  const output = {
    hardware: { ...hardware, gpuCount: budgetGpus(hardware).length },
    runtimes,
    catalog: {
      source: catalog.source,
      fetchedAt: catalog.fetchedAt === null ? null : new Date(catalog.fetchedAt).toISOString(),
      liveModels: catalog.liveModels,
    },
    recommendations: recommendations.map((r) => ({
      rank: r.rank,
      model: r.score.model.name,
      quantization: r.score.quantization.name,
      fitLevel: r.score.fitLevel,
      vramMb: r.score.quantization.vramMb,
      compositeScore: r.score.compositeScore,
      pullCommand: r.pullCommand,
      memory: memoryBreakdown(r.score.model, r.score.quantization, hardware),
    })),
  };

  console.log(JSON.stringify(output, null, 2));
}

async function scanCsv(options: ScanOptions, ollamaHost: string): Promise<void> {
  const [hardware, runtimes] = await Promise.all([
    detectHardware(),
    detectAllRuntimes(ollamaHost),
  ]);

  const recommendations = getRecommendations(hardware, {
    category: options.category,
    top: options.top,
    onlyFitting: true,
  });

  const headers = ["rank", "model", "quantization", "fitLevel", "vramMb", "compositeScore", "pullCommand", ...MEMORY_CSV_HEADERS];
  const rows = recommendations.map((r) => [
    r.rank,
    r.score.model.name,
    r.score.quantization.name,
    r.score.fitLevel,
    r.score.quantization.vramMb,
    r.score.compositeScore,
    r.pullCommand,
    ...memoryCsvCells(memoryBreakdown(r.score.model, r.score.quantization, hardware)),
  ]);

  console.log(toCsv(headers, rows));
}

function formatCpu(hw: HardwareProfile): string {
  const c = hw.cpu;
  // AVX2 is an x86-only instruction set. ARM CPUs (Apple Silicon, Snapdragon,
  // Ampere) use NEON instead — so "No AVX2" on arm64 is a category error, not
  // a warning. Surface NEON as a neutral fact and only warn on x86 without AVX2.
  const avx = c.hasAvx2
    ? theme.pass("AVX2 ✓")
    : c.architecture === "arm64"
      ? theme.muted("NEON")
      : theme.warning("No AVX2");
  // Speed may be null on Apple Silicon (systeminformation can't read it).
  // Show the accelerator class instead of a bogus GHz value.
  let speedLabel: string;
  if (c.speed === null) {
    speedLabel = hw.primaryGpu?.acceleratorType === "metal" ? "Metal" : "—";
  } else {
    speedLabel = `${c.speed} GHz`;
  }
  const lines = [
    keyValue("CPU", c.brand),
    subLine(`${c.threads} threads · ${avx} · ${speedLabel}`),
  ];
  return lines.join("\n");
}

function formatGpu(hw: HardwareProfile): string {
  if (!hw.primaryGpu || hw.primaryGpu.vramMb === 0) {
    return [
      keyValue("GPU", theme.warning("No dedicated GPU detected")),
      subLine("Models will run on CPU (slower)"),
    ].join("\n");
  }

  const ignored = ignoredGpu(hw);
  if (ignored) {
    return [
      keyValue("GPU", theme.warning("None — CPU inference")),
      ...wrappedSubLines(ignoredGpuLabel(ignored)),
    ].join("\n");
  }
  const budget = budgetGpus(hw);
  const g = budget[0];
  const vram = formatMb(budget.length > 1 ? budget.reduce((sum, gpu) => sum + gpu.vramMb, 0) : g.vramMb);
  const accelLabel = g.acceleratorType === "cuda" ? "CUDA"
    : g.acceleratorType === "rocm" ? "ROCm"
    : g.acceleratorType === "metal" ? "Metal" : null;
  const cuda = accelLabel && g.acceleratorVersion ? ` · ${accelLabel} ${g.acceleratorVersion}` : "";
  const lines = gpuNameRows(gpuNames(budget, gpuLabel));
  lines.push(subLine(`${vram} VRAM${cuda}`));

  if (g.utilizationPercent !== null) {
    lines.push(
      subLine(progressBar(g.utilizationPercent, 20, `${g.utilizationPercent}% utilized`)),
    );
  }

  return lines.join("\n");
}

function formatMemory(hw: HardwareProfile): string {
  const m = hw.memory;
  const total = formatMb(m.totalMb);
  const typeStr = m.type !== "Unknown" ? ` ${m.type}` : "";
  const speedStr = m.speedMhz ? ` @ ${m.speedMhz} MHz` : "";
  const lines = [
    keyValue("RAM", `${total}${typeStr}${speedStr}`),
    subLine(progressBar(m.usedPercent, 20, `${m.usedPercent}% used`)),
  ];
  return lines.join("\n");
}

function formatDisk(hw: HardwareProfile): string {
  const d = hw.disk;
  return keyValue("Disk", `${d.type} · ${d.freeGb} GiB free`);
}

function formatRuntime(rt: RuntimeInfo): string {
  if (rt.status === "not_found") {
    return `  ${theme.fail("✗")} ${theme.muted(`${rt.name} (not found)`)}`;
  }

  const icon = rt.status === "running" ? theme.pass("✓") : theme.warning("●");
  const version = rt.version ? ` v${rt.version}` : "";
  const status = rt.status === "running" ? theme.pass("running") : theme.warning("installed");
  let line = `  ${icon} ${rt.name}${version} (${status})`;

  if (rt.models.length > 0) {
    line += `\n${wrappedSubLines(`Models: ${rt.models.join(", ")}`).join("\n")}`;
  }

  return line;
}

// Context the tip is sized at: llama.cpp's --fit minimum (L common/common.h:506), also Ollama's smallest default
// tier. Sizing at the machine's default tier instead would make the tip shrink as memory grows (KV dominates).
const TIP_CONTEXT = 4096;

// ESTIMATE: the largest whole-billion size whose Q4_K_M weights, bucket KV cache and default-vocabulary compute
// buffer fit the budget at a 4K context. Real models of the same size differ in KV and vocabulary.
export function estimateMaxParams(hw: HardwareProfile): number {
  const budgetMb = getAvailableVram(hw);
  const q4kmBpw = quantFileBpw("Q4_K_M")!.bpw;
  const needMb = (paramsBillion: number) =>
    memoryNeedMb({ parametersBillion: paramsBillion }, { vramMb: (paramsBillion * 1e9 * q4kmBpw) / 8 / 2 ** 20 }, TIP_CONTEXT);

  let paramsBillion = 1;
  while (needMb(paramsBillion + 1) <= budgetMb) paramsBillion++;
  return paramsBillion;
}
