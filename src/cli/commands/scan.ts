import ora from "ora";
import { detectHardware } from "../../hardware/index.js";
import { detectAllRuntimes } from "../../runtimes/index.js";
import { fetchOllamaModels } from "../../models/ollama-models.js";
import { getRecommendations } from "../../analysis/recommender.js";
import { titleBox, sectionHeader, keyValue, subLine } from "../ui/boxes.js";
import { theme } from "../ui/colors.js";
import { progressBar, formatMb } from "../ui/progress.js";
import { recommendationTable } from "../ui/tables.js";
import type { ScanOptions, HardwareProfile, RuntimeInfo } from "../../core/types.js";

export async function scanCommand(options: ScanOptions): Promise<void> {
  // JSON mode: no spinners
  if (options.format === "json") {
    return scanJson(options);
  }

  const spinner = ora({ text: "Scanning hardware...", color: "cyan" }).start();

  const [hardware, runtimes, ollamaModels] = await Promise.all([
    detectHardware(),
    detectAllRuntimes(),
    fetchOllamaModels(),
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
    lines.push(recommendationTable(recommendations, installedTags));
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
  lines.push(
    `  ${theme.muted("Tip:")} You can comfortably run up to ${theme.highlight(`${maxParams}B`)} parameter models with Q4 quantization.`,
  );

  console.log(titleBox(lines.join("\n")));
}

async function scanJson(options: ScanOptions): Promise<void> {
  const [hardware, runtimes] = await Promise.all([
    detectHardware(),
    detectAllRuntimes(),
  ]);

  const recommendations = getRecommendations(hardware, {
    category: options.category,
    top: options.top,
    onlyFitting: true,
  });

  const output = {
    hardware,
    runtimes,
    recommendations: recommendations.map((r) => ({
      rank: r.rank,
      model: r.score.model.name,
      quantization: r.score.quantization.name,
      fitLevel: r.score.fitLevel,
      vramMb: r.score.quantization.vramMb,
      compositeScore: r.score.compositeScore,
      pullCommand: r.pullCommand,
    })),
  };

  console.log(JSON.stringify(output, null, 2));
}

function formatCpu(hw: HardwareProfile): string {
  const c = hw.cpu;
  const avx = c.hasAvx2 ? theme.pass("AVX2 ✓") : theme.warning("No AVX2");
  const lines = [
    keyValue("CPU", c.brand),
    subLine(`${c.threads} threads · ${avx} · ${c.speed} GHz`),
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

  const g = hw.primaryGpu;
  const vram = formatMb(g.vramMb);
  const cuda = g.cudaVersion ? ` · CUDA ${g.cudaVersion}` : "";
  const lines = [keyValue("GPU", `${g.vendor} ${g.model}`)];
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
  return keyValue("Disk", `${d.type} · ${d.freeGb} GB free`);
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
    line += `\n${subLine(`Models: ${rt.models.join(", ")}`)}`;
  }

  return line;
}

function estimateMaxParams(hw: HardwareProfile): number {
  const vramMb = hw.primaryGpu?.vramMb ?? hw.memory.availableMb * 0.7;
  // Rough estimate: Q4_K_M uses ~0.6 GB per billion params
  const maxB = Math.floor(vramMb / 600);
  return Math.max(1, maxB);
}
