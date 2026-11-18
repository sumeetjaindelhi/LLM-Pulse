import { detectHardware } from "../../hardware/index.js";
import { detectAllRuntimes } from "../../runtimes/index.js";
import { fetchOllamaModels } from "../../models/ollama-models.js";
import { getRecommendations } from "../../analysis/recommender.js";
import { runDiagnostics } from "../../analysis/doctor.js";
import { budgetExact, budgetGpus, getAvailableVram, ignoredGpu, ignoredGpuLabel, unifiedMemoryNote } from "../../analysis/scorer.js";
import { resolveOllamaHost } from "../../core/config.js";
import { VERSION } from "../../core/constants.js";
import { SEVERITY_ICONS } from "../ui/badges.js";
import { escapeMarkdownValue, toMarkdownTable } from "../ui/markdown.js";
import { formatMb, formatNeed, formatUsableMb, gpuLabel, gpuNames } from "../ui/progress.js";
import { NEED_NOTE, SPEED_NOTE } from "../ui/tables.js";
import { memoryBreakdown } from "../../analysis/context-capacity.js";
import type {
  GpuInfo,
  HardwareProfile,
  HealthReport,
  ModelCategory,
  Recommendation,
  RuntimeInfo,
  RuntimeStatus,
  ScanOptions,
} from "../../core/types.js";

export interface ReportInput {
  version: string;
  generatedAt: Date;
  platform: string;
  hardware: HardwareProfile;
  runtimes: RuntimeInfo[];
  installedTags: ReadonlySet<string>;
  recommendations: Recommendation[];
  health: HealthReport;
  availableVramMb: number;
  category: ModelCategory | "all";
  top: number;
}

const ACCELERATOR_LABELS: Partial<Record<NonNullable<GpuInfo["acceleratorType"]>, string>> = {
  cuda: "CUDA",
  rocm: "ROCm",
  metal: "Metal",
};

const RUNTIME_STATUS_LABELS: Record<RuntimeStatus, string> = {
  running: "running",
  installed: "installed",
  not_found: "not found",
};

function cpuDetail(hardware: HardwareProfile): string {
  const { cpu } = hardware;
  // AVX2 is x86-only; arm64 CPUs use NEON, so "No AVX2" there would be misleading.
  const simd = cpu.hasAvx2 ? "AVX2" : cpu.architecture === "arm64" ? "NEON" : "No AVX2";
  return `${cpu.brand} · ${cpu.cores} cores / ${cpu.threads} threads · ${simd}`;
}

// The GPUs the usable budget sums: "2× NVIDIA GeForce RTX 3090 · 48.0 GiB VRAM".
function gpuDetail(hardware: HardwareProfile): string {
  const ignored = ignoredGpu(hardware);
  if (ignored) return ignoredGpuLabel(ignored);
  const budget = budgetGpus(hardware);
  if (budget.length === 0) return "None — CPU inference";
  const gpu = budget[0];
  const label = gpu.acceleratorType ? ACCELERATOR_LABELS[gpu.acceleratorType] : undefined;
  const accelerator = label && gpu.acceleratorVersion ? ` · ${label} ${gpu.acceleratorVersion}` : "";
  const vramMb = budget.length > 1 ? budget.reduce((sum, g) => sum + g.vramMb, 0) : gpu.vramMb;
  return `${gpuNames(budget, gpuLabel)} · ${formatMb(vramMb)} VRAM${accelerator}`;
}

function memoryDetail(hardware: HardwareProfile): string {
  const { memory } = hardware;
  const type = memory.type !== "Unknown" ? ` ${memory.type}` : "";
  const speed = memory.speedMhz ? ` @ ${memory.speedMhz} MHz` : "";
  return `${formatMb(memory.totalMb)}${type}${speed}`;
}

// Pure markdown serialization of an already-collected report. Exported so it can
// be unit-tested without hardware detection or console output.
export function renderMarkdownReport(input: ReportInput): string {
  const { hardware, health } = input;
  const category = escapeMarkdownValue(input.category);
  const date = input.generatedAt.toISOString().slice(0, 10);

  const blocks: string[] = [
    "## llm-pulse report",
    `_llm-pulse v${escapeMarkdownValue(input.version)} · ${escapeMarkdownValue(input.platform)} · ${date}_`,
    "### Hardware",
    toMarkdownTable(["Component", "Detail"], [
      ["CPU", cpuDetail(hardware)],
      ["GPU", gpuDetail(hardware)],
      ["RAM", memoryDetail(hardware)],
      ["Disk", `${hardware.disk.type} · ${hardware.disk.freeGb} GiB free`],
      ["Usable for inference", formatUsableMb(input.availableVramMb, budgetExact(hardware))],
    ]),
  ];

  const memoryNote = unifiedMemoryNote(hardware);
  if (memoryNote) blocks.push(`_Usable for inference: ${memoryNote}._`);

  blocks.push(
    "### Runtimes",
    toMarkdownTable(
      ["Runtime", "Status", "Version", "Models"],
      // The model list is only populated when the runtime's API answered, so a
      // count is shown for running runtimes only. Names and paths are never printed.
      input.runtimes.map((runtime) => [
        runtime.name,
        RUNTIME_STATUS_LABELS[runtime.status],
        runtime.version,
        runtime.status === "running" ? String(runtime.models.length) : null,
      ]),
    ),
    `### Health — ${health.score}/100 · ${escapeMarkdownValue(health.summary)}`,
    health.checks
      .map((check) => `- ${SEVERITY_ICONS[check.severity]} ${escapeMarkdownValue(check.message)}`)
      .join("\n"),
  );

  if (health.topSuggestion !== null) {
    blocks.push(`> Suggestion: ${escapeMarkdownValue(health.topSuggestion)}`);
  }

  blocks.push(`### Recommended models (${category}, top ${input.top})`);
  if (input.recommendations.length > 0) {
    blocks.push(
      toMarkdownTable(
        ["#", "Model", "Quant", "Fit", "Need", "Speed", "Installed"],
        input.recommendations.map(({ rank, score }) => [
          rank,
          score.model.name,
          score.quantization.name,
          score.fitLevel,
          formatNeed(memoryBreakdown(score.model, score.quantization, hardware)),
          score.speedEstimate,
          score.model.ollamaTag !== null && input.installedTags.has(score.model.ollamaTag) ? "yes" : null,
        ]),
      ),
      `_${NEED_NOTE}. ${SPEED_NOTE}._`,
    );
  } else {
    blocks.push(
      `_No curated models fit this hardware with the current filters (category: ${category}, top: ${input.top})._`,
    );
  }

  blocks.push(
    "---",
    "_Generated by [llm-pulse](https://github.com/sumeetjaindelhi/LLM-Pulse) · `npx llm-pulse report`_",
  );

  return blocks.join("\n\n");
}

export async function reportCommand(options: Omit<ScanOptions, "format">): Promise<void> {
  const ollamaHost = resolveOllamaHost(options.host);

  // No spinner: stdout must stay clean for `report > rig.md` and `report | pbcopy`.
  const [hardware, runtimes, ollamaModels] = await Promise.all([
    detectHardware(),
    detectAllRuntimes(ollamaHost),
    fetchOllamaModels(ollamaHost),
  ]);

  console.log(renderMarkdownReport({
    version: VERSION,
    generatedAt: new Date(),
    platform: `${process.platform}/${process.arch}`,
    hardware,
    runtimes,
    installedTags: new Set(ollamaModels.map((model) => model.name)),
    recommendations: getRecommendations(hardware, {
      category: options.category,
      top: options.top,
      onlyFitting: true,
    }),
    health: runDiagnostics(hardware, runtimes),
    availableVramMb: getAvailableVram(hardware),
    category: options.category,
    top: options.top,
  }));
}
