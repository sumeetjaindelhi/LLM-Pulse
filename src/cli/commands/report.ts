import { detectHardware } from "../../hardware/index.js";
import { detectAllRuntimes } from "../../runtimes/index.js";
import { fetchOllamaModels } from "../../models/ollama-models.js";
import { getRecommendations } from "../../analysis/recommender.js";
import { runDiagnostics } from "../../analysis/doctor.js";
import { getAvailableVram } from "../../analysis/scorer.js";
import { resolveOllamaHost } from "../../core/config.js";
import { UNIFIED_MEMORY_HEADROOM_MB, VERSION } from "../../core/constants.js";
import { SEVERITY_ICONS } from "../ui/badges.js";
import { escapeMarkdownValue, toMarkdownTable } from "../ui/markdown.js";
import { formatMb } from "../ui/progress.js";
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

function gpuDetail(gpu: GpuInfo | null): string {
  if (!gpu || gpu.vramMb === 0) return "None — CPU inference";
  // Some drivers already include the vendor in the model string
  // ("NVIDIA GeForce RTX 4090", "Apple M2 Pro"); don't print it twice.
  const name = gpu.model.toLowerCase().startsWith(gpu.vendor.toLowerCase())
    ? gpu.model
    : `${gpu.vendor} ${gpu.model}`;
  const label = gpu.acceleratorType ? ACCELERATOR_LABELS[gpu.acceleratorType] : undefined;
  const accelerator = label && gpu.acceleratorVersion ? ` · ${label} ${gpu.acceleratorVersion}` : "";
  return `${name} · ${formatMb(gpu.vramMb)} VRAM${accelerator}`;
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
      ["GPU", gpuDetail(hardware.primaryGpu)],
      ["RAM", memoryDetail(hardware)],
      ["Disk", `${hardware.disk.type} · ${hardware.disk.freeGb} GB free`],
      ["Usable for inference", formatMb(input.availableVramMb)],
    ]),
  ];

  if (hardware.primaryGpu?.acceleratorType === "metal") {
    const reservedGb = (UNIFIED_MEMORY_HEADROOM_MB / 1024).toFixed(0);
    blocks.push(`_Unified memory — estimate reserves ~${reservedGb} GB for OS, editor, and other apps._`);
  }

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
        ["#", "Model", "Quant", "Fit", "VRAM", "Speed", "Installed"],
        input.recommendations.map(({ rank, score }) => [
          rank,
          score.model.name,
          score.quantization.name,
          score.fitLevel,
          formatMb(score.quantization.vramMb),
          score.speedEstimate,
          score.model.ollamaTag !== null && input.installedTags.has(score.model.ollamaTag) ? "yes" : null,
        ]),
      ),
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
