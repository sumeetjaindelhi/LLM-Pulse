import { isUnifiedMemory } from "../core/unified-memory.js";
import { FIT_THRESHOLDS } from "../core/constants.js";
import { assessFit, budgetMb, layersThatFit, ollamaDefaultContext, speedClass, type FitAssessment } from "./memory-model.js";
import { computeMb, kvCacheMb, memoryNeedMb, type KvCacheType } from "./context-capacity.js";
import type {
  GpuInfo,
  HardwareProfile,
  ModelEntry,
  QuantizationVariant,
  ModelScore,
  FitLevel,
  ModelCategory,
  Verdict,
  GpuOffloadSuggestion,
} from "../core/types.js";

export function deriveVerdict(fitLevel: FitLevel): Verdict {
  switch (fitLevel) {
    case "excellent":
    case "comfortable":
      return "yes";
    case "tight":
    case "barely":
      return "maybe";
    case "cannot_run":
      return "no";
  }
}

const FALSE_ENV_VALUES = new Set(["0", "f", "F", "FALSE", "false", "False"]);

/** OLLAMA_IGPU_ENABLE as O envconfig reads it: undefined when unset, false for a false boolean, else true (an unparsable value counts as true). */
function integratedGpuSetting(): boolean | undefined {
  const value = (process.env.OLLAMA_IGPU_ENABLE ?? "").replace(/^["' ]+|["' ]+$/g, "");
  return value === "" ? undefined : !FALSE_ENV_VALUES.has(value);
}

/**
 * Ollama's integrated-GPU admission (O discover/runner.go integratedGPUAdmission, 31-34, 382-435): an explicit
 * OLLAMA_IGPU_ENABLE decides alone, except Metal, which is never filtered; unset, it keeps CUDA (GB10, Jetson) and AMD gfx1151 (Strix Halo).
 */
function isAllowedIntegrated(gpu: GpuInfo): boolean {
  if (gpu.acceleratorType === "metal") return true;
  return integratedGpuSetting() ?? (gpu.acceleratorType === "cuda" || gpu.gfxTarget === "gfx1151");
}

/**
 * The GPUs Ollama schedules on: none without a primary GPU that reports memory; otherwise every discrete GPU
 * with VRAM plus the integrated GPUs Ollama admits, else just the primary if admitted. A dropped integrated GPU
 * leaves the machine CPU-only (O discover/runner.go:126, 382-435).
 */
function inferenceGpus(hardware: HardwareProfile): GpuInfo[] {
  const primary = hardware.primaryGpu;
  if (!primary || primary.vramMb <= 0) return [];
  const discrete = hardware.gpus.filter((gpu) => gpu.vramMb > 0 && !isUnifiedMemory(gpu));
  const integrated = discrete.length > 0 ? hardware.gpus.filter((gpu) => gpu.vramMb > 0 && isUnifiedMemory(gpu)) : [primary];
  return [...discrete, ...integrated.filter((gpu) => !isUnifiedMemory(gpu) || isAllowedIntegrated(gpu))];
}

/** True when no GPU contributes memory: none detected, only ones that report 0 VRAM, or only integrated ones Ollama drops. */
export function isCpuOnly(hardware: HardwareProfile): boolean {
  return inferenceGpus(hardware).length === 0;
}

/** The detected GPU Ollama will not schedule on (a dropped integrated GPU), else null. */
export function ignoredGpu(hardware: HardwareProfile): GpuInfo | null {
  const gpu = hardware.primaryGpu;
  return gpu && gpu.vramMb > 0 && isCpuOnly(hardware) ? gpu : null;
}

/** Why Ollama skips `gpu`, worded the same in every command. */
export function ignoredGpuLabel(gpu: GpuInfo): string {
  return `${gpu.model} not used by Ollama (integrated; OLLAMA_IGPU_ENABLE=1 enables it)`;
}

function isAmd(gpu: GpuInfo): boolean {
  const vendor = gpu.vendor.toLowerCase();
  return vendor.includes("amd") || vendor.includes("advanced micro");
}

/** Runtime library Ollama loads for the GPU: NVIDIA CUDA, AMD ROCm, Apple Metal, anything else (Intel) Vulkan. */
function runtimeLibrary(gpu: GpuInfo): string {
  if (gpu.acceleratorType === "metal") return "Metal";
  if (gpu.vendor.toLowerCase().includes("nvidia")) return "CUDA";
  return isAmd(gpu) ? "ROCm" : "Vulkan";
}

function gpuBudgetMb(gpu: GpuInfo, availableMb: number): number {
  if (!isUnifiedMemory(gpu)) return budgetMb({ kind: "discrete-gpu", vramMb: gpu.vramMb });
  if (gpu.acceleratorType === "metal") return budgetMb({ kind: "apple-metal", vramMb: gpu.vramMb });
  if (isAmd(gpu)) return budgetMb({ kind: "amd-apu", gttMb: gpu.vramMb, availableMb });
  return budgetMb({ kind: "unified", availableMb });
}

/**
 * The GPUs one model loads on. Ollama never splits a model across runtime libraries: it groups GPUs by library
 * and places on one group (O ml/device.go ByLibrary, O server/sched.go:1008-1128). This is the group with the
 * largest summed budget; empty when the machine is CPU-only.
 */
export function budgetGpus(hardware: HardwareProfile): GpuInfo[] {
  const availableMb = hardware.memory.availableMb;
  const groups = new Map<string, GpuInfo[]>();
  for (const gpu of inferenceGpus(hardware)) {
    const library = runtimeLibrary(gpu);
    groups.set(library, [...(groups.get(library) ?? []), gpu]);
  }
  const groupBudget = (group: GpuInfo[]): number => group.reduce((sum, gpu) => sum + gpuBudgetMb(gpu, availableMb), 0);
  const [first = [], ...rest] = groups.values();
  return rest.reduce((best, group) => (groupBudget(group) > groupBudget(best) ? group : best), first);
}

/** GPUs the model loads on; 1 when CPU-only (the compute buffer then lives in RAM). */
export function budgetGpuCount(hardware: HardwareProfile): number {
  return Math.max(1, budgetGpus(hardware).length);
}

/**
 * Memory (MiB) the model may use: the memory-model budget for this machine, after the runtime's fixed
 * reserves (see `budgetMb`). Discrete GPUs: the sum of each card's VRAM budget in the chosen runtime group
 * (L common/fit.cpp:748-752 applies the margin per device). Apple: the Metal working set. GB10-style unified
 * memory: available RAM. AMD APU: GTT (`vramMb`) capped by available RAM. No usable GPU: available RAM.
 */
export function getAvailableVram(hardware: HardwareProfile): number {
  const availableMb = hardware.memory.availableMb;
  const gpus = budgetGpus(hardware);
  if (gpus.length === 0) return budgetMb({ kind: "cpu-only", availableMb });
  return gpus.reduce((sum, gpu) => sum + gpuBudgetMb(gpu, availableMb), 0);
}

/**
 * False when getAvailableVram includes llm-pulse's estimated GPU context reserve (discrete GPUs, GB10, AMD APUs);
 * true on Apple Metal (working set minus a fixed 1 GiB) and CPU-only (available RAM, by definition).
 */
export function budgetExact(hardware: HardwareProfile): boolean {
  const [gpu] = budgetGpus(hardware);
  return !gpu || gpu.acceleratorType === "metal";
}

/** One line saying what the budget is on unified-memory hardware; null for discrete GPUs and CPU-only machines. */
export function unifiedMemoryNote(hardware: HardwareProfile): string | null {
  const [gpu] = budgetGpus(hardware);
  if (!gpu || !isUnifiedMemory(gpu)) return null;
  if (gpu.acceleratorType === "metal") return "Metal working set — macOS keeps the rest for the system";
  if (isAmd(gpu)) return "GPU pool = GTT";
  return "unified memory — budget is available RAM";
}

/** The model's training context when known; an estimated window is a guess and is not one. */
export function trainContextOf(model: ModelEntry): number | undefined {
  return model.arch?.contextLength ?? (model.contextWindowEstimated ? undefined : model.contextWindow);
}

/** Context Ollama allocates by default on this machine: total VRAM of the GPUs it uses, clamped to the training context. */
export function defaultContextFor(model: ModelEntry, hardware: HardwareProfile): number {
  const totalVramMb = inferenceGpus(hardware).reduce((sum, gpu) => sum + gpu.vramMb, 0);
  return ollamaDefaultContext(totalVramMb, trainContextOf(model));
}

/** Fit of one quant on this machine at Ollama's default context; `maxContext` is capped at the training context. */
export function assessModelFit(
  model: ModelEntry,
  quant: QuantizationVariant,
  hardware: HardwareProfile,
  kvType: KvCacheType = "f16",
): FitAssessment {
  const gpuCount = budgetGpuCount(hardware);
  return assessFit({
    budgetMb: getAvailableVram(hardware),
    cpuOnly: isCpuOnly(hardware),
    defaultContext: defaultContextFor(model, hardware),
    need: (ctx) => memoryNeedMb(model, quant, ctx, kvType, undefined, gpuCount),
    trainContext: trainContextOf(model),
  });
}

export function isFitting(level: FitLevel): boolean {
  return level !== "cannot_run";
}

export function isComfortable(level: FitLevel): boolean {
  return level === "excellent" || level === "comfortable";
}

// "Either can't run or will run painfully slow" — the threshold downstream
// rendering uses for a failure tone; memory-model's `speedClass` short-circuits the same levels to "slow".
export function isCriticalFit(level: FitLevel): boolean {
  return level === "cannot_run" || level === "barely";
}

export function classifyFit(availableVramMb: number, requiredVramMb: number): FitLevel {
  if (requiredVramMb <= 0) return "excellent";
  const ratio = availableVramMb / requiredVramMb;
  if (ratio >= FIT_THRESHOLDS.excellent) return "excellent";
  if (ratio >= FIT_THRESHOLDS.comfortable) return "comfortable";
  if (ratio >= FIT_THRESHOLDS.tight) return "tight";
  if (ratio >= FIT_THRESHOLDS.barely) return "barely";
  return "cannot_run";
}

const TIER_SCORES: Record<string, number> = {
  frontier: 1.0,
  strong: 0.8,
  good: 0.6,
  lightweight: 0.4,
};

const FIT_FACTORS: Record<FitLevel, number> = {
  excellent: 1.0,
  comfortable: 0.9,
  tight: 0.75,
  barely: 0.5,
  cannot_run: 0,
};

export function scoreModel(
  model: ModelEntry,
  quant: QuantizationVariant,
  hardware: HardwareProfile,
  category: ModelCategory | "all" = "all",
): ModelScore {
  const { level: fitLevel, fitRatio } = assessModelFit(model, quant, hardware);

  // Composite score: quality * quantization retention * tier * fit
  const tierScore = TIER_SCORES[model.qualityTier] ?? 0.5;
  const fitFactor = FIT_FACTORS[fitLevel];

  let qualityWeight = 0.4;
  let quantWeight = 0.2;
  let tierWeight = 0.2;
  let fitWeight = 0.2;

  // Category-specific weights
  if (category === "coding") {
    qualityWeight = 0.5;
    tierWeight = 0.25;
    quantWeight = 0.15;
    fitWeight = 0.1;
  } else if (category === "reasoning") {
    qualityWeight = 0.45;
    tierWeight = 0.25;
    quantWeight = 0.2;
    fitWeight = 0.1;
  }

  const compositeScore = Math.round(
    (model.qualityScore / 100) * qualityWeight * 100 +
    quant.qualityRetention * quantWeight * 100 +
    tierScore * tierWeight * 100 +
    fitFactor * fitWeight * 100,
  );

  return {
    model,
    quantization: quant,
    fitLevel,
    fitRatio,
    compositeScore,
    speedEstimate: speedClass({
      level: fitLevel,
      cpuOnly: isCpuOnly(hardware),
      weightsMb: quant.vramMb,
      activeParams: model.activeParametersBillion,
      totalParams: model.parametersBillion,
    }),
  };
}

// ESTIMATE, used only when the model's architecture facts (ModelEntry.arch) are unknown: the bucket is a
// rough transformer-block count for common model sizes, off by up to ~20% for a given architecture. Ollama's `num_gpu`
// maps to llama.cpp's `--n-gpu-layers`: how many blocks live on GPU, with
// the rest on CPU. Layer counts come from published configs for mainstream
// Llama / Qwen / Gemma / DeepSeek families. Exact counts vary ±20% across
// architectures, but the resulting VRAM/layer estimate is close enough to
// guide offload tuning — and param count is the only field the curated DB
// reliably carries.
function estimateTotalLayers(paramsBillion: number): number {
  if (paramsBillion <= 1.5) return 16;
  if (paramsBillion <= 4) return 28;
  if (paramsBillion <= 9) return 32;
  if (paramsBillion <= 15) return 40;
  if (paramsBillion <= 22) return 42;
  if (paramsBillion <= 35) return 48;
  if (paramsBillion <= 80) return 80;
  return 126;
}

// Tolerance for treating two qualityRetention values as "tied" — anything
// closer than 0.5% is noise from data-source rounding, not a real gap.
const QUALITY_TIE_EPSILON = 0.005;

// llama.cpp community heuristic — buy the most quality you can afford in VRAM,
// since gains at the high end are real but diminishing. Prefers comfortable
// fits, then tight fits (weights still fit the pool), and only then barely
// fits (weights overflow into CPU offload). Returns -1 when every quant
// overflows the budget. Ties on retention break to higher bitsPerWeight so the
// fuller-precision variant wins.
export function pickSweetSpot(scores: ModelScore[]): number {
  const fitting = scores.map((s, i) => ({ s, i })).filter((x) => isFitting(x.s.fitLevel));
  if (fitting.length === 0) return -1;

  const comfortable = fitting.filter((x) => isComfortable(x.s.fitLevel));
  const tight = fitting.filter((x) => x.s.fitLevel === "tight");
  const pool = comfortable.length > 0 ? comfortable : tight.length > 0 ? tight : fitting;
  return pool.reduce((best, cur) => {
    const delta = cur.s.quantization.qualityRetention - best.s.quantization.qualityRetention;
    if (delta > QUALITY_TIE_EPSILON) return cur;
    if (Math.abs(delta) <= QUALITY_TIE_EPSILON) {
      return cur.s.quantization.bitsPerWeight > best.s.quantization.bitsPerWeight ? cur : best;
    }
    return best;
  }).i;
}

/** The quant to headline for one model (check, compare, MCP): the sweet spot, else the best composite when nothing fits. */
export function headlineScore(scores: ModelScore[]): ModelScore {
  const idx = pickSweetSpot(scores);
  return idx >= 0 ? scores[idx] : scores.reduce((best, s) => (s.compositeScore > best.compositeScore ? s : best));
}

/** True when `quant` is the file `ollama pull <tag>` installs, or that file is unknown: pull, run and FROM with the tag
 *  load it. */
export function isTagFile(model: ModelEntry, quant: QuantizationVariant): boolean {
  const pulled = model.quantizations.find((q) => q.exact);
  return !pulled || pulled.name === quant.name;
}

export function suggestGpuOffload(
  model: ModelEntry,
  quant: QuantizationVariant,
  hardware: HardwareProfile,
  context: number = defaultContextFor(model, hardware),
): GpuOffloadSuggestion | null {
  const [gpu] = budgetGpus(hardware);
  if (!gpu) return null;
  // Unified memory: `num_gpu` is effectively "all" by default and the CPU/GPU
  // split isn't meaningful — weights live in the same RAM pool either way.
  if (isUnifiedMemory(gpu)) return null;

  const totalLayers = model.arch?.layers ?? estimateTotalLayers(model.parametersBillion);
  const weightsPerLayer = quant.vramMb / totalLayers;
  const layersOnGpu = layersThatFit({
    budgetMb: getAvailableVram(hardware),
    kvMb: kvCacheMb(model, context),
    computeMb: computeMb(model, context, undefined, budgetGpuCount(hardware), false),
    weightsMb: quant.vramMb,
    layers: totalLayers,
  });

  // `ollama run <tag>` loads the file the tag installs, so the commands hold only for that quant.
  const tag = isTagFile(model, quant) ? model.ollamaTag : null;

  if (layersOnGpu >= totalLayers) {
    return {
      gpuLayers: "all",
      totalLayers,
      estimatedVramUsedMb: quant.vramMb,
      reason: "full_fit",
      ollamaCommand: tag ? `ollama run ${tag}` : null,
    };
  }

  if (layersOnGpu < 1) {
    return {
      gpuLayers: 0,
      totalLayers,
      estimatedVramUsedMb: 0,
      reason: "too_small",
      ollamaCommand: null,
    };
  }

  return {
    gpuLayers: layersOnGpu,
    totalLayers,
    estimatedVramUsedMb: Math.round(layersOnGpu * weightsPerLayer),
    reason: "partial_offload",
    // Ollama sets `num_gpu` via an interactive `/set parameter` line after
    // `ollama run`, or via a Modelfile `PARAMETER` — there's no CLI flag.
    ollamaCommand: tag ? `/set parameter num_gpu ${layersOnGpu}` : null,
  };
}
