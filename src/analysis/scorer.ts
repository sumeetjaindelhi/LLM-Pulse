import { FIT_THRESHOLDS, UNIFIED_MEMORY_HEADROOM_MB } from "../core/constants.js";
import type {
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

export function getAvailableVram(hardware: HardwareProfile): number {
  // `primaryGpu.vramMb` is already the final usable cap — on Apple Silicon,
  // the wired-memory limit has been applied in `detectHardware`; on discrete
  // GPUs it's the actual VRAM reported by nvidia-smi / rocm-smi.
  //
  // Guard on vramMb > 0: a GPU with zero reported VRAM (e.g. an Intel iGPU
  // systeminformation couldn't probe) should fall back to the CPU/RAM path
  // rather than forcing every model into cannot_run.
  //
  // On unified memory (Apple Silicon), the wired-limit sysctl is a kernel
  // ceiling — it doesn't reflect that the OS, editor, browser, and every
  // other app are drawing from the same pool. We subtract a flat headroom
  // so fit scoring reports what the machine can run while staying usable,
  // not the theoretical max for an idle box. A 2 GB floor prevents the
  // headroom from flipping small-RAM systems into cannot_run for every model.
  if (hardware.primaryGpu && hardware.primaryGpu.vramMb > 0) {
    if (hardware.primaryGpu.acceleratorType === "metal") {
      return Math.max(hardware.primaryGpu.vramMb - UNIFIED_MEMORY_HEADROOM_MB, 2048);
    }
    return hardware.primaryGpu.vramMb;
  }
  return hardware.memory.availableMb;
}

export function isFitting(level: FitLevel): boolean {
  return level !== "cannot_run";
}

export function isComfortable(level: FitLevel): boolean {
  return level === "excellent" || level === "comfortable";
}

// "Either can't run or will run painfully slow" — the threshold downstream
// rendering uses for a failure tone, and `estimateSpeed` uses to short-circuit
// to "slow".
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

export function getFitRatio(availableVramMb: number, requiredVramMb: number): number {
  return requiredVramMb > 0 ? availableVramMb / requiredVramMb : 0;
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

function estimateSpeed(
  fitLevel: FitLevel,
  paramsBillion: number,
  hasGpu: boolean,
): "fast" | "moderate" | "slow" {
  if (isCriticalFit(fitLevel)) return "slow";
  if (!hasGpu) return paramsBillion <= 3 ? "moderate" : "slow";
  if (paramsBillion <= 7 && isComfortable(fitLevel)) return "fast";
  if (paramsBillion <= 14) return "moderate";
  return "slow";
}

export function scoreModel(
  model: ModelEntry,
  quant: QuantizationVariant,
  hardware: HardwareProfile,
  category: ModelCategory | "all" = "all",
): ModelScore {
  // Single source of truth for "what VRAM can this model actually use" —
  // applies the unified-memory headroom on Apple Silicon so fit ratings
  // match day-to-day usability, not the idle sysctl ceiling.
  const availableVramMb = getAvailableVram(hardware);

  const fitRatio = getFitRatio(availableVramMb, quant.vramMb);
  const fitLevel = classifyFit(availableVramMb, quant.vramMb);

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
    speedEstimate: estimateSpeed(fitLevel, model.parametersBillion, !!hardware.primaryGpu),
  };
}

// Rough transformer-block count for common model sizes. Ollama's `num_gpu`
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

// Leave ~10% of VRAM for KV cache, attention workspace, and runtime overhead.
// Without this a "perfect" weight-only fit OOMs the first time context grows.
const OFFLOAD_KV_HEADROOM = 0.9;

// Tolerance for treating two qualityRetention values as "tied" — anything
// closer than 0.5% is noise from data-source rounding, not a real gap.
const QUALITY_TIE_EPSILON = 0.005;

// llama.cpp community heuristic — buy the most quality you can afford in VRAM,
// since gains at the high end are real but diminishing. Falls back to the best
// fitting quant when nothing is comfortable. Returns -1 when every quant
// overflows the budget. Ties on retention break to higher bitsPerWeight so the
// fuller-precision variant wins.
export function pickSweetSpot(scores: ModelScore[]): number {
  const fitting = scores.map((s, i) => ({ s, i })).filter((x) => isFitting(x.s.fitLevel));
  if (fitting.length === 0) return -1;

  const comfortable = fitting.filter((x) => isComfortable(x.s.fitLevel));
  const pool = comfortable.length > 0 ? comfortable : fitting;
  return pool.reduce((best, cur) => {
    const delta = cur.s.quantization.qualityRetention - best.s.quantization.qualityRetention;
    if (delta > QUALITY_TIE_EPSILON) return cur;
    if (Math.abs(delta) <= QUALITY_TIE_EPSILON) {
      return cur.s.quantization.bitsPerWeight > best.s.quantization.bitsPerWeight ? cur : best;
    }
    return best;
  }).i;
}

export function suggestGpuOffload(
  model: ModelEntry,
  quant: QuantizationVariant,
  hardware: HardwareProfile,
): GpuOffloadSuggestion | null {
  const gpu = hardware.primaryGpu;
  if (!gpu || gpu.vramMb <= 0) return null;
  // Unified memory: `num_gpu` is effectively "all" by default and the CPU/GPU
  // split isn't meaningful — weights live in the same RAM pool either way.
  if (gpu.acceleratorType === "metal") return null;

  const availableVramMb = gpu.vramMb;
  const totalLayers = estimateTotalLayers(model.parametersBillion);
  const weightsPerLayer = quant.vramMb / totalLayers;
  const usableVramForWeights = availableVramMb * OFFLOAD_KV_HEADROOM;
  const layersThatFit = Math.floor(usableVramForWeights / weightsPerLayer);

  const tag = model.ollamaTag;

  if (layersThatFit >= totalLayers) {
    return {
      gpuLayers: "all",
      totalLayers,
      estimatedVramUsedMb: quant.vramMb,
      reason: "full_fit",
      ollamaCommand: tag ? `ollama run ${tag}` : null,
    };
  }

  if (layersThatFit < 1) {
    return {
      gpuLayers: 0,
      totalLayers,
      estimatedVramUsedMb: 0,
      reason: "too_small",
      ollamaCommand: null,
    };
  }

  return {
    gpuLayers: layersThatFit,
    totalLayers,
    estimatedVramUsedMb: Math.round(layersThatFit * weightsPerLayer),
    reason: "partial_offload",
    // Ollama sets `num_gpu` via an interactive `/set parameter` line after
    // `ollama run`, or via a Modelfile `PARAMETER` — there's no CLI flag.
    ollamaCommand: tag ? `/set parameter num_gpu ${layersThatFit}` : null,
  };
}
