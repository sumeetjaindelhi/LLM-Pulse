import { FIT_THRESHOLDS } from "../core/constants.js";
import type {
  HardwareProfile,
  ModelEntry,
  QuantizationVariant,
  ModelScore,
  FitLevel,
  ModelCategory,
} from "../core/types.js";

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
  if (fitLevel === "cannot_run" || fitLevel === "barely") return "slow";
  if (!hasGpu) return paramsBillion <= 3 ? "moderate" : "slow";
  if (paramsBillion <= 7 && (fitLevel === "excellent" || fitLevel === "comfortable")) return "fast";
  if (paramsBillion <= 14) return "moderate";
  return "slow";
}

export function scoreModel(
  model: ModelEntry,
  quant: QuantizationVariant,
  hardware: HardwareProfile,
  category: ModelCategory | "all" = "all",
): ModelScore {
  // Use GPU VRAM if available, otherwise use RAM (CPU inference)
  const availableVramMb = hardware.primaryGpu
    ? hardware.primaryGpu.vramMb
    : hardware.memory.availableMb;

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
