import { scoreModel, pickSweetSpot } from "./scorer.js";
import { maxContextTokensForVram } from "./context-capacity.js";
import type {
  ContextFitInput,
  ContextFitResult,
  HardwareProfile,
  ModelEntry,
  QuantizationVariant,
} from "../core/types.js";

const DEFAULT_RESPONSE_RESERVE = 512;

// Above this fraction of the effective ceiling a workload still fits but leaves
// little slack — flagged "tight" so callers can warn before it OOMs in practice.
const CONTEXT_TIGHT_FRACTION = 0.9;

// Sweet-spot quant for auto mode; falls back to the smallest (most compressed)
// quant when nothing fits, so the result still reports a meaningful ceiling
// rather than refusing to answer. Mirrors optimizer.ts's selection.
function resolveQuant(
  model: ModelEntry,
  hardware: HardwareProfile,
  forced?: QuantizationVariant,
): { quant: QuantizationVariant; forced: boolean } {
  if (forced) return { quant: forced, forced: true };
  const scores = model.quantizations
    .map((q) => scoreModel(model, q, hardware))
    .sort((a, b) => a.quantization.bitsPerWeight - b.quantization.bitsPerWeight);
  const idx = pickSweetSpot(scores);
  const quant = idx >= 0 ? scores[idx].quantization : scores[0].quantization;
  return { quant, forced: false };
}

// Highest-quality quant strictly smaller than `current` whose context ceiling
// covers `needed` on this hardware. null when no smaller quant qualifies.
function findSmallerQuantThatFits(
  model: ModelEntry,
  hardware: HardwareProfile,
  current: QuantizationVariant,
  needed: number,
): { name: string; effective: number } | null {
  const smaller = model.quantizations
    .filter((q) => q.bitsPerWeight < current.bitsPerWeight)
    .sort((a, b) => b.bitsPerWeight - a.bitsPerWeight); // least compression first
  for (const q of smaller) {
    const afforded = maxContextTokensForVram(model, q, hardware);
    const effective = Math.min(model.contextWindow, afforded);
    if (effective >= needed) {
      return { name: q.name, effective };
    }
  }
  return null;
}

function buildRemedy(
  model: ModelEntry,
  hardware: HardwareProfile,
  quant: QuantizationVariant,
  quantForced: boolean,
  limitedBy: "model" | "hardware",
  effectiveMax: number,
  responseTokens: number,
  needed: number,
): string {
  if (limitedBy === "model") {
    return `Exceeds the model's ${model.contextWindow.toLocaleString()}-token native window — use a longer-context model or trim the prompt.`;
  }
  if (!quantForced) {
    const smaller = findSmallerQuantThatFits(model, hardware, quant, needed);
    if (smaller) {
      return `Switch to ${smaller.name} — affords ${smaller.effective.toLocaleString()} tokens on this hardware.`;
    }
  }
  const trimTo = Math.max(0, effectiveMax - responseTokens);
  const offload = hardware.primaryGpu && hardware.primaryGpu.acceleratorType !== "metal"
    ? ", or offload fewer layers to the GPU"
    : "";
  const unpin = quantForced ? " (or drop --quant to try a smaller quant)" : "";
  return `Trim the prompt to ≤ ${trimTo.toLocaleString()} tokens${offload}${unpin}.`;
}

export function checkContextFit(
  model: ModelEntry,
  hardware: HardwareProfile,
  input: ContextFitInput,
): ContextFitResult {
  const { quant, forced } = resolveQuant(model, hardware, input.quant);
  const responseTokens = input.responseTokens ?? DEFAULT_RESPONSE_RESERVE;
  const neededTokens = input.promptTokens + responseTokens;

  const nativeWindow = model.contextWindow;
  const affordedByVramTokens = maxContextTokensForVram(model, quant, hardware);
  const effectiveMaxTokens = Math.min(nativeWindow, affordedByVramTokens);
  const limitedBy: "model" | "hardware" =
    affordedByVramTokens < nativeWindow ? "hardware" : "model";

  let verdict: "yes" | "tight" | "no";
  if (neededTokens > effectiveMaxTokens) verdict = "no";
  else if (neededTokens > CONTEXT_TIGHT_FRACTION * effectiveMaxTokens) verdict = "tight";
  else verdict = "yes";

  const remedy = verdict === "yes"
    ? null
    : buildRemedy(model, hardware, quant, forced, limitedBy, effectiveMaxTokens, responseTokens, neededTokens);

  return {
    model: {
      id: model.id,
      name: model.name,
      parametersBillion: model.parametersBillion,
      contextWindow: model.contextWindow,
    },
    quant: { name: quant.name, vramMb: quant.vramMb },
    quantForced: forced,
    promptTokens: input.promptTokens,
    responseTokens,
    neededTokens,
    nativeWindow,
    affordedByVramTokens,
    effectiveMaxTokens,
    limitedBy,
    verdict,
    headroomTokens: effectiveMaxTokens - neededTokens,
    remedy,
  };
}
