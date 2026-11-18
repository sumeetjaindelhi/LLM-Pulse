import { scoreModel, pickSweetSpot } from "./scorer.js";
import { maxContextTokensForVram, memoryBreakdown, type KvCacheType } from "./context-capacity.js";
import type {
  ContextFitInput,
  ContextFitResult,
  HardwareProfile,
  ModelEntry,
  QuantizationVariant,
} from "../core/types.js";
import { isUnifiedMemory } from "../core/unified-memory.js";

const DEFAULT_RESPONSE_RESERVE = 512;

// Above this fraction of the effective ceiling a workload still fits but leaves
// little slack — flagged "tight" so callers can warn before it OOMs in practice.
const CONTEXT_TIGHT_FRACTION = 0.9;

// An estimated native window is a guess, so it must not cap what the hardware affords.
function ceilingFor(model: ModelEntry, affordedTokens: number): number {
  return model.contextWindowEstimated ? affordedTokens : Math.min(model.contextWindow, affordedTokens);
}

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
    .sort((a, b) => a.quantization.vramMb - b.quantization.vramMb);
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
    .filter((q) => q.vramMb < current.vramMb)
    .sort((a, b) => b.vramMb - a.vramMb); // least compression first
  for (const q of smaller) {
    const afforded = maxContextTokensForVram(model, q, hardware);
    const effective = ceilingFor(model, afforded);
    if (effective >= needed) {
      return { name: q.name, effective };
    }
  }
  return null;
}

// Smallest KV-cache quantization (least quality loss first) whose ceiling lets
// `needed` fit. A "tight" input must also clear the tight band, or the remedy
// would trade one tight verdict for another. null when none does.
function findKvCacheTypeThatFits(
  model: ModelEntry,
  hardware: HardwareProfile,
  quant: QuantizationVariant,
  needed: number,
  verdict: "no" | "tight",
): { type: KvCacheType; effective: number } | null {
  const usableFraction = verdict === "tight" ? CONTEXT_TIGHT_FRACTION : 1;
  for (const type of ["q8_0", "q4_0"] as const) {
    const effective = ceilingFor(model, maxContextTokensForVram(model, quant, hardware, type));
    if (needed <= usableFraction * effective) return { type, effective };
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
  verdict: "no" | "tight",
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
  if (effectiveMax === 0) {
    return `${quant.name} weights alone exceed the memory pool — use a smaller quant or model.`;
  }
  const trimTo = effectiveMax - responseTokens;
  if (trimTo <= 0) {
    return `Response reserve (${responseTokens.toLocaleString()} tokens) leaves no room in the ${effectiveMax.toLocaleString()}-token ceiling — lower the response reserve.`;
  }
  const kvFit = findKvCacheTypeThatFits(model, hardware, quant, needed, verdict);
  if (kvFit) {
    return `Run Ollama with OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=${kvFit.type} — affords ${kvFit.effective.toLocaleString()} tokens on this hardware.`;
  }
  const offload = hardware.primaryGpu && !isUnifiedMemory(hardware.primaryGpu)
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
  const effectiveMaxTokens = ceilingFor(model, affordedByVramTokens);
  const limitedBy: "model" | "hardware" =
    model.contextWindowEstimated || affordedByVramTokens < nativeWindow ? "hardware" : "model";

  let verdict: "yes" | "tight" | "no";
  if (neededTokens > effectiveMaxTokens) verdict = "no";
  else if (neededTokens > CONTEXT_TIGHT_FRACTION * effectiveMaxTokens) verdict = "tight";
  else verdict = "yes";

  const remedy = verdict === "yes"
    ? null
    : buildRemedy(model, hardware, quant, forced, limitedBy, effectiveMaxTokens, responseTokens, neededTokens, verdict);

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
    memory: memoryBreakdown(model, quant, hardware, neededTokens),
    ...(model.contextWindowEstimated && { nativeWindowEstimated: true as const }),
  };
}
