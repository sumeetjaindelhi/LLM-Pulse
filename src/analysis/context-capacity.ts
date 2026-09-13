import { getAvailableVram } from "./scorer.js";
import type { HardwareProfile, ModelEntry, QuantizationVariant } from "../core/types.js";

// VRAM/RAM reserve for the inference compute buffer, allocator slack, and
// fragmentation — memory that lives alongside weights + KV cache. A fraction of
// the pool is used when that's larger, so big rigs reserve proportionally more.
export const CONTEXT_COMPUTE_RESERVE_MB = 768;
export const CONTEXT_RESERVE_FRACTION = 0.05;

// Estimated fp16 KV-cache size (Ollama's default) in MB per 1,000 tokens of
// context. Curated models whose architecture the bucket underestimates — no
// grouped-query attention, or a large head_dim — carry a figure derived from
// layers x kv_heads x head_dim in `kvMbPer1kTokens`. Everything else falls back
// to a parameter-count bucket that assumes a modern GQA architecture and is set
// deliberately high: over-estimating KV makes the context estimate err toward
// "safe" rather than "OOM". Same bucket shape as scorer.ts's estimateTotalLayers.
export function kvCacheMbPer1kTokens(model: ModelEntry): number {
  if (model.kvMbPer1kTokens !== undefined) return model.kvMbPer1kTokens;
  const paramsBillion = model.parametersBillion;
  if (paramsBillion <= 1.5) return 40;
  if (paramsBillion <= 4) return 80;
  if (paramsBillion <= 9) return 150;
  if (paramsBillion <= 15) return 210;
  if (paramsBillion <= 22) return 280;
  if (paramsBillion <= 35) return 360;
  if (paramsBillion <= 80) return 440;
  return 600;
}

// Largest context length (tokens) whose fp16 KV cache fits the memory pool
// alongside the weights for this quant. Returns 0 when the weights alone exceed
// the pool (the quant only loads via CPU offload). This is the raw capacity;
// callers either snap it to clean tiers (optimizer) or compare it to a concrete
// workload (context-fit).
export function maxContextTokensForVram(
  model: ModelEntry,
  quant: QuantizationVariant,
  hardware: HardwareProfile,
): number {
  const budgetMb = getAvailableVram(hardware);
  const reserveMb = Math.max(CONTEXT_COMPUTE_RESERVE_MB, budgetMb * CONTEXT_RESERVE_FRACTION);
  const kvBudgetMb = budgetMb - quant.vramMb - reserveMb;
  const kvPer1k = kvCacheMbPer1kTokens(model);
  return Math.max(0, Math.floor((kvBudgetMb / kvPer1k) * 1000));
}
