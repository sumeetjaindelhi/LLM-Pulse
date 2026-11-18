import { budgetExact, budgetGpuCount, defaultContextFor, getAvailableVram, trainContextOf } from "./scorer.js";
import { autoBatch, computeBufferMb, kqMaskMb, kvCacheBytes, maxGpuContext, padContext } from "./memory-model.js";
import type { HardwareProfile, MemoryBreakdown, ModelEntry, QuantizationVariant } from "../core/types.js";

const MIB = 1024 * 1024;

// OLLAMA_KV_CACHE_TYPE values and their size relative to f16, from ggml block
// layouts: q8_0 = 34 bytes per 32 values, q4_0 = 18 bytes per 32, f16 = 64.
export type KvCacheType = "f16" | "q8_0" | "q4_0";
export const KV_CACHE_SIZE_RATIO: Record<KvCacheType, number> = {
  f16: 1,
  q8_0: 34 / 64,
  q4_0: 18 / 64,
};

// Estimated fp16 KV-cache size (Ollama's default) in MB per 1,000 tokens of
// context. Curated models whose architecture the bucket underestimates — no
// grouped-query attention, or a large head_dim — carry a figure derived from
// layers x kv_heads x head_dim in `kvMbPer1kTokens`. Everything else falls back
// to a parameter-count bucket that assumes a modern GQA architecture and is set
// deliberately high: over-estimating KV makes the context estimate err toward
// "safe" rather than "OOM". Same bucket shape as scorer.ts's estimateTotalLayers.
export function kvCacheMbPer1kTokens(model: Pick<ModelEntry, "parametersBillion" | "kvMbPer1kTokens">, kvType: KvCacheType = "f16"): number {
  return fp16KvMbPer1kTokens(model) * KV_CACHE_SIZE_RATIO[kvType];
}

function fp16KvMbPer1kTokens(model: Pick<ModelEntry, "parametersBillion" | "kvMbPer1kTokens">): number {
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

type SizedModel = Pick<ModelEntry, "parametersBillion" | "kvMbPer1kTokens" | "arch">;

/** KV cache in MiB at context `ctx`: exact from architecture facts when known, else the per-1k-token estimate on the padded context.
 *  `ubatch` (sliding-window slots) defaults to Ollama's automatic batch. */
export function kvCacheMb(model: SizedModel, ctx: number, kvType: KvCacheType = "f16", ubatch?: number): number {
  if (model.arch) return kvCacheBytes(model.arch, ctx, kvType, ubatch).bytes / MIB;
  return (kvCacheMbPer1kTokens(model, kvType) * padContext(ctx)) / 1000;
}

/** Compute buffer in MiB at context `ctx`, sized by Ollama's automatic batch unless a num_batch (`ubatch`) is set:
 *  the activations stand-in plus the KQ mask copies on `gpuCount` GPUs (pipeline parallel: every layer on the GPUs). */
export function computeMb(model: Pick<ModelEntry, "arch">, ctx: number, ubatch = autoBatch(ctx), gpuCount = 1, pipelineParallel = true): number {
  return computeBufferMb(model.arch?.vocabSize, Math.min(ctx, ubatch)) + kqMaskMb(model.arch, ctx, ubatch, gpuCount, pipelineParallel);
}

/** Memory (MiB) a model needs at context `ctx`: weights + KV cache + compute buffer. The one need formula.
 *  `ubatch` is a num_batch that turns Ollama's automatic batch off (O server/routes.go:214-224); unset, the automatic one.
 *  `gpuCount` is the number of GPUs the model loads on (1 on CPU-only and unified memory). */
export function memoryNeedMb(
  model: SizedModel,
  quant: Pick<QuantizationVariant, "vramMb">,
  ctx: number,
  kvType: KvCacheType = "f16",
  ubatch?: number,
  gpuCount = 1,
): number {
  return quant.vramMb + kvCacheMb(model, ctx, kvType, ubatch) + computeMb(model, ctx, ubatch, gpuCount);
}

const round1 = (mb: number): number => Math.round(mb * 10) / 10;

/** memoryNeedMb split into its parts, which of them are exact, and the usable budget; at Ollama's default context unless given.
 *  KV, compute and total are rounded to 0.1 MiB. */
export function memoryBreakdown(
  model: ModelEntry,
  quant: QuantizationVariant,
  hardware: HardwareProfile,
  contextTokens: number = defaultContextFor(model, hardware),
  kvType: KvCacheType = "f16",
  ubatch?: number,
): MemoryBreakdown {
  const gpuCount = budgetGpuCount(hardware);
  return {
    contextTokens,
    weightsMb: quant.vramMb,
    weightsExact: quant.exact === true,
    kvCacheMb: round1(kvCacheMb(model, contextTokens, kvType, ubatch)),
    kvExact: model.arch !== undefined && kvCacheBytes(model.arch, contextTokens, kvType, ubatch).exact,
    computeMb: round1(computeMb(model, contextTokens, ubatch, gpuCount)),
    totalMb: round1(memoryNeedMb(model, quant, contextTokens, kvType, ubatch, gpuCount)),
    budgetMb: getAvailableVram(hardware),
    budgetExact: budgetExact(hardware),
  };
}

// Largest context length (tokens), a multiple of 256, whose weights + KV cache + compute buffer fit the
// memory budget for this quant, up to the training context when known. Returns 0 when even the smallest
// context does not fit (the quant only loads via CPU offload). Without a known training context the search
// stops at the largest Ollama default tier. This is the raw capacity; callers either snap it to clean tiers
// (optimizer) or compare it to a concrete workload (context-fit).
export function maxContextTokensForVram(
  model: ModelEntry,
  quant: QuantizationVariant,
  hardware: HardwareProfile,
  kvType: KvCacheType = "f16",
): number {
  const gpuCount = budgetGpuCount(hardware);
  return maxGpuContext(getAvailableVram(hardware), (ctx) => memoryNeedMb(model, quant, ctx, kvType, undefined, gpuCount), trainContextOf(model));
}
