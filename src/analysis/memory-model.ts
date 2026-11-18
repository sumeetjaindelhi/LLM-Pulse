// Pure memory model for fitting a model into a memory budget. No I/O.
// Sources are pinned: llama.cpp a300a9c33462f23d9d61e951650d249486f2b0f6 (L),
// ollama b8db8d5b5c8daedbbef7bb9ace2221a38bc384c2 (O).

import type { ArchFacts, FitLevel } from "../core/types.js";
import { KV_CACHE_SIZE_RATIO, type KvCacheType } from "./context-capacity.js";

const MIB = 1024 * 1024;

// L src/llama-context.cpp:409 (n_ctx padded to a multiple of 256)
const CONTEXT_PAD = 256;
// L common/common.h:506 (--fit minimum context); also Ollama's smallest default tier
const MIN_CONTEXT = 4096;
// L common/common.h:509 and common/fit.cpp:751 (--fit target: free memory - 1024 MiB)
const FIT_MARGIN_MB = 1024;
// estimate: memory the GPU driver/runtime context holds before llama.cpp reads free memory on an idle GPU.
// Ollama's former MinimumMemory (O ml/device.go:110-115); pinned Ollama only logs it (O server/sched.go:655-663).
const GPU_CONTEXT_MB = 457;
// O server/sched.go:1094 (single-GPU rule: predicted <= 0.80 x available)
const SINGLE_GPU_FILL = 0.8;
// Definition, not a physical constant: below 75% of the model on the GPU, CPU decode dominates.
const BARELY_MIN_FRACTION = 0.75;
// Largest vocabulary in common use (Gemma 3/4: 262144 in issues 21879, 21497); assumed when the vocabulary is unknown (calc spec F6).
const DEFAULT_VOCAB_SIZE = 262144;
// estimate: vocab x ubatch x 4 B is a conservative stand-in for the graph's working memory, not the logits (pinned
// llama-server reserves output rows for only n_parallel x (1 + draft) tokens: L tools/server/server-context.cpp:49-68,
// L common/speculative.cpp:3175-3184); the KQ mask term is exact. x1.10 is the estimate margin.
// Holds with flash attention on only (llama.cpp issue 21497 comment 5: 1870.01 MiB at ctx 50176 with it off).
const COMPUTE_BUFFER_EXCESS = 1.1;
// F16 KQ mask element (flash attention requires an f16 mask: L src/llama-graph.cpp:32-47)
const MASK_BYTES_PER_CELL = 2;
// Copies of a graph input per GPU split under pipeline parallelism (L ggml/src/ggml-backend.cpp:1171-1172, 2326, 1795-1806).
// The scheduler enables it with 2+ devices and every layer offloaded (L src/llama-context.cpp:575-580).
const PIPELINE_INPUT_COPIES = 4;
// O server/routes.go:2571-2589 (default context tiers by total VRAM)
const DEFAULT_CONTEXT_TIERS: readonly { minVramMb: number; context: number }[] = [
  { minVramMb: 47 * 1024, context: 262144 },
  { minVramMb: 23 * 1024, context: 32768 },
];
// O server/routes.go:2571-2589 (fallback below the lowest tier)
const DEFAULT_CONTEXT_FALLBACK = 4096;
// Largest context Ollama picks by default (O server/routes.go:2571-2589); search ceiling when the training context is unknown.
const MAX_CONTEXT = 262144;
// Context lengths llm-pulse recommends (num_ctx): powers of two users recognise, up to MAX_CONTEXT. Snapping a
// computed capacity down to one keeps the recommendation clear of the budget edge.
export const CONTEXT_TIERS = [2048, 4096, 8192, 16384, 32768, 65536, 131072, MAX_CONTEXT] as const;
// Decode is memory-bandwidth bound: tok/s ~ bandwidth / active bytes. GPUs offer 200-1000 GB/s, CPU RAM 50-120 GB/s,
// so these byte limits are a coarse class boundary, not a measurement.
const CPU_MODERATE_MAX_MB = 2 * 1024;
const GPU_FAST_MAX_MB = 6 * 1024;
const GPU_MODERATE_MAX_MB = 24 * 1024;

// L src/models/gemma3n.cpp:7 (n_layer_kv_from_start: only the first 20 layers own KV)
const GEMMA3N_KV_LAYERS = 20;

export type { ArchFacts };

interface SlidingDefault {
  /** Period N: layer il is SWA iff il % N < N-1 (il % N != 0 when denseFirst); 0 = every layer. */
  period: number;
  denseFirst?: boolean;
  /** No sliding window unless n_swa > 0. */
  onlyIfWindow?: boolean;
  onlyLayers?: number;
  /** Window used when the GGUF has none. */
  defaultWindow?: number;
  /** Window llama.cpp forces regardless of the GGUF value. */
  forcedWindow?: number;
}

// Mask rule: L src/llama-hparams.cpp:8-22 and src/llama-model.cpp:4173-4180. Per-arch rows from L src/models/<arch>.cpp:
// gemma2 :5-6, gemma3 :4-7, gemma3n :4-5, gpt-oss (openai-moe) :6-9, cohere2 :5, cohere2moe :27-28, exaone4 :4-7,
// laguna :37-42, smallthinker :4-16, olmo2 :6-9, mellum :13-16, afmoe :15-17, plamo3 :5-9, lfm2 :38-43, llama4 :8-19.
const SLIDING_DEFAULTS: Record<string, SlidingDefault> = {
  gemma2: { period: 2, defaultWindow: 4096 },
  gemma3: { period: 6, onlyIfWindow: true },
  gemma3n: { period: 5 },
  "gpt-oss": { period: 2 },
  cohere2: { period: 4 },
  cohere2moe: { period: 4, denseFirst: true },
  exaone4: { period: 4, onlyLayers: 64, defaultWindow: 4096 },
  laguna: { period: 4, denseFirst: true, onlyIfWindow: true },
  smallthinker: { period: 4, denseFirst: true, onlyIfWindow: true, forcedWindow: 4096 },
  olmo2: { period: 4, onlyIfWindow: true },
  mellum: { period: 4, onlyIfWindow: true },
  afmoe: { period: 4, onlyIfWindow: true },
  plamo3: { period: 8, onlyIfWindow: true },
  lfm2: { period: 0, onlyIfWindow: true },
  llama4: { period: 4, forcedWindow: 8192 },
};

// L src/models/phi3.cpp:13-24: sliding-window metadata is ignored
const FULL_ATTENTION_ARCHITECTURES = new Set(["phi3"]);

export interface SlidingLayout {
  /** mask[il] is true when layer il uses the sliding window. */
  mask: boolean[];
  windowTokens: number;
  /** Set when the layout could not be determined; the mask then falls back to full attention. */
  reason?: string;
}

export function resolveSliding(facts: ArchFacts): SlidingLayout {
  const architecture = facts.architecture.toLowerCase();
  const none = (reason?: string): SlidingLayout => ({ mask: new Array<boolean>(facts.layers).fill(false), windowTokens: 0, reason });
  if (FULL_ATTENTION_ARCHITECTURES.has(architecture)) return none();

  const def = SLIDING_DEFAULTS[architecture];
  if (def?.onlyLayers !== undefined && def.onlyLayers !== facts.layers) return none();
  // An explicit 0 disables sliding attention (L src/models/llama4.cpp:9-11, smallthinker.cpp:6,14).
  if (facts.slidingWindow === 0) return none();

  const ggufWindow = facts.slidingWindow && facts.slidingWindow > 0 ? facts.slidingWindow : undefined;
  const windowTokens =
    def?.onlyIfWindow && ggufWindow === undefined ? 0 : (def?.forcedWindow ?? ggufWindow ?? def?.defaultWindow ?? 0);
  if (windowTokens <= 0) {
    if (def?.onlyIfWindow) return none();
    return def || Array.isArray(facts.slidingWindowPattern) ? none("missing sliding-window size") : none();
  }

  const pattern = facts.slidingWindowPattern ?? def?.period;
  if (pattern === undefined) return none("unknown sliding-window layout");

  const mask = Array.isArray(pattern)
    ? Array.from({ length: facts.layers }, (_, il) => pattern[il] === true)
    : Array.from({ length: facts.layers }, (_, il) =>
        pattern === 0 ? true : def?.denseFirst ? il % pattern !== 0 : il % pattern < pattern - 1,
      );
  return { mask, windowTokens };
}

/** Pad a context length up to a multiple of 256 (L src/llama-context.cpp:409). */
export function padContext(ctx: number): number {
  return Math.ceil(ctx / CONTEXT_PAD) * CONTEXT_PAD;
}

/**
 * Ollama's generation batch size by context (O server/sched.go:900-909).
 * Upper bound: automaticGenerationBatch steps 2048 -> 1024 -> 512 while the batch does not fit
 * (sched.go:871-874, 911-936: 2048 needs <= 60% and 1024 <= 75% of free memory), so near the budget Ollama may run ubatch 512.
 * Compute buffer and SWA slots built from this are overstated by up to (batch - 512) x vocab x 4 x 1.1 B.
 */
export function autoBatch(ctx: number): number {
  if (ctx > 32768) return 2048;
  if (ctx > 4096) return 1024;
  return 512;
}

// Bytes per element of a ggml row = 2 B (f16) x ratio: q8_0 34 B per 32 (L ggml/src/ggml-common.h:256), q4_0 18 B per 32 (:199).
const rowBytesPerElement = (kvType: KvCacheType): number => 2 * KV_CACHE_SIZE_RATIO[kvType];

/** Cells of the sliding-window cache (L src/llama-kv-cache-iswa.cpp:73). */
const swaCells = (paddedCtx: number, windowTokens: number, ubatch: number): number =>
  padContext(Math.min(paddedCtx, windowTokens + ubatch));

const kvLayerCount = (facts: ArchFacts): number =>
  facts.kvLayers ?? (facts.architecture.toLowerCase() === "gemma3n" ? GEMMA3N_KV_LAYERS : facts.layers);

const kvHeads = (facts: ArchFacts, il: number): number =>
  Array.isArray(facts.headCountKv) ? (facts.headCountKv[il] ?? 0) : facts.headCountKv;

export interface KvCacheEstimate {
  bytes: number;
  /** False when the SWA layout was unknown, recurrent state size is missing, per-layer arrays do not match `layers`, or the facts carry an `inexactReason`. */
  exact: boolean;
  reason?: string;
}

/** KV cache bytes at context `ctx` (L src/llama-kv-cache.cpp:209-234, src/llama-hparams.cpp:156-165). */
export function kvCacheBytes(facts: ArchFacts, ctx: number, kvType: KvCacheType, ubatch = autoBatch(ctx)): KvCacheEstimate {
  const paddedCtx = padContext(ctx);
  const { mask, windowTokens, reason: slidingReason } = resolveSliding(facts);
  const swaSlots = swaCells(paddedCtx, windowTokens, ubatch);
  const bytesPerElement = rowBytesPerElement(kvType);
  const kvLayers = kvLayerCount(facts);

  let reason = slidingReason ?? facts.inexactReason;
  if (Array.isArray(facts.headCountKv) && facts.headCountKv.length !== facts.layers) reason ??= "per-layer array length mismatch";

  let bytes = facts.recurrentStateBytes ?? 0;
  let hasRecurrentLayer = false;
  for (let il = 0; il < Math.min(facts.layers, kvLayers); il++) {
    const heads = kvHeads(facts, il);
    if (heads === 0 || facts.recurrentLayers?.[il]) {
      hasRecurrentLayer = true;
      continue;
    }
    const swa = mask[il];
    const keyLength = swa ? (facts.keyLengthSwa ?? facts.keyLength) : facts.keyLength;
    const valueLength = swa ? (facts.valueLengthSwa ?? facts.valueLength) : facts.valueLength;
    const elementsPerToken = heads * (keyLength + (facts.mla ? 0 : valueLength));
    bytes += elementsPerToken * bytesPerElement * (swa ? swaSlots : paddedCtx);
  }
  if (hasRecurrentLayer && facts.recurrentStateBytes === undefined) reason ??= "missing recurrent state size";
  return { bytes, exact: reason === undefined, reason };
}

/**
 * Bytes of one copy of the graph's KQ mask inputs: F16 [n_kv, n_tokens] per attention mask, n_kv the KV cache
 * size at graph reserve (L src/llama-kv-cache.cpp:2981) and n_tokens the batch. An iSWA model has a second mask of
 * the SWA cache size. `facts` undefined (estimated model): one full mask.
 */
export function kqMaskBytes(facts: ArchFacts | undefined, ctx: number, ubatch: number): number {
  const paddedCtx = padContext(ctx);
  const nTokens = Math.min(ctx, ubatch);
  const maskBytes = (cells: number): number => MASK_BYTES_PER_CELL * cells * nTokens;
  if (!facts) return maskBytes(paddedCtx);

  const { mask, windowTokens } = resolveSliding(facts);
  let hasFull = false;
  let hasSwa = false;
  for (let il = 0; il < Math.min(facts.layers, kvLayerCount(facts)); il++) {
    if (kvHeads(facts, il) === 0 || facts.recurrentLayers?.[il]) continue;
    if (mask[il]) hasSwa = true;
    else hasFull = true;
  }
  return (hasFull ? maskBytes(paddedCtx) : 0) + (hasSwa ? maskBytes(swaCells(paddedCtx, windowTokens, ubatch)) : 0);
}

/**
 * KQ mask MiB held on the devices: the CPU-built input is copied into every GPU split that uses it, and with 2+
 * GPUs pipeline parallelism keeps PIPELINE_INPUT_COPIES copies per GPU. Without it (a partial offload leaves layers on
 * the CPU, L src/llama-context.cpp:575-578) each GPU keeps one. CPU-only, unified and single-GPU: one copy.
 */
export function kqMaskMb(facts: ArchFacts | undefined, ctx: number, ubatch: number, gpuCount: number, pipelineParallel = true): number {
  const copies = gpuCount < 2 ? 1 : pipelineParallel ? PIPELINE_INPUT_COPIES * gpuCount : gpuCount;
  return (copies * kqMaskBytes(facts, ctx, ubatch)) / MIB;
}

/** Activations stand-in in MiB: vocab x ubatch x 4 B (flash attention) times the 1.10 estimate margin. */
export function computeBufferMb(vocabSize: number | undefined, ubatch: number): number {
  return ((vocabSize ?? DEFAULT_VOCAB_SIZE) * ubatch * 4 * COMPUTE_BUFFER_EXCESS) / MIB;
}

/** Ollama's default context for the total VRAM across GPUs, clamped to the training context. */
export function ollamaDefaultContext(totalVramMb: number, trainContext?: number): number {
  const tier = DEFAULT_CONTEXT_TIERS.find((t) => totalVramMb >= t.minVramMb);
  const context = tier?.context ?? DEFAULT_CONTEXT_FALLBACK;
  // O llm/server.go:111-114 clamps only when the training context is known (> 0).
  return trainContext ? Math.min(context, trainContext) : context;
}

export type BudgetSource =
  | { kind: "discrete-gpu"; vramMb: number }
  | { kind: "apple-metal"; vramMb: number }
  | { kind: "unified"; availableMb: number }
  | { kind: "amd-apu"; gttMb: number; availableMb: number }
  | { kind: "cpu-only"; availableMb: number };

/**
 * Memory (MiB) available for weights + KV + compute buffer; never negative. llama.cpp's fit target is free memory
 * minus the margin (L common/fit.cpp:751). Metal's free memory is the recommended working set minus what is already
 * allocated, ~0 before load (L ggml-metal-device.m:1501-1506), so no driver reserve applies there.
 * CPU-only is `definition:` llm-pulse's rule (Ollama has no such refusal): the model must fit available RAM.
 */
export function budgetMb(source: BudgetSource): number {
  switch (source.kind) {
    case "discrete-gpu":
      return Math.max(0, source.vramMb - GPU_CONTEXT_MB - FIT_MARGIN_MB);
    case "apple-metal":
      return Math.max(0, source.vramMb - FIT_MARGIN_MB);
    case "unified":
      return Math.max(0, source.availableMb - GPU_CONTEXT_MB - FIT_MARGIN_MB);
    case "amd-apu":
      return Math.max(0, Math.min(source.gttMb, source.availableMb) - GPU_CONTEXT_MB - FIT_MARGIN_MB);
    case "cpu-only":
      return Math.max(0, source.availableMb);
  }
}

/** Largest multiple of 256 up to `ceiling` whose need fits the budget; 0 when none does. `need` must not decrease with context. */
export function maxGpuContext(budget: number, need: (ctx: number) => number, ceiling: number = MAX_CONTEXT): number {
  let lo = 0;
  let hi = Math.floor(ceiling / CONTEXT_PAD);
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (need(mid * CONTEXT_PAD) <= budget) lo = mid;
    else hi = mid - 1;
  }
  return lo * CONTEXT_PAD;
}

/** Round a context down to a display tier; values below the smallest tier are unchanged. */
export function snapToContextTier(ctx: number): number {
  const tier = [...CONTEXT_TIERS].reverse().find((t) => t <= ctx);
  return tier ?? ctx;
}

export interface FitAssessment {
  level: FitLevel;
  /** budget / need at the default context. */
  fitRatio: number;
  /** Largest 256-multiple context that fits the budget, up to the training context (MAX_CONTEXT when unknown). */
  maxContext: number;
}

export interface FitInput {
  budgetMb: number;
  cpuOnly: boolean;
  defaultContext: number;
  /** Total memory (MiB) needed at a context length. */
  need: (ctx: number) => number;
  /** Training context; Ollama never exceeds it (O llm/server.go:111-114). Omit when unknown. */
  trainContext?: number;
}

export function assessFit({ budgetMb: budget, cpuOnly, defaultContext, need, trainContext }: FitInput): FitAssessment {
  const needDefault = need(defaultContext);
  const needMin = need(Math.min(MIN_CONTEXT, defaultContext));
  let level: FitLevel;
  if (needDefault <= budget * SINGLE_GPU_FILL) level = "excellent";
  else if (needDefault <= budget) level = "comfortable";
  else if (needMin <= budget) level = "tight";
  else if (!cpuOnly && budget >= needMin * BARELY_MIN_FRACTION) level = "barely";
  else level = "cannot_run";
  return { level, fitRatio: budget / needDefault, maxContext: maxGpuContext(budget, need, trainContext) };
}

export type SpeedClass = "fast" | "moderate" | "slow";

export interface SpeedInput {
  level: FitLevel;
  cpuOnly: boolean;
  weightsMb: number;
  /** Billions of parameters active per token and in total (MoE); omit for dense models. */
  activeParams?: number;
  totalParams?: number;
}

export function speedClass({ level, cpuOnly, weightsMb, activeParams, totalParams }: SpeedInput): SpeedClass {
  if (level === "cannot_run" || level === "barely") return "slow";
  const activeMb = activeParams && totalParams ? (weightsMb * activeParams) / totalParams : weightsMb;
  if (cpuOnly) return activeMb <= CPU_MODERATE_MAX_MB ? "moderate" : "slow";
  if (activeMb <= GPU_FAST_MAX_MB) return "fast";
  return activeMb <= GPU_MODERATE_MAX_MB ? "moderate" : "slow";
}

export interface OffloadInput {
  budgetMb: number;
  kvMb: number;
  computeMb: number;
  weightsMb: number;
  layers: number;
}

/**
 * Layers that fit in the budget beside the compute buffer, clamped to [0, layers]. Each layer carries its share of
 * the weights and of the KV cache: llama.cpp allocates a layer's KV on that layer's device (L src/llama-kv-cache.cpp:216-221),
 * so the KV of CPU-resident layers lives in system RAM.
 */
export function layersThatFit({ budgetMb: budget, kvMb, computeMb, weightsMb, layers }: OffloadInput): number {
  const fit = Math.floor((budget - computeMb) / ((weightsMb + kvMb) / layers));
  return Math.min(layers, Math.max(0, fit));
}
