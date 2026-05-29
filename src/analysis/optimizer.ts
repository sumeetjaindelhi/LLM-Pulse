import {
  scoreModel,
  pickSweetSpot,
  classifyFit,
  getAvailableVram,
  isComfortable,
  suggestGpuOffload,
} from "./scorer.js";
import type {
  CpuInfo,
  FitLevel,
  HardwareProfile,
  ModelEntry,
  OptimizationProfile,
  OptimizedParameter,
  QuantizationVariant,
} from "../core/types.js";

// Ollama's num_ctx maps to llama.cpp's --ctx-size. We only ever recommend one of
// these "clean" tiers: users recognise them, and snapping down keeps the
// KV-cache estimate clear of the VRAM ceiling instead of landing on an odd
// boundary value that a small estimation error could push into OOM.
export const CONTEXT_TIERS = [2048, 4096, 8192, 16384, 32768, 65536, 131072] as const;

const MIN_CONTEXT = CONTEXT_TIERS[0];
const MAX_CONTEXT = CONTEXT_TIERS[CONTEXT_TIERS.length - 1];

// VRAM/RAM reserve for the inference compute buffer, allocator slack, and
// fragmentation — memory that lives alongside weights + KV cache. A fraction of
// the pool is used when that's larger, so big rigs reserve proportionally more.
const CONTEXT_COMPUTE_RESERVE_MB = 768;
const CONTEXT_RESERVE_FRACTION = 0.05;

// Ollama's default num_batch is 512 (prompt tokens processed per step). A larger
// batch speeds prompt eval but spikes VRAM; on tight fits we halve it so the
// spike doesn't tip the model into OOM.
const DEFAULT_BATCH = 512;
const TIGHT_BATCH = 256;

// Estimated KV-cache size in MB per 1,000 tokens of context, keyed by parameter
// count. A precise figure needs the model's n_kv_heads x head_dim and whether it
// uses grouped-query attention — neither is in the curated DB, which carries only
// parametersBillion. These values assume modern GQA-era architectures with an
// fp16 KV cache (Ollama's default) and are set deliberately high: over-estimating
// KV makes num_ctx err toward "safe" rather than "OOM", which is what we want.
// Same param-bucket shape as scorer.ts's estimateTotalLayers.
function kvCacheMbPer1kTokens(paramsBillion: number): number {
  if (paramsBillion <= 1.5) return 40;
  if (paramsBillion <= 4) return 80;
  if (paramsBillion <= 9) return 150;
  if (paramsBillion <= 15) return 210;
  if (paramsBillion <= 22) return 280;
  if (paramsBillion <= 35) return 360;
  if (paramsBillion <= 80) return 440;
  return 600;
}

// num_thread — physical performance cores. llama.cpp throughput peaks there:
// logical SMT threads add contention rather than speed, and efficiency cores drag
// the critical path. performanceCores is null on symmetric CPUs or when the
// platform didn't expose the P/E split, so fall back to the physical core count.
export function recommendThreads(cpu: CpuInfo): OptimizedParameter {
  const physical = cpu.performanceCores ?? cpu.cores;
  const value = Math.max(1, physical);
  const note = cpu.performanceCores != null && cpu.performanceCores < cpu.cores
    ? "performance cores (no E-cores/SMT)"
    : "physical cores (no SMT)";
  return { value, note };
}

// num_batch — keep Ollama's 512 default when there's comfortable headroom; halve
// it otherwise so the one-time prompt-eval VRAM spike doesn't OOM a tight fit.
export function recommendBatch(fitLevel: FitLevel): OptimizedParameter {
  if (isComfortable(fitLevel)) {
    return { value: DEFAULT_BATCH, note: "comfortable VRAM headroom" };
  }
  return { value: TIGHT_BATCH, note: "smaller — eases tight-fit VRAM spike" };
}

// num_ctx — largest context tier whose estimated KV cache fits the memory pool
// alongside the weights, capped at the model's native window and floored at 2048.
export function recommendContext(
  model: ModelEntry,
  quant: QuantizationVariant,
  hardware: HardwareProfile,
): OptimizedParameter {
  // getAvailableVram returns the right pool for the system: discrete VRAM, the
  // unified-memory cap on Apple Silicon, or system RAM on a CPU-only box.
  const budgetMb = getAvailableVram(hardware);
  const reserveMb = Math.max(CONTEXT_COMPUTE_RESERVE_MB, budgetMb * CONTEXT_RESERVE_FRACTION);
  const kvBudgetMb = budgetMb - quant.vramMb - reserveMb;

  const kvPer1k = kvCacheMbPer1kTokens(model.parametersBillion);
  const maxCtxByVram = (kvBudgetMb / kvPer1k) * 1000;
  const ceiling = Math.min(model.contextWindow, MAX_CONTEXT);

  // Largest tier that fits both the VRAM budget and the model's native window;
  // never drops below the 2048 floor even when the budget can't hold it.
  let chosen: number = MIN_CONTEXT;
  for (const tier of CONTEXT_TIERS) {
    if (tier <= maxCtxByVram && tier <= ceiling) chosen = tier;
  }

  let note: string;
  if (maxCtxByVram < MIN_CONTEXT) {
    note = "VRAM tight — little ctx room";
  } else if (chosen >= model.contextWindow) {
    note = "capped at native context";
  } else {
    note = "fits KV cache beside weights";
  }

  return { value: chosen, note };
}

// num_gpu — how many transformer blocks to put on the GPU. Reuses the offload
// math from scorer.ts. Returns null on Apple Silicon, where the CPU/GPU split
// isn't meaningful (shared memory pool) and the parameter is omitted entirely.
function recommendGpuLayers(
  model: ModelEntry,
  quant: QuantizationVariant,
  hardware: HardwareProfile,
): OptimizedParameter | null {
  const gpu = hardware.primaryGpu;
  if (!gpu || gpu.vramMb <= 0) {
    return { value: 0, note: "no GPU — CPU inference" };
  }
  if (gpu.acceleratorType === "metal") return null;

  const offload = suggestGpuOffload(model, quant, hardware);
  if (!offload) return null; // defensive: discrete GPU but offload not computable

  switch (offload.reason) {
    case "full_fit":
      return { value: "all", note: `full ${offload.totalLayers}/${offload.totalLayers} layers on GPU` };
    case "partial_offload":
      return {
        value: offload.gpuLayers,
        note: `${offload.gpuLayers} of ${offload.totalLayers} on GPU, rest on CPU`,
      };
    case "too_small":
      return { value: 0, note: "VRAM too small — CPU only" };
  }
}

// Builds a full set of recommended Ollama parameters for a model on the detected
// hardware. Auto-picks the sweet-spot quant unless one is forced. Returns null
// when no quantization fits at all (the caller renders a "nothing fits" message).
export function computeOptimization(
  model: ModelEntry,
  hardware: HardwareProfile,
  forcedQuant?: QuantizationVariant,
): OptimizationProfile | null {
  let quant: QuantizationVariant;
  if (forcedQuant) {
    quant = forcedQuant;
  } else {
    const scores = model.quantizations
      .map((q) => scoreModel(model, q, hardware))
      .sort((a, b) => a.quantization.bitsPerWeight - b.quantization.bitsPerWeight);
    const idx = pickSweetSpot(scores);
    if (idx < 0) return null;
    quant = scores[idx].quantization;
  }

  const fitLevel = classifyFit(getAvailableVram(hardware), quant.vramMb);

  return {
    quant,
    quantForced: forcedQuant !== undefined,
    fitLevel,
    numCtx: recommendContext(model, quant, hardware),
    numGpu: recommendGpuLayers(model, quant, hardware),
    numThread: recommendThreads(hardware.cpu),
    numBatch: recommendBatch(fitLevel),
  };
}
