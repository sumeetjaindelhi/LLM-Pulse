import {
  scoreModel,
  pickSweetSpot,
  assessModelFit,
  isComfortable,
  isCriticalFit,
  suggestGpuOffload,
  isCpuOnly,
} from "./scorer.js";
import { maxContextTokensForVram } from "./context-capacity.js";
import { autoBatch, CONTEXT_TIERS } from "./memory-model.js";
import type {
  CpuInfo,
  FitLevel,
  HardwareProfile,
  ModelEntry,
  OptimizationProfile,
  OptimizedParameter,
  QuantizationVariant,
} from "../core/types.js";

const MIN_CONTEXT = CONTEXT_TIERS[0];
const MAX_CONTEXT = CONTEXT_TIERS[CONTEXT_TIERS.length - 1];

// Unset num_batch, Ollama picks a batch from the context itself (autoBatch: 512 to 2048, O server/routes.go:214-224,
// server/sched.go:900-909). A larger batch speeds prompt eval but spikes memory, so on tight fits we pin a smaller one
// so the spike doesn't tip the model into OOM.
const TIGHT_BATCH = 256;

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

// num_batch — the batch Ollama picks automatically for `numCtx` when there's comfortable
// headroom; a pinned smaller one otherwise so the one-time prompt-eval spike doesn't OOM a tight fit.
export function recommendBatch(fitLevel: FitLevel, numCtx: number): OptimizedParameter {
  if (isComfortable(fitLevel)) {
    return { value: autoBatch(numCtx), note: "automatic — Ollama uses up to this" };
  }
  return { value: TIGHT_BATCH, note: "smaller — eases tight-fit memory spike" };
}

// num_ctx — largest context tier whose estimated KV cache fits the memory pool
// alongside the weights, capped at the model's native window and floored at 2048.
export function recommendContext(
  model: ModelEntry,
  quant: QuantizationVariant,
  hardware: HardwareProfile,
): OptimizedParameter {
  // Raw VRAM-afforded context capacity (shared with context-fit); snap it to a
  // clean tier below.
  const maxCtxByVram = maxContextTokensForVram(model, quant, hardware);
  const ceiling = Math.min(model.contextWindow, MAX_CONTEXT);

  // Largest tier that fits both the VRAM budget and the model's native window;
  // never drops below the 2048 floor even when the budget can't hold it.
  let chosen: number = MIN_CONTEXT;
  for (const tier of CONTEXT_TIERS) {
    if (tier <= maxCtxByVram && tier <= ceiling) chosen = tier;
  }

  let note: string;
  if (maxCtxByVram < MIN_CONTEXT) {
    note = "KV cache won't fit at 2048 — expect CPU offload";
  } else if (chosen >= model.contextWindow) {
    note = model.contextWindowEstimated
      ? "capped at est. window — pull to confirm"
      : "capped at native context";
  } else {
    note = "fits KV cache beside weights";
  }

  return { value: chosen, note };
}

// Context ceiling with OLLAMA_KV_CACHE_TYPE=q8_0, offered as a note when VRAM
// (not the model's window) is what holds num_ctx down. null when q8_0 would
// not above numCtx.
export function kvQ8MaxContext(
  model: ModelEntry,
  quant: QuantizationVariant,
  hardware: HardwareProfile,
  numCtx: number,
): number | null {
  const ceiling = Math.min(model.contextWindow, MAX_CONTEXT);
  if (maxContextTokensForVram(model, quant, hardware) >= ceiling) return null;
  const q8Max = Math.min(ceiling, maxContextTokensForVram(model, quant, hardware, "q8_0"));
  return q8Max > numCtx ? q8Max : null;
}

// num_gpu — how many transformer blocks to put on the GPU. Reuses the offload
// math from scorer.ts. Returns null on Apple Silicon, where the CPU/GPU split
// isn't meaningful (shared memory pool) and the parameter is omitted entirely.
// Sized at the context the Modelfile sets (`numCtx`), not Ollama's default, so the
// layers and the context recommended together fit together.
function recommendGpuLayers(
  model: ModelEntry,
  quant: QuantizationVariant,
  hardware: HardwareProfile,
  numCtx: number,
): OptimizedParameter | null {
  if (isCpuOnly(hardware)) {
    return { value: 0, note: "no GPU — CPU inference" };
  }

  const offload = suggestGpuOffload(model, quant, hardware, numCtx);
  if (!offload) return null; // unified memory: the CPU/GPU split is not meaningful

  switch (offload.reason) {
    case "full_fit":
      return { value: "all", note: `full ${offload.totalLayers}/${offload.totalLayers} layers on GPU` };
    case "partial_offload":
      return {
        value: offload.gpuLayers,
        note: `≈${offload.gpuLayers} of ${offload.totalLayers} on GPU, rest on CPU`, // even split: real placement differs (estimate)
      };
    case "too_small":
      return { value: 0, note: "VRAM too small — CPU only" };
  }
}

// Builds a full set of recommended Ollama parameters for a model on the detected
// hardware. Tunes the forced quant, else the file `ollama pull <tag>` installs
// when it fits whole (FROM, `ollama run` and OLLAMA_CONTEXT_LENGTH load that
// file), else the sweet spot. Returns null when no quantization fits at all
// (the caller renders a "nothing fits" message).
export function computeOptimization(
  model: ModelEntry,
  hardware: HardwareProfile,
  forcedQuant?: QuantizationVariant,
): OptimizationProfile | null {
  const pulled = model.quantizations.find((q) => q.exact);
  let quant: QuantizationVariant;
  if (forcedQuant) {
    quant = forcedQuant;
  } else if (pulled && !isCriticalFit(assessModelFit(model, pulled, hardware).level)) {
    quant = pulled;
  } else {
    const scores = model.quantizations
      .map((q) => scoreModel(model, q, hardware))
      .sort((a, b) => a.quantization.vramMb - b.quantization.vramMb);
    const idx = pickSweetSpot(scores);
    if (idx < 0) return null;
    quant = scores[idx].quantization;
  }

  const fitLevel = assessModelFit(model, quant, hardware).level;
  const numCtx = recommendContext(model, quant, hardware);

  return {
    quant,
    quantForced: forcedQuant !== undefined,
    fitLevel,
    numCtx,
    numGpu: recommendGpuLayers(model, quant, hardware, numCtx.value as number),
    numThread: recommendThreads(hardware.cpu),
    numBatch: recommendBatch(fitLevel, numCtx.value as number),
  };
}
