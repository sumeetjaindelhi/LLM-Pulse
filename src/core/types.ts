// ── Hardware ──────────────────────────────────

export interface CpuInfo {
  brand: string;
  manufacturer: string;
  cores: number;
  threads: number;
  // GHz base. null on Apple Silicon: systeminformation's macOS code path reads
  // hw.tbfrequency (the 24 MHz timer base) and multiplies by 100, producing a
  // bogus "2.4 GHz" for every M-series chip. There's no public API to read the
  // real P-core max, so we report null rather than a wrong number.
  speed: number | null;
  speedMax: number | null; // GHz boost; same null rule as `speed`
  architecture: string; // x64, arm64
  flags: string[]; // AVX, AVX2, AVX-512, etc.
  hasAvx2: boolean;
  // Split of performance vs. efficiency cores on hybrid CPUs (Apple Silicon
  // M-series, Intel Alder Lake+). Null on symmetric-core CPUs or when the
  // platform doesn't expose the topology. Sum may be less than `cores` on
  // SMT-enabled Intel — physical P-cores are counted, SMT threads aren't.
  performanceCores: number | null;
  efficiencyCores: number | null;
}

export interface GpuInfo {
  vendor: string; // NVIDIA, AMD, Intel, Apple
  model: string;
  vramMb: number;
  driverVersion: string;
  acceleratorVersion: string | null; // CUDA for NVIDIA, ROCm for AMD, Metal for Apple, oneAPI/Level Zero for Intel
  acceleratorType: "cuda" | "rocm" | "metal" | "oneapi" | null;
  /** GPU memory is system RAM or a shared pool, not dedicated VRAM. */
  unifiedMemory?: boolean;
  /** AMD gfx target from the KFD topology (e.g. "gfx1151"); Ollama admits an integrated AMD GPU by it. */
  gfxTarget?: string;
  utilizationPercent: number | null;
  temperatureCelsius: number | null;
  vramUsedMb: number | null;
}

export interface MemoryInfo {
  totalMb: number;
  availableMb: number;
  usedMb: number;
  usedPercent: number;
  type: string; // DDR4, DDR5, LPDDR5, etc.
  speedMhz: number | null;
}

export interface DiskInfo {
  type: "NVMe" | "SSD" | "HDD" | "Unknown";
  freeGb: number;
  totalGb: number;
}

export interface HardwareProfile {
  cpu: CpuInfo;
  gpus: GpuInfo[];
  memory: MemoryInfo;
  disk: DiskInfo;
  primaryGpu: GpuInfo | null; // Best GPU (most VRAM)
}

// ── Models ────────────────────────────────────

export type ModelCategory = "general" | "coding" | "reasoning" | "creative" | "multilingual";

export type QualityTier = "frontier" | "strong" | "good" | "lightweight";

export interface QuantizationVariant {
  name: string; // Q4_K_M, Q5_K_M, Q8_0, F16
  bitsPerWeight: number;
  vramMb: number; // Weights in MiB: estimated from bits per weight, or the real file size when `exact`
  qualityRetention: number; // 0.0 - 1.0 (1.0 = no loss)
  exact?: boolean; // this quant is the file `ollama pull <ollamaTag>` installs, and vramMb is its real size (registry manifest or installed file)
}

export interface ModelEntry {
  id: string; // e.g. "llama-3.1-8b"
  name: string; // e.g. "Llama 3.1 8B"
  provider: string; // Meta, DeepSeek, etc.
  parametersBillion: number;
  contextWindow: number;
  kvMbPer1kTokens?: number; // fp16 KV cache MB per 1K tokens, curated where the size bucket underestimates; unused when `arch` is set
  categories: ModelCategory[];
  qualityTier: QualityTier;
  qualityScore: number; // 0-100 (relative within tier)
  quantizations: QuantizationVariant[];
  ollamaTag: string | null; // e.g. "llama3.1:8b"
  releaseDate: string; // YYYY-MM
  source?: "curated" | "library"; // bundled data/models.json row vs synthesized from the live ollama.com catalog
  activeParametersBillion?: number; // mixture-of-experts: parameters active per token
  contextWindowEstimated?: boolean; // contextWindow is a conservative guess, not the model's real window
  arch?: ArchFacts; // exact architecture facts from the model's GGUF metadata, when known
}

/** Architecture facts read from a model's GGUF metadata (or Ollama /api/show). Omit fields that are unknown. */
export interface ArchFacts {
  architecture: string;
  layers: number;
  /** Scalar for every layer, or per layer (length must equal `layers`); 0 marks a layer with no KV. `layers` must be an integer >= 0. */
  headCountKv: number | number[];
  keyLength: number;
  valueLength: number;
  /** Head sizes of sliding-window layers; default to keyLength/valueLength (L src/llama-model.cpp:1666-1672). */
  keyLengthSwa?: number;
  valueLengthSwa?: number;
  /** Per-layer recurrent flag (qwen35, qwen3next: attention.recurrent_layers or full_attention_interval); those layers own no KV. */
  recurrentLayers?: boolean[];
  /** Only layers [0, kvLayers) own KV: layers - shared_kv_layers (gemma4). */
  kvLayers?: number;
  slidingWindow?: number;
  /** Per-layer mask (true = SWA) or the period N of the repeating pattern. */
  slidingWindowPattern?: number | boolean[];
  /** Multi-head latent attention: no V cache. */
  mla?: boolean;
  /** Total bytes of constant recurrent state (all recurrent layers). */
  recurrentStateBytes?: number;
  vocabSize?: number;
  contextLength?: number;
  /** Why the memory model is incomplete for this model (MTP drafting, parallel-hybrid SSM state); the KV estimate is then not exact. */
  inexactReason?: string;
}

// ── Ollama Live Models ───────────────────────

export interface OllamaModel {
  name: string; // "llama3.1:8b"
  size: number; // bytes (file size on disk)
  parameterSize: string; // "8.0B" from details
  quantization: string; // "Q4_K_M" from details
  family: string; // "llama" from details
}

export interface LibraryCatalogModel {
  slug: string; // "llama3.1"
  description: string;
  parameterSizes: string[]; // ["8b", "70b", "405b"]
  capabilities: string[]; // ["tools", "thinking"]
  pulls?: number; // all-time pull count shown on the card
  updated?: string; // YYYY-MM-DD of the card's last update (UTC)
  cloud?: boolean; // card carries the "cloud" badge (some or all tags run on Ollama's servers)
}

export type MergedModelSource = "curated" | "library" | "installed";

export interface MergedModel {
  entry: ModelEntry | null; // from curated DB (null if not curated)
  ollamaModel: OllamaModel | null; // from /api/tags (null if not installed)
  libraryModel: LibraryCatalogModel | null; // from ollama.com/library (null if not discovered)
  installed: boolean;
  ollamaTag: string | null; // canonical pull tag
  sources: MergedModelSource[]; // which backends contributed to this row
}

// ── Scoring ───────────────────────────────────

export type FitLevel = "excellent" | "comfortable" | "tight" | "barely" | "cannot_run";

/** Memory (MiB) one quant needs at a context, split into the parts of memoryNeedMb, with which parts are exact. */
export interface MemoryBreakdown {
  contextTokens: number;
  weightsMb: number;
  weightsExact: boolean; // QuantizationVariant.exact: the real file size, not params x bits per weight
  kvCacheMb: number;
  kvExact: boolean; // from the model's architecture facts; false for the size-bucket estimate
  computeMb: number; // always an estimate (memory-model.ts COMPUTE_BUFFER_EXCESS margin)
  totalMb: number; // weights + KV cache + compute buffer: memoryNeedMb at contextTokens
  budgetMb: number; // usable memory: VRAM (or RAM) minus the runtime's reserves
  budgetExact: boolean; // false when budgetMb includes llm-pulse's estimated GPU context reserve (scorer.ts budgetExact)
}

export interface ModelScore {
  model: ModelEntry;
  quantization: QuantizationVariant;
  fitLevel: FitLevel;
  fitRatio: number; // available VRAM / required VRAM
  compositeScore: number; // 0-100 combined score
  speedEstimate: "fast" | "moderate" | "slow";
}

export interface Recommendation {
  rank: number;
  score: ModelScore;
  pullCommand: string | null; // e.g. "ollama pull llama3.1:8b"
}

// ── Runtimes ──────────────────────────────────

export type RuntimeStatus = "running" | "installed" | "not_found";

export interface RuntimeInfo {
  name: string; // Ollama, llama.cpp, LM Studio
  status: RuntimeStatus;
  version: string | null;
  path: string | null;
  models: string[]; // Installed model names
}

// A model currently loaded in Ollama (/api/ps). arch holds its architecture facts from /api/show when they could be read.
export interface LoadedModel {
  name: string;
  contextLength: number | null;
  quantizationLevel?: string; // e.g. "Q4_K_M"
  arch?: ArchFacts;
}

// ── Doctor ────────────────────────────────────

export type CheckSeverity = "pass" | "warning" | "fail" | "info";

export interface FixAction {
  label: string; // e.g. "Start Ollama"
  command: string; // display-only, shown to the user (e.g. "curl -fsSL ... | sh")
  argv: string[]; // actual argv passed to execa; argv[0] is the binary
  description: string; // what the fix does
}

/** Verdict for one FixAction from `planFixes` — what `--fix` would do with it.
 *  "would-run" passes the runner's binary allowlist; "blocked" fails it;
 *  "malformed" has no argv to execute. */
export interface FixPlanEntry {
  label: string;
  command: string;
  description: string;
  status: "would-run" | "blocked" | "malformed";
}

export interface DiagnosticCheck {
  label: string;
  severity: CheckSeverity;
  message: string;
  suggestion?: string;
  fix?: FixAction;
}

/** State of the live ollama.com model catalog; timestamps are epoch milliseconds. */
export interface CatalogHealth {
  source: "live" | "bundled";
  fetchedAt: number | null;
  liveModels: number;
  lastError: string | null;
  lastSuccessAt: number | null;
  /** True when LLMPULSE_OFFLINE or the offline config disables fetching. */
  offline: boolean;
}

export interface HealthReport {
  score: number; // 0-100
  checks: DiagnosticCheck[];
  summary: string;
  topSuggestion: string | null;
}

// ── Monitor ──────────────────────────────────

export type MonitorTab = "overview" | "gpu" | "vram" | "models";

export interface ModelUsage {
  name: string; // e.g. "llama3.1:8b"
  avgTokPerSec: number;
  totalTokens: number;
  totalTimeMs: number;
  requests: number;
  startedAt: number; // Date.now()
}

export interface SessionStats {
  totalTokens: number;
  totalTimeMs: number;
  totalRequests: number;
  startedAt: number; // session start timestamp
  modelHistory: Map<string, ModelUsage>;
  lastModelSwapAt: number | null;
}

export interface VramBreakdown {
  totalMb: number;
  usedMb: number;
  freeMb: number;
  modelVramMb: number; // Ollama /api/ps size_vram of the loaded model, at most usedMb; 0 when none is loaded or not split
  otherMb: number; // usedMb - modelVramMb
  split: boolean; // false: size_vram may span GPUs the monitor does not read, so only used/free are meaningful (modelVramMb is 0)
}

export type AlertSeverity = "warning" | "info" | "success";

export interface SmartAlert {
  severity: AlertSeverity;
  icon: string; // ⚠ ✓ ℹ
  message: string;
}

// ── CLI Options ───────────────────────────────

export type OutputFormat = "table" | "json" | "csv";

export interface ScanOptions {
  format: OutputFormat;
  category: ModelCategory | "all";
  top: number;
  host?: string;
}

// ── Profile Command ──────────────────────────

export interface ProfileOptions {
  model: string;
  prompt?: string;
  contextSize: number;
  format: OutputFormat;
  host?: string;
}

export interface HardwareSnapshot {
  timestampMs: number;
  phase: "idle" | "prompt" | "generation" | "complete";
  gpuUtilPercent: number | null;
  gpuVramUsedMb: number | null;
  gpuVramTotalMb: number | null;
  gpuTempCelsius: number | null;
  gpuPowerWatt: number | null;
  cpuPercent: number | null;
  ramUsedMb: number | null;
}

export interface ProfileResult {
  model: string;
  prompt: string;
  contextSize: number;
  ttftMs: number;
  promptProcessMs: number;
  generationMs: number;
  totalMs: number;
  tokensGenerated: number;
  tokensPerSec: number;
  peakVramMb: number | null;
  avgGpuByPhase: Record<string, number | null>;
  snapshots: HardwareSnapshot[];
}

// ── Check Command ────────────────────────────

export type Verdict = "yes" | "maybe" | "no";

export interface CheckOptions {
  quant?: string;
  format: OutputFormat;
  verbose: boolean;
  host?: string;
}

export type OffloadReason = "full_fit" | "partial_offload" | "too_small";

export interface QuantAdviceOptions {
  format: OutputFormat;
  verbose: boolean;
}

export interface OptimizeOptions {
  quant?: string; // pin a specific quantization instead of auto-picking the sweet spot
  format: OutputFormat;
  verbose: boolean;
}

// ── Compare Command ──────────────────────────

export interface CompareOptions {
  format: OutputFormat;
  category: ModelCategory | "all";
  top: number;
  quant?: string;
  host?: string;
}

// GPU layer-offload guidance for a specific model+quant+GPU combo.
// `gpuLayers = "all"` means the full model fits on the GPU with headroom;
// a number means "put this many transformer blocks on the GPU, run the
// rest on CPU" (maps to `num_gpu` in Ollama / `--n-gpu-layers` in llama.cpp).
// Skipped entirely (caller gets null) on unified-memory systems and when
// there's no discrete GPU to offload to.
export interface GpuOffloadSuggestion {
  gpuLayers: number | "all";
  totalLayers: number;
  estimatedVramUsedMb: number;
  reason: OffloadReason;
  ollamaCommand: string | null; // null without a tag, or when `ollama run <tag>` loads another quant than the one sized
}

// ── Optimize Command ─────────────────────────

// One tuned Ollama runtime parameter plus a short, human-readable reason.
// `value` is a plain number for num_ctx / num_thread / num_batch; num_gpu uses
// the string "all" when the whole model fits on the GPU (Ollama offloads every
// layer by default in that case).
export interface OptimizedParameter {
  value: number | "all";
  note: string;
}

// A complete set of recommended Ollama runtime parameters for one model+quant on
// the detected hardware. `numGpu` is null on unified-memory systems (Apple
// Silicon): the CPU/GPU layer split isn't meaningful there, so the parameter is
// omitted from the generated Modelfile rather than guessed.
export interface OptimizationProfile {
  quant: QuantizationVariant;
  quantForced: boolean; // true when the user pinned --quant rather than auto-picking
  fitLevel: FitLevel;
  numCtx: OptimizedParameter;
  numGpu: OptimizedParameter | null;
  numThread: OptimizedParameter;
  numBatch: OptimizedParameter;
}

// ── Context-Fit Command ──────────────────────

// Input to checkContextFit. responseTokens defaults to 512 when omitted; quant
// is the already-resolved variant (the sweet-spot pick when the caller didn't
// force one).
export interface ContextFitInput {
  promptTokens: number;
  responseTokens?: number;
  quant?: QuantizationVariant;
}

// CLI-layer options (quant is the raw string the user typed, resolved in the
// command shell).
export interface ContextFitOptions {
  promptTokens: number;
  responseTokens?: number;
  quant?: string;
  format: OutputFormat;
}

// Whether a workload fits a model's context on this hardware. `nativeWindow` is
// the model's architectural limit; `affordedByVramTokens` is the largest KV
// cache that fits beside the weights for `quant`; `effectiveMaxTokens` is the
// binding minimum of the two. `limitedBy` names which ceiling binds — the field
// that tells a caller whether to switch models ("model") or switch quant / trim
// ("hardware").
export interface ContextFitResult {
  model: { id: string; name: string; parametersBillion: number; contextWindow: number };
  quant: { name: string; vramMb: number };
  quantForced: boolean;
  promptTokens: number;
  responseTokens: number;
  neededTokens: number;
  nativeWindow: number;
  affordedByVramTokens: number;
  effectiveMaxTokens: number;
  limitedBy: "model" | "hardware";
  verdict: "yes" | "tight" | "no";
  headroomTokens: number;
  remedy: string | null;
  memory: MemoryBreakdown; // need of `quant` at neededTokens
  nativeWindowEstimated?: true; // nativeWindow is a guess, so only the hardware ceiling applied
}
