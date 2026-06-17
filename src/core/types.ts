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
  vramMb: number; // Estimated VRAM needed
  qualityRetention: number; // 0.0 - 1.0 (1.0 = no loss)
}

export interface ModelEntry {
  id: string; // e.g. "llama-3.1-8b"
  name: string; // e.g. "Llama 3.1 8B"
  provider: string; // Meta, DeepSeek, etc.
  parametersBillion: number;
  contextWindow: number;
  categories: ModelCategory[];
  qualityTier: QualityTier;
  qualityScore: number; // 0-100 (relative within tier)
  quantizations: QuantizationVariant[];
  ollamaTag: string | null; // e.g. "llama3.1:8b"
  releaseDate: string; // YYYY-MM
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

export interface HealthReport {
  score: number; // 0-100
  checks: DiagnosticCheck[];
  summary: string;
  topSuggestion: string | null;
}

// ── Monitor ──────────────────────────────────

export type MonitorTab = "overview" | "inference" | "gpu" | "vram" | "models";

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
  modelWeightsMb: number; // from model database
  kvCacheMb: number; // estimated from context
  overheadMb: number; // usedMb - weights - kvCache
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
  ollamaCommand: string | null;
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
}
