// ── Hardware ──────────────────────────────────

export interface CpuInfo {
  brand: string;
  manufacturer: string;
  cores: number;
  threads: number;
  speed: number; // GHz base
  speedMax: number; // GHz boost
  architecture: string; // x64, arm64
  flags: string[]; // AVX, AVX2, AVX-512, etc.
  hasAvx2: boolean;
}

export interface GpuInfo {
  vendor: string; // NVIDIA, AMD, Intel, Apple
  model: string;
  vramMb: number;
  driverVersion: string;
  cudaVersion: string | null;
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

export interface DiagnosticCheck {
  label: string;
  severity: CheckSeverity;
  message: string;
  suggestion?: string;
}

export interface HealthReport {
  score: number; // 0-100
  checks: DiagnosticCheck[];
  summary: string;
  topSuggestion: string | null;
}

// ── CLI Options ───────────────────────────────

export type OutputFormat = "table" | "json";

export interface ScanOptions {
  format: OutputFormat;
  category: ModelCategory | "all";
  top: number;
  verbose: boolean;
}
