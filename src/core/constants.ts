import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { existsSync } from "node:fs";

// Resolve package.json from either src/core/ or dist/src/core/
const __dirname = dirname(fileURLToPath(import.meta.url));
const req = createRequire(import.meta.url);
const pkgPath = existsSync(resolve(__dirname, "../../package.json"))
  ? "../../package.json"
  : "../../../package.json";
const pkg = req(pkgPath) as { version: string };


export const VERSION = pkg.version;
export const APP_NAME = "LLM Pulse";

// Fit level thresholds (available VRAM / required VRAM)
export const FIT_THRESHOLDS = {
  excellent: 1.5, // 50%+ headroom
  comfortable: 1.15, // 15%+ headroom
  tight: 1.0, // Just fits
  barely: 0.75, // Needs CPU offloading
} as const;

// Doctor scoring
export const DOCTOR_WEIGHTS = {
  avx2: 10,
  gpuVram: 20,
  gpuDriver: 5,
  ramTotal: 15,
  ramSpeed: 5,
  diskType: 10,
  diskSpace: 10,
  runtimeInstalled: 15,
  coreCount: 10,
} as const;

// Minimum requirements for warnings
export const MIN_REQUIREMENTS = {
  ramMb: 8192, // 8 GB
  vramMb: 4096, // 4 GB
  diskFreeGb: 10,
  cpuCores: 4,
} as const;

// Apple Silicon wired-GPU-memory cap. macOS enforces `iogpu.wired_limit_mb`
// which defaults to ~67% of total RAM on Apple Silicon (the remainder is
// reserved for OS + CPU processes and cannot be wired for GPU use). The
// runtime reader `hardware/apple-memory.ts` prefers the live sysctl value;
// this constant is the fallback when sysctl is unreachable. 0.67 is the
// conservative default documented by Apple for long-running ML workloads.
// Previous value 0.75 was overoptimistic — it produced "fits" verdicts on
// models that hit OOM during real inference on high-memory Macs.
export const APPLE_UNIFIED_MEMORY_FACTOR_FALLBACK = 0.67;

// On unified-memory systems, the sysctl GPU-wired limit is the theoretical
// ceiling, but real-world usage has to share with the OS, IDE, Node, and
// every other app drawing from the same RAM pool. Subtracting this flat
// headroom from the max-params estimate keeps the "you can run up to NB"
// tip honest for daily-driver machines. 6 GB covers a modern dev setup
// (OS + browser + editor + Node + small tools) with margin for spikes.
export const UNIFIED_MEMORY_HEADROOM_MB = 6144;

// Runtime detection
export const OLLAMA_API_URL = "http://127.0.0.1:11434";
export const LMSTUDIO_API_URL = "http://127.0.0.1:1234";

// Monitor alert thresholds
export const ALERT_THRESHOLDS = {
  vramHighPercent: 85,
  tokSpeedDropPercent: 30,
  gpuUnderutilizedPercent: 20,
  noModelTimeoutMs: 5 * 60 * 1000, // 5 minutes
  sparklineHistory: 60, // number of data points
  gpuTempHighCelsius: 80, // vendor-agnostic default; prefer GPU_TEMP_THRESHOLDS_BY_VENDOR
} as const;

// Vendor-specific "this GPU is getting hot" thresholds. Apple Silicon runs
// fanless and thermally-throttles around 72°C; raising the alert line that
// high would miss real throttle events. NVIDIA/AMD desktops tolerate 85°C
// comfortably; laptop SKUs throttle earlier (around 78°C for mobile GPUs).
// Intel Arc dGPUs sit closer to NVIDIA desktop territory.
export const GPU_TEMP_THRESHOLDS_BY_VENDOR = {
  apple: 72,
  nvidia_desktop: 85,
  nvidia_mobile: 78,
  amd: 85,
  intel: 85,
  unknown: 80,
} as const;

export type GpuTempThresholdKey = keyof typeof GPU_TEMP_THRESHOLDS_BY_VENDOR;

/** Pick the right temp threshold for a GPU. `isMobile` is a best-effort hint
 *  from the model string — "Laptop GPU" / "Mobile" / "Max-Q" / "Max-P" in
 *  the name all raise the flag. Callers that don't know should pass false.
 */
export function pickGpuTempThreshold(vendor: string, isMobile: boolean): number {
  const v = vendor.toLowerCase();
  if (v.includes("apple")) return GPU_TEMP_THRESHOLDS_BY_VENDOR.apple;
  if (v.includes("nvidia")) {
    return isMobile
      ? GPU_TEMP_THRESHOLDS_BY_VENDOR.nvidia_mobile
      : GPU_TEMP_THRESHOLDS_BY_VENDOR.nvidia_desktop;
  }
  if (v.includes("amd") || v.includes("advanced micro")) {
    return GPU_TEMP_THRESHOLDS_BY_VENDOR.amd;
  }
  if (v.includes("intel")) return GPU_TEMP_THRESHOLDS_BY_VENDOR.intel;
  return GPU_TEMP_THRESHOLDS_BY_VENDOR.unknown;
}

// Expected tok/s baselines by GPU tier + model size (rough estimates)
// Format: { [vramTierGb]: { [paramBillion]: expectedTokPerSec } }
export const EXPECTED_TOK_PER_SEC: Record<string, Record<string, number>> = {
  "4": { "3": 25, "7": 10 },
  "6": { "3": 40, "7": 20, "8": 15 },
  "8": { "3": 60, "7": 35, "8": 30, "13": 15 },
  "10": { "3": 70, "7": 45, "8": 40, "13": 25 },
  "12": { "3": 80, "7": 50, "8": 45, "13": 30 },
  "16": { "3": 90, "7": 55, "8": 50, "13": 35, "34": 12 },
  "24": { "3": 100, "7": 65, "8": 60, "13": 45, "34": 20, "70": 8 },
} as const;

// LM Studio install-location hints. Non-absolute paths are prefixed with an
// env-var name (or the sentinel `HOME`, which resolves via `os.homedir()`).
// Resolution happens in `src/runtimes/lmstudio.ts` and refuses to build a path
// when the env var is missing — this prevents the old bug where an unset
// LOCALAPPDATA would produce `/LM Studio` at the filesystem root.
export const LMSTUDIO_PATH_HINTS = {
  win32: ["LOCALAPPDATA/LM Studio", "PROGRAMFILES/LM Studio"],
  darwin: ["/Applications/LM Studio.app"],
  linux: ["/opt/lm-studio", "HOME/.local/share/lm-studio"],
} as const;
