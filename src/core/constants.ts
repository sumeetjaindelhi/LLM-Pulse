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

// Apple Silicon unified-memory GPU budget, last resort only. The runtime
// reader `hardware/apple-memory.ts` prefers a user-raised
// `iogpu.wired_limit_mb`, then Metal's `recommendedMaxWorkingSetSize`
// (e.g. 74% of 24 GB). This conservative factor applies only when neither
// can be read, and it deliberately errs low: an overestimate produces "fits"
// verdicts on models that OOM during real inference.
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
