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

// Apple Silicon unified memory — ~25% reserved for OS + apps
export const APPLE_UNIFIED_MEMORY_FACTOR = 0.75;

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
} as const;

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

export const LMSTUDIO_PATHS = {
  win32: [
    `${process.env.LOCALAPPDATA || ""}/LM Studio`,
    `${process.env.PROGRAMFILES || ""}/LM Studio`,
  ],
  darwin: ["/Applications/LM Studio.app"],
  linux: ["/opt/lm-studio", `${process.env.HOME || ""}/.local/share/lm-studio`],
} as const;
