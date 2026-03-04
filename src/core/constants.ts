export const VERSION = "0.1.0";
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

// Runtime detection
export const OLLAMA_API_URL = "http://127.0.0.1:11434";

export const LMSTUDIO_PATHS = {
  win32: [
    `${process.env.LOCALAPPDATA || ""}/LM Studio`,
    `${process.env.PROGRAMFILES || ""}/LM Studio`,
  ],
  darwin: ["/Applications/LM Studio.app"],
  linux: ["/opt/lm-studio", `${process.env.HOME || ""}/.local/share/lm-studio`],
} as const;
