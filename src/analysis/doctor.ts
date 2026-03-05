import { MIN_REQUIREMENTS, DOCTOR_WEIGHTS, APPLE_UNIFIED_MEMORY_FACTOR } from "../core/constants.js";
import type {
  HardwareProfile,
  RuntimeInfo,
  DiagnosticCheck,
  HealthReport,
} from "../core/types.js";

export function runDiagnostics(
  hardware: HardwareProfile,
  runtimes: RuntimeInfo[],
): HealthReport {
  const checks: DiagnosticCheck[] = [];
  let score = 0;
  const maxScore = Object.values(DOCTOR_WEIGHTS).reduce((a, b) => a + b, 0);

  // ── CPU Checks ──
  if (hardware.cpu.hasAvx2) {
    checks.push({ label: "AVX2", severity: "pass", message: "CPU supports AVX2 — great for inference" });
    score += DOCTOR_WEIGHTS.avx2;
  } else if (hardware.cpu.architecture === "arm64") {
    // ARM CPUs (Apple Silicon, Snapdragon) use NEON instead of AVX2 — not a problem
    checks.push({ label: "SIMD", severity: "pass", message: "ARM CPU with NEON — optimized for inference" });
    score += DOCTOR_WEIGHTS.avx2;
  } else {
    checks.push({
      label: "AVX2",
      severity: "warning",
      message: "CPU lacks AVX2 — inference will be slower",
      suggestion: "AVX2 accelerates model inference significantly. Consider a newer CPU.",
    });
  }

  if (hardware.cpu.cores >= MIN_REQUIREMENTS.cpuCores) {
    checks.push({ label: "CPU Cores", severity: "pass", message: `${hardware.cpu.cores} cores — sufficient for inference` });
    score += DOCTOR_WEIGHTS.coreCount;
  } else {
    checks.push({
      label: "CPU Cores",
      severity: "warning",
      message: `Only ${hardware.cpu.cores} cores — may bottleneck inference`,
      suggestion: "4+ cores recommended for smooth LLM inference.",
    });
    score += DOCTOR_WEIGHTS.coreCount * 0.5;
  }

  // ── GPU Checks ──
  const gpu = hardware.primaryGpu;
  if (gpu && gpu.vramMb >= MIN_REQUIREMENTS.vramMb) {
    const vramGb = (gpu.vramMb / 1024).toFixed(0);
    const sizeHint = gpu.vramMb >= 16384 ? "7B-32B" : gpu.vramMb >= 8192 ? "7B-14B" : "3B-7B";
    checks.push({ label: "GPU VRAM", severity: "pass", message: `${vramGb} GB VRAM — can run ${sizeHint} models` });
    score += DOCTOR_WEIGHTS.gpuVram;
  } else if (gpu && gpu.vramMb > 0) {
    checks.push({
      label: "GPU VRAM",
      severity: "warning",
      message: `Only ${(gpu.vramMb / 1024).toFixed(1)} GB VRAM — limited to small models`,
      suggestion: "4+ GB VRAM recommended. Consider CPU-only inference or smaller models.",
    });
    score += DOCTOR_WEIGHTS.gpuVram * 0.3;
  } else {
    checks.push({
      label: "GPU",
      severity: "warning",
      message: "No dedicated GPU detected — models will run on CPU",
      suggestion: "A GPU with 6+ GB VRAM dramatically speeds up inference.",
    });
  }

  // Apple Silicon unified memory info
  if (gpu && gpu.vendor === "Apple") {
    const usableGb = Math.round((gpu.vramMb * APPLE_UNIFIED_MEMORY_FACTOR) / 1024);
    checks.push({
      label: "Unified Memory",
      severity: "info",
      message: `Apple Silicon unified memory — ~${usableGb} GB usable for inference (75% of ${Math.round(gpu.vramMb / 1024)} GB)`,
    });
  }

  // AMD ROCm availability check
  if (gpu && gpu.vendor === "AMD" && !gpu.acceleratorVersion) {
    checks.push({
      label: "ROCm",
      severity: "warning",
      message: "AMD GPU detected but rocm-smi not found — VRAM monitoring unavailable",
      suggestion: "Install ROCm to enable GPU monitoring: https://rocm.docs.amd.com",
    });
  }

  if (gpu && gpu.driverVersion) {
    const driverMajor = parseInt(gpu.driverVersion.split(".")[0], 10) || 0;
    if (gpu.vendor === "NVIDIA" && driverMajor < 550) {
      checks.push({
        label: "GPU Driver",
        severity: "warning",
        message: `GPU driver ${gpu.driverVersion} — update to 550+ recommended`,
        suggestion: "Newer drivers improve CUDA performance and compatibility.",
      });
      score += DOCTOR_WEIGHTS.gpuDriver * 0.5;
    } else {
      checks.push({ label: "GPU Driver", severity: "pass", message: `GPU driver ${gpu.driverVersion} — up to date` });
      score += DOCTOR_WEIGHTS.gpuDriver;
    }
  }

  // ── Memory Checks ──
  if (hardware.memory.totalMb >= MIN_REQUIREMENTS.ramMb) {
    const ramGb = (hardware.memory.totalMb / 1024).toFixed(0);
    checks.push({ label: "RAM", severity: "pass", message: `${ramGb} GB RAM — good headroom for CPU offloading` });
    score += DOCTOR_WEIGHTS.ramTotal;
  } else {
    checks.push({
      label: "RAM",
      severity: "fail",
      message: `Only ${(hardware.memory.totalMb / 1024).toFixed(1)} GB RAM — insufficient`,
      suggestion: "8+ GB RAM recommended. 16+ GB ideal for larger models.",
    });
  }

  if (hardware.memory.speedMhz && hardware.memory.speedMhz >= 3200) {
    checks.push({ label: "RAM Speed", severity: "pass", message: `${hardware.memory.speedMhz} MHz — fast memory` });
    score += DOCTOR_WEIGHTS.ramSpeed;
  } else if (hardware.memory.speedMhz) {
    checks.push({
      label: "RAM Speed",
      severity: "info",
      message: `${hardware.memory.speedMhz} MHz RAM — adequate`,
    });
    score += DOCTOR_WEIGHTS.ramSpeed * 0.7;
  }

  // ── Disk Checks ──
  if (hardware.disk.type === "NVMe") {
    checks.push({ label: "Disk", severity: "pass", message: "NVMe SSD — fast model loading" });
    score += DOCTOR_WEIGHTS.diskType;
  } else if (hardware.disk.type === "SSD") {
    checks.push({ label: "Disk", severity: "pass", message: "SSD — decent model loading speed" });
    score += DOCTOR_WEIGHTS.diskType * 0.8;
  } else {
    checks.push({
      label: "Disk",
      severity: "warning",
      message: "HDD detected — model loading will be slow",
      suggestion: "An SSD significantly improves model load times.",
    });
    score += DOCTOR_WEIGHTS.diskType * 0.3;
  }

  if (hardware.disk.freeGb >= MIN_REQUIREMENTS.diskFreeGb) {
    checks.push({ label: "Disk Space", severity: "pass", message: `${hardware.disk.freeGb} GB free — plenty for models` });
    score += DOCTOR_WEIGHTS.diskSpace;
  } else {
    checks.push({
      label: "Disk Space",
      severity: "warning",
      message: `Only ${hardware.disk.freeGb} GB free — models need 4-40 GB each`,
      suggestion: "Free up disk space. A 7B Q4 model needs ~4 GB.",
    });
  }

  // ── Runtime Checks ──
  const hasRuntime = runtimes.some((r) => r.status !== "not_found");
  if (hasRuntime) {
    const running = runtimes.find((r) => r.status === "running");
    if (running) {
      checks.push({ label: "Runtime", severity: "pass", message: `${running.name} installed and running` });
    } else {
      const installed = runtimes.find((r) => r.status !== "not_found");
      checks.push({ label: "Runtime", severity: "pass", message: `${installed?.name ?? "Runtime"} installed` });
    }
    score += DOCTOR_WEIGHTS.runtimeInstalled;
  } else {
    checks.push({
      label: "Runtime",
      severity: "fail",
      message: "No LLM runtime installed",
      suggestion: "Install Ollama (easiest): https://ollama.com",
    });
  }

  // ── Compute final score ──
  const normalizedScore = Math.round((score / maxScore) * 100);

  const warnings = checks.filter((c) => c.severity === "warning" || c.severity === "fail");
  const topSuggestion = warnings.find((w) => w.suggestion)?.suggestion ?? null;

  let summary: string;
  if (normalizedScore >= 80) summary = "Great for local LLMs";
  else if (normalizedScore >= 60) summary = "Good — can run most small-medium models";
  else if (normalizedScore >= 40) summary = "Basic — limited to small models";
  else summary = "Limited — consider hardware upgrades";

  return { score: normalizedScore, checks, summary, topSuggestion };
}
