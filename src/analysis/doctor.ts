import { MIN_REQUIREMENTS, DOCTOR_WEIGHTS } from "../core/constants.js";
import type {
  HardwareProfile,
  RuntimeInfo,
  DiagnosticCheck,
  HealthReport,
  FixAction,
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

  // On hybrid CPUs (Apple Silicon, Intel Alder Lake+), weight E-cores at 60%
  // of a P-core for inference capability. E-cores run at lower clocks and
  // lack some SIMD features — treating them as equivalent inflates the
  // capability score and produces over-optimistic recommendations.
  const { cores: totalCores, performanceCores, efficiencyCores } = hardware.cpu;
  // `!= null` catches both null and undefined (older fixtures / remote
  // profiles missing the field entirely) without false-positive on 0.
  const hasSplit =
    performanceCores != null && efficiencyCores != null;
  const effectiveCores = hasSplit
    ? performanceCores + efficiencyCores * 0.6
    : totalCores;
  const hybridNote =
    hasSplit && efficiencyCores > 0
      ? ` (${performanceCores}P + ${efficiencyCores}E — weighted)`
      : "";

  if (effectiveCores >= MIN_REQUIREMENTS.cpuCores) {
    checks.push({
      label: "CPU Cores",
      severity: "pass",
      message: `${totalCores} cores${hybridNote} — sufficient for inference`,
    });
    score += DOCTOR_WEIGHTS.coreCount;
  } else {
    checks.push({
      label: "CPU Cores",
      severity: "warning",
      message: `Only ${totalCores} cores${hybridNote} — may bottleneck inference`,
      suggestion: "4+ performance cores recommended for smooth LLM inference.",
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

  // Apple Silicon unified memory info. `gpu.vramMb` is already the resolved
  // usable cap (sysctl iogpu.wired_limit_mb, or 67% fallback) — no further
  // multiplier needed. We show both the cap and the host RAM so the user can
  // see how much is wired to the GPU vs. reserved for the OS.
  if (gpu && gpu.vendor === "Apple") {
    const usableGb = Math.round(gpu.vramMb / 1024);
    const totalGb = Math.round(hardware.memory.totalMb / 1024);
    const pct = hardware.memory.totalMb > 0
      ? Math.round((gpu.vramMb / hardware.memory.totalMb) * 100)
      : 0;
    checks.push({
      label: "Unified Memory",
      severity: "info",
      message: `Apple Silicon unified memory — ~${usableGb} GB wired to GPU (${pct}% of ${totalGb} GB total)`,
    });
  }

  // AMD ROCm availability check
  if (gpu && gpu.vendor === "AMD" && !gpu.acceleratorVersion) {
    checks.push({
      label: "ROCm",
      severity: "warning",
      message: "AMD GPU detected but rocm-smi not found — VRAM monitoring unavailable",
      suggestion: "Install ROCm to enable GPU monitoring: https://rocm.docs.amd.com",
      fix: process.platform === "linux" ? {
        label: "Install ROCm",
        command: "sudo apt install rocm-smi-lib",
        argv: ["sudo", "apt", "install", "-y", "rocm-smi-lib"],
        description: "Installs ROCm SMI for AMD GPU monitoring (Ubuntu/Debian)",
      } : undefined,
    });
  }

  // Driver version check is NVIDIA-specific — we have no age criteria for
  // AMD/Intel/Apple drivers, so it's dishonest to claim they're "up to date".
  // Only emit a driver check for NVIDIA.
  if (gpu && gpu.driverVersion && gpu.vendor === "NVIDIA") {
    const driverMajor = parseInt(gpu.driverVersion.split(".")[0], 10) || 0;
    if (driverMajor < 550) {
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
  } else if (hardware.disk.type === "HDD") {
    checks.push({
      label: "Disk",
      severity: "warning",
      message: "HDD detected — model loading will be slow",
      suggestion: "An SSD significantly improves model load times.",
    });
    score += DOCTOR_WEIGHTS.diskType * 0.3;
  } else {
    // "Unknown" — detection couldn't classify the drive. Don't nag the user
    // about a problem we can't confirm; emit a neutral info message and award
    // half the diskType weight.
    checks.push({
      label: "Disk",
      severity: "info",
      message: "Disk type could not be detected",
    });
    score += DOCTOR_WEIGHTS.diskType * 0.5;
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

      // Check if Ollama has any models installed
      const ollamaRuntime = runtimes.find((r) => r.name === "Ollama" && r.status === "running");
      if (ollamaRuntime && ollamaRuntime.models.length === 0) {
        checks.push({
          label: "Models",
          severity: "warning",
          message: "Ollama is running but no models are installed",
          suggestion: "Pull a model to get started: ollama pull llama3.2:3b",
          fix: {
            label: "Pull a starter model",
            command: "ollama pull llama3.2:3b",
            argv: ["ollama", "pull", "llama3.2:3b"],
            description: "Downloads Llama 3.2 3B (~2 GB) — a great starter model",
          },
        });
      }
    } else {
      const installed = runtimes.find((r) => r.status !== "not_found");
      checks.push({
        label: "Runtime",
        severity: "warning",
        message: `${installed?.name ?? "Runtime"} installed but not running`,
        suggestion: `Start it with: ${installed?.name === "Ollama" ? "ollama serve" : `${installed?.name ?? "runtime"}`}`,
        fix: installed?.name === "Ollama" ? {
          label: "Start Ollama",
          command: "ollama serve",
          argv: ["ollama", "serve"],
          description: "Starts the Ollama server in the background",
        } : undefined,
      });
    }
    score += DOCTOR_WEIGHTS.runtimeInstalled;
  } else {
    // Platform-specific install fixes. For Linux we need a shell pipe, which
    // doesn't survive naive argv splitting — invoke `sh -c` explicitly so the
    // curl output streams into sh. The command string is a compile-time
    // constant, so there is no interpolation / injection risk.
    let installCmd: string;
    let installArgv: string[];
    if (process.platform === "win32") {
      installCmd = "winget install Ollama.Ollama";
      installArgv = ["winget", "install", "Ollama.Ollama"];
    } else if (process.platform === "darwin") {
      installCmd = "brew install ollama";
      installArgv = ["brew", "install", "ollama"];
    } else {
      installCmd = "curl -fsSL https://ollama.com/install.sh | sh";
      installArgv = ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"];
    }
    checks.push({
      label: "Runtime",
      severity: "fail",
      message: "No LLM runtime installed",
      suggestion: "Install Ollama (easiest): https://ollama.com",
      fix: {
        label: "Install Ollama",
        command: installCmd,
        argv: installArgv,
        description: "Installs Ollama — the easiest way to run LLMs locally",
      },
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
