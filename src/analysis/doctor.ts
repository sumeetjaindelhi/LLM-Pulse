import { MIN_REQUIREMENTS, DOCTOR_WEIGHTS, AGENT_MIN_CONTEXT } from "../core/constants.js";
import { resolveModel } from "../models/database.js";
import { computeOptimization } from "./optimizer.js";
import { CONTEXT_TIERS, snapToContextTier } from "./memory-model.js";
import { maxContextTokensForVram } from "./context-capacity.js";
import { ignoredGpu, ignoredGpuLabel } from "./scorer.js";
import type {
  HardwareProfile,
  RuntimeInfo,
  DiagnosticCheck,
  HealthReport,
  FixAction,
  FixPlanEntry,
  LoadedModel,
  CatalogHealth,
} from "../core/types.js";

// Defense-in-depth: every FixAction literal today is a hardcoded constant in
// this file, but the type system can't enforce that across future refactors.
// If FixAction ever gets populated from config or an API, this allowlist
// blocks arbitrary command execution at the exec boundary. Lives here (not in
// the CLI runner) so `planFixes` previews and `--fix` execution share one gate.
const ALLOWED_FIX_BINARIES = new Set([
  "ollama",
  "brew",
  "winget",
  "sudo",
  "apt",
  "sh",
]);

/** Classify each fix exactly the way the `--fix` runner will treat it, without
 *  executing anything. This is what `--dry-run` renders — and because the
 *  runner consumes the same plan, the preview can never drift from reality. */
export function planFixes(fixes: FixAction[]): FixPlanEntry[] {
  return fixes.map((fix) => {
    let status: FixPlanEntry["status"];
    if (!fix.argv || fix.argv.length === 0) {
      status = "malformed";
    } else if (!ALLOWED_FIX_BINARIES.has(fix.argv[0])) {
      status = "blocked";
    } else {
      status = "would-run";
    }
    return { label: fix.label, command: fix.command, description: fix.description, status };
  });
}

// Warns about each loaded model whose allocated context sits below both Ollama's
// 64K agent guidance and what this machine could hold. Advisory only: it never
// feeds the score and offers no auto-fix, since it needs a server restart.
function contextChecks(hardware: HardwareProfile, loadedModels: LoadedModel[]): DiagnosticCheck[] {
  const checks: DiagnosticCheck[] = [];
  for (const loaded of loadedModels) {
    const current = loaded.contextLength;
    const entry = resolveModel(loaded.name);
    if (current === null || !entry) continue;

    const model = loaded.arch ? { ...entry, arch: loaded.arch } : entry;
    const level = loaded.quantizationLevel?.toLowerCase();
    const quant = model.quantizations.find((q) => q.name.toLowerCase() === level) ?? computeOptimization(model, hardware)?.quant;
    if (!quant) continue;
    const target = snapToContextTier(maxContextTokensForVram(model, quant, hardware));
    if (target < CONTEXT_TIERS[0] || current >= AGENT_MIN_CONTEXT || current >= target) continue;

    checks.push({
      label: "Context",
      severity: "warning",
      message: `${loaded.name} is loaded with ${current.toLocaleString()} tokens of context; this machine can hold ${target.toLocaleString()}`,
      suggestion: `Restart Ollama with: OLLAMA_CONTEXT_LENGTH=${target} ollama serve (agents and coding tools need at least ${AGENT_MIN_CONTEXT.toLocaleString()} tokens)`,
    });
  }
  return checks;
}

const MS_PER_HOUR = 60 * 60 * 1000;
const MS_PER_DAY = 24 * MS_PER_HOUR;
// A failing refresh only matters once the catalog on disk has gone this stale.
const CATALOG_STALE_AFTER_MS = 7 * MS_PER_DAY;

function formatAge(ms: number): string {
  const days = Math.floor(ms / MS_PER_DAY);
  if (days >= 1) return `${days} day${days === 1 ? "" : "s"}`;
  const hours = Math.floor(ms / MS_PER_HOUR);
  if (hours < 1) return "less than an hour";
  return `${hours} hour${hours === 1 ? "" : "s"}`;
}

// Informational only: the bundled model list always works, so this never feeds the score.
function catalogCheck(catalog: CatalogHealth): DiagnosticCheck {
  const now = Date.now();
  const staleOrMissing = catalog.lastSuccessAt === null || now - catalog.lastSuccessAt > CATALOG_STALE_AFTER_MS;
  if (catalog.lastError !== null && staleOrMissing) {
    return {
      label: "Model catalog",
      severity: "warning",
      message: `Model catalog refresh is failing: ${catalog.lastError}`,
      suggestion: "Run `llm-pulse models --refresh` to retry; the built-in list keeps working meanwhile. To stop fetching, set \"offline\": true or LLMPULSE_OFFLINE=1.",
    };
  }
  const message = catalog.source === "live" && catalog.fetchedAt !== null
    ? `${catalog.liveModels} live models from ollama.com, updated ${formatAge(now - catalog.fetchedAt)} ago`
    : catalog.offline
      ? "offline: built-in list (LLMPULSE_OFFLINE / offline config)"
      : "online, but no catalog cached yet: built-in list";
  const suggestion = !catalog.offline && catalog.source !== "live" ? "Run `llm-pulse models --refresh` to fetch it" : undefined;
  return { label: "Model catalog", severity: "info", message, ...(suggestion && { suggestion }) };
}

export function runDiagnostics(
  hardware: HardwareProfile,
  runtimes: RuntimeInfo[],
  loadedModels: LoadedModel[] = [],
  catalog?: CatalogHealth,
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
  const ignored = ignoredGpu(hardware);
  if (ignored) {
    checks.push({
      label: "GPU",
      severity: "warning",
      message: `${ignoredGpuLabel(ignored)} — models will run on CPU`,
      suggestion: "Set OLLAMA_IGPU_ENABLE=1 to let Ollama use it; it shares system memory.",
    });
  } else if (gpu && gpu.vramMb >= MIN_REQUIREMENTS.vramMb) {
    const vramGb = (gpu.vramMb / 1024).toFixed(0);
    checks.push({ label: "GPU VRAM", severity: "pass", message: `${vramGb} GiB VRAM — see \`llm-pulse scan\` for the models that fit` });
    score += DOCTOR_WEIGHTS.gpuVram;
  } else if (gpu && gpu.vramMb > 0) {
    checks.push({
      label: "GPU VRAM",
      severity: "warning",
      message: `Only ${(gpu.vramMb / 1024).toFixed(1)} GiB VRAM — limited to small models`,
      suggestion: "4+ GiB VRAM recommended. Consider CPU-only inference or smaller models.",
    });
    score += DOCTOR_WEIGHTS.gpuVram * 0.3;
  } else {
    checks.push({
      label: "GPU",
      severity: "warning",
      message: "No dedicated GPU detected — models will run on CPU",
      suggestion: "A GPU with 6+ GiB VRAM dramatically speeds up inference.",
    });
  }

  // Apple Silicon unified memory info. `gpu.vramMb` is already the resolved
  // usable cap (iogpu.wired_limit_mb override, Metal working set, or 67%
  // fallback) — no further multiplier needed. We show both the cap and the
  // host RAM so the user can see how much is wired to the GPU vs. reserved for
  // the OS.
  if (gpu && gpu.vendor === "Apple") {
    const usableGb = Math.round(gpu.vramMb / 1024);
    const totalGb = Math.round(hardware.memory.totalMb / 1024);
    const pct = hardware.memory.totalMb > 0
      ? Math.round((gpu.vramMb / hardware.memory.totalMb) * 100)
      : 0;
    checks.push({
      label: "Unified Memory",
      severity: "info",
      message: `Apple Silicon unified memory — ~${usableGb} GiB wired to GPU (${pct}% of ${totalGb} GiB total)`,
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
    checks.push({ label: "RAM", severity: "pass", message: `${ramGb} GiB RAM — good headroom for CPU offloading` });
    score += DOCTOR_WEIGHTS.ramTotal;
  } else {
    checks.push({
      label: "RAM",
      severity: "fail",
      message: `Only ${(hardware.memory.totalMb / 1024).toFixed(1)} GiB RAM — insufficient`,
      suggestion: "8+ GiB RAM recommended. 16+ GiB ideal for larger models.",
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
    checks.push({ label: "Disk Space", severity: "pass", message: `${hardware.disk.freeGb} GiB free — plenty for models` });
    score += DOCTOR_WEIGHTS.diskSpace;
  } else {
    checks.push({
      label: "Disk Space",
      severity: "warning",
      message: `Only ${hardware.disk.freeGb} GiB free — models need 4-40 GiB each`,
      suggestion: "Free up disk space. A 7B Q4 model needs ~4 GiB.",
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
            description: "Downloads Llama 3.2 3B (~2 GiB) — a great starter model",
          },
        });
      }

      if (ollamaRuntime) checks.push(...contextChecks(hardware, loadedModels));
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

  if (catalog) checks.push(catalogCheck(catalog));

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
