import si from "systeminformation";
import { execa } from "execa";
import { z } from "zod";
import { retry } from "./retry.js";
import { detectIntelGpu } from "./intel-gpu.js";
import { resolveNvidiaSmi } from "./nvidia-smi-path.js";
import type { GpuInfo } from "../core/types.js";

// Retry on transient failures (driver contention under load). Skip retry when
// the binary itself is missing (ENOENT) — waiting won't make it appear. Total
// worst-case sleep on a retry storm: 50 + 100 = 150ms, plus each attempt's
// 5s execa timeout budget.
const SMI_RETRY = {
  attempts: 3,
  delayMs: 50,
  backoff: 2,
  shouldRetry: (err: unknown) => {
    const code = (err as { code?: string })?.code;
    return code !== "ENOENT";
  },
} as const;

interface NvidiaSmiResult {
  vramTotalMb: number;
  vramUsedMb: number;
  utilizationPercent: number;
  temperatureCelsius: number;
  driverVersion: string;
  cudaVersion: string;
}

interface RocmSmiResult {
  // Per-GPU stats, indexed by the order rocm-smi reports them (typically
  // card0, card1, …). Length >= 1 when detection succeeds.
  gpus: RocmGpuStats[];
  rocmVersion: string;
}

export interface RocmGpuStats {
  vramTotalMb: number;
  vramUsedMb: number;
  utilizationPercent: number;
  temperatureCelsius: number;
}

// ── ROCm CSV parser (legacy / fallback) ───────────────────────────────
// Old `rocm-smi --csv` output path. ROCm 5.x emits field names like
// "VRAM Total Memory (B)". ROCm 6.x+ ships `--json` which is more stable,
// but we keep this parser as a fallback when --json returns empty or is
// unsupported by the installed rocm-smi version.

/** Shared CSV parser for rocm-smi legacy output. Returns ONE GPU's aggregate.
 *  On multi-GPU boxes, this collapses stats across cards (each row is
 *  processed, the last one wins). Prefer parseRocmJsonOutput for multi-GPU.
 */
export function parseRocmCsv(stdout: string): RocmGpuStats {
  let vramTotalMb = 0;
  let vramUsedMb = 0;
  let utilizationPercent = 0;
  let temperatureCelsius = 0;

  for (const line of stdout.trim().split("\n")) {
    const lower = line.toLowerCase();
    if (lower.includes("vram total")) {
      const m = line.match(/([\d.]+)/);
      if (m) vramTotalMb = Math.round(parseFloat(m[1]) / (1024 * 1024));
    } else if (lower.includes("vram used")) {
      const m = line.match(/([\d.]+)/);
      if (m) vramUsedMb = Math.round(parseFloat(m[1]) / (1024 * 1024));
    } else if (lower.includes("gpu use") || lower.includes("gpu utilization")) {
      // ROCm 6.x emits floats here (e.g. "73.5"). parseInt would silently
      // truncate the fraction; the JSON path uses parseFloat — keep them in
      // sync so the same metric reports the same precision either way.
      const m = line.match(/([\d.]+)\s*%?/);
      if (m) utilizationPercent = Math.round(parseFloat(m[1]));
    } else if (lower.includes("temperature") || lower.includes("temp")) {
      const m = line.match(/([\d.]+)\s*c?/i);
      if (m) temperatureCelsius = parseFloat(m[1]);
    }
  }

  return { vramTotalMb, vramUsedMb, utilizationPercent, temperatureCelsius };
}

// ── ROCm JSON parser (preferred; ROCm 5.5+) ──────────────────────────
// rocm-smi --json produces `{ "card0": {...}, "card1": {...} }`. Field names
// vary slightly between versions:
//   5.x: "VRAM Total Memory (B)", "VRAM Total Used Memory (B)"
//   6.x: "VRAM Total (B)",        "VRAM Total Used (B)"
// We accept both. Values are strings (rocm-smi emits JSON with stringified
// numbers), so we coerce per field.

const RocmJsonGpu = z
  .object({
    "VRAM Total Memory (B)": z.string().optional(),
    "VRAM Total (B)": z.string().optional(),
    "VRAM Total Used Memory (B)": z.string().optional(),
    "VRAM Total Used (B)": z.string().optional(),
    "GPU use (%)": z.string().optional(),
    "Temperature (Sensor edge) (C)": z.string().optional(),
    "Temperature (Sensor junction) (C)": z.string().optional(),
    "Temperature (C)": z.string().optional(),
  })
  .passthrough();
const RocmJsonSchema = z.record(z.string(), RocmJsonGpu);

function pickBytesToMb(values: (string | undefined)[]): number {
  for (const v of values) {
    if (!v) continue;
    const n = Number(v);
    if (Number.isFinite(n) && n > 0) {
      return Math.round(n / (1024 * 1024));
    }
  }
  return 0;
}

function pickNumber(values: (string | undefined)[]): number {
  for (const v of values) {
    if (!v) continue;
    const n = parseFloat(v);
    if (Number.isFinite(n)) return n;
  }
  return 0;
}

export function parseRocmJsonOutput(stdout: string): RocmGpuStats[] {
  let raw: unknown;
  try {
    raw = JSON.parse(stdout);
  } catch {
    return [];
  }
  const parsed = RocmJsonSchema.safeParse(raw);
  if (!parsed.success) return [];

  const out: RocmGpuStats[] = [];
  // Sort by card index so results are stable across invocations
  const entries = Object.entries(parsed.data)
    .filter(([k]) => /^card\d+$/.test(k))
    .sort(([a], [b]) => parseInt(a.slice(4), 10) - parseInt(b.slice(4), 10));

  for (const [, gpu] of entries) {
    out.push({
      vramTotalMb: pickBytesToMb([gpu["VRAM Total Memory (B)"], gpu["VRAM Total (B)"]]),
      vramUsedMb: pickBytesToMb([
        gpu["VRAM Total Used Memory (B)"],
        gpu["VRAM Total Used (B)"],
      ]),
      utilizationPercent: Math.round(pickNumber([gpu["GPU use (%)"]])),
      temperatureCelsius: pickNumber([
        gpu["Temperature (Sensor edge) (C)"],
        gpu["Temperature (Sensor junction) (C)"],
        gpu["Temperature (C)"],
      ]),
    });
  }
  return out;
}

async function parseNvidiaSmi(): Promise<NvidiaSmiResult | null> {
  const binPath = await resolveNvidiaSmi();
  if (!binPath) return null;

  return retry(async () => {
    const { stdout } = await execa(binPath, [
      "--query-gpu=memory.total,memory.used,utilization.gpu,temperature.gpu,driver_version",
      "--format=csv,noheader,nounits",
    ], { timeout: 5000 });

    // Also get CUDA version from nvidia-smi header (best-effort, non-fatal).
    let cudaVersion = "";
    try {
      const { stdout: header } = await execa(binPath, [], { timeout: 5000 });
      const cudaMatch = header.match(/CUDA Version:\s*([\d.]+)/);
      if (cudaMatch) cudaVersion = cudaMatch[1];
    } catch {
      // ignore
    }

    const line = stdout.trim().split("\n")[0]; // First GPU
    const [vramTotal, vramUsed, utilization, temp, driver] = line
      .split(",")
      .map((s) => s.trim());

    const vramTotalMb = parseInt(vramTotal, 10);
    const vramUsedMb = parseInt(vramUsed, 10);
    if (isNaN(vramTotalMb) || isNaN(vramUsedMb)) {
      throw new Error("nvidia-smi returned unparseable output");
    }

    return {
      vramTotalMb,
      vramUsedMb,
      utilizationPercent: parseInt(utilization, 10) || 0,
      temperatureCelsius: parseInt(temp, 10) || 0,
      driverVersion: driver,
      cudaVersion,
    };
  }, SMI_RETRY);
}

async function parseRocmSmi(): Promise<RocmSmiResult | null> {
  return retry(async () => {
    // Try --json first: per-GPU, version-independent field set. If the
    // installed rocm-smi doesn't support --json (very old 4.x), we fall
    // through to the CSV path.
    let gpus: RocmGpuStats[] = [];
    try {
      const { stdout: jsonOut } = await execa(
        "rocm-smi",
        ["--showmeminfo", "vram", "--showtemp", "--showuse", "--json"],
        { timeout: 5000 },
      );
      gpus = parseRocmJsonOutput(jsonOut);
    } catch {
      // old rocm-smi or transient failure — retry with CSV
    }

    if (gpus.length === 0) {
      const { stdout } = await execa(
        "rocm-smi",
        ["--showmeminfo", "vram", "--showtemp", "--showuse", "--csv"],
        { timeout: 5000 },
      );
      const single = parseRocmCsv(stdout);
      if (single.vramTotalMb > 0) gpus = [single];
    }

    if (gpus.length === 0 || gpus.every((g) => g.vramTotalMb === 0)) {
      throw new Error("rocm-smi returned no usable GPU stats");
    }

    // Get ROCm version (best-effort, non-fatal).
    let rocmVersion = "";
    try {
      const { stdout: versionOut } = await execa("rocm-smi", ["--showversion"], { timeout: 5000 });
      const vMatch = versionOut.match(/ROCm[- ]?SMI version:\s*([\d.]+)/i)
        ?? versionOut.match(/([\d]+\.[\d]+\.[\d]+)/);
      if (vMatch) rocmVersion = vMatch[1];
    } catch {
      // ignore
    }

    return { gpus, rocmVersion };
  }, SMI_RETRY);
}

export async function detectGpus(): Promise<GpuInfo[]> {
  const graphics = await si.graphics();

  // Detect vendor first to decide which SMI tool(s) to call.
  const vendors = graphics.controllers
    .filter((c) => c.model && !c.model.includes("Microsoft"))
    .map((c) => detectVendor(c.vendor || ""));

  const hasNvidia = vendors.includes("NVIDIA");
  const hasAmd = vendors.includes("AMD");
  const hasIntel = vendors.includes("Intel");

  // Run probes in parallel based on detected vendors. Intel probe is cheap
  // (sysfs reads only) so we always run it when there's any Intel controller.
  const [nvidiaSmi, rocmSmi, intelGpu] = await Promise.all([
    hasNvidia ? parseNvidiaSmi() : Promise.resolve(null),
    hasAmd ? parseRocmSmi() : Promise.resolve(null),
    hasIntel ? detectIntelGpu() : Promise.resolve(null),
  ]);

  const gpus: GpuInfo[] = [];
  let amdIndex = 0;

  for (const controller of graphics.controllers) {
    // Skip virtual/display-only adapters
    if (!controller.model || controller.model.includes("Microsoft")) continue;

    const vendor = detectVendor(controller.vendor || "");
    const isNvidia = vendor === "NVIDIA";
    const isAmd = vendor === "AMD";
    const isApple = vendor === "Apple";
    const isIntel = vendor === "Intel";

    if (isNvidia && nvidiaSmi) {
      gpus.push({
        vendor,
        model: controller.model,
        vramMb: nvidiaSmi.vramTotalMb,
        driverVersion: nvidiaSmi.driverVersion,
        acceleratorVersion: nvidiaSmi.cudaVersion || null,
        acceleratorType: "cuda",
        utilizationPercent: nvidiaSmi.utilizationPercent,
        temperatureCelsius: nvidiaSmi.temperatureCelsius,
        vramUsedMb: nvidiaSmi.vramUsedMb,
      });
    } else if (isAmd && rocmSmi) {
      // Multi-AMD: match each controller to the next per-GPU row from
      // rocm-smi --json (stable order via card0/card1/…). If rocm-smi
      // reported fewer rows than systeminformation saw controllers (rare —
      // usually means one GPU is powered off), fall back to the first row.
      const stats = rocmSmi.gpus[amdIndex] ?? rocmSmi.gpus[0];
      amdIndex++;
      gpus.push({
        vendor,
        model: controller.model,
        vramMb: stats.vramTotalMb,
        // ROCm tooling doesn't expose the kernel driver version via rocm-smi
        // itself — prefer the rocmVersion over systeminformation's field
        // for diagnostics; doctor.ts uses acceleratorVersion anyway.
        driverVersion: controller.driverVersion || rocmSmi.rocmVersion || "",
        acceleratorVersion: rocmSmi.rocmVersion || null,
        acceleratorType: "rocm",
        utilizationPercent: stats.utilizationPercent,
        temperatureCelsius: stats.temperatureCelsius,
        vramUsedMb: stats.vramUsedMb,
      });
    } else if (isIntel && intelGpu && intelGpu.vramMb > 0) {
      // Intel Arc / discrete Intel GPU: sysfs gave us a real VRAM number.
      // Mark as oneapi so the scorer knows it's a GPU, not CPU-only.
      gpus.push({
        vendor,
        model: controller.model,
        vramMb: intelGpu.vramMb,
        driverVersion: controller.driverVersion || "",
        acceleratorVersion: intelGpu.hasOneApiRuntime ? "oneapi" : null,
        acceleratorType: "oneapi",
        utilizationPercent: null,
        temperatureCelsius: controller.temperatureGpu ?? null,
        vramUsedMb: null,
      });
    } else {
      // Generic path: Apple Metal, Intel iGPU, anything we don't have a
      // specialized probe for. For Intel iGPUs, vramMb stays 0 and the
      // scorer falls back to CPU/RAM — correct behaviour (iGPUs share
      // system RAM, not dedicated VRAM).
      gpus.push({
        vendor,
        model: controller.model,
        vramMb: controller.vram || 0,
        driverVersion: controller.driverVersion || "",
        acceleratorVersion: null,
        acceleratorType: isApple ? "metal" : null,
        utilizationPercent: null,
        temperatureCelsius: controller.temperatureGpu ?? null,
        vramUsedMb: null,
      });
    }
  }

  return gpus;
}

function detectVendor(vendor: string): string {
  const v = vendor.toLowerCase();
  if (v.includes("nvidia")) return "NVIDIA";
  if (v.includes("amd") || v.includes("advanced micro")) return "AMD";
  if (v.includes("intel")) return "Intel";
  if (v.includes("apple")) return "Apple";
  return vendor || "Unknown";
}
