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

interface NvidiaGpuStats {
  vramTotalMb: number;
  vramUsedMb: number;
  utilizationPercent: number;
  temperatureCelsius: number;
  driverVersion: string;
}

interface NvidiaSmiResult {
  // One entry per nvidia-smi row, in the order nvidia-smi reports them; null
  // for a row that could not be parsed, so row positions stay aligned.
  gpus: (NvidiaGpuStats | null)[];
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

function toRocmGpuStats(gpu: z.infer<typeof RocmJsonGpu>): RocmGpuStats {
  return {
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
  };
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

  // Sort by card index so results are stable across invocations
  return Object.entries(parsed.data)
    .filter(([k]) => /^card\d+$/.test(k))
    .sort(([a], [b]) => parseInt(a.slice(4), 10) - parseInt(b.slice(4), 10))
    .map(([, gpu]) => toRocmGpuStats(gpu));
}

// ── ROCm CSV parser (fallback for rocm-smi builds without --json) ────
// `rocm-smi --csv` prints a header row with the same field names the JSON
// output uses, then one row per card:
//   device,Temperature (Sensor edge) (C),GPU use (%),VRAM Total Memory (B),…
//   card0,45.0,12,17163091968,…
export function parseRocmCsv(stdout: string): RocmGpuStats[] {
  const [header, ...rows] = stdout.trim().split("\n");
  const columns = header.split(",").map((c) => c.trim());
  return rows
    .map((row) => row.split(",").map((v) => v.trim()))
    .filter((values) => /^card\d+$/.test(values[0]))
    .map((values) =>
      toRocmGpuStats(Object.fromEntries(columns.map((column, i) => [column, values[i]]))),
    );
}

const ROCM_STAT_ARGS = ["--showmeminfo", "vram", "--showtemp", "--showuse"];

/** Per-card rocm-smi stats. Tries --json first (per-GPU, version-independent
 *  field set) and falls back to --csv for rocm-smi builds without --json.
 */
export async function readRocmGpuStats(): Promise<RocmGpuStats[]> {
  try {
    const { stdout } = await execa("rocm-smi", [...ROCM_STAT_ARGS, "--json"], { timeout: 5000 });
    const gpus = parseRocmJsonOutput(stdout);
    if (gpus.length > 0) return gpus;
  } catch {
    // old rocm-smi or transient failure — try CSV
  }
  const { stdout } = await execa("rocm-smi", [...ROCM_STAT_ARGS, "--csv"], { timeout: 5000 });
  return parseRocmCsv(stdout);
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

    const gpus = stdout.trim().split("\n").map((line) => {
      const [vramTotal, vramUsed, utilization, temp, driver] = line
        .split(",")
        .map((s) => s.trim());

      const vramTotalMb = parseInt(vramTotal, 10);
      const vramUsedMb = parseInt(vramUsed, 10);
      if (isNaN(vramTotalMb) || isNaN(vramUsedMb)) return null;

      return {
        vramTotalMb,
        vramUsedMb,
        utilizationPercent: parseInt(utilization, 10) || 0,
        temperatureCelsius: parseInt(temp, 10) || 0,
        driverVersion: driver,
      };
    });
    if (gpus.every((g) => g === null)) {
      throw new Error("nvidia-smi returned unparseable output");
    }

    return { gpus, cudaVersion };
  }, SMI_RETRY);
}

async function parseRocmSmi(): Promise<RocmSmiResult | null> {
  return retry(async () => {
    const gpus = await readRocmGpuStats();
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
  let nvidiaIndex = 0;
  let amdIndex = 0;

  for (const controller of graphics.controllers) {
    // Skip virtual/display-only adapters
    if (!controller.model || controller.model.includes("Microsoft")) continue;

    const vendor = detectVendor(controller.vendor || "");
    const isNvidia = vendor === "NVIDIA";
    const isAmd = vendor === "AMD";
    const isApple = vendor === "Apple";
    const isIntel = vendor === "Intel";

    // Multi-NVIDIA: same controller-to-row matching as the AMD path below. A
    // row nvidia-smi could not report on sends that controller down the
    // generic path instead of borrowing another card's stats.
    const stats = isNvidia && nvidiaSmi
      ? (nvidiaIndex < nvidiaSmi.gpus.length ? nvidiaSmi.gpus[nvidiaIndex] : nvidiaSmi.gpus[0])
      : null;
    if (isNvidia) nvidiaIndex++;

    if (isNvidia && nvidiaSmi && stats) {
      gpus.push({
        vendor,
        model: controller.model,
        vramMb: stats.vramTotalMb,
        driverVersion: stats.driverVersion,
        acceleratorVersion: nvidiaSmi.cudaVersion || null,
        acceleratorType: "cuda",
        utilizationPercent: stats.utilizationPercent,
        temperatureCelsius: stats.temperatureCelsius,
        vramUsedMb: stats.vramUsedMb,
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
      // specialized probe for. Intel iGPUs share system RAM; the aperture size
      // systeminformation reports for them is not usable VRAM, so vramMb is 0
      // and the scorer falls back to CPU/RAM. Discrete Arc cards (A/B-series
      // model numbers, e.g. A770, B580) have no sysfs figure off Linux but do
      // report real dedicated VRAM, so keep it. Arc-branded iGPUs ("Arc
      // Graphics", "Arc 140V") carry no such number.
      gpus.push({
        vendor,
        model: controller.model,
        vramMb: isIntel && !/\bArc\b.*\b[AB]\d{3}M?\b/i.test(controller.model) ? 0 : controller.vram || 0,
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
