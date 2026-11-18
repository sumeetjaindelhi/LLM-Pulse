import si from "systeminformation";
import { execa } from "execa";
import { z } from "zod";
import { retry } from "./retry.js";
import { detectIntelGpu } from "./intel-gpu.js";
import { readAmdSysfsMemory } from "./amd-sysfs.js";
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
  // 0 when nvidia-smi reports no total (unified memory only).
  vramTotalMb: number;
  // null when memory.used is "[N/A]" (unified-memory GPUs).
  vramUsedMb: number | null;
  unifiedMemory: boolean;
  utilizationPercent: number;
  temperatureCelsius: number;
  driverVersion: string;
}

interface NvidiaSmiResult {
  // One entry per nvidia-smi row, in the order nvidia-smi reports them; null
  // for a row that could not be parsed, so row positions stay aligned.
  gpus: (NvidiaGpuStats | null)[];
  // Normalised PCI bus id of each row (see normalizeBusId), same order as `gpus`.
  busIds: (string | undefined)[];
  cudaVersion: string;
}

interface RocmSmiResult {
  // Per-GPU stats, indexed by the order rocm-smi reports them (typically
  // card0, card1, …). Length >= 1 when detection succeeds.
  gpus: RocmGpuStats[];
  rocmVersion: string;
}

export interface RocmGpuStats {
  // Normalised PCI bus id; undefined when rocm-smi did not report one.
  busId?: string;
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
    "PCI Bus": z.string().optional(),
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

/** "00000000:0A:00.0" and "0a:00.0" both become "0:0a:00.0": domain as a number (missing = 0), lower case. undefined when not a PCI bus id. */
export function normalizeBusId(raw: string | undefined): string | undefined {
  const m = raw?.trim().toLowerCase().match(/^(?:([0-9a-f]{1,8}):)?([0-9a-f]{2}:[0-9a-f]{2}\.[0-7])$/);
  return m ? `${parseInt(m[1] ?? "0", 16)}:${m[2]}` : undefined;
}

function toRocmGpuStats(gpu: z.infer<typeof RocmJsonGpu>): RocmGpuStats {
  const busId = normalizeBusId(gpu["PCI Bus"]);
  return {
    ...(busId && { busId }),
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

const ROCM_STAT_ARGS = ["--showmeminfo", "vram", "--showtemp", "--showuse", "--showbus"];

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

// nvidia-smi prints "[N/A]" / "Not Supported" instead of a number for fields
// the GPU cannot report; parseInt turns those into NaN.
export function parseSmiInt(field: string): number | null {
  const value = parseInt(field, 10);
  return isNaN(value) ? null : value;
}

// GB10 / Jetson (arm64) have no dedicated VRAM, so nvidia-smi cannot report
// memory.used. On any other host such a row is just a broken reading.
async function parseNvidiaSmi(arch: string): Promise<NvidiaSmiResult | null> {
  const binPath = await resolveNvidiaSmi();
  if (!binPath) return null;

  return retry(async () => {
    const { stdout } = await execa(binPath, [
      "--query-gpu=memory.total,memory.used,utilization.gpu,temperature.gpu,driver_version,pci.bus_id",
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

    const rows = stdout.trim().split("\n").map((line) => line.split(",").map((s) => s.trim()));
    // Unified-memory NVIDIA systems are single-GPU, so a lone memory-less row
    // is GB10; a memory-less row beside a healthy one is just a sick card.
    const hasDedicatedRow = rows.some((row) => parseSmiInt(row[1]) !== null);
    const gpus = rows.map(([vramTotal, vramUsed, utilization, temp, driver]) => {
      const vramTotalMb = parseSmiInt(vramTotal);
      const vramUsedMb = parseSmiInt(vramUsed);
      const unifiedMemory =
        arch === "arm64" && vramUsedMb === null && driver !== undefined && !hasDedicatedRow;
      if (!unifiedMemory && (vramTotalMb === null || vramUsedMb === null)) return null;

      return {
        vramTotalMb: vramTotalMb ?? 0,
        vramUsedMb,
        unifiedMemory,
        utilizationPercent: parseInt(utilization, 10) || 0,
        temperatureCelsius: parseInt(temp, 10) || 0,
        driverVersion: driver,
      };
    });
    if (gpus.every((g) => g === null)) {
      throw new Error("nvidia-smi returned unparseable output");
    }

    return { gpus, busIds: rows.map((row) => normalizeBusId(row[5])), cudaVersion };
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

// Intel ARK naming: discrete Arc cards are A3xx/A5xx/A7xx (A730M mobile), B5xx and up (B570, B580),
// embedded A3xxE/A5xxE, and Arc Pro A/B with two digits (A60, B60). Arc B3xx and Arc Pro B3xx (B390, B370:
// Panther Lake iGPUs), "Arc 1xxV/1xxT" and plain "Graphics" are iGPUs that share system RAM and report no VRAM.
export function isDiscreteIntel(model: string): boolean {
  return /\bArc(?:\(TM\))?\s+(?:Pro\s+[AB]\d{2}M?|A[3-7]\d{2}[ME]?|B[5-9]\d{2})\b/i.test(model);
}

export async function detectGpus(arch: string = process.arch): Promise<GpuInfo[]> {
  const graphics = await si.graphics();

  // Detect vendor first to decide which SMI tool(s) to call.
  const vendors = graphics.controllers
    .filter((c) => c.model && !c.model.includes("Microsoft"))
    .map((c) => detectVendor(c.vendor || ""));

  const hasNvidia = vendors.includes("NVIDIA");
  const hasAmd = vendors.includes("AMD");
  const hasIntel = vendors.includes("Intel");

  // Run probes in parallel based on detected vendors. Intel probe only checks
  // for the Level Zero loader, so we always run it when there's any Intel controller.
  const [nvidiaSmi, rocmSmi, intelGpu, amdSysfs] = await Promise.all([
    hasNvidia ? parseNvidiaSmi(arch) : Promise.resolve(null),
    hasAmd ? parseRocmSmi() : Promise.resolve(null),
    hasIntel ? detectIntelGpu() : Promise.resolve(null),
    hasAmd ? readAmdSysfsMemory() : Promise.resolve([]),
  ]);

  const amdControllerCount = vendors.filter((v) => v === "AMD").length;
  const gpus: GpuInfo[] = [];
  let nvidiaIndex = 0;
  let amdIndex = 0;

  // An smi row belongs to the controller on the same PCI bus: a card the driver does not use (vfio-bound) must not
  // shift the pairing. Position is the fallback only when no bus ids are available at all.
  const controllerBuses = graphics.controllers.map((c) => normalizeBusId(c.busAddress));
  const nvidiaByBus = nvidiaSmi?.busIds.some(Boolean) === true && controllerBuses.some(Boolean);
  const rocmBusIds = rocmSmi?.gpus.map((g) => g.busId) ?? [];
  const rocmByBus = rocmBusIds.some(Boolean) && controllerBuses.some(Boolean);
  // A row serves one controller: a second controller on the same bus (a clinfo duplicate of an lspci entry) is dropped.
  const usedNvidiaRows = new Set<number>();
  const usedRocmRows = new Set<number>();
  const rowFor = (byBus: boolean, busIds: (string | undefined)[], bus: string | undefined, position: number, used: Set<number>): number => {
    const row = byBus ? busIds.findIndex((id) => id !== undefined && id === bus) : position < busIds.length ? position : -1;
    if (row < 0 || used.has(row)) return -1;
    used.add(row);
    return row;
  };

  for (const [controllerIndex, controller] of graphics.controllers.entries()) {
    // Skip virtual/display-only adapters
    if (!controller.model || controller.model.includes("Microsoft")) continue;

    const vendor = detectVendor(controller.vendor || "");
    const isNvidia = vendor === "NVIDIA";
    const isAmd = vendor === "AMD";
    const isApple = vendor === "Apple";
    const isIntel = vendor === "Intel";

    // Multi-NVIDIA: the row on the controller's PCI bus (by position without bus ids). A controller with no row
    // (vfio-bound, unsupported by the driver) is not usable and is dropped, never given another card's row. A row
    // nvidia-smi could not report on sends that controller down the generic path.
    const nvidiaRow = isNvidia && nvidiaSmi ? rowFor(nvidiaByBus, nvidiaSmi.busIds, controllerBuses[controllerIndex], nvidiaIndex, usedNvidiaRows) : 0;
    if (isNvidia && nvidiaSmi && nvidiaRow < 0) continue;
    const stats = isNvidia && nvidiaSmi ? (nvidiaSmi.gpus[nvidiaRow] ?? null) : null;
    if (isNvidia) nvidiaIndex++;

    // Position among AMD controllers, shared by the rocm-smi rows and sysfs cards.
    // Skipped when the counts differ: a card without a DRM node (e.g. bound to
    // vfio-pci) would shift every later pairing (ollama discover/amd.go:163-167).
    const amdPosition = isAmd ? amdIndex++ : 0;
    const sysfs = isAmd && amdSysfs.length === amdControllerCount ? amdSysfs[amdPosition] ?? null : null;
    const apu = sysfs?.apu ?? null;

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
        ...(stats.unifiedMemory && { unifiedMemory: true }),
      });
    } else if (isAmd && rocmSmi) {
      // Multi-AMD: match each controller to the rocm-smi row on its PCI bus (by position without bus ids; stable
      // order via card0/card1/…). A controller with no row (one GPU powered off or unsupported) is dropped.
      const stats = rocmSmi.gpus[rowFor(rocmByBus, rocmBusIds, controllerBuses[controllerIndex], amdPosition, usedRocmRows)];
      if (!stats) continue;
      gpus.push({
        vendor,
        model: controller.model,
        vramMb: apu?.gttTotalMb ?? stats.vramTotalMb,
        // ROCm tooling doesn't expose the kernel driver version via rocm-smi
        // itself — prefer the rocmVersion over systeminformation's field
        // for diagnostics; doctor.ts uses acceleratorVersion anyway.
        driverVersion: controller.driverVersion || rocmSmi.rocmVersion || "",
        acceleratorVersion: rocmSmi.rocmVersion || null,
        acceleratorType: "rocm",
        utilizationPercent: stats.utilizationPercent,
        temperatureCelsius: stats.temperatureCelsius,
        vramUsedMb: apu?.gttUsedMb ?? stats.vramUsedMb,
        ...(apu && { unifiedMemory: true }),
        ...(sysfs?.gfxTarget && { gfxTarget: sysfs.gfxTarget }),
      });
    } else if (isIntel && intelGpu?.hasOneApiRuntime && isDiscreteIntel(controller.model)) {
      // Discrete Intel Arc with a Level Zero loader present (probed on Linux only). There controller.vram is an
      // lspci BAR parse (16 with ReBAR, else 256; i915/xe expose no VRAM size), never a VRAM figure.
      gpus.push({
        vendor,
        model: controller.model,
        vramMb: 0,
        driverVersion: controller.driverVersion || "",
        acceleratorVersion: "oneapi",
        acceleratorType: "oneapi",
        utilizationPercent: null,
        temperatureCelsius: controller.temperatureGpu ?? null,
        vramUsedMb: null,
      });
    } else {
      // Generic path: Apple Metal, Intel iGPU, anything we don't have a
      // specialized probe for. Intel iGPUs share system RAM; the aperture size
      // systeminformation reports for them is not usable VRAM, so vramMb is 0
      // and the scorer falls back to CPU/RAM. Off Linux, discrete Arc cards
      // (A/B-series model numbers, e.g. A770, B580) keep the driver's dedicated
      // VRAM figure from systeminformation; on Linux that figure is an lspci
      // BAR parse, so it is 0 there too.
      gpus.push({
        vendor,
        model: controller.model,
        // Prefer the kernel's amdgpu vram_total over systeminformation's
        // lspci BAR-size figure (lib/graphics.js reads `size=` regions).
        vramMb: apu?.gttTotalMb
          ?? sysfs?.vramTotalMb
          ?? (isIntel && (process.platform === "linux" || !isDiscreteIntel(controller.model)) ? 0 : controller.vram || 0),
        driverVersion: controller.driverVersion || "",
        acceleratorVersion: null,
        acceleratorType: isApple ? "metal" : null,
        utilizationPercent: null,
        temperatureCelsius: controller.temperatureGpu ?? null,
        vramUsedMb: null,
        ...((isApple || apu) && { unifiedMemory: true }),
        ...(sysfs?.gfxTarget && { gfxTarget: sysfs.gfxTarget }),
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
