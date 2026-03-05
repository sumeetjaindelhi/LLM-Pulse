import si from "systeminformation";
import { execa } from "execa";
import type { GpuInfo } from "../core/types.js";

interface NvidiaSmiResult {
  vramTotalMb: number;
  vramUsedMb: number;
  utilizationPercent: number;
  temperatureCelsius: number;
  driverVersion: string;
  cudaVersion: string;
}

interface RocmSmiResult {
  vramTotalMb: number;
  vramUsedMb: number;
  utilizationPercent: number;
  temperatureCelsius: number;
  rocmVersion: string;
}

export interface RocmGpuStats {
  vramTotalMb: number;
  vramUsedMb: number;
  utilizationPercent: number;
  temperatureCelsius: number;
}

/** Shared parser for rocm-smi CSV output — used by both detection and monitor polling */
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
      const m = line.match(/([\d.]+)\s*%?/);
      if (m) utilizationPercent = parseInt(m[1], 10);
    } else if (lower.includes("temperature") || lower.includes("temp")) {
      const m = line.match(/([\d.]+)\s*c?/i);
      if (m) temperatureCelsius = parseFloat(m[1]);
    }
  }

  return { vramTotalMb, vramUsedMb, utilizationPercent, temperatureCelsius };
}

async function parseNvidiaSmi(): Promise<NvidiaSmiResult | null> {
  try {
    const { stdout } = await execa("nvidia-smi", [
      "--query-gpu=memory.total,memory.used,utilization.gpu,temperature.gpu,driver_version",
      "--format=csv,noheader,nounits",
    ], { timeout: 5000 });

    // Also get CUDA version from nvidia-smi header
    let cudaVersion = "";
    try {
      const { stdout: header } = await execa("nvidia-smi", [], { timeout: 5000 });
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
    if (isNaN(vramTotalMb) || isNaN(vramUsedMb)) return null;

    return {
      vramTotalMb,
      vramUsedMb,
      utilizationPercent: parseInt(utilization, 10) || 0,
      temperatureCelsius: parseInt(temp, 10) || 0,
      driverVersion: driver,
      cudaVersion,
    };
  } catch {
    return null;
  }
}

async function parseRocmSmi(): Promise<RocmSmiResult | null> {
  try {
    const { stdout } = await execa("rocm-smi", [
      "--showmeminfo", "vram",
      "--showtemp",
      "--showuse",
      "--csv",
    ], { timeout: 5000 });

    const stats = parseRocmCsv(stdout);
    if (stats.vramTotalMb === 0) return null;

    // Get ROCm version
    let rocmVersion = "";
    try {
      const { stdout: versionOut } = await execa("rocm-smi", ["--showversion"], { timeout: 5000 });
      const vMatch = versionOut.match(/ROCm[- ]?SMI version:\s*([\d.]+)/i)
        ?? versionOut.match(/([\d]+\.[\d]+\.[\d]+)/);
      if (vMatch) rocmVersion = vMatch[1];
    } catch {
      // ignore
    }

    return { ...stats, rocmVersion };
  } catch {
    return null;
  }
}

export async function detectGpus(): Promise<GpuInfo[]> {
  const graphics = await si.graphics();

  // Detect vendor first to decide which SMI tool to call
  const vendors = graphics.controllers
    .filter((c) => c.model && !c.model.includes("Microsoft"))
    .map((c) => detectVendor(c.vendor || ""));

  const hasNvidia = vendors.includes("NVIDIA");
  const hasAmd = vendors.includes("AMD");

  // Run SMI tools in parallel based on detected vendors
  const [nvidiaSmi, rocmSmi] = await Promise.all([
    hasNvidia ? parseNvidiaSmi() : Promise.resolve(null),
    hasAmd ? parseRocmSmi() : Promise.resolve(null),
  ]);

  const gpus: GpuInfo[] = [];

  for (const controller of graphics.controllers) {
    // Skip virtual/display-only adapters
    if (!controller.model || controller.model.includes("Microsoft")) continue;

    const vendor = detectVendor(controller.vendor || "");
    const isNvidia = vendor === "NVIDIA";
    const isAmd = vendor === "AMD";
    const isApple = vendor === "Apple";

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
      gpus.push({
        vendor,
        model: controller.model,
        vramMb: rocmSmi.vramTotalMb,
        driverVersion: controller.driverVersion || "",
        acceleratorVersion: rocmSmi.rocmVersion || null,
        acceleratorType: "rocm",
        utilizationPercent: rocmSmi.utilizationPercent,
        temperatureCelsius: rocmSmi.temperatureCelsius,
        vramUsedMb: rocmSmi.vramUsedMb,
      });
    } else {
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
