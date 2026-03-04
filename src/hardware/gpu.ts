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

async function parseNvidiaSmi(): Promise<NvidiaSmiResult | null> {
  try {
    const { stdout } = await execa("nvidia-smi", [
      "--query-gpu=memory.total,memory.used,utilization.gpu,temperature.gpu,driver_version",
      "--format=csv,noheader,nounits",
    ]);

    // Also get CUDA version from nvidia-smi header
    let cudaVersion = "";
    try {
      const { stdout: header } = await execa("nvidia-smi");
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

export async function detectGpus(): Promise<GpuInfo[]> {
  const graphics = await si.graphics();
  const nvidiaSmi = await parseNvidiaSmi();

  const gpus: GpuInfo[] = [];

  for (const controller of graphics.controllers) {
    // Skip virtual/display-only adapters
    if (!controller.model || controller.model.includes("Microsoft")) continue;

    const vendor = detectVendor(controller.vendor || "");
    const isNvidia = vendor === "NVIDIA";

    gpus.push({
      vendor,
      model: controller.model,
      vramMb: isNvidia && nvidiaSmi
        ? nvidiaSmi.vramTotalMb
        : controller.vram || 0,
      driverVersion: isNvidia && nvidiaSmi
        ? nvidiaSmi.driverVersion
        : controller.driverVersion || "",
      cudaVersion: isNvidia && nvidiaSmi ? nvidiaSmi.cudaVersion : null,
      utilizationPercent: isNvidia && nvidiaSmi
        ? nvidiaSmi.utilizationPercent
        : null,
      temperatureCelsius: isNvidia && nvidiaSmi
        ? nvidiaSmi.temperatureCelsius
        : (controller.temperatureGpu ?? null),
      vramUsedMb: isNvidia && nvidiaSmi ? nvidiaSmi.vramUsedMb : null,
    });
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
