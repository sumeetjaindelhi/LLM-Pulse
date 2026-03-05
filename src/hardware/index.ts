import { detectCpu } from "./cpu.js";
import { detectGpus } from "./gpu.js";
import { detectMemory } from "./memory.js";
import { detectDisk } from "./disk.js";
import type { HardwareProfile } from "../core/types.js";

let cachedProfile: HardwareProfile | null = null;

export async function detectHardware(): Promise<HardwareProfile> {
  if (cachedProfile) return cachedProfile;

  const [cpu, gpus, memory, disk] = await Promise.all([
    detectCpu(),
    detectGpus(),
    detectMemory(),
    detectDisk(),
  ]);

  // Primary GPU = the one with the most VRAM
  const primaryGpu =
    gpus.length > 0
      ? gpus.reduce((best, g) => (g.vramMb > best.vramMb ? g : best))
      : null;

  cachedProfile = { cpu, gpus, memory, disk, primaryGpu };
  return cachedProfile;
}

export function clearHardwareCache(): void {
  cachedProfile = null;
}

export { detectCpu } from "./cpu.js";
export { detectGpus } from "./gpu.js";
export { detectMemory } from "./memory.js";
export { detectDisk } from "./disk.js";
