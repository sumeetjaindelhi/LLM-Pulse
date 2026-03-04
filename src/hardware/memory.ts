import si from "systeminformation";
import type { MemoryInfo } from "../core/types.js";

export async function detectMemory(): Promise<MemoryInfo> {
  const [mem, layout] = await Promise.all([si.mem(), si.memLayout()]);

  const totalMb = Math.round(mem.total / (1024 * 1024));
  const usedMb = Math.round(mem.used / (1024 * 1024));
  const availableMb = Math.round(mem.available / (1024 * 1024));

  // Get memory type and speed from first module
  const firstModule = layout[0];
  const type = firstModule?.type || "Unknown";
  const speedMhz = firstModule?.clockSpeed ?? null;

  return {
    totalMb,
    availableMb,
    usedMb,
    usedPercent: Math.round((usedMb / totalMb) * 100),
    type,
    speedMhz,
  };
}
