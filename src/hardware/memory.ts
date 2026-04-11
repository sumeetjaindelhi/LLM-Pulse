import si from "systeminformation";
import type { MemoryInfo } from "../core/types.js";

export async function detectMemory(): Promise<MemoryInfo> {
  const [mem, layout] = await Promise.all([si.mem(), si.memLayout()]);

  const totalMb = Math.round(mem.total / (1024 * 1024));
  const availableMb = Math.round(mem.available / (1024 * 1024));
  // mem.used = total - free, which counts reclaimable buffcache as "used" and
  // always approaches 100% on macOS/Linux. Use (total - available) so that
  // reclaimable memory is excluded — this is the metric every Mac/Linux-aware
  // tool reports as "memory pressure".
  const usedMb = Math.max(0, totalMb - availableMb);

  // Get memory type and speed from first module
  const firstModule = layout[0];
  const type = firstModule?.type || "Unknown";
  const speedMhz = firstModule?.clockSpeed ?? null;

  return {
    totalMb,
    availableMb,
    usedMb,
    usedPercent: totalMb > 0 ? Math.round((usedMb / totalMb) * 100) : 0,
    type,
    speedMhz,
  };
}
