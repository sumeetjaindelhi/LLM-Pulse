import { detectCpu } from "./cpu.js";
import { detectGpus } from "./gpu.js";
import { detectMemory } from "./memory.js";
import { detectDisk } from "./disk.js";
import { readAppleVramLimit } from "./apple-memory.js";
import type { HardwareProfile } from "../core/types.js";

let cachedProfile: HardwareProfile | null = null;
let cachedAt = 0;
// 60 seconds — long enough that chained MCP tool calls share one detect pass
// (saving 300-800ms per subsequent call), short enough that restarting Ollama
// or plugging in an eGPU mid-session is picked up within a minute. The live
// monitor TUI has its own 2-second polling that doesn't go through this path,
// so real-time metrics are unaffected.
const CACHE_TTL_MS = 60_000;

export async function detectHardware(): Promise<HardwareProfile> {
  const now = Date.now();
  if (cachedProfile && now - cachedAt < CACHE_TTL_MS) {
    return cachedProfile;
  }

  const [cpu, gpus, memory, disk] = await Promise.all([
    detectCpu(),
    detectGpus(),
    detectMemory(),
    detectDisk(),
  ]);

  // Apple Silicon uses unified memory: systeminformation reports vram: 0 for
  // Metal GPUs because there is no discrete VRAM pool. The *actual* cap on
  // wired GPU memory is `sysctl iogpu.wired_limit_mb` (defaults to ~67% of
  // total RAM). We resolve it once here so downstream callers can use
  // `primaryGpu.vramMb` as the final usable cap without further multipliers.
  for (const g of gpus) {
    if (g.acceleratorType === "metal" && g.vramMb === 0 && memory.totalMb > 0) {
      const limit = await readAppleVramLimit(memory.totalMb * 1024 * 1024);
      g.vramMb =
        limit.vramMb ?? Math.round(memory.totalMb * limit.factor);
    }
  }

  // Primary GPU = the one with the most VRAM
  const primaryGpu =
    gpus.length > 0
      ? gpus.reduce((best, g) => (g.vramMb > best.vramMb ? g : best))
      : null;

  cachedProfile = { cpu, gpus, memory, disk, primaryGpu };
  cachedAt = now;
  return cachedProfile;
}

export function clearHardwareCache(): void {
  cachedProfile = null;
  cachedAt = 0;
}

export { detectCpu } from "./cpu.js";
export { detectGpus } from "./gpu.js";
export { detectMemory } from "./memory.js";
export { detectDisk } from "./disk.js";
