import si from "systeminformation";
import { readCgroupMemoryLimit } from "./cgroups.js";
import type { MemoryInfo } from "../core/types.js";

export async function detectMemory(): Promise<MemoryInfo> {
  const [mem, layout, cgroup] = await Promise.all([
    si.mem(),
    si.memLayout(),
    readCgroupMemoryLimit(),
  ]);

  // Host view — what systeminformation reports from the kernel
  let totalBytes = mem.total;
  let availableBytes = mem.available;

  // Container awareness: if we're inside a cgroup with a memory limit,
  // report that limit instead of the host's total RAM. Without this, a
  // container with `--memory=8g` on a 64 GB host would have the scorer
  // recommend 70B models that never fit, because si.mem() returns the
  // HOST memory, not the container's slice.
  if (cgroup.limitBytes !== null && cgroup.limitBytes < totalBytes) {
    totalBytes = cgroup.limitBytes;
    // The kernel's "available" number includes reclaimable pagecache etc.
    // from outside the container view; clamp it to the container ceiling.
    if (availableBytes > totalBytes) availableBytes = totalBytes;
  }

  const totalMb = Math.round(totalBytes / (1024 * 1024));
  const availableMb = Math.round(availableBytes / (1024 * 1024));
  // mem.used = total - free, which counts reclaimable buffcache as "used" and
  // always approaches 100% on macOS/Linux. Use (total - available) so that
  // reclaimable memory is excluded — this is the metric every Mac/Linux-aware
  // tool reports as "memory pressure".
  const usedMb = Math.max(0, totalMb - availableMb);

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
