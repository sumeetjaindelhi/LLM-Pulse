import { readFile } from "node:fs/promises";

const CGROUPS_V2_MEMORY_MAX = "/sys/fs/cgroup/memory.max";
const CGROUPS_V1_MEMORY_LIMIT = "/sys/fs/cgroup/memory/memory.limit_in_bytes";

// cgroups v1 reports "no limit" as a kernel sentinel near LONG_MAX — observed
// values are e.g. 9223372036854771712 (LONG_MAX rounded down to the page
// boundary). Anything above 1 PiB (2^50 bytes) is effectively unlimited for
// memory purposes, so that's a safe "this isn't a real limit" threshold.
const UNLIMITED_THRESHOLD_BYTES = 1n << 50n;

export interface CgroupMemory {
  /** Container memory limit in bytes, or null if unlimited / not in a container. */
  limitBytes: number | null;
  /** Which cgroup version we read from (for diagnostics). */
  source: "v2" | "v1" | "none";
}

async function tryRead(path: string): Promise<string | null> {
  try {
    return await readFile(path, "utf-8");
  } catch {
    return null;
  }
}

function parseLimit(raw: string): number | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  if (trimmed === "max") return null; // cgroups v2 string literal
  let n: bigint;
  try {
    n = BigInt(trimmed);
  } catch {
    return null;
  }
  if (n <= 0n) return null;
  if (n >= UNLIMITED_THRESHOLD_BYTES) return null;
  // Safe to Number-cast: any valid container limit fits in MAX_SAFE_INTEGER
  // (MAX_SAFE_INTEGER = 9 PiB, containers are capped far below that in practice).
  return Number(n);
}

/** Read the memory limit imposed by the surrounding cgroup, if any.
 *
 *  Prefers cgroups v2 (`/sys/fs/cgroup/memory.max`) which is what modern
 *  Docker, containerd, and Kubernetes use. Falls back to v1 for older hosts.
 *  Returns null when running outside a container (limit == "max" or the
 *  kernel sentinel value) so callers can keep using the host RAM reading.
 */
export async function readCgroupMemoryLimit(): Promise<CgroupMemory> {
  if (process.platform !== "linux") {
    return { limitBytes: null, source: "none" };
  }

  const v2 = await tryRead(CGROUPS_V2_MEMORY_MAX);
  if (v2 !== null) {
    const limit = parseLimit(v2);
    return { limitBytes: limit, source: "v2" };
  }

  const v1 = await tryRead(CGROUPS_V1_MEMORY_LIMIT);
  if (v1 !== null) {
    const limit = parseLimit(v1);
    return { limitBytes: limit, source: "v1" };
  }

  return { limitBytes: null, source: "none" };
}
