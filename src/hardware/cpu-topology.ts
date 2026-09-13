import { execa } from "execa";
import { readFile } from "node:fs/promises";

export interface CpuTopology {
  performanceCores: number | null;
  efficiencyCores: number | null;
}

// Parse a Linux cpus list like "0-7" or "0,2,4-6" into the listed CPU indices.
function parseCpuList(list: string): number[] {
  const cpus: number[] = [];
  for (const part of list.trim().split(",")) {
    const trimmed = part.trim();
    if (!trimmed) continue;
    if (trimmed.includes("-")) {
      const [a, b] = trimmed.split("-").map((n) => parseInt(n, 10));
      if (Number.isFinite(a) && Number.isFinite(b) && b >= a) {
        for (let cpu = a; cpu <= b; cpu++) cpus.push(cpu);
      }
    } else {
      const cpu = parseInt(trimmed, 10);
      if (Number.isFinite(cpu)) cpus.push(cpu);
    }
  }
  return cpus;
}

// Count physical cores behind a set of logical CPUs. SMT siblings share one
// core_cpus_list (thread_siblings_list on kernels older than 5.x), so the
// number of distinct lists is the core count whether or not SMT is present.
// Returns null when any CPU's topology is unreadable.
async function countPhysicalCores(cpus: number[]): Promise<number | null> {
  const siblingLists = await Promise.all(
    cpus.map((cpu) => {
      const topology = `/sys/devices/system/cpu/cpu${cpu}/topology`;
      return readFile(`${topology}/core_cpus_list`, "utf-8")
        .catch(() => readFile(`${topology}/thread_siblings_list`, "utf-8"))
        .catch(() => null);
    }),
  );
  if (siblingLists.includes(null)) return null;
  return new Set(siblingLists.map((list) => list?.trim())).size;
}

async function readDarwinTopology(): Promise<CpuTopology> {
  try {
    const { stdout: p } = await execa("sysctl", ["-n", "hw.perflevel0.physicalcpu"], { timeout: 2000 });
    const { stdout: e } = await execa("sysctl", ["-n", "hw.perflevel1.physicalcpu"], { timeout: 2000 });
    const pCores = parseInt(p.trim(), 10);
    const eCores = parseInt(e.trim(), 10);
    return {
      performanceCores: Number.isFinite(pCores) && pCores > 0 ? pCores : null,
      efficiencyCores: Number.isFinite(eCores) && eCores >= 0 ? eCores : null,
    };
  } catch {
    return { performanceCores: null, efficiencyCores: null };
  }
}

async function readLinuxTopology(): Promise<CpuTopology> {
  // Alder Lake+ and newer hybrid Intel chips expose two "CPU type" devices
  // under sysfs: `cpu_core` (P-cores) and `cpu_atom` (E-cores). The `cpus`
  // file in each lists the logical CPU indices. On systems without hybrid
  // cores, these paths don't exist and we report null.
  try {
    const [pRaw, eRaw] = await Promise.all([
      readFile("/sys/devices/cpu_core/cpus", "utf-8").catch(() => null),
      readFile("/sys/devices/cpu_atom/cpus", "utf-8").catch(() => null),
    ]);
    if (pRaw === null && eRaw === null) {
      return { performanceCores: null, efficiencyCores: null };
    }
    // The `cpus` listing counts logical CPUs. P-cores may or may not have SMT
    // (Core Ultra 200S drops it), so resolve physical P-cores from per-CPU
    // topology. E-cores have no SMT, so their logical count is the core count.
    const pCores = pRaw !== null ? await countPhysicalCores(parseCpuList(pRaw)) : null;
    const eCores = eRaw !== null ? parseCpuList(eRaw).length : 0;
    return {
      performanceCores: pCores !== null && pCores > 0 ? pCores : null,
      efficiencyCores: eCores > 0 ? eCores : null,
    };
  } catch {
    return { performanceCores: null, efficiencyCores: null };
  }
}

/** Detect performance vs. efficiency core counts on hybrid CPUs.
 *  Apple Silicon (P+E layout since M1) and Intel Alder Lake+ (P+E since 2021)
 *  are the two supported platforms. Returns nulls on symmetric-core CPUs and
 *  on Windows (detection there needs WMI/PowerShell — out of scope for this
 *  pass; callers should treat null as "E-core weighting unknown, use full count").
 */
export async function detectCpuTopology(): Promise<CpuTopology> {
  if (process.platform === "darwin" && process.arch === "arm64") {
    return readDarwinTopology();
  }
  if (process.platform === "linux") {
    return readLinuxTopology();
  }
  return { performanceCores: null, efficiencyCores: null };
}
