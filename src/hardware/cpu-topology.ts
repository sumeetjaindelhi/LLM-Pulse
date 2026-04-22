import { execa } from "execa";
import { readFile } from "node:fs/promises";

export interface CpuTopology {
  performanceCores: number | null;
  efficiencyCores: number | null;
}

// Parse a Linux cpus list like "0-7" or "0,2,4-6" into the count of listed CPUs.
function countCpusInList(list: string): number {
  let total = 0;
  for (const part of list.trim().split(",")) {
    const trimmed = part.trim();
    if (!trimmed) continue;
    if (trimmed.includes("-")) {
      const [a, b] = trimmed.split("-").map((n) => parseInt(n, 10));
      if (Number.isFinite(a) && Number.isFinite(b) && b >= a) total += b - a + 1;
    } else {
      if (Number.isFinite(parseInt(trimmed, 10))) total += 1;
    }
  }
  return total;
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
    // The `cpus` listing counts logical CPUs. Each hybrid P-core has SMT
    // (2 threads), E-cores don't. We report physical cores, so halve the
    // P-core count when both topology entries exist (meaning SMT applies).
    const pLogical = pRaw !== null ? countCpusInList(pRaw) : 0;
    const eLogical = eRaw !== null ? countCpusInList(eRaw) : 0;
    return {
      performanceCores: pLogical > 0 ? Math.max(1, Math.floor(pLogical / 2)) : null,
      efficiencyCores: eLogical > 0 ? eLogical : null,
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
