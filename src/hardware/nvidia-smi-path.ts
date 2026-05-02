import { execa } from "execa";
import { access } from "node:fs/promises";
import { constants } from "node:fs";
import { join } from "node:path";

let cached: string | null = null;
let cacheValid = false;

// NVIDIA's installer historically dropped nvidia-smi.exe into
// `%ProgramFiles%\NVIDIA Corporation\NVSMI\`. Modern drivers (since ~2021)
// moved it to `%SYSTEMROOT%\System32\`. A fresh install with PATH not yet
// updated — or a custom install — leaves the binary invisible to execa's
// PATH lookup. The actual probed paths are inlined in `resolveWindowsPath`
// below to avoid a constant that nothing else reads.

async function existsAt(path: string): Promise<boolean> {
  try {
    await access(path, constants.X_OK);
    return true;
  } catch {
    return false;
  }
}

async function resolveWindowsPath(): Promise<string | null> {
  const programFiles = process.env.ProgramFiles || "C:\\Program Files";
  const systemRoot = process.env.SystemRoot || "C:\\Windows";
  const candidates = [
    join(programFiles, "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe"),
    join(systemRoot, "System32", "nvidia-smi.exe"),
    // 32-bit install on 64-bit Windows
    join(process.env["ProgramFiles(x86)"] || "C:\\Program Files (x86)",
         "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe"),
  ];
  for (const c of candidates) {
    if (await existsAt(c)) return c;
  }
  return null;
}

async function resolveWithWhich(): Promise<string | null> {
  const cmd = process.platform === "win32" ? "where" : "which";
  try {
    const { stdout } = await execa(cmd, ["nvidia-smi"], { timeout: 2000 });
    const first = stdout.trim().split(/\r?\n/)[0];
    return first || null;
  } catch {
    return null;
  }
}

/** Return the best path to invoke nvidia-smi by. `null` means no resolver
 *  found one and the caller should skip NVIDIA detection. The result is
 *  cached per-process — nvidia-smi doesn't move while we're running.
 */
export async function resolveNvidiaSmi(): Promise<string | null> {
  if (cacheValid) return cached;

  const byWhich = await resolveWithWhich();
  if (byWhich && (await existsAt(byWhich))) {
    cached = byWhich;
    cacheValid = true;
    return cached;
  }

  if (process.platform === "win32") {
    const win = await resolveWindowsPath();
    if (win) {
      cached = win;
      cacheValid = true;
      return cached;
    }
  }

  // Fall through — "nvidia-smi" on PATH (existing behaviour). If it doesn't
  // exist, execa will throw ENOENT and detectGpus treats that as "no NVIDIA".
  cached = "nvidia-smi";
  cacheValid = true;
  return cached;
}

// Exposed for tests — lets a test force a re-resolution after mutating PATH
// or the Program Files env var.
export function _resetNvidiaSmiPathCache(): void {
  cached = null;
  cacheValid = false;
}
