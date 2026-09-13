import { execa } from "execa";
import { APPLE_UNIFIED_MEMORY_FACTOR_FALLBACK } from "../core/constants.js";

export interface AppleMemoryLimit {
  /** Usable GPU memory in megabytes; null only off macOS, where there is nothing to read. */
  vramMb: number | null;
  /** Provenance: "sysctl" (user-raised wired limit), "metal" (Metal's recommended working set), or "fallback" (fixed estimate). */
  source: "sysctl" | "metal" | "fallback";
  /** Fraction of total RAM that ended up usable as GPU memory (for diagnostics). */
  factor: number;
}

const BYTES_PER_MB = 1024 * 1024;

// JXA snippet printing MTLDevice.recommendedMaxWorkingSetSize in bytes, the
// budget Ollama and llama.cpp's Metal backend size models against. Prints
// "undefined" or "null" when there is no Metal device (e.g. Intel Macs).
const METAL_WORKING_SET_SCRIPT =
  'ObjC.import("Metal"); String($.MTLCreateSystemDefaultDevice().recommendedMaxWorkingSetSize)';

/** Resolve the usable unified-memory GPU budget on Apple Silicon.
 *
 *  1. `sysctl iogpu.wired_limit_mb` when > 0: the user has raised the wired
 *     limit, so that is the real cap. It reads 0 when left at the default.
 *  2. Metal's `recommendedMaxWorkingSetSize` via osascript (no Xcode needed).
 *  3. A fixed 67% of total RAM when neither can be read.
 *
 *  Off macOS, vramMb is null and factor is the fallback, so the caller can
 *  apply it to total memory itself.
 */
export async function readAppleVramLimit(totalBytes: number): Promise<AppleMemoryLimit> {
  if (process.platform !== "darwin") {
    return { vramMb: null, source: "fallback", factor: APPLE_UNIFIED_MEMORY_FACTOR_FALLBACK };
  }

  try {
    const { stdout } = await execa("sysctl", ["-n", "iogpu.wired_limit_mb"], {
      timeout: 2000,
    });
    const parsed = parseInt(stdout.trim(), 10);
    if (Number.isFinite(parsed) && parsed > 0) {
      return {
        vramMb: parsed,
        source: "sysctl",
        factor: totalBytes > 0 ? (parsed * BYTES_PER_MB) / totalBytes : 0,
      };
    }
  } catch {
    // sysctl missing, key not present (Intel Mac), or timeout — try Metal.
  }

  try {
    const { stdout } = await execa("osascript", ["-l", "JavaScript", "-e", METAL_WORKING_SET_SCRIPT], {
      timeout: 2000,
    });
    const text = stdout.trim();
    const bytes = /^\d+$/.test(text) ? Number(text) : 0;
    if (Number.isSafeInteger(bytes) && bytes > 0) {
      return {
        vramMb: Math.round(bytes / BYTES_PER_MB),
        source: "metal",
        factor: totalBytes > 0 ? bytes / totalBytes : 0,
      };
    }
  } catch {
    // osascript missing, no Metal framework, or timeout — use fallback.
  }

  const totalMb = Math.round(totalBytes / BYTES_PER_MB);
  return {
    vramMb: Math.round(totalMb * APPLE_UNIFIED_MEMORY_FACTOR_FALLBACK),
    source: "fallback",
    factor: APPLE_UNIFIED_MEMORY_FACTOR_FALLBACK,
  };
}
