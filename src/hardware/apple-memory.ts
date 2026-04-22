import { execa } from "execa";
import { APPLE_UNIFIED_MEMORY_FACTOR_FALLBACK } from "../core/constants.js";

export interface AppleMemoryLimit {
  /** VRAM cap in megabytes, or null if we can't determine one. */
  vramMb: number | null;
  /** Human-readable provenance — "sysctl" if read live, "fallback" if estimated. */
  source: "sysctl" | "fallback";
  /** Fraction of total RAM that ended up usable as GPU memory (for diagnostics). */
  factor: number;
}

/** Resolve the real unified-memory cap on Apple Silicon.
 *
 *  macOS limits "wired" GPU memory (pages that can't be evicted during
 *  inference) to ~67% of total RAM. The exact value is `iogpu.wired_limit_mb`,
 *  a sysctl readable from user-space. Users can raise it (e.g. for ML work)
 *  so a hardcoded 0.75 multiplier is both wrong AND blind to what the user
 *  has configured. We prefer to read the real number; fall back to 0.67.
 *
 *  Returns null vramMb when the reported limit is 0 (some macOS versions
 *  report 0 to mean "use the default 67%"), so the caller can apply the
 *  fallback factor to total memory themselves.
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
        factor: totalBytes > 0 ? (parsed * 1024 * 1024) / totalBytes : 0,
      };
    }
  } catch {
    // sysctl missing, key not present (Intel Mac), or timeout — use fallback.
  }

  const totalMb = Math.round(totalBytes / (1024 * 1024));
  return {
    vramMb: Math.round(totalMb * APPLE_UNIFIED_MEMORY_FACTOR_FALLBACK),
    source: "fallback",
    factor: APPLE_UNIFIED_MEMORY_FACTOR_FALLBACK,
  };
}
