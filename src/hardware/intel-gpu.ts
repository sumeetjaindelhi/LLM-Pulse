import { readFile, readdir } from "node:fs/promises";
import { join } from "node:path";

export interface IntelGpuInfo {
  vramMb: number;
  hasOneApiRuntime: boolean;
}

const DRM_ROOT = "/sys/class/drm";

// Intel's discrete GPUs (Arc A-series, Battlemage) are exposed through the
// `xe` or `i915` kernel driver. On Linux, per-GPU VRAM sits at
// `/sys/class/drm/card<N>/device/mem_info_vram_total` in bytes. systeminformation
// doesn't read this, so Arc users ship with vramMb = 0 and the scorer falls
// back to CPU-only. We probe the sysfs path directly.
async function readIntelArcVramLinux(): Promise<number> {
  try {
    const entries = await readdir(DRM_ROOT);
    let bestMb = 0;
    for (const entry of entries) {
      // Only look at real card nodes ("card0", "card1"…) not the render nodes
      // ("renderD128") which share the same VRAM pool and would double-count.
      if (!/^card\d+$/.test(entry)) continue;
      const path = join(DRM_ROOT, entry, "device", "mem_info_vram_total");
      try {
        const raw = (await readFile(path, "utf-8")).trim();
        const bytes = Number(raw);
        if (Number.isFinite(bytes) && bytes > 0) {
          const mb = Math.round(bytes / (1024 * 1024));
          if (mb > bestMb) bestMb = mb;
        }
      } catch {
        // no VRAM info on this card (integrated GPU or non-Intel dGPU — skip)
      }
    }
    return bestMb;
  } catch {
    return 0;
  }
}

// oneAPI / Level Zero runtime presence hint. We don't try to interrogate it
// for VRAM — on Linux the sysfs path is authoritative; on Windows/macOS the
// runtime presence is just a signal that GPU compute is wired up.
async function detectOneApiRuntime(): Promise<boolean> {
  if (process.platform === "linux") {
    const candidates = [
      "/usr/lib/x86_64-linux-gnu/libze_loader.so.1",
      "/opt/intel/oneapi/redist/lib/intel64/libze_loader.so.1",
    ];
    for (const path of candidates) {
      try {
        await readFile(path);
        return true;
      } catch {
        // not present at this path
      }
    }
    return false;
  }
  return false;
}

/** Probe Intel Arc / discrete-GPU VRAM and oneAPI presence. Returns zeros
 *  on non-Linux platforms for now — Windows WMI probing is doable but needs
 *  PowerShell fallback work that's out of scope for the first pass.
 */
export async function detectIntelGpu(): Promise<IntelGpuInfo> {
  if (process.platform !== "linux") {
    return { vramMb: 0, hasOneApiRuntime: false };
  }
  const [vramMb, hasOneApiRuntime] = await Promise.all([
    readIntelArcVramLinux(),
    detectOneApiRuntime(),
  ]);
  return { vramMb, hasOneApiRuntime };
}
