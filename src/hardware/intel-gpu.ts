import { readFile } from "node:fs/promises";

export interface IntelGpuInfo {
  hasOneApiRuntime: boolean;
}

// oneAPI / Level Zero runtime presence hint. Intel exposes no VRAM size in
// sysfs (the amdgpu-only mem_info_vram_total does not exist for i915/xe), so
// the runtime is probed for presence only.
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

/** Probe oneAPI runtime presence. Linux only for now; other platforms report none. */
export async function detectIntelGpu(): Promise<IntelGpuInfo> {
  return { hasOneApiRuntime: await detectOneApiRuntime() };
}
