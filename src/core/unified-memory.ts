import type { GpuInfo } from "./types.js";

/** True when the GPU draws from system RAM instead of a dedicated VRAM pool.
 *  Apple Metal GPUs always do; `acceleratorType === "metal"` covers profiles
 *  built without the explicit flag (older fixtures, API callers). */
export function isUnifiedMemory(gpu: GpuInfo): boolean {
  return gpu.unifiedMemory === true || gpu.acceleratorType === "metal";
}
