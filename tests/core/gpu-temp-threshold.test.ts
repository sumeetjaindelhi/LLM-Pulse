import { describe, it, expect } from "vitest";
import { pickGpuTempThreshold, GPU_TEMP_THRESHOLDS_BY_VENDOR } from "../../src/core/constants.js";

describe("pickGpuTempThreshold", () => {
  it("returns Apple's lower threshold for Apple Silicon", () => {
    expect(pickGpuTempThreshold("Apple", false)).toBe(GPU_TEMP_THRESHOLDS_BY_VENDOR.apple);
  });

  it("distinguishes NVIDIA desktop vs. mobile", () => {
    expect(pickGpuTempThreshold("NVIDIA", false)).toBe(
      GPU_TEMP_THRESHOLDS_BY_VENDOR.nvidia_desktop,
    );
    expect(pickGpuTempThreshold("NVIDIA", true)).toBe(
      GPU_TEMP_THRESHOLDS_BY_VENDOR.nvidia_mobile,
    );
  });

  it("treats AMD as desktop-class by default", () => {
    expect(pickGpuTempThreshold("AMD", false)).toBe(GPU_TEMP_THRESHOLDS_BY_VENDOR.amd);
    expect(pickGpuTempThreshold("Advanced Micro Devices", false)).toBe(
      GPU_TEMP_THRESHOLDS_BY_VENDOR.amd,
    );
  });

  it("treats Intel Arc / dGPU as desktop-class", () => {
    expect(pickGpuTempThreshold("Intel", false)).toBe(GPU_TEMP_THRESHOLDS_BY_VENDOR.intel);
  });

  it("falls back to the generic threshold for unknown vendors", () => {
    expect(pickGpuTempThreshold("Matrox", false)).toBe(GPU_TEMP_THRESHOLDS_BY_VENDOR.unknown);
    expect(pickGpuTempThreshold("", false)).toBe(GPU_TEMP_THRESHOLDS_BY_VENDOR.unknown);
  });

  it("Apple threshold is strictly lower than NVIDIA desktop (real-world calibration)", () => {
    expect(GPU_TEMP_THRESHOLDS_BY_VENDOR.apple).toBeLessThan(
      GPU_TEMP_THRESHOLDS_BY_VENDOR.nvidia_desktop,
    );
  });
});
