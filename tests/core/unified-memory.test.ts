import { describe, it, expect } from "vitest";
import { isUnifiedMemory } from "../../src/core/unified-memory.js";
import type { GpuInfo, HardwareProfile } from "../../src/core/types.js";
import appleM2 from "../fixtures/hardware-profiles/apple-m2.json";

const baseGpu: GpuInfo = {
  vendor: "NVIDIA",
  model: "NVIDIA GB10",
  vramMb: 100_000,
  driverVersion: "580.95.05",
  acceleratorVersion: "13.0",
  acceleratorType: "cuda",
  utilizationPercent: null,
  temperatureCelsius: null,
  vramUsedMb: null,
};

describe("isUnifiedMemory", () => {
  it("is true for the explicit flag and for Apple Metal GPUs without it", () => {
    expect(isUnifiedMemory({ ...baseGpu, unifiedMemory: true })).toBe(true);
    expect((appleM2 as HardwareProfile).primaryGpu!.unifiedMemory).toBeUndefined();
    expect(isUnifiedMemory((appleM2 as HardwareProfile).primaryGpu!)).toBe(true);
  });

  it("is false for a discrete GPU", () => {
    expect(isUnifiedMemory(baseGpu)).toBe(false);
  });
});
