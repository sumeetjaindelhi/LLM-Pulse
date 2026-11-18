import { describe, it, expect } from "vitest";
import { estimateMaxParams } from "../../src/cli/commands/scan.js";
import type { HardwareProfile } from "../../src/core/types.js";
import cpuOnly from "../fixtures/hardware-profiles/cpu-only.json";

const zeroVram = () => ({
  vendor: "Intel",
  model: "UHD Graphics",
  vramMb: 0,
  driverVersion: "",
  acceleratorVersion: null,
  acceleratorType: null,
  utilizationPercent: null,
  temperatureCelsius: null,
  vramUsedMb: null,
});

describe("estimateMaxParams", () => {
  it("falls back to RAM when the GPU reports no VRAM instead of computing from 0", () => {
    const hw = cpuOnly as HardwareProfile;
    const zeroVramGpu = zeroVram();
    const withGpu = { ...hw, gpus: [zeroVramGpu], primaryGpu: zeroVramGpu };
    expect(estimateMaxParams(withGpu)).toBe(estimateMaxParams({ ...hw, primaryGpu: null }));
    expect(estimateMaxParams(withGpu)).toBeGreaterThan(1);
  });

  it("returns the largest size whose Q4_K_M weights, bucket KV and compute fit at a 4K context", () => {
    // RTX 4090 (24564 MiB): budget 24564 - 457 - 1024 = 23083; context 4096, compute 262144*512*4*1.1/2^20 = 563.2,
    // weights 583.43 MiB per billion (4.8944 bpw), KV bucket 360 MiB per 1k tokens at 22-35B = 1474.6 MiB.
    // 35B: 20420 + 1475 + 563 = 22458 <= 23083. 36B: 21004 + 1475 + 563 = 23042 would fit but the bucket is 360 only up to 35B;
    // above it 440 MiB per 1k tokens (1802): 36B = 21004 + 1802 + 563 = 23369 > 23083.
    const hw = cpuOnly as HardwareProfile;
    const gpu = { ...zeroVram(), vendor: "NVIDIA", model: "RTX 4090", vramMb: 24564, acceleratorType: "cuda" as const };
    expect(estimateMaxParams({ ...hw, gpus: [gpu], primaryGpu: gpu })).toBe(35);
  });

  it("never shrinks as memory grows (16 <= 24 <= 48 GB, Apple 24 <= 64 GB)", () => {
    const hw = cpuOnly as HardwareProfile;
    const sized = (vramMb: number, apple = false) => {
      const gpu = { ...zeroVram(), vendor: apple ? "Apple" : "NVIDIA", vramMb, acceleratorType: apple ? ("metal" as const) : ("cuda" as const) };
      return estimateMaxParams({ ...hw, gpus: [gpu], primaryGpu: gpu });
    };
    const discrete = [16376, 24564, 49140].map((mb) => sized(mb));
    expect(discrete).toEqual([...discrete].sort((a, b) => a - b));
    expect(sized(18186, true)).toBeLessThanOrEqual(sized(49152, true));
  });
});
