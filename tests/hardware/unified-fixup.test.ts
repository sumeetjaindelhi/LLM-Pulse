import { describe, it, expect, vi, beforeEach } from "vitest";
import type { GpuInfo } from "../../src/core/types.js";

const detectGpus = vi.fn();

vi.mock("../../src/hardware/cpu.js", () => ({ detectCpu: vi.fn().mockResolvedValue({}) }));
vi.mock("../../src/hardware/gpu.js", () => ({ detectGpus }));
vi.mock("../../src/hardware/disk.js", () => ({ detectDisk: vi.fn().mockResolvedValue({}) }));
vi.mock("../../src/hardware/memory.js", () => ({
  detectMemory: vi.fn().mockResolvedValue({ totalMb: 124_000, availableMb: 100_000 }),
}));

const gb10: GpuInfo = {
  vendor: "NVIDIA",
  model: "NVIDIA GB10",
  vramMb: 0,
  driverVersion: "580.95.05",
  acceleratorVersion: "13.0",
  acceleratorType: "cuda",
  unifiedMemory: true,
  utilizationPercent: 12,
  temperatureCelsius: 45,
  vramUsedMb: null,
};

describe("detectHardware unified-memory fixup", () => {
  beforeEach(async () => {
    const { clearHardwareCache } = await import("../../src/hardware/index.js");
    clearHardwareCache();
  });

  it("sizes a non-Apple unified GPU with no VRAM figure to system memory", async () => {
    detectGpus.mockResolvedValue([{ ...gb10 }]);
    const { detectHardware } = await import("../../src/hardware/index.js");
    const hw = await detectHardware();
    expect(hw.primaryGpu?.vramMb).toBe(124_000);
  });

  it("keeps a unified GPU that already has a pool size", async () => {
    detectGpus.mockResolvedValue([{ ...gb10, vramMb: 65_536 }]);
    const { detectHardware } = await import("../../src/hardware/index.js");
    const hw = await detectHardware();
    expect(hw.primaryGpu?.vramMb).toBe(65_536);
  });

  it("prefers a dedicated GPU over a larger unified one as primary", async () => {
    const rtx: GpuInfo = { ...gb10, model: "RTX 4070", vramMb: 12_282, unifiedMemory: undefined };
    detectGpus.mockResolvedValue([{ ...gb10, vramMb: 32_768 }, rtx]);
    const { detectHardware } = await import("../../src/hardware/index.js");
    const hw = await detectHardware();
    expect(hw.primaryGpu?.model).toBe("RTX 4070");
  });
});
