import { describe, it, expect, vi, beforeEach } from "vitest";
import { detectHardware, clearHardwareCache } from "../../src/hardware/index.js";

// Mock all hardware detection modules
vi.mock("../../src/hardware/cpu.js", () => ({
  detectCpu: vi.fn().mockResolvedValue({
    brand: "Mock CPU",
    manufacturer: "Mock",
    cores: 4,
    threads: 8,
    speed: 3.0,
    speedMax: 4.0,
    architecture: "x64",
    flags: ["avx2"],
    hasAvx2: true,
  }),
}));

vi.mock("../../src/hardware/gpu.js", () => ({
  detectGpus: vi.fn().mockResolvedValue([]),
  parseRocmCsv: vi.fn(),
}));

vi.mock("../../src/hardware/memory.js", () => ({
  detectMemory: vi.fn().mockResolvedValue({
    totalMb: 16384,
    availableMb: 10000,
    usedMb: 6384,
    usedPercent: 39,
    type: "DDR4",
    speedMhz: 3200,
  }),
}));

vi.mock("../../src/hardware/disk.js", () => ({
  detectDisk: vi.fn().mockResolvedValue({
    type: "SSD",
    freeGb: 100,
    totalGb: 500,
  }),
}));

describe("detectHardware caching", () => {
  beforeEach(() => {
    clearHardwareCache();
  });

  it("returns cached result on second call", async () => {
    const first = await detectHardware();
    const second = await detectHardware();
    expect(first).toBe(second); // Same reference — cached
  });

  it("returns fresh result after clearHardwareCache()", async () => {
    const first = await detectHardware();
    clearHardwareCache();
    const second = await detectHardware();
    // Same shape, but different object reference
    expect(first).not.toBe(second);
    expect(first.cpu.brand).toBe(second.cpu.brand);
  });
});
