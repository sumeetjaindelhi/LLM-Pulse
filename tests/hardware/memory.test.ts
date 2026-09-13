import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;

describe("detectMemory inside a cgroup", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.doMock("systeminformation", () => ({
      default: {
        mem: vi.fn().mockResolvedValue({ total: 64 * GIB, available: 60 * GIB }),
        memLayout: vi.fn().mockResolvedValue([]),
      },
    }));
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("subtracts the container's own usage from its limit", async () => {
    vi.doMock("../../src/hardware/cgroups.js", () => ({
      readCgroupMemoryLimit: vi.fn().mockResolvedValue({ limitBytes: 8 * GIB, usageBytes: 6 * GIB, source: "v2" }),
    }));
    const { detectMemory } = await import("../../src/hardware/memory.js");
    const mem = await detectMemory();

    expect(mem.totalMb).toBe(8192);
    expect(mem.availableMb).toBe(2048);
    expect(mem.usedMb).toBe(6144);
  });

  it("never reports negative available memory when usage exceeds the limit", async () => {
    vi.doMock("../../src/hardware/cgroups.js", () => ({
      readCgroupMemoryLimit: vi.fn().mockResolvedValue({ limitBytes: 8 * GIB, usageBytes: 9 * GIB, source: "v1" }),
    }));
    const { detectMemory } = await import("../../src/hardware/memory.js");
    const mem = await detectMemory();

    expect(mem.availableMb).toBe(0);
  });

  it("clamps to the limit when container usage is unreadable", async () => {
    vi.doMock("../../src/hardware/cgroups.js", () => ({
      readCgroupMemoryLimit: vi.fn().mockResolvedValue({ limitBytes: 8 * GIB, usageBytes: null, source: "v2" }),
    }));
    const { detectMemory } = await import("../../src/hardware/memory.js");
    const mem = await detectMemory();

    expect(mem.availableMb).toBe(8192);
  });
});
