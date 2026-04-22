import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

describe("readCgroupMemoryLimit", () => {
  const originalPlatform = process.platform;

  beforeEach(() => {
    vi.resetModules();
  });

  afterEach(() => {
    Object.defineProperty(process, "platform", { value: originalPlatform });
    vi.restoreAllMocks();
  });

  it("returns source 'none' on non-linux platforms", async () => {
    Object.defineProperty(process, "platform", { value: "darwin" });
    const { readCgroupMemoryLimit } = await import("../../src/hardware/cgroups.js");
    const result = await readCgroupMemoryLimit();
    expect(result.source).toBe("none");
    expect(result.limitBytes).toBeNull();
  });

  it("parses cgroups v2 'max' as null (no limit)", async () => {
    Object.defineProperty(process, "platform", { value: "linux" });
    vi.doMock("node:fs/promises", () => ({
      readFile: vi.fn().mockImplementation(async (path: string) => {
        if (path.endsWith("memory.max")) return "max\n";
        throw Object.assign(new Error("ENOENT"), { code: "ENOENT" });
      }),
    }));
    const { readCgroupMemoryLimit } = await import("../../src/hardware/cgroups.js");
    const result = await readCgroupMemoryLimit();
    expect(result.source).toBe("v2");
    expect(result.limitBytes).toBeNull();
  });

  it("parses a concrete cgroups v2 limit", async () => {
    Object.defineProperty(process, "platform", { value: "linux" });
    vi.doMock("node:fs/promises", () => ({
      readFile: vi.fn().mockImplementation(async (path: string) => {
        if (path.endsWith("memory.max")) return "8589934592\n"; // 8 GiB
        throw Object.assign(new Error("ENOENT"), { code: "ENOENT" });
      }),
    }));
    const { readCgroupMemoryLimit } = await import("../../src/hardware/cgroups.js");
    const result = await readCgroupMemoryLimit();
    expect(result.source).toBe("v2");
    expect(result.limitBytes).toBe(8589934592);
  });

  it("falls back to cgroups v1 when v2 is missing", async () => {
    Object.defineProperty(process, "platform", { value: "linux" });
    vi.doMock("node:fs/promises", () => ({
      readFile: vi.fn().mockImplementation(async (path: string) => {
        if (path.endsWith("memory.limit_in_bytes")) return "4294967296\n"; // 4 GiB
        throw Object.assign(new Error("ENOENT"), { code: "ENOENT" });
      }),
    }));
    const { readCgroupMemoryLimit } = await import("../../src/hardware/cgroups.js");
    const result = await readCgroupMemoryLimit();
    expect(result.source).toBe("v1");
    expect(result.limitBytes).toBe(4294967296);
  });

  it("treats the cgroups v1 kernel sentinel as unlimited", async () => {
    Object.defineProperty(process, "platform", { value: "linux" });
    vi.doMock("node:fs/promises", () => ({
      readFile: vi.fn().mockImplementation(async (path: string) => {
        if (path.endsWith("memory.limit_in_bytes")) {
          return "9223372036854771712\n"; // 2^63 rounded to page — "no limit"
        }
        throw Object.assign(new Error("ENOENT"), { code: "ENOENT" });
      }),
    }));
    const { readCgroupMemoryLimit } = await import("../../src/hardware/cgroups.js");
    const result = await readCgroupMemoryLimit();
    expect(result.source).toBe("v1");
    expect(result.limitBytes).toBeNull();
  });
});
