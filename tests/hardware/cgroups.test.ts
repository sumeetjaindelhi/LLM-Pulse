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

  it("reads the container's current usage from memory.current (v2)", async () => {
    Object.defineProperty(process, "platform", { value: "linux" });
    vi.doMock("node:fs/promises", () => ({
      readFile: vi.fn().mockImplementation(async (path: string) => {
        if (path.endsWith("memory.max")) return "8589934592\n";
        if (path.endsWith("memory.current")) return "6442450944\n"; // 6 GiB
        throw Object.assign(new Error("ENOENT"), { code: "ENOENT" });
      }),
    }));
    const { readCgroupMemoryLimit } = await import("../../src/hardware/cgroups.js");
    const result = await readCgroupMemoryLimit();
    expect(result.usageBytes).toBe(6442450944);
  });

  it("reads the container's current usage from memory.usage_in_bytes (v1)", async () => {
    Object.defineProperty(process, "platform", { value: "linux" });
    vi.doMock("node:fs/promises", () => ({
      readFile: vi.fn().mockImplementation(async (path: string) => {
        if (path.endsWith("memory.limit_in_bytes")) return "4294967296\n";
        if (path.endsWith("memory.usage_in_bytes")) return "1073741824\n"; // 1 GiB
        throw Object.assign(new Error("ENOENT"), { code: "ENOENT" });
      }),
    }));
    const { readCgroupMemoryLimit } = await import("../../src/hardware/cgroups.js");
    const result = await readCgroupMemoryLimit();
    expect(result.usageBytes).toBe(1073741824);
  });

  it("reports null usage when the usage file is unreadable", async () => {
    Object.defineProperty(process, "platform", { value: "linux" });
    vi.doMock("node:fs/promises", () => ({
      readFile: vi.fn().mockImplementation(async (path: string) => {
        if (path.endsWith("memory.max")) return "8589934592\n";
        throw Object.assign(new Error("ENOENT"), { code: "ENOENT" });
      }),
    }));
    const { readCgroupMemoryLimit } = await import("../../src/hardware/cgroups.js");
    const result = await readCgroupMemoryLimit();
    expect(result.usageBytes).toBeNull();
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

  it("excludes reclaimable page cache (inactive_file) from v2 usage", async () => {
    Object.defineProperty(process, "platform", { value: "linux" });
    vi.doMock("node:fs/promises", () => ({
      readFile: vi.fn().mockImplementation(async (path: string) => {
        if (path.endsWith("memory.max")) return "8589934592\n";
        if (path.endsWith("memory.current")) return "6442450944\n"; // 6 GiB
        if (path.endsWith("memory.stat")) return "anon 1073741824\nactive_file 1\ninactive_file 2147483648\n"; // 2 GiB
        throw Object.assign(new Error("ENOENT"), { code: "ENOENT" });
      }),
    }));
    const { readCgroupMemoryLimit } = await import("../../src/hardware/cgroups.js");
    const result = await readCgroupMemoryLimit();
    expect(result.usageBytes).toBe(4294967296);
  });

  it("excludes hierarchical total_inactive_file from v1 usage", async () => {
    Object.defineProperty(process, "platform", { value: "linux" });
    vi.doMock("node:fs/promises", () => ({
      readFile: vi.fn().mockImplementation(async (path: string) => {
        if (path.endsWith("memory.limit_in_bytes")) return "4294967296\n";
        if (path.endsWith("memory.usage_in_bytes")) return "2147483648\n"; // 2 GiB
        if (path.endsWith("memory.stat")) return "inactive_file 5\ntotal_inactive_file 1073741824\n"; // 1 GiB
        throw Object.assign(new Error("ENOENT"), { code: "ENOENT" });
      }),
    }));
    const { readCgroupMemoryLimit } = await import("../../src/hardware/cgroups.js");
    const result = await readCgroupMemoryLimit();
    expect(result.usageBytes).toBe(1073741824);
  });
});
