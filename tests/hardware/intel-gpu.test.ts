import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

describe("detectIntelGpu", () => {
  const originalPlatform = process.platform;

  beforeEach(() => {
    vi.resetModules();
  });

  afterEach(() => {
    Object.defineProperty(process, "platform", { value: originalPlatform });
    vi.restoreAllMocks();
  });

  it("returns zero on non-linux platforms (first-pass scope)", async () => {
    Object.defineProperty(process, "platform", { value: "win32" });
    const { detectIntelGpu } = await import("../../src/hardware/intel-gpu.js");
    const r = await detectIntelGpu();
    expect(r.vramMb).toBe(0);
    expect(r.hasOneApiRuntime).toBe(false);
  });

  it("reads VRAM from sysfs drm entries on Linux (Arc A770)", async () => {
    Object.defineProperty(process, "platform", { value: "linux" });
    vi.doMock("node:fs/promises", () => ({
      readdir: vi.fn().mockResolvedValue(["card0", "card1", "renderD128"]),
      readFile: vi.fn().mockImplementation(async (path: string) => {
        if (path.includes("card0") && path.endsWith("mem_info_vram_total")) {
          return "17179869184\n"; // 16 GB
        }
        if (path.includes("card1")) {
          // Integrated — no vram_total
          throw Object.assign(new Error("ENOENT"), { code: "ENOENT" });
        }
        throw Object.assign(new Error("ENOENT"), { code: "ENOENT" });
      }),
    }));
    const { detectIntelGpu } = await import("../../src/hardware/intel-gpu.js");
    const r = await detectIntelGpu();
    expect(r.vramMb).toBe(16384);
  });

  it("picks the largest card's VRAM when multiple dGPUs are present", async () => {
    Object.defineProperty(process, "platform", { value: "linux" });
    vi.doMock("node:fs/promises", () => ({
      readdir: vi.fn().mockResolvedValue(["card0", "card1"]),
      readFile: vi.fn().mockImplementation(async (path: string) => {
        if (path.includes("card0")) return "8589934592\n"; // 8 GB
        if (path.includes("card1")) return "17179869184\n"; // 16 GB
        throw Object.assign(new Error("ENOENT"), { code: "ENOENT" });
      }),
    }));
    const { detectIntelGpu } = await import("../../src/hardware/intel-gpu.js");
    const r = await detectIntelGpu();
    expect(r.vramMb).toBe(16384);
  });

  it("skips render nodes (shouldn't double-count shared VRAM)", async () => {
    Object.defineProperty(process, "platform", { value: "linux" });
    const readFileMock = vi.fn().mockImplementation(async (path: string) => {
      if (path.endsWith("mem_info_vram_total")) return "17179869184\n";
      throw Object.assign(new Error("ENOENT"), { code: "ENOENT" });
    });
    vi.doMock("node:fs/promises", () => ({
      readdir: vi.fn().mockResolvedValue(["card0", "renderD128", "renderD129"]),
      readFile: readFileMock,
    }));
    const { detectIntelGpu } = await import("../../src/hardware/intel-gpu.js");
    const r = await detectIntelGpu();
    expect(r.vramMb).toBe(16384);
    // Only card0 should have been queried for VRAM.
    const vramReads = readFileMock.mock.calls.filter((c) =>
      String(c[0]).endsWith("mem_info_vram_total"),
    );
    expect(vramReads).toHaveLength(1);
  });

  it("returns zero when readdir fails entirely", async () => {
    Object.defineProperty(process, "platform", { value: "linux" });
    vi.doMock("node:fs/promises", () => ({
      readdir: vi.fn().mockRejectedValue(Object.assign(new Error("ENOENT"), { code: "ENOENT" })),
      readFile: vi.fn().mockRejectedValue(Object.assign(new Error("ENOENT"), { code: "ENOENT" })),
    }));
    const { detectIntelGpu } = await import("../../src/hardware/intel-gpu.js");
    const r = await detectIntelGpu();
    expect(r.vramMb).toBe(0);
  });
});
