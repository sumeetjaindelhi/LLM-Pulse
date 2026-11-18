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

  it("reports no runtime on non-linux platforms (first-pass scope)", async () => {
    Object.defineProperty(process, "platform", { value: "win32" });
    const { detectIntelGpu } = await import("../../src/hardware/intel-gpu.js");
    const r = await detectIntelGpu();
    expect(r).toEqual({ hasOneApiRuntime: false });
  });

  it("never reads amdgpu-only VRAM sysfs files (an AMD card must not be attributed to Intel)", async () => {
    Object.defineProperty(process, "platform", { value: "linux" });
    const readFileMock = vi.fn().mockImplementation(async (path: string) => {
      if (path.endsWith("mem_info_vram_total")) return "17179869184\n";
      throw Object.assign(new Error("ENOENT"), { code: "ENOENT" });
    });
    const readdirMock = vi.fn().mockResolvedValue(["card0", "card1"]);
    vi.doMock("node:fs/promises", () => ({ readdir: readdirMock, readFile: readFileMock }));
    const { detectIntelGpu } = await import("../../src/hardware/intel-gpu.js");
    const r = await detectIntelGpu();
    expect(r).toEqual({ hasOneApiRuntime: false });
    expect(readdirMock).not.toHaveBeenCalled();
    expect(readFileMock.mock.calls.some((c) => String(c[0]).includes("/sys/class/drm"))).toBe(false);
  });

  it("detects the Level Zero loader on Linux", async () => {
    Object.defineProperty(process, "platform", { value: "linux" });
    vi.doMock("node:fs/promises", () => ({
      readFile: vi.fn().mockImplementation(async (path: string) => {
        if (path === "/opt/intel/oneapi/redist/lib/intel64/libze_loader.so.1") return Buffer.alloc(0);
        throw Object.assign(new Error("ENOENT"), { code: "ENOENT" });
      }),
    }));
    const { detectIntelGpu } = await import("../../src/hardware/intel-gpu.js");
    expect(await detectIntelGpu()).toEqual({ hasOneApiRuntime: true });
  });
});
