import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

describe("readAppleVramLimit", () => {
  const originalPlatform = process.platform;

  beforeEach(() => {
    vi.resetModules();
  });

  afterEach(() => {
    Object.defineProperty(process, "platform", { value: originalPlatform });
    vi.restoreAllMocks();
  });

  it("returns fallback factor on non-darwin platforms", async () => {
    Object.defineProperty(process, "platform", { value: "linux" });
    const { readAppleVramLimit } = await import("../../src/hardware/apple-memory.js");
    const r = await readAppleVramLimit(16 * 1024 * 1024 * 1024);
    expect(r.source).toBe("fallback");
    expect(r.factor).toBeCloseTo(0.67, 2);
    expect(r.vramMb).toBeNull();
  });

  it("reads sysctl iogpu.wired_limit_mb when available", async () => {
    Object.defineProperty(process, "platform", { value: "darwin" });
    vi.doMock("execa", () => ({
      execa: vi.fn().mockResolvedValue({ stdout: "24576\n" }),
    }));
    const { readAppleVramLimit } = await import("../../src/hardware/apple-memory.js");
    const r = await readAppleVramLimit(32 * 1024 * 1024 * 1024);
    expect(r.source).toBe("sysctl");
    expect(r.vramMb).toBe(24576);
    expect(r.factor).toBeCloseTo(24576 / (32 * 1024), 2);
  });

  it("falls back to 0.67 when sysctl fails", async () => {
    Object.defineProperty(process, "platform", { value: "darwin" });
    vi.doMock("execa", () => ({
      execa: vi.fn().mockRejectedValue(Object.assign(new Error("ENOENT"), { code: "ENOENT" })),
    }));
    const { readAppleVramLimit } = await import("../../src/hardware/apple-memory.js");
    const totalBytes = 16 * 1024 * 1024 * 1024;
    const r = await readAppleVramLimit(totalBytes);
    expect(r.source).toBe("fallback");
    expect(r.factor).toBeCloseTo(0.67, 2);
    // 67% of 16 GB = 10977 MB
    expect(r.vramMb).toBe(Math.round(16 * 1024 * 0.67));
  });

  it("falls back when sysctl returns 0 (disabled/unset key)", async () => {
    Object.defineProperty(process, "platform", { value: "darwin" });
    vi.doMock("execa", () => ({
      execa: vi.fn().mockResolvedValue({ stdout: "0\n" }),
    }));
    const { readAppleVramLimit } = await import("../../src/hardware/apple-memory.js");
    const r = await readAppleVramLimit(16 * 1024 * 1024 * 1024);
    expect(r.source).toBe("fallback");
  });
});
