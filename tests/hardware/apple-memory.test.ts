import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

const GIB = 1024 * 1024 * 1024;

type CommandResult = { stdout: string } | Error;

/** Mock execa so each command (sysctl / osascript) gets its own canned result. */
function mockExeca(results: { sysctl: CommandResult; osascript?: CommandResult }) {
  const execa = vi.fn(async (command: string) => {
    const result = command === "sysctl" ? results.sysctl : results.osascript;
    if (result === undefined) throw new Error(`unexpected command: ${command}`);
    if (result instanceof Error) throw result;
    return result;
  });
  vi.doMock("execa", () => ({ execa }));
  return execa;
}

describe("readAppleVramLimit", () => {
  const originalPlatform = process.platform;

  beforeEach(() => {
    vi.resetModules();
  });

  afterEach(() => {
    Object.defineProperty(process, "platform", { value: originalPlatform });
    vi.doUnmock("execa");
    vi.restoreAllMocks();
  });

  it("returns fallback factor on non-darwin platforms", async () => {
    Object.defineProperty(process, "platform", { value: "linux" });
    const { readAppleVramLimit } = await import("../../src/hardware/apple-memory.js");
    const r = await readAppleVramLimit(16 * GIB);
    expect(r.source).toBe("fallback");
    expect(r.factor).toBeCloseTo(0.67, 2);
    expect(r.vramMb).toBeNull();
  });

  it("uses a positive sysctl iogpu.wired_limit_mb override without asking Metal", async () => {
    Object.defineProperty(process, "platform", { value: "darwin" });
    const execa = mockExeca({ sysctl: { stdout: "24576\n" }, osascript: { stdout: "19069665280\n" } });
    const { readAppleVramLimit } = await import("../../src/hardware/apple-memory.js");
    const r = await readAppleVramLimit(32 * GIB);
    expect(r.source).toBe("sysctl");
    expect(r.vramMb).toBe(24576);
    expect(r.factor).toBeCloseTo(24576 / (32 * 1024), 2);
    expect(execa).not.toHaveBeenCalledWith("osascript", expect.anything(), expect.anything());
  });

  it("reads Metal recommendedMaxWorkingSetSize when sysctl reports 0", async () => {
    Object.defineProperty(process, "platform", { value: "darwin" });
    const execa = mockExeca({ sysctl: { stdout: "0\n" }, osascript: { stdout: "19069665280\n" } });
    const { readAppleVramLimit } = await import("../../src/hardware/apple-memory.js");
    const r = await readAppleVramLimit(24 * GIB);
    expect(r.source).toBe("metal");
    expect(r.vramMb).toBe(18186);
    expect(r.factor).toBeCloseTo(0.74, 2);
    expect(execa).toHaveBeenCalledWith(
      "osascript",
      ["-l", "JavaScript", "-e", expect.stringContaining("recommendedMaxWorkingSetSize")],
      expect.objectContaining({ timeout: 2000 }),
    );
  });

  it("still reads Metal when sysctl fails", async () => {
    Object.defineProperty(process, "platform", { value: "darwin" });
    const enoent = Object.assign(new Error("ENOENT"), { code: "ENOENT" });
    mockExeca({ sysctl: enoent, osascript: { stdout: "19069665280\n" } });
    const { readAppleVramLimit } = await import("../../src/hardware/apple-memory.js");
    const r = await readAppleVramLimit(24 * GIB);
    expect(r.source).toBe("metal");
    expect(r.vramMb).toBe(18186);
  });

  it("falls back to 0.67 when both sysctl and osascript fail", async () => {
    Object.defineProperty(process, "platform", { value: "darwin" });
    const enoent = Object.assign(new Error("ENOENT"), { code: "ENOENT" });
    mockExeca({ sysctl: enoent, osascript: enoent });
    const { readAppleVramLimit } = await import("../../src/hardware/apple-memory.js");
    const r = await readAppleVramLimit(16 * GIB);
    expect(r.source).toBe("fallback");
    expect(r.factor).toBeCloseTo(0.67, 2);
    // 67% of 16 GB = 10977 MB
    expect(r.vramMb).toBe(Math.round(16 * 1024 * 0.67));
  });

  it.each(["undefined", "null", "", "0", "-1", "1.5e10", "12abc"])(
    "falls back to 0.67 when Metal prints %j",
    async (stdout) => {
      Object.defineProperty(process, "platform", { value: "darwin" });
      mockExeca({ sysctl: { stdout: "0\n" }, osascript: { stdout: `${stdout}\n` } });
      const { readAppleVramLimit } = await import("../../src/hardware/apple-memory.js");
      const r = await readAppleVramLimit(16 * GIB);
      expect(r.source).toBe("fallback");
      expect(r.vramMb).toBe(Math.round(16 * 1024 * 0.67));
    },
  );
});
