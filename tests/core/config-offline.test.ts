import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

function mockRcFile(rc: Record<string, unknown> | null) {
  vi.doMock("node:fs", async (importOriginal) => {
    const actual = await importOriginal<typeof import("node:fs")>();
    return {
      ...actual,
      readFileSync: vi.fn((path: Parameters<typeof actual.readFileSync>[0], ...rest: unknown[]) => {
        if (String(path).endsWith(".llmpulserc")) {
          if (rc === null) throw Object.assign(new Error("ENOENT"), { code: "ENOENT" });
          return JSON.stringify(rc);
        }
        return (actual.readFileSync as (...args: unknown[]) => unknown)(path, ...rest);
      }),
    };
  });
}

describe("isOffline", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.stubEnv("LLMPULSE_OFFLINE", "");
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.doUnmock("node:fs");
  });

  it("is false with no env var and no config", async () => {
    mockRcFile(null);
    const { isOffline } = await import("../../src/core/config.js");
    expect(isOffline()).toBe(false);
  });

  it("is true when LLMPULSE_OFFLINE is 1", async () => {
    mockRcFile(null);
    vi.stubEnv("LLMPULSE_OFFLINE", "1");
    const { isOffline } = await import("../../src/core/config.js");
    expect(isOffline()).toBe(true);
  });

  it("ignores other LLMPULSE_OFFLINE values", async () => {
    mockRcFile(null);
    vi.stubEnv("LLMPULSE_OFFLINE", "true");
    const { isOffline } = await import("../../src/core/config.js");
    expect(isOffline()).toBe(false);
  });

  it("is true when .llmpulserc sets offline: true", async () => {
    mockRcFile({ offline: true });
    const { isOffline } = await import("../../src/core/config.js");
    expect(isOffline()).toBe(true);
  });

  it("is false when .llmpulserc sets offline: false", async () => {
    mockRcFile({ offline: false });
    const { isOffline } = await import("../../src/core/config.js");
    expect(isOffline()).toBe(false);
  });
});
