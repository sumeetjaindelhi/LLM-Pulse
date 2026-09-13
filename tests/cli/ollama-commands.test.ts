import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

vi.mock("systeminformation", () => ({
  default: {
    currentLoad: vi.fn().mockResolvedValue({ currentLoad: 10 }),
    mem: vi.fn().mockResolvedValue({ total: 16 * 1024 ** 3, available: 8 * 1024 ** 3 }),
  },
}));

vi.mock("execa", () => ({
  execa: vi.fn().mockRejectedValue(new Error("nvidia-smi not found")),
}));

import { benchmarkCommand } from "../../src/cli/commands/benchmark.js";
import { profileCommand } from "../../src/cli/commands/profile.js";

let stdout: string[];

beforeEach(() => {
  stdout = [];
  vi.spyOn(console, "log").mockImplementation((...args) => { stdout.push(args.join(" ")); });
  process.exitCode = undefined;
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  process.exitCode = undefined;
});

// /api/version answers; /api/generate fails the way a daemon that died mid-run does.
function ollamaThatFailsGenerate() {
  return vi.fn(async (url: string) => {
    if (url.endsWith("/api/version")) return { ok: true } as Response;
    throw new TypeError("fetch failed");
  });
}

function expectNoRedirectFollowing(fetchMock: ReturnType<typeof vi.fn>) {
  expect(fetchMock).toHaveBeenCalled();
  for (const [, init] of fetchMock.mock.calls) {
    expect(init).toMatchObject({ redirect: "error" });
  }
}

const host = "http://127.0.0.1:11434";

describe("benchmark", () => {
  it("json: Ollama not running prints the error payload and exits 1", async () => {
    const fetchMock = vi.fn().mockRejectedValue(new TypeError("fetch failed"));
    vi.stubGlobal("fetch", fetchMock);
    await benchmarkCommand({ model: "tinyllama", rounds: 1, format: "json", host });
    expect(JSON.parse(stdout.join("\n"))).toEqual({ error: "Ollama is not running" });
    expect(process.exitCode).toBe(1);
    expectNoRedirectFollowing(fetchMock);
  });

  it("json: all rounds failing exits 1 and never follows redirects", async () => {
    const fetchMock = ollamaThatFailsGenerate();
    vi.stubGlobal("fetch", fetchMock);
    await benchmarkCommand({ model: "tinyllama", rounds: 2, format: "json", host });
    expect(JSON.parse(stdout.join("\n"))).toEqual({ error: "All rounds failed" });
    expect(process.exitCode).toBe(1);
    expectNoRedirectFollowing(fetchMock);
  });
});

describe("profile", () => {
  it("json: Ollama not running prints the error payload and exits 1", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("fetch failed")));
    await profileCommand({ model: "tinyllama", contextSize: 2048, format: "json", host });
    expect(JSON.parse(stdout.join("\n"))).toEqual({ error: "Ollama is not running" });
    expect(process.exitCode).toBe(1);
  });

  it("clears every polling interval when inference throws, exits 1, never follows redirects", async () => {
    const fetchMock = ollamaThatFailsGenerate();
    vi.stubGlobal("fetch", fetchMock);
    const setIntervalSpy = vi.spyOn(globalThis, "setInterval");
    const clearIntervalSpy = vi.spyOn(globalThis, "clearInterval");

    await profileCommand({ model: "tinyllama", contextSize: 2048, format: "json", host });

    expect(JSON.parse(stdout.join("\n"))).toEqual({ error: "All prompts failed" });
    expect(process.exitCode).toBe(1);
    const created = setIntervalSpy.mock.results.map((r) => r.value);
    expect(created.length).toBeGreaterThan(0);
    const cleared = clearIntervalSpy.mock.calls.map(([id]) => id);
    for (const id of created) expect(cleared).toContain(id);
    expectNoRedirectFollowing(fetchMock);
  });
});
