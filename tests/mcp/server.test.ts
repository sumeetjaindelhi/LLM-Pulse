import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock systeminformation, execa, and fetch so HardwareMonitor.takeSnapshot()
// runs deterministically without touching real hardware or network.
vi.mock("systeminformation", () => ({
  default: {
    currentLoad: vi.fn().mockResolvedValue({ currentLoad: 42 }),
    cpuTemperature: vi.fn().mockResolvedValue({ main: 55 }),
    mem: vi.fn().mockResolvedValue({ total: 16 * 1024 * 1024 * 1024, used: 8 * 1024 * 1024 * 1024 }),
    graphics: vi.fn().mockResolvedValue({
      controllers: [{ vendor: "NVIDIA", model: "RTX 4090" }],
    }),
  },
}));

vi.mock("execa", () => ({
  execa: vi.fn().mockResolvedValue({
    // nvidia-smi CSV: util, temp, vramUsed, vramTotal, power, clock
    stdout: "37, 62, 4096, 24576, 180.5, 2520",
  }),
}));

// Stub fetch for the Ollama /api/ps call — return empty model list.
const originalFetch = globalThis.fetch;
beforeEach(() => {
  globalThis.fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ models: [] }),
  }) as unknown as typeof fetch;
});

describe("HardwareMonitor.takeSnapshot", () => {
  it("returns a snapshot with the expected shape", async () => {
    const { HardwareMonitor } = await import("../../src/hardware/monitor.js");
    const monitor = new HardwareMonitor();
    const snap = await monitor.takeSnapshot();

    expect(snap).toMatchObject({
      cpuPercent: 42,
      cpuTemp: 55,
      ramTotalMb: 16384,
      gpuPercent: 37,
      gpuTemp: 62,
      gpuVramUsedMb: 4096,
      gpuVramTotalMb: 24576,
      gpuVendor: "NVIDIA",
      gpuModel: "RTX 4090",
      activeModel: null,
      tokensPerSec: null,
    });
  });

  it("does not start the polling loop", async () => {
    const { HardwareMonitor } = await import("../../src/hardware/monitor.js");
    const monitor = new HardwareMonitor();
    await monitor.takeSnapshot();

    // Internal flags should remain in the "stopped" state.
    expect((monitor as unknown as { running: boolean }).running).toBe(false);
    expect((monitor as unknown as { timeout: unknown }).timeout).toBe(null);
  });

  it("does not mutate session stats", async () => {
    const { HardwareMonitor } = await import("../../src/hardware/monitor.js");
    const monitor = new HardwareMonitor();
    await monitor.takeSnapshot();

    expect(monitor.session.totalTokens).toBe(0);
    expect(monitor.session.totalRequests).toBe(0);
    expect(monitor.session.modelHistory.size).toBe(0);
  });

  it("can be called multiple times in a row", async () => {
    const { HardwareMonitor } = await import("../../src/hardware/monitor.js");
    const monitor = new HardwareMonitor();
    const a = await monitor.takeSnapshot();
    const b = await monitor.takeSnapshot();
    expect(a.cpuPercent).toBe(b.cpuPercent);
    expect(a.gpuPercent).toBe(b.gpuPercent);
  });
});

describe("MCP server module", () => {
  it("imports without throwing", async () => {
    const mod = await import("../../src/mcp/server.js");
    expect(typeof mod.startServer).toBe("function");
  });
});

// Restore fetch after the test file finishes.
afterAll(() => {
  globalThis.fetch = originalFetch;
});

// vitest globals — declare afterAll for type checker
declare const afterAll: (fn: () => void) => void;
