import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

const GIB = 1024 * 1024 * 1024;

function mockSystem(vendor: string, model: string) {
  vi.doMock("systeminformation", () => ({
    default: {
      currentLoad: vi.fn().mockResolvedValue({ currentLoad: 10 }),
      cpuTemperature: vi.fn().mockResolvedValue({ main: 50 }),
      mem: vi.fn().mockResolvedValue({ total: 24 * GIB, available: 12 * GIB }),
      graphics: vi.fn().mockResolvedValue({ controllers: [{ vendor, model }] }),
    },
  }));
}

const readAmdSysfsMemory = vi.fn();

describe("HardwareMonitor", () => {
  const originalPlatform = process.platform;
  const originalFetch = globalThis.fetch;
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.resetModules();
    fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ models: [] }) });
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    // Keep the AMD tests independent of the host's real /sys/class/drm.
    readAmdSysfsMemory.mockReset().mockResolvedValue([]);
    vi.doMock("../../src/hardware/amd-sysfs.js", () => ({ readAmdSysfsMemory }));
  });

  afterEach(() => {
    Object.defineProperty(process, "platform", { value: originalPlatform });
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("reads the context length from /api/ps and never reports tokens/sec", async () => {
    mockSystem("NVIDIA", "RTX 4090");
    vi.doMock("execa", () => ({ execa: vi.fn().mockRejectedValue(new Error("no gpu tool")) }));
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        models: [{
          name: "llama3.1:8b",
          model: "llama3.1:8b",
          size: 5 * GIB,
          digest: "abc",
          details: { format: "gguf", family: "llama", parameter_size: "8.0B", quantization_level: "Q4_K_M" },
          expires_at: "2026-09-13T10:00:00Z",
          size_vram: 5 * GIB,
          context_length: 4096,
        }],
      }),
    });

    const { HardwareMonitor } = await import("../../src/hardware/monitor.js");
    const snap = await new HardwareMonitor("http://127.0.0.1:11434").takeSnapshot();

    expect(snap.activeModel).toBe("llama3.1:8b");
    expect(snap.modelContextLength).toBe(4096);
    expect(snap.modelQuantization).toBe("Q4_K_M");
    expect(snap.modelSize).toBe("5.0 GiB");
    expect(snap.modelVramMb).toBe(5 * 1024);
    expect(snap.tokensPerSec).toBeNull();
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:11434/api/ps",
      expect.objectContaining({ redirect: "error" }),
    );
  });

  it("uses the Metal wired limit, not total RAM, as Apple Silicon VRAM total", async () => {
    Object.defineProperty(process, "platform", { value: "darwin" });
    mockSystem("Apple", "Apple M4 Pro");
    vi.doMock("execa", () => ({
      execa: vi.fn().mockImplementation(async (cmd: string) => {
        if (cmd === "sysctl") return { stdout: "18186\n" };
        if (cmd === "ioreg") {
          return { stdout: '"PerformanceStatistics" = {"Device Utilization %"=27,"In use system memory"=1073741824}' };
        }
        throw new Error(`unexpected ${cmd}`);
      }),
    }));

    const { HardwareMonitor } = await import("../../src/hardware/monitor.js");
    const snap = await new HardwareMonitor().takeSnapshot();

    expect(snap.gpuVramTotalMb).toBe(18186);
    expect(snap.gpuVramUsedMb).toBe(1024);
  });

  it("reads AMD stats from rocm-smi --json", async () => {
    mockSystem("Advanced Micro Devices, Inc.", "Radeon RX 7900 XTX");
    vi.doMock("execa", () => ({
      execa: vi.fn().mockImplementation(async (_cmd: string, args: string[]) => {
        if (args.includes("--json")) {
          return {
            stdout: JSON.stringify({
              card0: { "VRAM Total (B)": "25769803776", "VRAM Total Used (B)": "1073741824", "GPU use (%)": "42", "Temperature (Sensor edge) (C)": "55" },
            }),
          };
        }
        throw new Error("unexpected rocm-smi call");
      }),
    }));

    const { HardwareMonitor } = await import("../../src/hardware/monitor.js");
    const snap = await new HardwareMonitor().takeSnapshot();

    expect(snap).toMatchObject({ gpuVramTotalMb: 24576, gpuVramUsedMb: 1024, gpuPercent: 42, gpuTemp: 55 });
  });

  it("reports the AMD card count rocm-smi lists while the figures stay card0's", async () => {
    mockSystem("Advanced Micro Devices, Inc.", "Radeon RX 7900 XTX");
    vi.doMock("execa", () => ({
      execa: vi.fn().mockImplementation(async (_cmd: string, args: string[]) => {
        if (!args.includes("--json")) throw new Error("unexpected rocm-smi call");
        const card = (used: string) => ({ "VRAM Total (B)": "25769803776", "VRAM Total Used (B)": used, "GPU use (%)": "42" });
        return { stdout: JSON.stringify({ card0: card("1073741824"), card1: card("0") }) };
      }),
    }));

    const { HardwareMonitor } = await import("../../src/hardware/monitor.js");
    const snap = await new HardwareMonitor().takeSnapshot();

    expect(snap).toMatchObject({ gpuCount: 2, gpuVramUsedMb: 1024, gpuVramTotalMb: 24576 });
  });

  it("falls back to rocm-smi --csv for AMD when --json is unsupported", async () => {
    mockSystem("Advanced Micro Devices, Inc.", "Radeon RX 7900 XTX");
    vi.doMock("execa", () => ({
      execa: vi.fn().mockImplementation(async (_cmd: string, args: string[]) => {
        if (args.includes("--json")) throw new Error("unrecognized arguments: --json");
        return {
          stdout: "device,Temperature (Sensor edge) (C),GPU use (%),VRAM Total Memory (B),VRAM Total Used Memory (B)\ncard0,45.0,12,17163091968,1277042688",
        };
      }),
    }));

    const { HardwareMonitor } = await import("../../src/hardware/monitor.js");
    const snap = await new HardwareMonitor().takeSnapshot();

    expect(snap).toMatchObject({ gpuVramTotalMb: 16368, gpuVramUsedMb: 1218, gpuPercent: 12, gpuTemp: 45 });
  });

  it("keeps NVIDIA utilization and temperature when memory fields are [N/A]", async () => {
    mockSystem("NVIDIA", "NVIDIA GB10");
    vi.doMock("execa", () => ({
      execa: vi.fn().mockResolvedValue({ stdout: "12, 45, [N/A], [N/A], 20.5, 1500\n" }),
    }));

    const { HardwareMonitor } = await import("../../src/hardware/monitor.js");
    const snap = await new HardwareMonitor().takeSnapshot();

    expect(snap).toMatchObject({ gpuPercent: 12, gpuTemp: 45, gpuVramUsedMb: null, gpuVramTotalMb: null });
  });

  it("reports AMD APU memory from GTT instead of the BIOS carve-out", async () => {
    mockSystem("Advanced Micro Devices, Inc.", "Radeon 8060S");
    const MIB = 1024 * 1024;
    readAmdSysfsMemory.mockResolvedValue([{ vramTotalMb: 512, apu: { gttTotalMb: 65536, gttUsedMb: 8192 } }]);
    vi.doMock("execa", () => ({
      execa: vi.fn().mockResolvedValue({
        stdout: JSON.stringify({
          card0: { "VRAM Total (B)": String(512 * MIB), "VRAM Total Used (B)": String(435 * MIB), "GPU use (%)": "30", "Temperature (Sensor edge) (C)": "50" },
        }),
      }),
    }));

    const { HardwareMonitor } = await import("../../src/hardware/monitor.js");
    const snap = await new HardwareMonitor().takeSnapshot();

    expect(snap).toMatchObject({ gpuVramTotalMb: 65536, gpuVramUsedMb: 8192, gpuPercent: 30, gpuTemp: 50, gpuUnifiedMemory: true, gpuCount: 1 });
  });

  it("reports the GPU count nvidia-smi lists while the figures stay the first GPU's", async () => {
    mockSystem("NVIDIA", "RTX 3090");
    vi.doMock("execa", () => ({
      execa: vi.fn().mockResolvedValue({ stdout: "30, 60, 21504, 24576, 300, 1700\n5, 40, 512, 24576, 20, 210\n" }),
    }));

    const { HardwareMonitor } = await import("../../src/hardware/monitor.js");
    const snap = await new HardwareMonitor().takeSnapshot();

    expect(snap).toMatchObject({ gpuCount: 2, gpuVramUsedMb: 21504, gpuVramTotalMb: 24576, gpuUnifiedMemory: false });
  });
});
