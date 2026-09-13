import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

function mockControllers(controllers: Array<Record<string, unknown>>) {
  vi.doMock("systeminformation", () => ({
    default: { graphics: vi.fn().mockResolvedValue({ controllers }) },
  }));
}

describe("detectGpus", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.doMock("../../src/hardware/intel-gpu.js", () => ({
      detectIntelGpu: vi.fn().mockResolvedValue({ vramMb: 0, hasOneApiRuntime: false }),
    }));
    vi.doMock("../../src/hardware/nvidia-smi-path.js", () => ({
      resolveNvidiaSmi: vi.fn().mockResolvedValue("nvidia-smi"),
    }));
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("assigns each NVIDIA controller its own nvidia-smi row", async () => {
    mockControllers([
      { vendor: "NVIDIA", model: "RTX 4090" },
      { vendor: "NVIDIA", model: "RTX 3060" },
    ]);
    vi.doMock("execa", () => ({
      execa: vi.fn().mockImplementation(async (_bin: string, args: string[]) => {
        if (args.length === 0) return { stdout: "CUDA Version: 12.4" };
        return { stdout: "24564, 1024, 30, 60, 550.54\n12288, 512, 5, 40, 550.54\n" };
      }),
    }));

    const { detectGpus } = await import("../../src/hardware/gpu.js");
    const gpus = await detectGpus();

    expect(gpus.map((g) => [g.model, g.vramMb, g.vramUsedMb, g.utilizationPercent])).toEqual([
      ["RTX 4090", 24564, 1024, 30],
      ["RTX 3060", 12288, 512, 5],
    ]);
    expect(gpus.every((g) => g.acceleratorVersion === "12.4")).toBe(true);
  });

  it("reports Intel integrated GPUs without a sysfs VRAM figure as vramMb 0", async () => {
    mockControllers([{ vendor: "Intel Corporation", model: "UHD Graphics 770", vram: 1024 }]);
    vi.doMock("execa", () => ({ execa: vi.fn().mockRejectedValue(new Error("unused")) }));

    const { detectGpus } = await import("../../src/hardware/gpu.js");
    const [gpu] = await detectGpus();

    expect(gpu.vramMb).toBe(0);
    expect(gpu.acceleratorType).toBeNull();
  });

  it("falls back to rocm-smi CSV when --json yields nothing", async () => {
    mockControllers([{ vendor: "Advanced Micro Devices, Inc.", model: "Radeon RX 7900 XTX" }]);
    vi.doMock("execa", () => ({
      execa: vi.fn().mockImplementation(async (_bin: string, args: string[]) => {
        if (args.includes("--json")) throw new Error("unrecognized arguments: --json");
        if (args.includes("--csv")) {
          return {
            stdout: "device,GPU use (%),VRAM Total Memory (B),VRAM Total Used Memory (B)\ncard0,12,25769803776,1073741824",
          };
        }
        return { stdout: "ROCM-SMI version: 6.1.0" };
      }),
    }));

    const { detectGpus } = await import("../../src/hardware/gpu.js");
    const [gpu] = await detectGpus();

    expect(gpu.vramMb).toBe(24576);
    expect(gpu.vramUsedMb).toBe(1024);
    expect(gpu.acceleratorType).toBe("rocm");
  });

  it("keeps CUDA detection for healthy NVIDIA rows when another row is unparseable", async () => {
    mockControllers([
      { vendor: "NVIDIA", model: "RTX 4090" },
      { vendor: "NVIDIA", model: "RTX 3060", vram: 12288 },
    ]);
    vi.doMock("execa", () => ({
      execa: vi.fn().mockImplementation(async (_bin: string, args: string[]) => {
        if (args.length === 0) return { stdout: "CUDA Version: 12.4" };
        return { stdout: "24564, 1024, 30, 60, 550.54\n[N/A], [N/A], [N/A], [N/A], 550.54\n" };
      }),
    }));

    const { detectGpus } = await import("../../src/hardware/gpu.js");
    const gpus = await detectGpus();

    expect(gpus.map((g) => [g.model, g.vramMb, g.acceleratorType])).toEqual([
      ["RTX 4090", 24564, "cuda"],
      ["RTX 3060", 12288, null],
    ]);
  });

  it("keeps the reported VRAM of Intel Arc cards that have no sysfs figure", async () => {
    mockControllers([
      { vendor: "Intel Corporation", model: "Intel(R) Arc(TM) A770 Graphics", vram: 16384 },
      { vendor: "Intel Corporation", model: "Intel(R) Arc(TM) B580 Graphics", vram: 12288 },
      // Meteor Lake iGPU: Arc-branded, but shares system RAM.
      { vendor: "Intel Corporation", model: "Intel(R) Arc(TM) Graphics", vram: 128 },
    ]);
    vi.doMock("execa", () => ({ execa: vi.fn().mockRejectedValue(new Error("unused")) }));

    const { detectGpus } = await import("../../src/hardware/gpu.js");
    const gpus = await detectGpus();

    expect(gpus.map((g) => g.vramMb)).toEqual([16384, 12288, 0]);
  });
});
