import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

function mockControllers(controllers: Array<Record<string, unknown>>) {
  vi.doMock("systeminformation", () => ({
    default: { graphics: vi.fn().mockResolvedValue({ controllers }) },
  }));
}

const readAmdSysfsMemory = vi.fn();
const detectIntelGpu = vi.fn();

describe("detectGpus", () => {
  beforeEach(() => {
    vi.resetModules();
    detectIntelGpu.mockReset().mockResolvedValue({ hasOneApiRuntime: false });
    vi.doMock("../../src/hardware/intel-gpu.js", () => ({ detectIntelGpu }));
    vi.doMock("../../src/hardware/nvidia-smi-path.js", () => ({
      resolveNvidiaSmi: vi.fn().mockResolvedValue("nvidia-smi"),
    }));
    // Keep the AMD tests independent of the host's real /sys/class/drm.
    readAmdSysfsMemory.mockReset().mockResolvedValue([]);
    vi.doMock("../../src/hardware/amd-sysfs.js", () => ({ readAmdSysfsMemory }));
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

  it("drops NVIDIA controllers beyond the nvidia-smi rows instead of copying another card's row", async () => {
    // e.g. a vfio-bound card: systeminformation lists 2 controllers, nvidia-smi reports 1 row.
    mockControllers([
      { vendor: "NVIDIA", model: "RTX 4090" },
      { vendor: "NVIDIA", model: "RTX 3090" },
    ]);
    vi.doMock("execa", () => ({
      execa: vi.fn().mockImplementation(async (_bin: string, args: string[]) =>
        args.length === 0 ? { stdout: "CUDA Version: 12.4" } : { stdout: "24564, 1024, 30, 60, 550.54\n" },
      ),
    }));

    const { detectGpus } = await import("../../src/hardware/gpu.js");
    const gpus = await detectGpus();

    expect(gpus.map((g) => [g.model, g.vramMb])).toEqual([["RTX 4090", 24564]]);
  });

  it("drops AMD controllers beyond the rocm-smi cards instead of copying card0's row", async () => {
    mockControllers([
      { vendor: "Advanced Micro Devices, Inc.", model: "Radeon RX 7900 XTX" },
      { vendor: "Advanced Micro Devices, Inc.", model: "Radeon RX 6800" },
    ]);
    vi.doMock("execa", () => ({
      execa: vi.fn().mockImplementation(async (_bin: string, args: string[]) =>
        args.includes("--json")
          ? { stdout: JSON.stringify({ card0: { "VRAM Total (B)": String(24576 * 1024 * 1024), "VRAM Total Used (B)": "0" } }) }
          : { stdout: "ROCM-SMI version: 6.1.0" },
      ),
    }));

    const { detectGpus } = await import("../../src/hardware/gpu.js");
    const gpus = await detectGpus();

    expect(gpus.map((g) => [g.model, g.vramMb])).toEqual([["Radeon RX 7900 XTX", 24576]]);
  });

  // A vfio-bound card listed first must not take the next card's row: rows pair with controllers by PCI bus id
  // ("00000000:0A:00.0" from nvidia-smi is "0a:00.0" in systeminformation).
  it("pairs nvidia-smi rows with controllers by PCI bus id, not position", async () => {
    mockControllers([
      { vendor: "NVIDIA", model: "RTX 3090", busAddress: "01:00.0" },
      { vendor: "NVIDIA", model: "RTX 4090", busAddress: "0a:00.0" },
    ]);
    vi.doMock("execa", () => ({
      execa: vi.fn().mockImplementation(async (_bin: string, args: string[]) =>
        args.length === 0
          ? { stdout: "CUDA Version: 12.4" }
          : { stdout: `24564, 1024, 30, 60, 550.54${args[0].includes("pci.bus_id") ? ", 00000000:0A:00.0" : ""}\n` },
      ),
    }));

    const { detectGpus } = await import("../../src/hardware/gpu.js");
    const gpus = await detectGpus();

    expect(gpus.map((g) => [g.model, g.vramMb])).toEqual([["RTX 4090", 24564]]);
  });

  it("never pairs one nvidia-smi row with two controllers (lspci and clinfo entries of one card)", async () => {
    mockControllers([
      { vendor: "NVIDIA", model: "AD102 [GeForce RTX 4090]", busAddress: "0000:01:00.0" },
      { vendor: "NVIDIA", model: "NVIDIA GeForce RTX 4090", busAddress: "01:00.0" },
    ]);
    vi.doMock("execa", () => ({
      execa: vi.fn().mockImplementation(async (_bin: string, args: string[]) =>
        args.length === 0
          ? { stdout: "CUDA Version: 12.4" }
          : { stdout: "24564, 0, 0, 40, 550.54, 00000000:01:00.0\n" },
      ),
    }));

    const { detectGpus } = await import("../../src/hardware/gpu.js");
    const gpus = await detectGpus();

    expect(gpus.map((g) => [g.model, g.vramMb])).toEqual([["AD102 [GeForce RTX 4090]", 24564]]);
  });

  it("keeps the PCI domain, so equal bus:dev.fn on two domains are two cards", async () => {
    mockControllers([
      { vendor: "NVIDIA", model: "GPU-A", busAddress: "0009:01:00.0" },
      { vendor: "NVIDIA", model: "GPU-B", busAddress: "0019:01:00.0" },
    ]);
    vi.doMock("execa", () => ({
      execa: vi.fn().mockImplementation(async (_bin: string, args: string[]) =>
        args.length === 0
          ? { stdout: "CUDA Version: 12.4" }
          : { stdout: "24564, 0, 0, 40, 550.54, 00000009:01:00.0\n81920, 0, 0, 40, 550.54, 00000019:01:00.0\n" },
      ),
    }));

    const { detectGpus } = await import("../../src/hardware/gpu.js");
    const gpus = await detectGpus();

    expect(gpus.map((g) => [g.model, g.vramMb])).toEqual([["GPU-A", 24564], ["GPU-B", 81920]]);
  });

  it("pairs rocm-smi cards with controllers by PCI bus id, not position", async () => {
    mockControllers([
      { vendor: "Advanced Micro Devices, Inc.", model: "Radeon RX 6800", busAddress: "03:00.0" },
      { vendor: "Advanced Micro Devices, Inc.", model: "Radeon RX 7900 XTX", busAddress: "0000:0A:00.0" },
    ]);
    vi.doMock("execa", () => ({
      execa: vi.fn().mockImplementation(async (_bin: string, args: string[]) =>
        args.includes("--json")
          ? { stdout: JSON.stringify({ card0: { "VRAM Total (B)": String(24576 * 1024 * 1024), "VRAM Total Used (B)": "0", ...(args.includes("--showbus") && { "PCI Bus": "0000:0a:00.0" }) } }) }
          : { stdout: "ROCM-SMI version: 6.1.0" },
      ),
    }));

    const { detectGpus } = await import("../../src/hardware/gpu.js");
    const gpus = await detectGpus();

    expect(gpus.map((g) => [g.model, g.vramMb])).toEqual([["Radeon RX 7900 XTX", 24576]]);
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
    // Pinned: an arm64 host would read the [N/A] row as unified memory.
    const gpus = await detectGpus("x64");

    expect(gpus.map((g) => [g.model, g.vramMb, g.acceleratorType])).toEqual([
      ["RTX 4090", 24564, "cuda"],
      ["RTX 3060", 12288, null],
    ]);
  });

  it("keeps the driver's VRAM of discrete Arc cards off Linux and leaves acceleratorType null", async () => {
    vi.spyOn(process, "platform", "get").mockReturnValue("win32");
    mockControllers([
      { vendor: "Intel Corporation", model: "Intel(R) Arc(TM) A770 Graphics", vram: 16384 },
      { vendor: "Intel Corporation", model: "Intel(R) Arc(TM) B580 Graphics", vram: 12288 },
      // Meteor Lake iGPU: Arc-branded, but shares system RAM.
      { vendor: "Intel Corporation", model: "Intel(R) Arc(TM) Graphics", vram: 128 },
    ]);
    vi.doMock("execa", () => ({ execa: vi.fn().mockRejectedValue(new Error("unused")) }));

    const { detectGpus } = await import("../../src/hardware/gpu.js");
    const gpus = await detectGpus();

    expect(gpus.map((g) => [g.vramMb, g.acceleratorType])).toEqual([[16384, null], [12288, null], [0, null]]);
  });

  it("never uses the lspci BAR size (16 or 256) as an Intel card's VRAM on Linux", async () => {
    vi.spyOn(process, "platform", "get").mockReturnValue("linux");
    mockControllers([
      { vendor: "Intel Corporation", model: "Intel(R) Arc(TM) A770 Graphics", vram: 256 },
      { vendor: "Intel Corporation", model: "Intel(R) Arc(TM) B580 Graphics", vram: 16 },
    ]);
    vi.doMock("execa", () => ({ execa: vi.fn().mockRejectedValue(new Error("unused")) }));

    const { detectGpus } = await import("../../src/hardware/gpu.js");
    const gpus = await detectGpus();

    expect(gpus.map((g) => g.vramMb)).toEqual([0, 0]);
  });

  it("marks discrete Arc cards oneapi only when the Level Zero loader is present", async () => {
    detectIntelGpu.mockResolvedValue({ hasOneApiRuntime: true });
    mockControllers([
      { vendor: "Intel Corporation", model: "Intel(R) Arc(TM) A770 Graphics", vram: 16 },
      { vendor: "Intel Corporation", model: "Intel(R) Arc(TM) Graphics", vram: 128 },
    ]);
    vi.doMock("execa", () => ({ execa: vi.fn().mockRejectedValue(new Error("unused")) }));

    const { detectGpus } = await import("../../src/hardware/gpu.js");
    const gpus = await detectGpus();

    // controller.vram is an lspci BAR parse on Linux (16 with ReBAR, else 256): never a VRAM size, so 0 (CPU budget).
    expect(gpus.map((g) => [g.vramMb, g.acceleratorType, g.acceleratorVersion])).toEqual([
      [0, "oneapi", "oneapi"],
      [0, null, null],
    ]);
  });

  it("flags Apple GPUs as unified memory", async () => {
    mockControllers([{ vendor: "Apple", model: "Apple M4 Pro" }]);
    vi.doMock("execa", () => ({ execa: vi.fn().mockRejectedValue(new Error("unused")) }));

    const { detectGpus } = await import("../../src/hardware/gpu.js");
    const [gpu] = await detectGpus();

    expect(gpu).toMatchObject({ acceleratorType: "metal", unifiedMemory: true });
  });

  describe("NVIDIA unified memory (DGX Spark / GB10)", () => {
    function mockSmi(row: string) {
      vi.doMock("execa", () => ({
        execa: vi.fn().mockImplementation(async (_bin: string, args: string[]) => {
          if (args.length === 0) return { stdout: "CUDA Version: 13.0" };
          return { stdout: `${row}\n` };
        }),
      }));
    }

    it.each(["[N/A]", "N/A", "Not Supported", "[Not Supported]"])(
      "treats an arm64 row with memory.used %s as unified CUDA memory",
      async (na) => {
        mockControllers([{ vendor: "NVIDIA", model: "NVIDIA GB10", vram: 0 }]);
        mockSmi(`${na}, ${na}, 12, 45, 580.95.05`);

        const { detectGpus } = await import("../../src/hardware/gpu.js");
        const [gpu] = await detectGpus("arm64");

        expect(gpu).toMatchObject({
          acceleratorType: "cuda",
          unifiedMemory: true,
          vramMb: 0,
          vramUsedMb: null,
          utilizationPercent: 12,
          temperatureCelsius: 45,
          driverVersion: "580.95.05",
        });
      },
    );

    it("keeps a numeric memory.total on an arm64 unified row", async () => {
      mockControllers([{ vendor: "NVIDIA", model: "NVIDIA GB10" }]);
      mockSmi("122502, [N/A], 0, 40, 580.95.05");

      const { detectGpus } = await import("../../src/hardware/gpu.js");
      const [gpu] = await detectGpus("arm64");

      expect(gpu).toMatchObject({ unifiedMemory: true, vramMb: 122502, vramUsedMb: null });
    });

    it("leaves the same row on x64 as an unusable row (generic path)", async () => {
      mockControllers([{ vendor: "NVIDIA", model: "NVIDIA GB10", vram: 4096 }]);
      mockSmi("[N/A], [N/A], 12, 45, 580.95.05");

      const { detectGpus } = await import("../../src/hardware/gpu.js");
      const [gpu] = await detectGpus("x64");

      expect(gpu.acceleratorType).toBeNull();
      expect(gpu.unifiedMemory).toBeUndefined();
      expect(gpu.vramMb).toBe(4096);
    });

    it("does not mark a healthy discrete arm64 GPU as unified", async () => {
      mockControllers([{ vendor: "NVIDIA", model: "RTX 4090" }]);
      mockSmi("24564, 1024, 30, 60, 550.54");

      const { detectGpus } = await import("../../src/hardware/gpu.js");
      const [gpu] = await detectGpus("arm64");

      expect(gpu.unifiedMemory).toBeUndefined();
      expect(gpu).toMatchObject({ vramMb: 24564, vramUsedMb: 1024, acceleratorType: "cuda" });
    });

    it("keeps a degraded arm64 row on the generic path when another row reports memory", async () => {
      mockControllers([
        { vendor: "NVIDIA", model: "RTX 4090" },
        { vendor: "NVIDIA", model: "RTX 3060", vram: 12288 },
      ]);
      mockSmi("24564, 1024, 30, 60, 550.54\n[N/A], [N/A], [N/A], [N/A], 550.54");

      const { detectGpus } = await import("../../src/hardware/gpu.js");
      const gpus = await detectGpus("arm64");

      expect(gpus[1]).toMatchObject({ vramMb: 12288, acceleratorType: null });
      expect(gpus[1].unifiedMemory).toBeUndefined();
    });

    it("still throws and retries on unparseable arm64 output", async () => {
      mockControllers([{ vendor: "NVIDIA", model: "NVIDIA GB10", vram: 4096 }]);
      const execa = vi.fn().mockImplementation(async (_bin: string, args: string[]) =>
        args.length === 0 ? { stdout: "" } : { stdout: "garbage\n" },
      );
      vi.doMock("execa", () => ({ execa }));

      const { detectGpus } = await import("../../src/hardware/gpu.js");
      const [gpu] = await detectGpus("arm64");

      expect(gpu.acceleratorType).toBeNull();
      expect(execa.mock.calls.filter(([, args]) => args.length > 0)).toHaveLength(3);
    });

    it("does not retry an arm64 unified row", async () => {
      mockControllers([{ vendor: "NVIDIA", model: "NVIDIA GB10" }]);
      const execa = vi.fn().mockImplementation(async (_bin: string, args: string[]) =>
        args.length === 0 ? { stdout: "" } : { stdout: "[N/A], [N/A], 12, 45, 580.95.05\n" },
      );
      vi.doMock("execa", () => ({ execa }));

      const { detectGpus } = await import("../../src/hardware/gpu.js");
      await detectGpus("arm64");

      expect(execa.mock.calls.filter(([, args]) => args.length > 0)).toHaveLength(1);
    });
  });

  describe("AMD APU unified memory", () => {
    const MIB = 1024 * 1024;
    const GIB = 1024 * MIB;

    function mockApuMemory(entries: ({ vramTotalMb: number | null; apu: { gttTotalMb: number; gttUsedMb: number } | null })[]) {
      readAmdSysfsMemory.mockResolvedValue(entries);
    }

    const rocmCarveOut = JSON.stringify({
      card0: { "VRAM Total (B)": String(512 * MIB), "VRAM Total Used (B)": String(435 * MIB), "GPU use (%)": "3" },
    });

    it("uses GTT as the pool and keeps rocm when rocm-smi works", async () => {
      mockControllers([{ vendor: "Advanced Micro Devices, Inc.", model: "Radeon 8060S" }]);
      mockApuMemory([{ vramTotalMb: 512, apu: { gttTotalMb: 65536, gttUsedMb: 0 } }]);
      vi.doMock("execa", () => ({
        execa: vi.fn().mockImplementation(async (_bin: string, args: string[]) =>
          args.includes("--json") ? { stdout: rocmCarveOut } : { stdout: "ROCM-SMI version: 6.1.0" },
        ),
      }));

      const { detectGpus } = await import("../../src/hardware/gpu.js");
      const [gpu] = await detectGpus();

      expect(gpu).toMatchObject({ vramMb: 65536, unifiedMemory: true, acceleratorType: "rocm" });
    });

    it("applies the APU pool on the generic path when rocm-smi is absent", async () => {
      mockControllers([{ vendor: "Advanced Micro Devices, Inc.", model: "Radeon 8060S", vram: 512 }]);
      mockApuMemory([{ vramTotalMb: 512, apu: { gttTotalMb: 65536, gttUsedMb: 0 } }]);
      vi.doMock("execa", () => ({ execa: vi.fn().mockRejectedValue(Object.assign(new Error("ENOENT"), { code: "ENOENT" })) }));

      const { detectGpus } = await import("../../src/hardware/gpu.js");
      const [gpu] = await detectGpus();

      expect(gpu).toMatchObject({ vramMb: 65536, unifiedMemory: true, acceleratorType: null });
    });

    it("leaves a discrete AMD card unchanged", async () => {
      mockControllers([{ vendor: "Advanced Micro Devices, Inc.", model: "Radeon RX 7900 XTX" }]);
      mockApuMemory([{ vramTotalMb: 24576, apu: null }]);
      vi.doMock("execa", () => ({
        execa: vi.fn().mockImplementation(async (_bin: string, args: string[]) =>
          args.includes("--json")
            ? { stdout: JSON.stringify({ card0: { "VRAM Total (B)": String(24 * GIB), "VRAM Total Used (B)": "0" } }) }
            : { stdout: "ROCM-SMI version: 6.1.0" },
        ),
      }));

      const { detectGpus } = await import("../../src/hardware/gpu.js");
      const [gpu] = await detectGpus();

      expect(gpu.vramMb).toBe(24576);
      expect(gpu.unifiedMemory).toBeUndefined();
    });

    it("uses kernel vram_total over the lspci BAR size for a discrete AMD card when rocm-smi is absent", async () => {
      mockControllers([{ vendor: "Advanced Micro Devices, Inc.", model: "Radeon RX 7900 XTX", vram: 256 }]);
      mockApuMemory([{ vramTotalMb: 24576, apu: null }]);
      vi.doMock("execa", () => ({ execa: vi.fn().mockRejectedValue(Object.assign(new Error("ENOENT"), { code: "ENOENT" })) }));

      const { detectGpus } = await import("../../src/hardware/gpu.js");
      const [gpu] = await detectGpus();

      expect(gpu.vramMb).toBe(24576);
      expect(gpu.unifiedMemory).toBeUndefined();
    });

    it("pairs each AMD card with its own sysfs entry", async () => {
      mockControllers([
        { vendor: "Advanced Micro Devices, Inc.", model: "Radeon RX 7900 XTX", vram: 1 },
        { vendor: "Advanced Micro Devices, Inc.", model: "Radeon RX 6800", vram: 1 },
      ]);
      mockApuMemory([{ vramTotalMb: 24576, apu: null }, { vramTotalMb: 16384, apu: null }]);
      vi.doMock("execa", () => ({ execa: vi.fn().mockRejectedValue(Object.assign(new Error("ENOENT"), { code: "ENOENT" })) }));

      const { detectGpus } = await import("../../src/hardware/gpu.js");
      const gpus = await detectGpus();

      expect(gpus.map((g) => g.vramMb)).toEqual([24576, 16384]);
    });

    it("ignores sysfs when its card count differs from the AMD controller count", async () => {
      mockControllers([
        { vendor: "Advanced Micro Devices, Inc.", model: "Radeon RX 7900 XTX", vram: 1 },
        { vendor: "Advanced Micro Devices, Inc.", model: "Radeon RX 6600", vram: 8192 },
      ]);
      mockApuMemory([{ vramTotalMb: 8192, apu: null }]);
      vi.doMock("execa", () => ({ execa: vi.fn().mockRejectedValue(Object.assign(new Error("ENOENT"), { code: "ENOENT" })) }));

      const { detectGpus } = await import("../../src/hardware/gpu.js");
      const gpus = await detectGpus();

      expect(gpus.map((g) => g.vramMb)).toEqual([1, 8192]);
    });

    it("falls back to the reported VRAM when sysfs has no figure for the AMD card", async () => {
      mockControllers([{ vendor: "Advanced Micro Devices, Inc.", model: "Radeon RX 6800", vram: 16384 }]);
      mockApuMemory([{ vramTotalMb: null, apu: null }]);
      vi.doMock("execa", () => ({ execa: vi.fn().mockRejectedValue(Object.assign(new Error("ENOENT"), { code: "ENOENT" })) }));

      const { detectGpus } = await import("../../src/hardware/gpu.js");
      const [gpu] = await detectGpus();

      expect(gpu.vramMb).toBe(16384);
    });
  });
});
