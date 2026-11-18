import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

const MIB = 1024 * 1024;
const GIB = 1024 * MIB;

type CardFiles = Record<string, Record<string, number | string>>;

function mockSysfs(cards: CardFiles) {
  vi.doMock("node:fs/promises", () => ({
    readdir: vi.fn().mockResolvedValue([...Object.keys(cards), "renderD128"]),
    readFile: vi.fn().mockImplementation(async (path: string) => {
      const match = /\/(card\d+)\/device\/(\w+)$/.exec(path);
      const value = match && cards[match[1]]?.[match[2]];
      if (value === undefined) throw Object.assign(new Error("ENOENT"), { code: "ENOENT" });
      return `${value}\n`;
    }),
  }));
}

describe("readAmdSysfsMemory", () => {
  const originalPlatform = process.platform;

  beforeEach(() => {
    vi.resetModules();
    Object.defineProperty(process, "platform", { value: "linux" });
  });

  afterEach(() => {
    Object.defineProperty(process, "platform", { value: originalPlatform });
    vi.restoreAllMocks();
  });

  it("reports GTT as the pool for an APU with a small BIOS carve-out", async () => {
    mockSysfs({
      card0: { vendor: "0x1002", mem_info_vram_total: 512 * MIB, mem_info_gtt_total: 64 * GIB, mem_info_gtt_used: 2 * GIB },
    });
    const { readAmdSysfsMemory } = await import("../../src/hardware/amd-sysfs.js");
    expect(await readAmdSysfsMemory()).toEqual([{ vramTotalMb: 512, apu: { gttTotalMb: 65536, gttUsedMb: 2048 } }]);
  });

  it("reports kernel VRAM and no APU pool for a discrete AMD card even when GTT exceeds VRAM", async () => {
    mockSysfs({
      card0: { vendor: "0x1002", mem_info_vram_vendor: "samsung", mem_info_vram_total: 24 * GIB, mem_info_gtt_total: 32 * GIB, mem_info_gtt_used: 0 },
    });
    const { readAmdSysfsMemory } = await import("../../src/hardware/amd-sysfs.js");
    expect(await readAmdSysfsMemory()).toEqual([{ vramTotalMb: 24576, apu: null }]);
  });

  it("ignores non-AMD vendors", async () => {
    mockSysfs({
      card0: { vendor: "0x8086", mem_info_vram_total: 512 * MIB, mem_info_gtt_total: 64 * GIB, mem_info_gtt_used: 0 },
    });
    const { readAmdSysfsMemory } = await import("../../src/hardware/amd-sysfs.js");
    expect(await readAmdSysfsMemory()).toEqual([]);
  });

  it("reports null VRAM for an AMD card with unreadable memory files", async () => {
    mockSysfs({ card0: { vendor: "0x1002" } });
    const { readAmdSysfsMemory } = await import("../../src/hardware/amd-sysfs.js");
    expect(await readAmdSysfsMemory()).toEqual([{ vramTotalMb: null, apu: null }]);
  });

  it("treats a 2 GiB UMA / 16 GiB GTT card without discrete markers as an APU", async () => {
    mockSysfs({
      card0: { vendor: "0x1002", mem_info_vram_total: 2 * GIB, mem_info_gtt_total: 16 * GIB, mem_info_gtt_used: 0 },
    });
    const { readAmdSysfsMemory } = await import("../../src/hardware/amd-sysfs.js");
    expect(await readAmdSysfsMemory()).toEqual([{ vramTotalMb: 2048, apu: { gttTotalMb: 16384, gttUsedMb: 0 } }]);
  });

  it("treats a 4 GiB UMA / 32 GiB GTT card without discrete markers as an APU", async () => {
    mockSysfs({
      card0: { vendor: "0x1002", mem_info_vram_total: 4 * GIB, mem_info_gtt_total: 32 * GIB, mem_info_gtt_used: 0 },
    });
    const { readAmdSysfsMemory } = await import("../../src/hardware/amd-sysfs.js");
    expect((await readAmdSysfsMemory())[0].apu).not.toBeNull();
  });

  it.each(["mem_info_vram_vendor", "board_info"])("treats a small-VRAM card exposing %s as discrete", async (marker) => {
    mockSysfs({
      card0: { vendor: "0x1002", [marker]: "x", mem_info_vram_total: 2 * GIB, mem_info_gtt_total: 16 * GIB, mem_info_gtt_used: 0 },
    });
    const { readAmdSysfsMemory } = await import("../../src/hardware/amd-sysfs.js");
    expect(await readAmdSysfsMemory()).toEqual([{ vramTotalMb: 2048, apu: null }]);
  });

  it("returns nothing off Linux", async () => {
    Object.defineProperty(process, "platform", { value: "win32" });
    mockSysfs({ card0: { vendor: "0x1002", mem_info_vram_total: 512 * MIB, mem_info_gtt_total: 64 * GIB } });
    const { readAmdSysfsMemory } = await import("../../src/hardware/amd-sysfs.js");
    expect(await readAmdSysfsMemory()).toEqual([]);
  });

  // O discover/amd.go readROCmLinuxSysfsDevices: a KFD node's gfx_target_version, matched to the card by PCI slot
  // through the node's drm_render_minor.
  describe("gfx target", () => {
    const files: Record<string, string> = {
      "/sys/class/drm/card0/device/vendor": "0x1002",
      "/sys/class/drm/card0/device/mem_info_vram_total": String(512 * MIB),
      "/sys/class/drm/card0/device/mem_info_gtt_total": String(64 * GIB),
      "/sys/class/drm/card0/device/uevent": "DRIVER=amdgpu\nPCI_SLOT_NAME=0000:c5:00.0\n",
      "/sys/class/drm/renderD128/device/uevent": "DRIVER=amdgpu\nPCI_SLOT_NAME=0000:c5:00.0\n",
      "/sys/class/kfd/kfd/topology/nodes/0/properties": "cpu_cores_count 32\ngfx_target_version 0\n",
      "/sys/class/kfd/kfd/topology/nodes/1/properties":
        "vendor_id 4098\ndevice_id 5510\ndrm_render_minor 128\ngfx_target_version 110501\n",
    };

    function mockFiles(map: Record<string, string>) {
      vi.doMock("node:fs/promises", () => ({
        readdir: vi.fn().mockImplementation(async (path: string) => {
          if (path === "/sys/class/drm") return ["card0", "renderD128"];
          if (path === "/sys/class/kfd/kfd/topology/nodes") return ["1", "0"];
          throw Object.assign(new Error("ENOENT"), { code: "ENOENT" });
        }),
        readFile: vi.fn().mockImplementation(async (path: string) => {
          if (map[path] === undefined) throw Object.assign(new Error("ENOENT"), { code: "ENOENT" });
          return map[path];
        }),
      }));
    }

    it("reads gfx1151 (kfd version 110501) for the matching card", async () => {
      mockFiles(files);
      const { readAmdSysfsMemory } = await import("../../src/hardware/amd-sysfs.js");
      expect((await readAmdSysfsMemory())[0]).toMatchObject({ gfxTarget: "gfx1151" });
    });

    it("reads gfx1103 (kfd version 110003) for a Radeon 780M", async () => {
      mockFiles({
        ...files,
        "/sys/class/kfd/kfd/topology/nodes/1/properties": "vendor_id 4098\ndevice_id 5510\ndrm_render_minor 128\ngfx_target_version 110003\n",
      });
      const { readAmdSysfsMemory } = await import("../../src/hardware/amd-sysfs.js");
      expect((await readAmdSysfsMemory())[0].gfxTarget).toBe("gfx1103");
    });

    it("leaves the target unset without KFD", async () => {
      mockFiles(Object.fromEntries(Object.entries(files).filter(([path]) => !path.includes("/kfd/"))));
      const { readAmdSysfsMemory } = await import("../../src/hardware/amd-sysfs.js");
      expect((await readAmdSysfsMemory())[0].gfxTarget).toBeUndefined();
    });
  });
});
