import { readFile, readdir } from "node:fs/promises";
import { join } from "node:path";

export interface AmdApuMemory {
  gttTotalMb: number;
  gttUsedMb: number;
}

export interface AmdSysfsMemory {
  /** Kernel-reported dedicated VRAM; null when the attribute is unreadable. */
  vramTotalMb: number | null;
  /** The GTT pool when the card is an APU, otherwise null. */
  apu: AmdApuMemory | null;
  /** gfx target from the KFD topology, e.g. "gfx1151"; absent without KFD. */
  gfxTarget?: string;
}

const DRM_ROOT = "/sys/class/drm";
const KFD_NODES_ROOT = "/sys/class/kfd/kfd/topology/nodes";
// ollama discover/amd.go:342
const AMD_PCI_VENDOR = "0x1002";
const MIB = 1024 * 1024;
// rocm-smi reports only the BIOS carve-out as VRAM on an APU, while the GPU
// can address the much larger GTT pool (a slice of system RAM). Ported from
// ollama's integrated-GPU rule: ollama discover/amd.go:370-377 calls a card
// integrated iff 0 < vram <= 4 GiB && gtt >= 8 GiB && gtt >= 4 * vram.
const APU_MAX_VRAM_MB = 4096;
const APU_MIN_GTT_MB = 8192;
const APU_MIN_GTT_TO_VRAM_RATIO = 4;
// Discrete-card markers: ollama discover/amd.go:355-359 treats a card with
// either attribute as discrete. The kernel hides board_info iff the device is
// an APU (amdgpu_device.c:379-390) and vram_vendor when unknown
// (amdgpu_vram_mgr.c:236-238).
const DISCRETE_MARKERS = ["mem_info_vram_vendor", "board_info"];

async function readMb(path: string): Promise<number | null> {
  try {
    const bytes = Number((await readFile(path, "utf-8")).trim());
    return Number.isFinite(bytes) ? Math.round(bytes / MIB) : null;
  } catch {
    return null;
  }
}

async function readPciSlot(deviceDir: string): Promise<string | null> {
  try {
    return /^PCI_SLOT_NAME=(\S+)$/m.exec(await readFile(join(deviceDir, "uevent"), "utf-8"))?.[1].toLowerCase() ?? null;
  } catch {
    return null;
  }
}

/** KFD gfx_target_version (major*10000 + minor*100 + stepping) as "gfx" + decimal major + hex minor + hex stepping
 *  (O discover/amd.go gfxTargetFromKFDVersion). */
function gfxTargetFromKfdVersion(version: number): string | null {
  const minor = Math.floor(version / 100) % 100;
  const stepping = version % 100;
  if (version <= 0 || minor > 0xf || stepping > 0xf) return null;
  return `gfx${Math.floor(version / 10000)}${minor.toString(16)}${stepping.toString(16)}`;
}

/** gfx target by PCI slot for every KFD GPU node: the node's drm_render_minor names the renderD device whose PCI
 *  slot matches a card's (O discover/amd.go readROCmLinuxSysfsDevices). */
async function readGfxTargetsByPciSlot(): Promise<Map<string, string>> {
  const targets = new Map<string, string>();
  let nodes: string[];
  try {
    nodes = await readdir(KFD_NODES_ROOT);
  } catch {
    return targets;
  }
  for (const node of nodes) {
    let properties: string;
    try {
      properties = await readFile(join(KFD_NODES_ROOT, node, "properties"), "utf-8");
    } catch {
      continue;
    }
    const value = (key: string): number => Number(new RegExp(`^${key}\\s+(\\d+)`, "m").exec(properties)?.[1] ?? 0);
    const renderMinor = value("drm_render_minor");
    const gfxTarget = gfxTargetFromKfdVersion(value("gfx_target_version"));
    if (renderMinor === 0 || gfxTarget === null) continue;
    const slot = await readPciSlot(join(DRM_ROOT, `renderD${renderMinor}`, "device"));
    if (slot) targets.set(slot, gfxTarget);
  }
  return targets;
}

/** One entry per AMD card in card-number order (matching rocm-smi's card0,
 *  card1, …). `mem_info_vram_total` is an amdgpu attribute
 *  (docs.kernel.org/gpu/amdgpu/driver-misc.html). Empty off Linux or when
 *  sysfs is unavailable. */
export async function readAmdSysfsMemory(): Promise<AmdSysfsMemory[]> {
  if (process.platform !== "linux") return [];
  let entries: string[];
  try {
    entries = await readdir(DRM_ROOT);
  } catch {
    return [];
  }
  // Card nodes only; renderD* nodes share the same device.
  const cards = entries
    .filter((entry) => /^card\d+$/.test(entry))
    .sort((a, b) => parseInt(a.slice(4), 10) - parseInt(b.slice(4), 10));

  const gfxTargets = await readGfxTargetsByPciSlot();
  const result: AmdSysfsMemory[] = [];
  for (const card of cards) {
    const device = join(DRM_ROOT, card, "device");
    let vendor: string;
    try {
      vendor = (await readFile(join(device, "vendor"), "utf-8")).trim().toLowerCase();
    } catch {
      continue;
    }
    if (vendor !== AMD_PCI_VENDOR) continue;

    const [vramTotalMb, gttTotalMb, gttUsedMb] = await Promise.all([
      readMb(join(device, "mem_info_vram_total")),
      readMb(join(device, "mem_info_gtt_total")),
      readMb(join(device, "mem_info_gtt_used")),
    ]);
    const hasDiscreteMarker = (
      await Promise.all(DISCRETE_MARKERS.map((name) => readFile(join(device, name)).then(() => true, () => false)))
    ).some(Boolean);
    const isApu =
      !hasDiscreteMarker &&
      vramTotalMb !== null &&
      gttTotalMb !== null &&
      vramTotalMb > 0 &&
      vramTotalMb <= APU_MAX_VRAM_MB &&
      gttTotalMb >= APU_MIN_GTT_MB &&
      gttTotalMb >= APU_MIN_GTT_TO_VRAM_RATIO * vramTotalMb;
    const slot = await readPciSlot(device);
    const gfxTarget = slot ? gfxTargets.get(slot) : undefined;
    result.push({ vramTotalMb, apu: isApu ? { gttTotalMb, gttUsedMb: gttUsedMb ?? 0 } : null, ...(gfxTarget && { gfxTarget }) });
  }
  return result;
}
