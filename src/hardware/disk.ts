import { homedir } from "node:os";
import si from "systeminformation";
import type { DiskInfo } from "../core/types.js";

export async function detectDisk(): Promise<DiskInfo> {
  const [disks, fs] = await Promise.all([si.diskLayout(), si.fsSize()]);

  // Determine disk type from primary disk. The canonical `type` strings emitted
  // by systeminformation (verified from node_modules/systeminformation/lib/filesystem.js)
  // are: macOS → "SSD" | "HD" | "NVMe" | "USB";
  //      Linux → "SSD" | "HD" | "NVMe";
  //      Windows → "SSD" | "HD" | "SCM" | "Virtual" (+ raw MediaType codes).
  // We match every known value explicitly — no platform-specific fallbacks.
  const primaryDisk = disks[0];
  let type: DiskInfo["type"] = "Unknown";
  if (primaryDisk) {
    const iface = (primaryDisk.interfaceType || "").toLowerCase();
    const name = (primaryDisk.name || "").toLowerCase();
    const rawType = (primaryDisk.type || "").toLowerCase();
    if (rawType === "nvme" || iface.includes("nvme") || name.includes("nvme")) {
      type = "NVMe";
    } else if (rawType === "scm") {
      // Windows Storage Class Memory (e.g. Intel Optane persistent memory) — NVMe-class speed.
      type = "NVMe";
    } else if (rawType === "ssd" || rawType === "usb") {
      // USB drives are non-rotational; scoring them as SSD avoids a false HDD warning.
      type = "SSD";
    } else if (rawType === "hd" || rawType === "hdd") {
      type = "HDD";
    } else if (rawType === "virtual") {
      // VM / Hyper-V / VHD — can't infer underlying speed. Leave as Unknown
      // so doctor's Unknown branch emits a neutral info message.
      type = "Unknown";
    } else if (iface.includes("sata")) {
      type = "SSD";
    }
  }

  // Pick the filesystem that actually holds the user's home directory — that's
  // where Ollama stores models (`~/.ollama/models`), so "free space for models"
  // is what we want to report. Use longest-prefix match so Linux setups with a
  // separate /home mount work correctly. Also avoids APFS synthetic-volume
  // double-counting (on macOS, si.fsSize() returns 9 entries all reporting the
  // same pool, summing them gives a 4-5× overcount).
  const home = homedir();
  const isWin = process.platform === "win32";
  const normalize = (s: string) => (isWin ? s.toLowerCase() : s);
  const normHome = normalize(home);
  let chosen: (typeof fs)[number] | undefined;
  let bestPrefixLen = -1;
  for (const f of fs) {
    if (!f.mount) continue;
    const m = normalize(f.mount);
    if (normHome.startsWith(m) && m.length > bestPrefixLen) {
      chosen = f;
      bestPrefixLen = m.length;
    }
  }
  // Fall back to root mount, then to first entry, then to zero (honest unknown).
  if (!chosen) chosen = fs.find((f) => f.mount === "/");
  if (!chosen) chosen = fs[0];
  const freeBytes = chosen ? chosen.available || 0 : 0;
  const totalBytes = chosen ? chosen.size || 0 : 0;

  return {
    type,
    freeGb: Math.round(freeBytes / (1024 * 1024 * 1024)),
    totalGb: Math.round(totalBytes / (1024 * 1024 * 1024)),
  };
}
