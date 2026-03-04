import si from "systeminformation";
import type { DiskInfo } from "../core/types.js";

export async function detectDisk(): Promise<DiskInfo> {
  const [disks, fs] = await Promise.all([si.diskLayout(), si.fsSize()]);

  // Determine disk type from primary disk
  const primaryDisk = disks[0];
  let type: DiskInfo["type"] = "Unknown";
  if (primaryDisk) {
    const iface = (primaryDisk.interfaceType || "").toLowerCase();
    const name = (primaryDisk.name || "").toLowerCase();
    if (iface.includes("nvme") || name.includes("nvme")) {
      type = "NVMe";
    } else if (primaryDisk.type === "SSD" || iface.includes("sata") && primaryDisk.type !== "HD") {
      type = "SSD";
    } else if (primaryDisk.type === "HD") {
      type = "HDD";
    }
  }

  // Sum free space across all filesystems
  const totalFreeBytes = fs.reduce((sum, f) => sum + (f.available || 0), 0);
  const totalBytes = fs.reduce((sum, f) => sum + f.size, 0);

  return {
    type,
    freeGb: Math.round(totalFreeBytes / (1024 * 1024 * 1024)),
    totalGb: Math.round(totalBytes / (1024 * 1024 * 1024)),
  };
}
