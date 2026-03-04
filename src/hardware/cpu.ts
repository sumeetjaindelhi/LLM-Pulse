import si from "systeminformation";
import type { CpuInfo } from "../core/types.js";

export async function detectCpu(): Promise<CpuInfo> {
  const [cpu, flagsRaw] = await Promise.all([si.cpu(), si.cpuFlags()]);

  const flags = flagsRaw.split(" ").filter(Boolean);
  const hasAvx2 = flags.includes("avx2");

  return {
    brand: cpu.brand,
    manufacturer: cpu.manufacturer,
    cores: cpu.physicalCores,
    threads: cpu.cores, // logical cores = threads
    speed: cpu.speed,
    speedMax: cpu.speedMax || cpu.speed,
    architecture: process.arch,
    flags,
    hasAvx2,
  };
}
