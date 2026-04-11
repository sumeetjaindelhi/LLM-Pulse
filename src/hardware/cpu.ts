import si from "systeminformation";
import type { CpuInfo } from "../core/types.js";

export async function detectCpu(): Promise<CpuInfo> {
  const [cpu, flagsRaw] = await Promise.all([si.cpu(), si.cpuFlags()]);

  const flags = flagsRaw.split(" ").filter(Boolean);
  const hasAvx2 = flags.includes("avx2");

  // systeminformation's macOS CPU path parses `machdep.cpu.brand_string`, which
  // on Apple Silicon is literally "Apple M{N} Pro" with no `@ X GHz` suffix.
  // It then falls back to `hw.tbfrequency / 1e9 * 100` → a constant 2.4 GHz for
  // every M-series chip (that's the timer base, not the CPU clock). There is
  // no public API to read the real P-core max clock, so we report null rather
  // than a wrong value. No lookup table, no hardcoded fallback.
  const isAppleSilicon = process.platform === "darwin" && process.arch === "arm64";
  const speed: number | null = isAppleSilicon ? null : cpu.speed;
  const speedMax: number | null = isAppleSilicon
    ? null
    : cpu.speedMax || cpu.speed;

  return {
    brand: cpu.brand,
    manufacturer: cpu.manufacturer,
    cores: cpu.physicalCores,
    threads: cpu.cores, // logical cores = threads
    speed,
    speedMax,
    architecture: process.arch,
    flags,
    hasAvx2,
  };
}
