import { describe, it, expect } from "vitest";
import { parseRocmCsv } from "../../src/hardware/gpu.js";

describe("parseRocmCsv", () => {
  it("parses realistic rocm-smi CSV output", () => {
    // rocm-smi outputs label + value on the same line
    const csv = [
      "device,VRAM Total Memory (B),25769803776",
      "device,VRAM Used Memory (B),838860800",
      "device,GPU use (%),42",
      "device,Temperature (edge),55.0",
    ].join("\n");

    const stats = parseRocmCsv(csv);

    // 25769803776 / 1024 / 1024 ≈ 24576 MB
    expect(stats.vramTotalMb).toBe(24576);
    // 838860800 / 1024 / 1024 ≈ 800 MB
    expect(stats.vramUsedMb).toBe(800);
    expect(stats.utilizationPercent).toBe(42);
    expect(stats.temperatureCelsius).toBe(55);
  });

  it("returns zeros for empty output", () => {
    const stats = parseRocmCsv("");
    expect(stats.vramTotalMb).toBe(0);
    expect(stats.vramUsedMb).toBe(0);
    expect(stats.utilizationPercent).toBe(0);
    expect(stats.temperatureCelsius).toBe(0);
  });

  it("handles output with GPU utilization label variant", () => {
    const csv = [
      "device,VRAM Total Memory (B),8589934592",
      "device,VRAM Used Memory (B),2147483648",
      "device,GPU utilization (%),75",
    ].join("\n");

    const stats = parseRocmCsv(csv);
    expect(stats.vramTotalMb).toBe(8192);
    expect(stats.vramUsedMb).toBe(2048);
    expect(stats.utilizationPercent).toBe(75);
  });
});
