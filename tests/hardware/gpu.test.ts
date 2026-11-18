import { describe, it, expect } from "vitest";
import { parseRocmCsv, isDiscreteIntel } from "../../src/hardware/gpu.js";

describe("parseRocmCsv", () => {
  it("parses real rocm-smi --csv output (header row + one row per card)", () => {
    const csv = [
      "device,Temperature (Sensor edge) (C),GPU use (%),VRAM Total Memory (B),VRAM Total Used Memory (B)",
      "card0,45.0,12,17163091968,1277042688",
      "card1,61.5,88,25769803776,10737418240",
    ].join("\n");

    const gpus = parseRocmCsv(csv);

    expect(gpus).toEqual([
      { vramTotalMb: 16368, vramUsedMb: 1218, utilizationPercent: 12, temperatureCelsius: 45 },
      { vramTotalMb: 24576, vramUsedMb: 10240, utilizationPercent: 88, temperatureCelsius: 61.5 },
    ]);
  });

  it("does not confuse the VRAM Total and VRAM Total Used columns", () => {
    const csv = [
      "device,VRAM Total Used (B),VRAM Total (B)",
      "card0,2147483648,8589934592",
    ].join("\n");

    const [gpu] = parseRocmCsv(csv);
    expect(gpu.vramTotalMb).toBe(8192);
    expect(gpu.vramUsedMb).toBe(2048);
  });

  it("returns an empty list for empty or header-only output", () => {
    expect(parseRocmCsv("")).toEqual([]);
    expect(parseRocmCsv("device,VRAM Total (B)")).toEqual([]);
  });
});

describe("isDiscreteIntel", () => {
  it.each([
    "Intel(R) Arc(TM) A770 Graphics",
    "Intel(R) Arc(TM) A380 Graphics",
    "Intel(R) Arc(TM) A730M Graphics",
    "Intel(R) Arc(TM) B580 Graphics",
    "Intel(R) Arc(TM) Pro B60 Graphics",
    "Intel(R) Arc(TM) Pro A60 Graphics",
    "Intel(R) Arc(TM) Pro B70 Graphics",
    "Intel(R) Arc(TM) A380E Graphics",
  ])("treats %s as discrete", (model) => {
    expect(isDiscreteIntel(model)).toBe(true);
  });

  it.each([
    "Intel(R) Arc(TM) B390 Graphics",
    "Intel(R) Arc(TM) Pro B390 GPU",
    "Intel(R) Arc(TM) Pro B370 GPU",
    "Intel(R) Arc(TM) 140V Graphics",
    "Intel(R) Arc(TM) Graphics",
    "Intel(R) Iris(R) Xe Graphics",
  ])("treats %s as integrated", (model) => {
    expect(isDiscreteIntel(model)).toBe(false);
  });
});
