import { describe, it, expect } from "vitest";
import { parseRocmJsonOutput } from "../../src/hardware/gpu.js";

describe("parseRocmJsonOutput", () => {
  it("parses ROCm 6.x output with per-GPU fields", () => {
    const json = JSON.stringify({
      card0: {
        "VRAM Total (B)": "25769803776",
        "VRAM Total Used (B)": "1073741824",
        "GPU use (%)": "42",
        "Temperature (Sensor edge) (C)": "55.5",
      },
      card1: {
        "VRAM Total (B)": "51539607552",
        "VRAM Total Used (B)": "10737418240",
        "GPU use (%)": "88",
        "Temperature (Sensor junction) (C)": "72.0",
      },
    });

    const gpus = parseRocmJsonOutput(json);
    expect(gpus).toHaveLength(2);

    expect(gpus[0].vramTotalMb).toBe(24576);
    expect(gpus[0].vramUsedMb).toBe(1024);
    expect(gpus[0].utilizationPercent).toBe(42);
    expect(gpus[0].temperatureCelsius).toBeCloseTo(55.5, 1);

    expect(gpus[1].vramTotalMb).toBe(49152);
    expect(gpus[1].vramUsedMb).toBe(10240);
    expect(gpus[1].utilizationPercent).toBe(88);
    expect(gpus[1].temperatureCelsius).toBe(72);
  });

  it("accepts the ROCm 5.x field names too", () => {
    const json = JSON.stringify({
      card0: {
        "VRAM Total Memory (B)": "8589934592",
        "VRAM Total Used Memory (B)": "2147483648",
        "GPU use (%)": "75",
        "Temperature (C)": "60",
      },
    });
    const gpus = parseRocmJsonOutput(json);
    expect(gpus[0].vramTotalMb).toBe(8192);
    expect(gpus[0].vramUsedMb).toBe(2048);
    expect(gpus[0].utilizationPercent).toBe(75);
    expect(gpus[0].temperatureCelsius).toBe(60);
  });

  it("sorts entries by card index", () => {
    const json = JSON.stringify({
      card10: { "VRAM Total (B)": "8589934592" },
      card0: { "VRAM Total (B)": "17179869184" },
      card2: { "VRAM Total (B)": "25769803776" },
    });
    const gpus = parseRocmJsonOutput(json);
    expect(gpus).toHaveLength(3);
    expect(gpus[0].vramTotalMb).toBe(16384); // card0
    expect(gpus[1].vramTotalMb).toBe(24576); // card2
    expect(gpus[2].vramTotalMb).toBe(8192);  // card10
  });

  it("returns empty array for malformed JSON", () => {
    expect(parseRocmJsonOutput("not json")).toEqual([]);
    expect(parseRocmJsonOutput("{}")).toEqual([]);
    expect(parseRocmJsonOutput("[1,2,3]")).toEqual([]);
  });

  it("ignores non-card entries", () => {
    const json = JSON.stringify({
      card0: { "VRAM Total (B)": "8589934592" },
      system: { "some meta": "foo" },
    });
    const gpus = parseRocmJsonOutput(json);
    expect(gpus).toHaveLength(1);
  });
});
