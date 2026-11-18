import { describe, it, expect } from "vitest";
import { vramBreakdown, fitsInFreeVram } from "../../src/tui/vram-breakdown.js";
import { getCuratedModels } from "../../src/models/database.js";
import { memoryNeedMb } from "../../src/analysis/context-capacity.js";

describe("vramBreakdown", () => {
  const snap = (over: Partial<Parameters<typeof vramBreakdown>[0]>) =>
    vramBreakdown({ gpuVramUsedMb: 6000, gpuVramTotalMb: 16000, modelVramMb: 4500, gpuCount: 1, ...over });

  it("splits used VRAM into Ollama's model figure and the rest when the monitor reads the only GPU", () => {
    expect(snap({})).toEqual({
      totalMb: 16000, usedMb: 6000, freeMb: 10000, modelVramMb: 4500, otherMb: 1500, split: true,
    });
  });

  it("clamps the model figure to the used memory and a missing figure to 0", () => {
    // size_vram 5000 > used 4000: the loaded model cannot exceed what the GPU reports in use.
    expect(snap({ gpuVramUsedMb: 4000, gpuVramTotalMb: 8000, modelVramMb: 5000 })).toMatchObject({ modelVramMb: 4000, otherMb: 0 });
    expect(snap({ gpuVramUsedMb: 4000, gpuVramTotalMb: 8000, modelVramMb: null })).toMatchObject({ modelVramMb: 0, otherMb: 4000 });
  });

  it("shows used/free only when size_vram may span GPUs the monitor does not read", () => {
    // A 41 GiB model split over two 24 GiB cards: card 0 uses 21 GiB, size_vram is 41 GiB.
    expect(snap({ gpuVramUsedMb: 21504, gpuVramTotalMb: 24576, modelVramMb: 41984, gpuCount: 2 })).toEqual({
      totalMb: 24576, usedMb: 21504, freeMb: 3072, modelVramMb: 0, otherMb: 21504, split: false,
    });
    expect(snap({ gpuCount: null })).toMatchObject({ split: false });
  });

  it("is null without GPU figures", () => {
    expect(snap({ gpuVramUsedMb: null, modelVramMb: null })).toBeNull();
  });
});

describe("fitsInFreeVram on an AMD APU", () => {
  // The pool is GTT, a slice of system RAM: the scan's amd-apu budget = min(GTT, available RAM) - 457 - 1024.
  it("caps the free GTT at the RAM available", () => {
    // free GTT 32668, RAM available 8000: budget 8000 - 1481 = 6519
    const fits = fitsInFreeVram(32668, "AMD", 8000);
    for (const f of fits) expect(f.needMb).toBeLessThanOrEqual(6519);
    // with RAM to spare the same GTT lists models up to 32668 - 1481 = 31187
    expect(Math.max(...fitsInFreeVram(32668, "AMD", 100_000).map((f) => f.needMb))).toBeGreaterThan(6519);
  });

  it("a discrete AMD card (no RAM figure) keeps the discrete budget", () => {
    expect(Math.max(...fitsInFreeVram(32668, "AMD").map((f) => f.needMb))).toBeGreaterThan(6519);
  });
});

describe("fitsInFreeVram", () => {
  it("lists at most 3 curated exact quants whose need at 4096 tokens fits, largest first", () => {
    const fits = fitsInFreeVram(12000, "NVIDIA");
    expect(fits.length).toBeGreaterThan(0);
    expect(fits.length).toBeLessThanOrEqual(3);
    for (const f of fits) expect(f.needMb).toBeLessThanOrEqual(12000 - 457 - 1024);
    expect(fits.map((f) => f.needMb)).toEqual([...fits.map((f) => f.needMb)].sort((a, b) => b - a));
    const m = getCuratedModels().find((c) => c.name === fits[0].name)!;
    const q = m.quantizations.find((x) => x.exact)!;
    expect(fits[0].needMb).toBe(memoryNeedMb(m, q, 4096));
  });

  it("is empty when nothing fits", () => {
    expect(fitsInFreeVram(10, "NVIDIA")).toEqual([]);
  });

  // llama.cpp's fit target is free memory minus 1024 MiB; a discrete GPU also holds ~457 MiB (estimate) before it reads it.
  it("leaves the runtime's reserves out of the free memory", () => {
    const smallest = Math.min(...getCuratedModels().flatMap((m) => {
      const q = m.quantizations.find((x) => x.exact);
      return q && m.ollamaTag ? [memoryNeedMb(m, q, 4096)] : [];
    }));
    expect(fitsInFreeVram(Math.floor(smallest) + 1480, "NVIDIA")).toEqual([]);
    expect(fitsInFreeVram(Math.ceil(smallest) + 1481, "NVIDIA").at(-1)?.needMb).toBe(smallest);
    expect(fitsInFreeVram(Math.floor(smallest) + 1023, "Apple")).toEqual([]);
    expect(fitsInFreeVram(Math.ceil(smallest) + 1024, "Apple").at(-1)?.needMb).toBe(smallest);
  });
});
