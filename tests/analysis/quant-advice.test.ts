import { describe, it, expect } from "vitest";
import { scoreModel, pickSweetSpot } from "../../src/analysis/scorer.js";
import { describeQuant } from "../../src/cli/commands/quant-advice.js";
import { getModelById } from "../../src/models/database.js";
import type { HardwareProfile, ModelScore } from "../../src/core/types.js";
import highEnd from "../fixtures/hardware-profiles/high-end-nvidia.json";
import cpuOnly from "../fixtures/hardware-profiles/cpu-only.json";
import appleM2 from "../fixtures/hardware-profiles/apple-m2.json";

function makeScore(overrides: Partial<ModelScore["quantization"]> & { fitLevel?: ModelScore["fitLevel"] }): ModelScore {
  return {
    model: getModelById("llama-3.1-8b")!,
    quantization: {
      name: "Q?",
      bitsPerWeight: 5,
      vramMb: 5000,
      qualityRetention: 0.9,
      ...overrides,
    },
    fitLevel: overrides.fitLevel ?? "comfortable",
    fitRatio: 1.5,
    compositeScore: 70,
    speedEstimate: "moderate",
  };
}

describe("quant-advice sweet-spot picker", () => {
  it("picks F16 on a high-end GPU — every quant fits comfortably, max quality wins", () => {
    const model = getModelById("llama-3.1-8b")!;
    const scores = model.quantizations
      .map((q) => scoreModel(model, q, highEnd as HardwareProfile))
      .sort((a, b) => a.quantization.bitsPerWeight - b.quantization.bitsPerWeight);

    const idx = pickSweetSpot(scores);
    expect(idx).toBeGreaterThanOrEqual(0);
    // With 24 GB VRAM, 8B F16 at 16 GB fits comfortably — that's the max-quality winner.
    expect(scores[idx].quantization.name).toBe("F16");
  });

  it("picks a smaller quant on tighter VRAM — largest that fits comfortably", () => {
    const model = getModelById("llama-3.1-8b")!;
    const scores = model.quantizations
      .map((q) => scoreModel(model, q, appleM2 as HardwareProfile))
      .sort((a, b) => a.quantization.bitsPerWeight - b.quantization.bitsPerWeight);

    const idx = pickSweetSpot(scores);
    // M2 Pro effective VRAM ~5 GB after headroom; F16 overflows, so Q8_0 or
    // smaller should win. Either way it must NOT be F16, and the picked quant
    // must actually fit.
    expect(scores[idx].quantization.name).not.toBe("F16");
    expect(scores[idx].fitLevel).not.toBe("cannot_run");
  });

  it("returns -1 when no quantization fits at all", () => {
    const model = getModelById("llama-3.1-70b")!;
    const scores = model.quantizations
      .map((q) => scoreModel(model, q, cpuOnly as HardwareProfile))
      .sort((a, b) => a.quantization.bitsPerWeight - b.quantization.bitsPerWeight);

    // cpu-only has 10 GB available RAM — every 70B quant is >39 GB.
    const idx = pickSweetSpot(scores);
    expect(idx).toBe(-1);
  });

  it("falls back to tight fit when nothing is comfortable", () => {
    // Fabricate scores where only a tight-fit quant is possible.
    const fakeModel = getModelById("llama-3.1-8b")!;
    const tightScores: ModelScore[] = [
      {
        model: fakeModel,
        quantization: { name: "Q4_K_M", bitsPerWeight: 4.83, vramMb: 5000, qualityRetention: 0.92 },
        fitLevel: "cannot_run",
        fitRatio: 0.8,
        compositeScore: 30,
        speedEstimate: "slow",
      },
      {
        model: fakeModel,
        quantization: { name: "Q3_K_M", bitsPerWeight: 3.5, vramMb: 3500, qualityRetention: 0.85 },
        fitLevel: "tight",
        fitRatio: 1.05,
        compositeScore: 60,
        speedEstimate: "moderate",
      },
    ];

    const idx = pickSweetSpot(tightScores);
    expect(idx).toBe(1); // Q3_K_M (tight) is the only fitting option.
    expect(tightScores[idx].fitLevel).toBe("tight");
  });

  it("breaks ties by higher bitsPerWeight — prefers fuller precision when quality is equal", () => {
    const fakeModel = getModelById("llama-3.1-8b")!;
    const tiedScores: ModelScore[] = [
      {
        model: fakeModel,
        quantization: { name: "Q5_0", bitsPerWeight: 5.5, vramMb: 5500, qualityRetention: 0.95 },
        fitLevel: "comfortable",
        fitRatio: 1.5,
        compositeScore: 80,
        speedEstimate: "moderate",
      },
      {
        model: fakeModel,
        quantization: { name: "Q5_K_M", bitsPerWeight: 5.69, vramMb: 5800, qualityRetention: 0.95 },
        fitLevel: "comfortable",
        fitRatio: 1.4,
        compositeScore: 80,
        speedEstimate: "moderate",
      },
    ];

    const idx = pickSweetSpot(tiedScores);
    expect(tiedScores[idx].quantization.name).toBe("Q5_K_M");
  });
});

describe("describeQuant note classification", () => {
  it("flags overflow as 'Too big — overflows VRAM'", () => {
    const score = makeScore({ name: "F16", bitsPerWeight: 16, fitLevel: "cannot_run" });
    const note = describeQuant(score, false, { bits: 8.5, retention: 0.99 });
    expect(note.text).toContain("overflows VRAM");
    expect(note.tone).toBe("fail");
  });

  it("flags the recommended quant as the sweet spot", () => {
    const score = makeScore({ name: "Q8_0", bitsPerWeight: 8.5, qualityRetention: 0.99 });
    const note = describeQuant(score, true, { bits: 8.5, retention: 0.99 });
    expect(note.text).toContain("Sweet spot");
    expect(note.tone).toBe("pass");
  });

  it("differentiates Q4 from Q5 when Q8 is recommended (the original bug)", () => {
    const q4 = makeScore({ name: "Q4_K_M", bitsPerWeight: 4.83, qualityRetention: 0.92 });
    const q5 = makeScore({ name: "Q5_K_M", bitsPerWeight: 5.69, qualityRetention: 0.95 });
    const ref = { bits: 8.5, retention: 0.99 };

    const q4Note = describeQuant(q4, false, ref);
    const q5Note = describeQuant(q5, false, ref);

    expect(q4Note.text).not.toBe(q5Note.text);
    expect(q4Note.text).toContain("Much smaller");
    expect(q5Note.text).toMatch(/^Smaller —/);
  });

  it("flags F16 over Q8 as 'Overkill' (negligible quality gain)", () => {
    const f16 = makeScore({ name: "F16", bitsPerWeight: 16, qualityRetention: 1.0 });
    const note = describeQuant(f16, false, { bits: 8.5, retention: 0.99 });
    expect(note.text).toContain("Overkill");
    expect(note.tone).toBe("muted");
  });

  it("flags a bigger fitting quant with meaningful quality gain as tighter-fit upside", () => {
    // Recommended is Q4 (because Q5+ are tight-fit). Q5 fits but isn't comfortable.
    const q5 = makeScore({ name: "Q5_K_M", bitsPerWeight: 5.69, qualityRetention: 0.95, fitLevel: "tight" });
    const note = describeQuant(q5, false, { bits: 4.83, retention: 0.92 });
    expect(note.text).toContain("Bigger");
    expect(note.tone).toBe("warning");
  });

  it("flags Q2/Q3-tier retention as 'Noticeably lossy'", () => {
    const q2 = makeScore({ name: "Q2_K", bitsPerWeight: 2.6, qualityRetention: 0.78 });
    const note = describeQuant(q2, false, { bits: 8.5, retention: 0.99 });
    expect(note.text).toContain("lossy");
    expect(note.tone).toBe("warning");
  });

  it("returns 'Fits' when no quant is recommended (recommended ref is null)", () => {
    const score = makeScore({ name: "Q4_K_M", bitsPerWeight: 4.83, qualityRetention: 0.92 });
    const note = describeQuant(score, false, null);
    expect(note.text).toBe("Fits");
  });
});
