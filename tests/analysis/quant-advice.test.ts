import { describe, it, expect } from "vitest";
import { scoreModel, pickSweetSpot } from "../../src/analysis/scorer.js";
import { describeQuant } from "../../src/cli/commands/quant-advice.js";
import { quantRetention } from "../../src/models/quant-facts.js";
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
      vramMb: (overrides.bitsPerWeight ?? 5) * 1000,
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

  it("prefers a tight quant over a higher-quality barely quant that needs CPU offload", () => {
    const tight = makeScore({ name: "Q4_K_M", bitsPerWeight: 4.83, qualityRetention: 0.92, fitLevel: "tight" });
    const barely = makeScore({ name: "Q5_K_M", bitsPerWeight: 5.69, qualityRetention: 0.95, fitLevel: "barely" });
    const scores = [tight, barely];

    expect(scores[pickSweetSpot(scores)].quantization.name).toBe("Q4_K_M");
  });

  it("falls back to a barely quant only when nothing is comfortable or tight", () => {
    const cannot = makeScore({ name: "Q8_0", bitsPerWeight: 8.5, qualityRetention: 0.99, fitLevel: "cannot_run" });
    const barely = makeScore({ name: "Q4_K_M", bitsPerWeight: 4.83, qualityRetention: 0.92, fitLevel: "barely" });
    const scores = [barely, cannot];

    expect(pickSweetSpot(scores)).toBe(0);
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
  // The budget is VRAM minus reserves, or RAM on CPU-only machines: usable memory, not VRAM.
  it("flags overflow as 'Too big — overflows usable memory'", () => {
    const score = makeScore({ name: "F16", bitsPerWeight: 16, fitLevel: "cannot_run" });
    const note = describeQuant(score, false, { weightsMb: 8500, retention: 0.99 });
    expect(note.text).toBe("Too big — overflows usable memory");
    expect(note.tone).toBe("fail");
  });

  it("flags the recommended quant as the sweet spot", () => {
    const score = makeScore({ name: "Q8_0", bitsPerWeight: 8.5, qualityRetention: 0.99 });
    const note = describeQuant(score, true, { weightsMb: 8500, retention: 0.99 });
    expect(note.text).toContain("Sweet spot");
    expect(note.tone).toBe("pass");
  });

  it("differentiates Q4 from Q5 when Q8 is recommended (the original bug)", () => {
    const q4 = makeScore({ name: "Q4_K_M", bitsPerWeight: 4.83, qualityRetention: 0.92 });
    // Rule: a smaller quant is a "real" drop only past 0.02 retention (exp(-KLD) scale); Q5 is 0.015 below, Q4 0.07 below.
    const q5 = makeScore({ name: "Q5_K_M", bitsPerWeight: 5.69, qualityRetention: 0.975 });
    const ref = { weightsMb: 8500, retention: 0.99 };

    const q4Note = describeQuant(q4, false, ref);
    const q5Note = describeQuant(q5, false, ref);

    expect(q4Note.text).not.toBe(q5Note.text);
    expect(q4Note.text).toContain("Much smaller");
    expect(q5Note.text).toMatch(/^Smaller —/);
  });

  it("flags F16 over Q8 as 'Overkill' (negligible quality gain)", () => {
    // Rule: a bigger quant gaining under 0.005 retention is negligible.
    const f16 = makeScore({ name: "F16", bitsPerWeight: 16, qualityRetention: 1.0 });
    const note = describeQuant(f16, false, { weightsMb: 8500, retention: 0.996 });
    expect(note.text).toContain("Overkill");
    expect(note.tone).toBe("muted");
  });

  it("flags a bigger fitting quant with meaningful quality gain as tighter-fit upside", () => {
    // Recommended is Q4 (because Q5+ are tight-fit). Q5 fits but isn't comfortable.
    const q5 = makeScore({ name: "Q5_K_M", bitsPerWeight: 5.69, qualityRetention: 0.95, fitLevel: "tight" });
    const note = describeQuant(q5, false, { weightsMb: 4830, retention: 0.92 });
    expect(note.text).toContain("Bigger");
    expect(note.tone).toBe("warning");
  });

  it("flags Q2/Q3-tier retention as 'Noticeably lossy'", () => {
    const q2 = makeScore({ name: "Q2_K", bitsPerWeight: 2.6, qualityRetention: 0.78 });
    const note = describeQuant(q2, false, { weightsMb: 8500, retention: 0.99 });
    expect(note.text).toContain("lossy");
    expect(note.tone).toBe("warning");
  });

  it("returns 'Fits' when no quant is recommended (recommended ref is null)", () => {
    const score = makeScore({ name: "Q4_K_M", bitsPerWeight: 4.83, qualityRetention: 0.92 });
    const note = describeQuant(score, false, null);
    expect(note.text).toBe("Fits");
  });

  it("flags retention just under exp(-0.1) as lossy and just over as not", () => {
    const ref = { weightsMb: 8500, retention: 0.99 };
    expect(describeQuant(makeScore({ name: "Q3_K_M", bitsPerWeight: 3.9, qualityRetention: 0.904 }), false, ref).text).toContain("lossy");
    expect(describeQuant(makeScore({ name: "Q4_K_S", bitsPerWeight: 4.6, qualityRetention: 0.906 }), false, ref).text).not.toContain("lossy");
  });

  it("says 'no published quality data' for a quant the table does not cover", () => {
    const note = describeQuant(makeScore({ name: "Q?", bitsPerWeight: 5 }), false, { weightsMb: 8500, retention: 0.99 });
    expect(note.text).toBe("No published quality data");
    expect(note.tone).toBe("muted");
  });

  // Rules pinned on the published table: Q5_K_M->Q8_0 is +0.0094 retention (above the 0.005 negligible-gain cut),
  // Q5_K_M->Q4_K_M is -0.0201 (above the 0.02 notable-drop cut); README.md:51-59.
  const published = (name: string, bitsPerWeight: number) =>
    makeScore({ name, bitsPerWeight, qualityRetention: quantRetention(name).retention });
  const q5Ref = { weightsMb: 5690, retention: quantRetention("Q5_K_M").retention };

  it("calls Q8_0 over a Q5_K_M sweet spot 'Bigger', not 'Overkill'", () => {
    expect(describeQuant(published("Q8_0", 8.5), false, q5Ref).text).toMatch(/^Bigger/);
  });

  it("calls Q4_K_M under a Q5_K_M sweet spot 'Much smaller', not 'Smaller'", () => {
    expect(describeQuant(published("Q4_K_M", 4.83), false, q5Ref).text).toMatch(/^Much smaller/);
  });

  // C1: the smaller file decides "bigger" or "smaller", not the nominal bits: mistral-small:22b's exact Q4_0
  // (11987 MiB) is estimated at 4.9 bpw like Q4_K_M (12800 MiB est.), yet it is the smaller file.
  it("compares by weights, not bits: a Q4_0 file smaller than the Q4_K_M sweet spot is 'Smaller'", () => {
    const q40 = makeScore({ name: "Q4_0", bitsPerWeight: 4.9, vramMb: 11987, qualityRetention: quantRetention("Q4_0").retention });
    const ref = { weightsMb: 12800, retention: quantRetention("Q4_K_M").retention };
    expect(describeQuant(q40, false, ref).text).toMatch(/^(Much )?[Ss]maller/);
  });

  it("names an unpublished sweet spot without a quality claim", () => {
    const note = describeQuant(makeScore({ name: "default", bitsPerWeight: 4.83 }), true, { weightsMb: 4830, retention: 1 });
    expect(note.text).toBe("★ Sweet spot — largest that fits (quality n/a)");
  });
});
