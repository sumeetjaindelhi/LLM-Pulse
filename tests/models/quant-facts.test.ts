import { describe, it, expect } from "vitest";
import { quantFileBpw, quantRetention, ESTIMATED_BPW_EXCESS } from "../../src/models/quant-facts.js";

describe("quantFileBpw", () => {
  it("returns measured Llama-3.1-8B file bpw, not estimated", () => {
    expect(quantFileBpw("Q4_K_M")).toEqual({ bpw: 4.8944, estimated: false });
    expect(quantFileBpw("Q8_0")).toEqual({ bpw: 8.5008, estimated: false });
    expect(quantFileBpw("Q2_K")).toEqual({ bpw: 3.1593, estimated: false });
    expect(quantFileBpw("Q3_K_S")).toEqual({ bpw: 3.6429, estimated: false });
    expect(quantFileBpw("Q5_K_S")).toEqual({ bpw: 5.5704, estimated: false });
    expect(quantFileBpw("IQ4_NL")).toEqual({ bpw: 4.6818, estimated: false });
  });

  it("is case-insensitive", () => {
    expect(quantFileBpw("q6_k")).toEqual({ bpw: 6.5633, estimated: false });
  });

  it("falls back to nominal + 0.4 for types without a measured file bpw", () => {
    // Q4_0 nominal 4.5 (18 B per 32 weights) + 0.4
    const q4 = quantFileBpw("Q4_0");
    expect(q4?.estimated).toBe(true);
    expect(q4?.bpw).toBeCloseTo(4.5 + ESTIMATED_BPW_EXCESS, 10);
    expect(ESTIMATED_BPW_EXCESS).toBe(0.4);
    // MXFP4 nominal 4.25 + 0.4
    expect(quantFileBpw("MXFP4")?.bpw).toBeCloseTo(4.65, 10);
  });

  it("returns undefined for unknown names", () => {
    expect(quantFileBpw("NOT_A_QUANT")).toBeUndefined();
  });
});

describe("quantRetention", () => {
  it("is exp(-meanKLD) for published types", () => {
    // Q4_K_M: exp(-0.031273) = 0.969211
    const r = quantRetention("Q4_K_M");
    expect(r.published).toBe(true);
    expect(r.retention).toBeCloseTo(Math.exp(-0.031273), 10);
    expect(r.retention).toBeCloseTo(0.96921, 4);
  });

  it("is exactly 1 for F16 and BF16 (KLD 0)", () => {
    expect(quantRetention("F16")).toEqual({ retention: 1, published: true });
    expect(quantRetention("BF16")).toEqual({ retention: 1, published: true });
  });

  it("orders retention by published KLD", () => {
    expect(quantRetention("Q8_0").retention).toBeGreaterThan(quantRetention("Q4_K_M").retention);
    expect(quantRetention("Q4_K_M").retention).toBeGreaterThan(quantRetention("Q2_K").retention);
    // Q2_K: exp(-0.445132) = 0.64074
    expect(quantRetention("Q2_K").retention).toBeCloseTo(0.6407, 4);
  });

  it("uses 1.0 and published=false for MXFP4 and unknown types", () => {
    expect(quantRetention("MXFP4")).toEqual({ retention: 1, published: false });
    expect(quantRetention("TQ1_0")).toEqual({ retention: 1, published: false });
  });
});
