import { describe, it, expect } from "vitest";
import { getModelById } from "../../src/models/database.js";

describe("curated quant normalization", () => {
  it("takes bpw, weights size and retention from the quant table by name", () => {
    // Rule: Q5_K_M file bpw 5.7036 (L tools/quantize/README.md), vramMb = 8e9 * 5.7036 / 8 / 2^20 = 5439.4 -> 5439,
    // retention = exp(-0.010762) (L tools/perplexity/README.md).
    const q = getModelById("llama-3.1-8b")!.quantizations.find((x) => x.name === "Q5_K_M")!;
    expect(q.bitsPerWeight).toBe(5.7036);
    expect(q.vramMb).toBe(5439);
    expect(q.qualityRetention).toBeCloseTo(Math.exp(-0.010762), 10);
  });

  it("keeps an exact quant's registry size and still takes bpw and retention from the table", () => {
    // llama3.1:8b is a Q4_K_M file of 4920738944 B in the registry manifest: 4692.8 MiB -> 4693.
    const q = getModelById("llama-3.1-8b")!.quantizations.find((x) => x.name === "Q4_K_M")!;
    expect(q).toMatchObject({ exact: true, vramMb: 4693, bitsPerWeight: 4.8944 });
    expect(q.qualityRetention).toBeCloseTo(Math.exp(-0.031273), 10);
  });

  it("sizes F16 with its measured file bpw and full retention", () => {
    // 8e9 * 16.0005 / 8 / 2^20 = 15259.1 -> 15259; KLD 0 -> retention 1
    const q = getModelById("llama-3.1-8b")!.quantizations.find((x) => x.name === "F16")!;
    expect(q.vramMb).toBe(15259);
    expect(q.qualityRetention).toBe(1);
  });
});
