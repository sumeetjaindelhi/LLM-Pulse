import { describe, it, expect } from "vitest";
import { formatBits, formatBreakdown, formatMb, formatNeed, formatNeedAt, formatQualityPct, formatUsable, formatWeights } from "../../src/cli/ui/progress.js";
import type { MemoryBreakdown } from "../../src/core/types.js";

describe("formatMb", () => {
  it("prints MiB / 1024 as GiB, the binary unit it is", () => {
    // Rule: 23095 / 1024 = 22.55 -> 22.6.
    expect(formatMb(23095)).toBe("22.6 GiB");
    expect(formatMb(1024)).toBe("1.0 GiB");
    expect(formatMb(512)).toBe("512 MiB");
  });

  it("rounds a fractional MiB figure below 1 GiB to whole MiB", () => {
    expect(formatMb(551.1)).toBe("551 MiB");
  });
});

// llama3.1:8b Q8_0 at 32768 tokens on a 24564 MiB GPU (see memoryBreakdown in context-capacity.test.ts).
const q8: MemoryBreakdown = {
  contextTokens: 32768,
  weightsMb: 8107,
  weightsExact: false,
  kvCacheMb: 4096,
  kvExact: true,
  computeMb: 551.1,
  totalMb: 12754.1,
  budgetMb: 23083,
  budgetExact: false,
};

describe("memory labels", () => {
  // 12754.1 / 1024 = 12.45 -> 12.5; 23083 / 1024 = 22.54 -> 22.5.
  it("names the context and carries the est. marker; the usable budget has its own label", () => {
    expect(formatNeedAt(q8)).toBe(`12.5 GiB est. at ${(32768).toLocaleString()} tokens`);
    expect(formatNeedAt({ ...q8, weightsExact: true })).toBe(`12.5 GiB at ${(32768).toLocaleString()} tokens`);
    expect(formatUsable(q8)).toBe("of 22.5 GiB usable est.");
    expect(formatUsable({ ...q8, budgetExact: true })).toBe("of 22.5 GiB usable");
  });

  it("marks each estimated part est. and the compute buffer ≈", () => {
    expect(formatBreakdown(q8)).toBe("weights 7.9 GiB est. · KV 4.0 GiB · compute ≈551 MiB");
    expect(formatBreakdown({ ...q8, weightsExact: true, kvExact: false })).toBe("weights 7.9 GiB · KV 4.0 GiB est. · compute ≈551 MiB");
  });

  it("marks a total est. when its weights or KV cache are estimated", () => {
    expect(formatNeed(q8)).toBe("12.5 GiB est.");
    expect(formatNeed({ ...q8, weightsExact: true, kvExact: false })).toBe("12.5 GiB est.");
    expect(formatNeed({ ...q8, weightsExact: true })).toBe("12.5 GiB");
  });

  it("marks weights est. unless they are the real file size", () => {
    expect(formatWeights({ name: "Q8_0", bitsPerWeight: 8.5008, vramMb: 8107, qualityRetention: 0.99 })).toBe("7.9 GiB est.");
    expect(formatWeights({ name: "Q4_K_M", bitsPerWeight: 4.8944, vramMb: 4693, qualityRetention: 0.97, exact: true })).toBe("4.6 GiB");
  });
});

describe("formatQualityPct", () => {
  // exp(-0.001355) = 0.99865 -> 99.8 (floored to 0.1), never rounded up to the F16 reference's 100.
  it("shows one decimal and never rounds a value below 1 up to 100%", () => {
    expect(formatQualityPct(Math.exp(-0.001355))).toBe("99.8%");
    expect(formatQualityPct(1)).toBe("100.0%");
    expect(formatQualityPct(0.975)).toBe("97.5%");
  });
});

describe("formatBits", () => {
  const quant = (name: string) => ({ name, bitsPerWeight: 4.9, vramMb: 1, qualityRetention: 1 });
  it("marks bits derived from the nominal figure, and library placeholders, with ≈", () => {
    expect(formatBits(quant("Q4_0"))).toBe("≈4.9");
    expect(formatBits(quant("default"))).toBe("≈4.9");
    expect(formatBits(quant("Q4_K_M"))).toBe("4.9");
  });
});
