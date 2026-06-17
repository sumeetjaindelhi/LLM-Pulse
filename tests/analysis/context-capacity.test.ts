import { describe, it, expect } from "vitest";
import {
  kvCacheMbPer1kTokens,
  maxContextTokensForVram,
} from "../../src/analysis/context-capacity.js";
import { getModelById } from "../../src/models/database.js";
import type { HardwareProfile, ModelEntry, QuantizationVariant } from "../../src/core/types.js";
import highEnd from "../fixtures/hardware-profiles/high-end-nvidia.json";
import cpuOnly from "../fixtures/hardware-profiles/cpu-only.json";

const HIGH_END = highEnd as HardwareProfile;
const CPU_ONLY = cpuOnly as HardwareProfile;

const llama8b = getModelById("llama-3.1-8b")!;
const llama70b = getModelById("llama-3.1-70b")!;

function quantOf(model: ModelEntry, name: string): QuantizationVariant {
  const q = model.quantizations.find((x) => x.name === name);
  if (!q) throw new Error(`fixture model missing quant ${name}`);
  return q;
}

describe("kvCacheMbPer1kTokens", () => {
  it("maps param-count buckets to fp16 KV-cache MB per 1k tokens", () => {
    expect(kvCacheMbPer1kTokens(1.5)).toBe(40);
    expect(kvCacheMbPer1kTokens(4)).toBe(80);
    expect(kvCacheMbPer1kTokens(8)).toBe(150);
    expect(kvCacheMbPer1kTokens(15)).toBe(210);
    expect(kvCacheMbPer1kTokens(22)).toBe(280);
    expect(kvCacheMbPer1kTokens(35)).toBe(360);
    expect(kvCacheMbPer1kTokens(80)).toBe(440);
    expect(kvCacheMbPer1kTokens(81)).toBe(600);
  });
});

describe("maxContextTokensForVram", () => {
  it("computes raw token capacity on a discrete GPU", () => {
    // 24576 - 5000 - max(768, 1228.8) = 18347.2; /150*1000 = 122314.67 -> 122314
    const q4 = quantOf(llama8b, "Q4_K_M");
    expect(maxContextTokensForVram(llama8b, q4, HIGH_END)).toBe(122314);
  });

  it("uses available system RAM on a CPU-only box", () => {
    // 10000 - 5000 - max(768, 500) = 4232; /150*1000 = 28213.33 -> 28213
    const q4 = quantOf(llama8b, "Q4_K_M");
    expect(maxContextTokensForVram(llama8b, q4, CPU_ONLY)).toBe(28213);
  });

  it("returns 0 when the weights alone exceed the memory pool", () => {
    const q70 = quantOf(llama70b, "Q4_K_M"); // 40000 MB on a 10000 MB box
    expect(maxContextTokensForVram(llama70b, q70, CPU_ONLY)).toBe(0);
  });

  it("affords more context for a smaller quant", () => {
    const q4 = quantOf(llama8b, "Q4_K_M");
    const f16 = quantOf(llama8b, "F16");
    expect(maxContextTokensForVram(llama8b, q4, HIGH_END)).toBeGreaterThan(
      maxContextTokensForVram(llama8b, f16, HIGH_END),
    );
  });

  it("affords more context on a larger memory budget", () => {
    const q4 = quantOf(llama8b, "Q4_K_M");
    expect(maxContextTokensForVram(llama8b, q4, HIGH_END)).toBeGreaterThan(
      maxContextTokensForVram(llama8b, q4, CPU_ONLY),
    );
  });
});
