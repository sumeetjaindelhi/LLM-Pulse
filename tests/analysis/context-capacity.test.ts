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
  const withParams = (parametersBillion: number): ModelEntry => ({ ...llama8b, parametersBillion });

  it("maps param-count buckets to fp16 KV-cache MB per 1k tokens", () => {
    expect(kvCacheMbPer1kTokens(withParams(1.5))).toBe(40);
    expect(kvCacheMbPer1kTokens(withParams(4))).toBe(80);
    expect(kvCacheMbPer1kTokens(withParams(8))).toBe(150);
    expect(kvCacheMbPer1kTokens(withParams(15))).toBe(210);
    expect(kvCacheMbPer1kTokens(withParams(22))).toBe(280);
    expect(kvCacheMbPer1kTokens(withParams(35))).toBe(360);
    expect(kvCacheMbPer1kTokens(withParams(80))).toBe(440);
    expect(kvCacheMbPer1kTokens(withParams(81))).toBe(600);
  });

  it("uses the model's architecture-derived kvMbPer1kTokens over the bucket", () => {
    expect(kvCacheMbPer1kTokens({ ...withParams(3.8), kvMbPer1kTokens: 375 })).toBe(375);
  });

  it("carries real KV sizes for curated non-GQA models the bucket underestimates", () => {
    // phi-3-mini: 32 layers x 32 KV heads x 96 head_dim x 4 bytes = 375 MiB / 1k tokens
    expect(kvCacheMbPer1kTokens(getModelById("phi-3-mini")!)).toBeGreaterThanOrEqual(375);
    // codellama-7b: 32 x 32 x 128 x 4 bytes = 500 MiB / 1k tokens
    expect(kvCacheMbPer1kTokens(getModelById("codellama-7b")!)).toBeGreaterThanOrEqual(500);
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
