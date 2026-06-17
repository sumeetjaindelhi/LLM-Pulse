import { describe, it, expect } from "vitest";
import { toContextFitCsv } from "../../src/cli/commands/context-fit.js";
import type { ContextFitResult } from "../../src/core/types.js";

function result(over: Partial<ContextFitResult> = {}): ContextFitResult {
  return {
    model: { id: "llama-3.1-8b", name: "Llama 3.1 8B", parametersBillion: 8, contextWindow: 131072 },
    quant: { name: "Q5_K_M", vramMb: 5800 },
    quantForced: false,
    promptTokens: 50000,
    responseTokens: 1024,
    neededTokens: 51024,
    nativeWindow: 131072,
    affordedByVramTokens: 16384,
    effectiveMaxTokens: 16384,
    limitedBy: "hardware",
    verdict: "no",
    headroomTokens: -34640,
    remedy: "Switch to Q4_K_M — affords 24,576 tokens on this hardware.",
    ...over,
  };
}

describe("toContextFitCsv", () => {
  it("emits a header row and exactly one data row", () => {
    const rows = toContextFitCsv(result()).split("\n");
    expect(rows).toHaveLength(2);
    expect(rows[0]).toBe(
      "model,quant,quantForced,promptTokens,responseTokens,neededTokens,nativeWindow,affordedByVramTokens,effectiveMaxTokens,limitedBy,verdict,headroomTokens,remedy",
    );
  });

  it("quotes the remedy cell because it contains commas", () => {
    const csv = toContextFitCsv(result());
    expect(csv).toContain('"Switch to Q4_K_M — affords 24,576 tokens on this hardware."');
  });

  it("renders a null remedy as a trailing empty cell", () => {
    const csv = toContextFitCsv(result({ verdict: "yes", remedy: null }));
    expect(csv.split("\n")[1].endsWith(",")).toBe(true);
  });
});
