import { describe, it, expect, vi, afterEach } from "vitest";

vi.mock("../../src/hardware/index.js", async () => ({
  detectHardware: async () => (await import("../fixtures/hardware-profiles/high-end-nvidia.json")).default,
}));

import { contextFitCommand, toContextFitCsv } from "../../src/cli/commands/context-fit.js";
import type { ContextFitResult } from "../../src/core/types.js";

describe("contextFitCommand exit code", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    process.exitCode = undefined;
  });

  it("json: an unknown --quant prints the error payload and exits 1", async () => {
    const stdout: string[] = [];
    vi.spyOn(console, "log").mockImplementation((...args) => { stdout.push(args.join(" ")); });
    await contextFitCommand("llama3.1:8b", { promptTokens: 1000, responseTokens: 0, quant: "Q9_Z", format: "json" });
    expect(JSON.parse(stdout.join("\n")).availableQuantizations).toContain("Q4_K_M");
    expect(process.exitCode).toBe(1);
  });

  it("leaves the exit code alone on success", async () => {
    vi.spyOn(console, "log").mockImplementation(() => {});
    await contextFitCommand("llama3.1:8b", { promptTokens: 1000, responseTokens: 0, format: "json" });
    expect(process.exitCode).toBeUndefined();
  });
});

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
