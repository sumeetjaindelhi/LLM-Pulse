import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import chalk from "chalk";

vi.mock("../../src/hardware/index.js", async () => ({
  detectHardware: async () => (await import("../fixtures/hardware-profiles/high-end-nvidia.json")).default,
}));

// No live Ollama or registry in unit tests: exact-fact lookups pass the model through unless a test says otherwise.
const withExactFacts = vi.hoisted(() => vi.fn(async (_host: string, model: ModelEntry) => model));
vi.mock("../../src/runtimes/ollama.js", () => ({ withExactFacts }));

import { contextFitCommand, toContextFitCsv } from "../../src/cli/commands/context-fit.js";
import type { ContextFitResult, ModelEntry } from "../../src/core/types.js";
import { EXACT_Q8_MB, withExactQ8 } from "../fixtures/exact-facts.js";

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

  it("json: --quant is matched on the model with exact facts", async () => {
    const stdout: string[] = [];
    vi.spyOn(console, "log").mockImplementation((...args) => { stdout.push(args.join(" ")); });
    withExactFacts.mockImplementationOnce(withExactQ8);
    await contextFitCommand("llama3.1:8b", { promptTokens: 1000, responseTokens: 0, quant: "Q8_0", format: "json" });
    expect(JSON.parse(stdout.join("\n")).quant.vramMb).toBe(EXACT_Q8_MB);
  });
});

// 50,000 + 512 tokens of llama3.1:8b on the 24576 MiB fixture GPU (budget 23095 MiB, 22.6 GiB), sweet spot F16
// (15259 MiB, an estimate). At 50512 tokens: KV from arch 32 x 8 x 256 x 2 B x 50688 (padded) = 6336 MiB;
// batch 2048, compute 128256 x 2048 x 4 x 1.1 / 2^20 = 1102.2; total 22697.2 MiB.
describe("context-fit memory at the workload's needed tokens", () => {
  const workload = { promptTokens: 50000, responseTokens: 512 };
  let stdout: string[];
  const originalChalkLevel = chalk.level;

  beforeEach(() => {
    stdout = [];
    chalk.level = 0;
    vi.spyOn(console, "log").mockImplementation((...args) => { stdout.push(args.join(" ")); });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    chalk.level = originalChalkLevel;
  });

  it("json: adds memory; every pre-existing field keeps its value", async () => {
    await contextFitCommand("llama3.1:8b", { ...workload, format: "json" });
    const { memory, ...json } = JSON.parse(stdout.join("\n"));
    // context-fit --format json before `memory` was added. Above 32768 tokens the batch is 2048: compute 1102.2 and a KQ mask of
    // 2 B x 2048 per context cell, so F16 affords (23095 - 15259 - 1102.2) / (0.125 + 0.00390625) = 52240 -> 52224;
    // Q8_0 (8107) affords (23095 - 8107 - 1102.2) / 0.12890625 = 107722 -> 107520.
    expect(json).toEqual({
      model: { id: "llama-3.1-8b", name: "Llama 3.1 8B", parametersBillion: 8, contextWindow: 131072 },
      quant: { name: "F16", vramMb: 15259 },
      quantForced: false,
      promptTokens: 50000,
      responseTokens: 512,
      neededTokens: 50512,
      nativeWindow: 131072,
      affordedByVramTokens: 52224,
      effectiveMaxTokens: 52224,
      limitedBy: "hardware",
      verdict: "tight",
      headroomTokens: 1712,
      remedy: `Switch to Q8_0 — affords ${(107520).toLocaleString()} tokens on this hardware.`,
    });
    // compute 1102.2 + mask 2 B x 50688 x 2048 (198) = 1300.2; total 15259 + 6336 + 1300.2 = 22895.2
    expect(memory).toEqual({
      contextTokens: 50512, weightsMb: 15259, weightsExact: false, kvCacheMb: 6336, kvExact: true,
      computeMb: expect.closeTo(1300.2, 6), totalMb: expect.closeTo(22895.2, 6), budgetMb: 23095, budgetExact: false,
    });
  });

  it("csv: memory cells appended after the existing ones, which keep their values", async () => {
    await contextFitCommand("llama3.1:8b", { ...workload, format: "csv" });
    const row = stdout.join("\n").split("\n")[1];
    // The remedy cell between them is quoted only when the locale groups digits with commas.
    expect(row.startsWith("llama-3.1-8b,F16,false,50000,512,50512,131072,52224,52224,hardware,tight,1712,")).toBe(true);
    expect(row).toContain(`Switch to Q8_0 — affords ${(107520).toLocaleString()} tokens on this hardware.`);
    expect(row.endsWith(",50512,15259,false,6336,true,1300.2,22895.2,23095,false")).toBe(true);
  });

  it("table: breakdown at the needed tokens", async () => {
    await contextFitCommand("llama3.1:8b", { ...workload, format: "table" });
    const out = stdout.join("\n");
    expect(out).toContain(`22.4 GiB est. at ${(50512).toLocaleString()} tokens`);
    expect(out).toContain("of 22.6 GiB usable est.");
    expect(out).toContain("weights 14.9 GiB est. · KV 6.2 GiB · compute ≈1.3 GiB");
  });

  // Labels wider than keyValue's 6-column pad ran into their values ("Native window131,072 tok").
  it("table: ceiling labels are separated from their values and aligned", async () => {
    await contextFitCommand("llama3.1:8b", { ...workload, format: "table" });
    const out = stdout.join("\n");
    const at = (label: string) => new RegExp(`${label} +\\d`).exec(out);
    for (const label of ["Native window", "Hardware-afforded", "Effective max"]) expect(at(label), label).not.toBeNull();
    const columns = ["Native window", "Hardware-afforded", "Effective max"].map((label) => {
      const line = out.split("\n").find((l) => l.includes(label))!;
      return line.search(/\d/);
    });
    expect(new Set(columns).size).toBe(1);
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
    memory: {
      contextTokens: 51024, weightsMb: 5800, weightsExact: false, kvCacheMb: 6400, kvExact: true,
      computeMb: 1102.2, totalMb: 13302.2, budgetMb: 23095, budgetExact: false,
    },
    ...over,
  };
}

describe("toContextFitCsv", () => {
  it("emits a header row and exactly one data row", () => {
    const rows = toContextFitCsv(result()).split("\n");
    expect(rows).toHaveLength(2);
    // The memory columns follow the pre-existing ones (CSV is additive only).
    expect(rows[0]).toBe(
      "model,quant,quantForced,promptTokens,responseTokens,neededTokens,nativeWindow,affordedByVramTokens,effectiveMaxTokens,limitedBy,verdict,headroomTokens,remedy," +
      "contextTokens,weightsMb,weightsExact,kvCacheMb,kvExact,computeMb,totalMb,budgetMb,budgetExact",
    );
  });

  it("quotes the remedy cell because it contains commas", () => {
    const csv = toContextFitCsv(result());
    expect(csv).toContain('"Switch to Q4_K_M — affords 24,576 tokens on this hardware."');
  });

  it("renders a null remedy as an empty cell", () => {
    const csv = toContextFitCsv(result({ verdict: "yes", remedy: null }));
    expect(csv.split("\n")[1].split(",")[12]).toBe("");
  });
});
