import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import chalk from "chalk";

vi.mock("../../src/hardware/index.js", async () => ({
  detectHardware: async () => (await import("../fixtures/hardware-profiles/apple-m2.json")).default,
}));

// No live Ollama or registry in unit tests: exact-fact lookups pass the model through unless a test says otherwise.
const withExactFacts = vi.hoisted(() => vi.fn(async (_host: string, model: ModelEntry) => model));
vi.mock("../../src/runtimes/ollama.js", () => ({ withExactFacts }));

import { checkCommand } from "../../src/cli/commands/check.js";
import { headlineScore } from "../../src/analysis/scorer.js";
import { getModelById } from "../../src/models/database.js";
import { memoryNeedMb } from "../../src/analysis/context-capacity.js";
import { MEMORY_CSV_HEADERS } from "../../src/cli/ui/csv.js";
import type { ModelEntry, ModelScore } from "../../src/core/types.js";
import { quantAdviceCommand } from "../../src/cli/commands/quant-advice.js";
import { EXACT_Q8_MB, withExactQ8 } from "../fixtures/exact-facts.js";

let stdout: string[];
const originalChalkLevel = chalk.level;

beforeEach(() => {
  stdout = [];
  // Theme colours off, as in a pipe, so any escape code left comes from cli-table3.
  chalk.level = 0;
  vi.spyOn(console, "log").mockImplementation((...args) => { stdout.push(args.join(" ")); });
});

afterEach(() => {
  vi.restoreAllMocks();
  chalk.level = originalChalkLevel;
});

describe("check table on Apple Silicon", () => {
  it("describes unified memory without a stale percentage", async () => {
    await checkCommand("llama3.1:8b", { format: "table", verbose: false });
    const out = stdout.join("\n");
    expect(out).toContain("(unified memory — shared with system RAM)");
    expect(out).not.toContain("75% usable");
  });

  it("emits no ANSI escape codes when colours are off", async () => {
    await checkCommand("llama3.1:8b", { format: "table", verbose: false });
    expect(stdout.join("\n")).not.toContain("[");
  });
});

describe("quant-advice table", () => {
  it("emits no ANSI escape codes when colours are off", async () => {
    await quantAdviceCommand("llama3.1:8b", { format: "table", verbose: false });
    expect(stdout.join("\n")).not.toContain("[");
  });

  it("labels sizes as weights, est. unless they are the pulled file, against the usable budget", async () => {
    // Apple M2 Pro: budget 10977 - 1024 = 9953 MiB (9.7 GiB). Q4_K_M is the pulled file (4693 MiB);
    // Q8_0's 8107 MiB is params x bits per weight. The sweet spot is Q8_0.
    await quantAdviceCommand("llama3.1:8b", { format: "table", verbose: false });
    const out = stdout.join("\n");
    expect(out).toContain("Weights");
    expect(out).toMatch(/Q4_K_M {2,}4\.9 {2,}4\.6 GiB {2,}/);
    expect(out).toMatch(/Q8_0 {2,}8\.5 {2,}7\.9 GiB est\./);
    expect(out).toContain("Usable memory: 9.7 GiB");
    expect(out).toContain("Speed: rough (decode is bandwidth-bound); measure: llm-pulse benchmark");
    expect(out).toContain("★ Recommended: Q8_0 — 99.8% quality, weights 7.9 GiB est.");
    expect(out).not.toMatch(/\bGB\b/);
  });

  it("names the file `ollama pull` installs and where the sweet spot's tag is", async () => {
    await quantAdviceCommand("llama3.1:8b", { format: "table", verbose: false });
    const out = stdout.join("\n");
    expect(out).toContain("→ ollama pull llama3.1:8b  installs Q4_K_M (4.6 GiB)");
    expect(out).toContain("For Q8_0, pick its tag at");
    expect(out).toContain("https://ollama.com/library/llama3.1/tags");
  });
});

describe("exact model facts", () => {
  it("check looks them up on --host and matches --quant on the result", async () => {
    withExactFacts.mockImplementationOnce(withExactQ8);
    await checkCommand("llama3.1:8b", { format: "json", verbose: false, quant: "Q8_0", host: "http://127.0.0.1:11500" });
    expect(withExactFacts).toHaveBeenLastCalledWith("http://127.0.0.1:11500", expect.objectContaining({ id: "llama-3.1-8b" }));
    expect(JSON.parse(stdout.join("\n")).bestQuantization.vramMb).toBe(EXACT_Q8_MB);
  });

  it("quant-advice sizes the quants of the model with exact facts", async () => {
    withExactFacts.mockImplementationOnce(withExactQ8);
    await quantAdviceCommand("llama3.1:8b", { format: "json", verbose: false });
    const q8 = JSON.parse(stdout.join("\n")).quantizations.find((q: { name: string }) => q.name === "Q8_0");
    expect(q8.vramMb).toBe(EXACT_Q8_MB);
  });

  it("json: each quant says whether its quality is measured and carries its memory breakdown", async () => {
    withExactFacts.mockImplementationOnce(withExactQ8);
    await quantAdviceCommand("llama3.1:8b", { format: "json", verbose: false });
    const out = JSON.parse(stdout.join("\n"));
    const model = await withExactQ8("", getModelById("llama-3.1-8b")!);
    for (const q of [...out.quantizations, out.recommended]) {
      expect(q.qualityMeasured).toBe(true);
      // totalMb is rounded to 0.1 MiB (memoryBreakdown)
      expect(q.memory.totalMb).toBeCloseTo(memoryNeedMb(model, model.quantizations.find((v) => v.name === q.name)!, q.memory.contextTokens), 0);
    }
    const q8 = out.quantizations.find((q: { name: string }) => q.name === "Q8_0");
    expect(q8.memory.weightsExact).toBe(true);
    expect(out.quantizations.find((q: { name: string }) => q.name === "Q4_K_M").memory.weightsExact).toBe(false);
  });

  it("csv: memory columns are appended after the existing ones", async () => {
    await quantAdviceCommand("llama3.1:8b", { format: "csv", verbose: false });
    const header = stdout.join("\n").split("\n")[0];
    expect(header).toBe("quantization,bitsPerWeight,vramMb,qualityRetention,vramAvailable,fitLevel,speedEstimate,isRecommended,qualityMeasured," + MEMORY_CSV_HEADERS.join(","));
  });
});

describe("headlineScore", () => {
  const base = getModelById("llama-3.1-8b")!;
  const score = (name: string, retention: number, bpw: number, composite: number, fitLevel: ModelScore["fitLevel"]): ModelScore => ({
    model: base,
    quantization: { name, bitsPerWeight: bpw, vramMb: 5000, qualityRetention: retention },
    fitLevel,
    fitRatio: 1.5,
    compositeScore: composite,
    speedEstimate: "moderate",
  });

  it("picks the sweet spot (most quality that fits comfortably), not the top composite", () => {
    const q8 = score("Q8_0", 0.99, 8.5, 70, "comfortable");
    const q4 = score("Q4_K_M", 0.95, 4.8, 90, "excellent");
    expect(headlineScore([q4, q8])).toBe(q8);
  });

  it("falls back to the best composite when nothing fits", () => {
    const a = score("Q8_0", 0.99, 8.5, 20, "cannot_run");
    const b = score("Q4_K_M", 0.95, 4.8, 30, "cannot_run");
    expect(headlineScore([a, b])).toBe(b);
  });
});
