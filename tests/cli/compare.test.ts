import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import chalk from "chalk";

vi.mock("../../src/hardware/index.js", async () => ({
  detectHardware: async () => (await import("../fixtures/hardware-profiles/high-end-nvidia.json")).default,
}));

// No live Ollama or registry in unit tests: exact-fact lookups pass the model through unless a test says otherwise.
const withExactFacts = vi.hoisted(() => vi.fn(async (_host: string, model: ModelEntry) => model));
vi.mock("../../src/runtimes/ollama.js", () => ({ withExactFacts }));

import { compareCommand } from "../../src/cli/commands/compare.js";
import type { CompareOptions, ModelEntry } from "../../src/core/types.js";
import { getModelById } from "../../src/models/database.js";
import { memoryNeedMb } from "../../src/analysis/context-capacity.js";
import { MEMORY_CSV_HEADERS } from "../../src/cli/ui/csv.js";
import { EXACT_Q8_MB, withExactQ8 } from "../fixtures/exact-facts.js";

let stdout: string[];
let stderr: string[];

beforeEach(() => {
  stdout = [];
  stderr = [];
  vi.spyOn(console, "log").mockImplementation((...args) => { stdout.push(args.join(" ")); });
  vi.spyOn(console, "error").mockImplementation((...args) => { stderr.push(args.join(" ")); });
  process.exitCode = undefined;
});

afterEach(() => {
  vi.restoreAllMocks();
  process.exitCode = undefined;
});

function options(over: Partial<CompareOptions> = {}): CompareOptions {
  return { format: "json", category: "all", top: 3, ...over };
}

describe("compare", () => {
  it("matches --quant case-insensitively", async () => {
    await compareCommand(["llama3.1:8b", "qwen2.5:7b"], options({ quant: "q4_k_m" }));
    const output = JSON.parse(stdout.join("\n"));
    expect(output.models.map((m: { quantization: string }) => m.quantization)).toEqual(["Q4_K_M", "Q4_K_M"]);
    expect(stderr).toEqual([]);
  });

  it("compares each model with its exact facts, looked up on --host, and matches --quant on them", async () => {
    withExactFacts.mockClear().mockImplementationOnce(withExactQ8).mockImplementationOnce(withExactQ8);
    await compareCommand(["llama3.1:8b", "qwen2.5:7b"], options({ quant: "Q8_0", host: "http://127.0.0.1:11500" }));
    expect(JSON.parse(stdout.join("\n")).models.map((m: { vramMb: number }) => m.vramMb)).toEqual([EXACT_Q8_MB, EXACT_Q8_MB]);
    expect(withExactFacts.mock.calls.map(([host]) => host)).toEqual(["http://127.0.0.1:11500", "http://127.0.0.1:11500"]);
  });

  it("json: an unavailable --quant warns on stderr and stdout stays parseable", async () => {
    await compareCommand(["llama3.1:8b", "qwen2.5:7b"], options({ quant: "Q9_Z" }));
    expect(() => JSON.parse(stdout.join("\n"))).not.toThrow();
    expect(stderr.join("\n")).toContain("quant Q9_Z not available, using best fit");
  });

  it("json: an unknown model is reported on stderr, stdout stays parseable, exit code 1", async () => {
    await compareCommand(["llama3.1:8b", "nosuchmodelxyz", "qwen2.5:7b"], options());
    expect(JSON.parse(stdout.join("\n")).models).toHaveLength(2);
    expect(stderr.join("\n")).toContain("Model not found: nosuchmodelxyz");
    expect(process.exitCode).toBe(1);
  });

  it("csv: diagnostics stay off stdout", async () => {
    await compareCommand(["llama3.1:8b", "nosuchmodelxyz", "qwen2.5:7b"], options({ format: "csv", quant: "Q9_Z" }));
    expect(stdout.join("\n").startsWith("id,name,")).toBe(true);
  });

  it("json: fewer than 2 models prints the error payload and exits 1", async () => {
    await compareCommand(["llama3.1:8b"], options());
    expect(JSON.parse(stdout.join("\n"))).toEqual({ error: "Need at least 2 models to compare" });
    expect(process.exitCode).toBe(1);
  });

  it("csv: fewer than 2 models writes the error to stderr only", async () => {
    await compareCommand(["llama3.1:8b"], options({ format: "csv" }));
    expect(stdout).toEqual([]);
    expect(stderr.join("\n")).toContain("Need at least 2 models to compare");
    expect(process.exitCode).toBe(1);
  });

  it("leaves the exit code alone on success", async () => {
    await compareCommand(["llama3.1:8b", "qwen2.5:7b"], options());
    expect(process.exitCode).toBeUndefined();
  });

  // Both score 87; Qwen 2.5 7B wins on fit at Q8_0, but `ollama pull qwen2.5:7b` installs its Q4_K_M file (4466 MiB).
  it("table: the winner's pull line names the file it installs and where the compared quant's tag is", async () => {
    const originalChalkLevel = chalk.level;
    chalk.level = 0;
    try {
      await compareCommand(["llama3.1:8b", "qwen2.5:7b"], options({ format: "table" }));
    } finally {
      chalk.level = originalChalkLevel;
    }
    const out = stdout.join("\n");
    expect(out).toContain("Winner: Qwen 2.5 7B");
    expect(out).toContain("→ ollama pull qwen2.5:7b  installs Q4_K_M (4.4 GiB)");
    expect(out).toContain("For Q8_0, pick its tag at");
    expect(out).toContain("https://ollama.com/library/qwen2.5/tags");
  });
});

describe("compare memory", () => {
  it("json: each model carries the memory breakdown of its headline quant", async () => {
    await compareCommand(["llama3.1:8b", "qwen2.5:7b"], options({ quant: "Q4_K_M" }));
    const { models } = JSON.parse(stdout.join("\n"));
    for (const m of models) {
      const model = getModelById(m.id)!;
      const quant = model.quantizations.find((q) => q.name === m.quantization)!;
      expect(m.memory.weightsMb).toBe(m.vramMb);
      expect(m.memory.totalMb).toBe(memoryNeedMb(model, quant, m.memory.contextTokens));
    }
  });

  it("csv: memory columns are appended after the existing ones", async () => {
    await compareCommand(["llama3.1:8b", "qwen2.5:7b"], options({ format: "csv" }));
    const header = stdout.join("\n").split("\n")[0];
    expect(header).toBe("id,name,provider,parametersBillion,contextWindow,qualityTier,quantization,vramMb,availableVramMb,fitLevel,compositeScore,speedEstimate,isWinner," + MEMORY_CSV_HEADERS.join(",") + ",contextWindowEstimated");
  });
});
