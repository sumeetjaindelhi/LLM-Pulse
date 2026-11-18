import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import chalk from "chalk";
import type { ModelEntry } from "../../src/core/types.js";

const libraryRow = vi.hoisted(() => ({ model: null as ModelEntry | null }));

vi.mock("../../src/hardware/index.js", async () => ({
  detectHardware: async () => (await import("../fixtures/hardware-profiles/high-end-nvidia.json")).default,
}));
vi.mock("../../src/models/database.js", () => ({ resolveModel: () => libraryRow.model }));
// No live Ollama or registry in unit tests: exact-fact lookups pass the model through.
vi.mock("../../src/runtimes/ollama.js", () => ({
  withExactFacts: async (_host: string, model: unknown) => model,
}));

import { quantAdviceCommand } from "../../src/cli/commands/quant-advice.js";

function row(quantName: string): ModelEntry {
  return {
    id: "lib",
    name: "Library Row",
    provider: "Test",
    parametersBillion: 8,
    contextWindow: 32768,
    categories: ["general"],
    qualityTier: "strong",
    qualityScore: 70,
    quantizations: [{ name: quantName, bitsPerWeight: 4.83, vramMb: 5000, qualityRetention: 1 }],
    ollamaTag: "lib:8b",
    releaseDate: "2025-06",
    source: "library",
  };
}

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

describe("quant-advice on quants without published quality data", () => {
  it.each(["default", "ternary"])("makes no quality claim for a %s quant", async (name) => {
    libraryRow.model = row(name);
    await quantAdviceCommand("lib", { format: "table", verbose: false });
    const out = stdout.join("\n");
    expect(out).toContain("quality n/a");
    expect(out).toContain(`Recommended: ${name}`);
    expect(out).not.toContain("%");
  });
});

describe("quant-advice json on a library row", () => {
  it("marks a sole default quant as not quality-measured", async () => {
    libraryRow.model = row("default");
    await quantAdviceCommand("lib", { format: "json", verbose: false });
    const out = JSON.parse(stdout.join("\n"));
    expect(out.recommended.qualityMeasured).toBe(false);
    expect(out.quantizations[0].qualityMeasured).toBe(false);
  });
});
