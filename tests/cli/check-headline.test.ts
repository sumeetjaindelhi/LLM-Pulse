import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import chalk from "chalk";
import type { ModelEntry } from "../../src/core/types.js";

vi.mock("../../src/hardware/index.js", async () => ({
  detectHardware: async () => (await import("../fixtures/hardware-profiles/high-end-nvidia.json")).default,
}));

// No live Ollama or registry in unit tests: exact-fact lookups pass the model through unless a test says otherwise.
const withExactFacts = vi.hoisted(() => vi.fn(async (_host: string, model: ModelEntry) => model));
vi.mock("../../src/runtimes/ollama.js", () => ({ withExactFacts }));

import { checkCommand } from "../../src/cli/commands/check.js";

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

const withoutMemory = ({ memory: _memory, ...rest }: Record<string, unknown>) => rest;

describe("check headline quantization", () => {
  // Rule: the headline is the sweet spot (F16 fits comfortably on 24 GiB), not the top composite (Q5_K_M).
  it("reports the sweet spot as bestQuantization and marks it recommended in CSV", async () => {
    await checkCommand("llama3.1:8b", { format: "json", verbose: false });
    const json = JSON.parse(stdout.join("\n"));
    expect(json.bestQuantization.name).toBe("F16");

    stdout = [];
    await checkCommand("llama3.1:8b", { format: "csv", verbose: false });
    const recommended = stdout.join("\n").split("\n").filter((line) => line.split(",")[7] === "true");
    expect(recommended).toHaveLength(1);
    expect(recommended[0].startsWith("F16,")).toBe(true);
  });
});

describe("check offload hint", () => {
  // llama3.1:70b fits no quant on 24 GiB; the headline is the best composite, Q5_K_M, but `ollama run llama3.1:70b`
  // loads Q4_K_M, so a num_gpu sized for Q5_K_M must not be offered for it.
  it("table: no run command for layers sized for a quant the tag does not install", async () => {
    await checkCommand("llama3.1:70b", { format: "table", verbose: false });
    const out = stdout.join("\n");
    expect(out).toMatch(/Put ≈\d+ of 80 layers on GPU/);
    expect(out).not.toContain("ollama run");
  });
});

// llama3.1:8b on the 24576 MiB fixture GPU: budget 24576 - 457 - 1024 = 23095 MiB (22.6 GiB); Ollama's default context
// is 32768. KV from arch: 32 x 8 x 256 x 2 B x 32768 = 4096 MiB (exact); compute 128256 x 1024 x 4 x 1.1 / 2^20 = 551.1.
// Q4_K_M is the file `ollama pull llama3.1:8b` installs (4693 MiB, exact); the other quants' weights are estimates.
describe("check memory provenance", () => {
  it("table: need per quant at the named default context, est. where estimated, breakdown in the headline", async () => {
    await checkCommand("llama3.1:8b", { format: "table", verbose: false });
    const out = stdout.join("\n");
    const ctx = (32768).toLocaleString();
    expect(out).toContain(`Need at ${ctx} tok`);
    // compute 551.1 + KQ mask 2 B x 32768 x 1024 (64) = 615.1. 4693 + 4096 + 615.1 = 9404.1 MiB, all exact but compute: no marker.
    // Q8_0: 8107 + 4711.1 = 12818.1 MiB.
    expect(out).toMatch(/Q4_K_M {2,}9\.2 GiB {2,}22\.6 GiB/);
    expect(out).toMatch(/Q8_0 {2,}12\.5 GiB est\. {2,}22\.6 GiB/);
    // F16: 15259 + 4096 + 615.1 = 19970.1 MiB.
    expect(out).toContain(`★ Recommended: F16 — 19.5 GiB est. at ${ctx} tokens`);
    expect(out).toContain("of 22.6 GiB usable");
    expect(out).toContain("weights 14.9 GiB est. · KV 4.0 GiB · compute ≈615 MiB");
    expect(out).toContain("Speed: rough (decode is bandwidth-bound); measure: llm-pulse benchmark");
    expect(out).not.toMatch(/VRAM|\bGB\b|\bMB\b/);
  });

  it("table: the pull line names the file it installs and where to find the headline quant's tag", async () => {
    await checkCommand("llama3.1:8b", { format: "table", verbose: false });
    let out = stdout.join("\n");
    expect(out).toContain("→ ollama pull llama3.1:8b  installs Q4_K_M (4.6 GiB)");
    expect(out).toContain("For F16, pick its tag at");
    expect(out).toContain("https://ollama.com/library/llama3.1/tags");

    stdout = [];
    await checkCommand("llama3.1:8b", { format: "table", verbose: false, quant: "Q4_K_M" });
    out = stdout.join("\n");
    expect(out).toContain("installs Q4_K_M (4.6 GiB)");
    expect(out).not.toContain("pick its tag");
  });

  it("table: names no file when the one `ollama pull` installs is unknown", async () => {
    withExactFacts.mockImplementationOnce(async (_host, model) => ({
      ...model,
      quantizations: model.quantizations.map(({ exact: _exact, ...quant }) => quant),
    }));
    await checkCommand("llama3.1:8b", { format: "table", verbose: false });
    const out = stdout.join("\n");
    expect(out).toMatch(/→ ollama pull llama3\.1:8b\s*│/);
    expect(out).not.toContain("installs");
    expect(out).not.toContain("pick its tag");
  });

  it("json: memory per quant at the default context; every pre-existing field keeps its value", async () => {
    await checkCommand("llama3.1:8b", { format: "json", verbose: false });
    const json = JSON.parse(stdout.join("\n"));
    // check --format json before `memory` was added, except gpuOffload.ollamaCommand: it was "ollama run llama3.1:8b",
    // which loads Q4_K_M, not the F16 the suggestion is sized for.
    expect({
      ...json,
      bestQuantization: withoutMemory(json.bestQuantization),
      allQuantizations: json.allQuantizations.map(withoutMemory),
    }).toEqual({
      model: { id: "llama-3.1-8b", name: "Llama 3.1 8B", provider: "Meta", parametersBillion: 8, contextWindow: 131072, contextWindowEstimated: false, qualityTier: "strong", ollamaTag: "llama3.1:8b" },
      hardware: { gpu: "GeForce RTX 4090", gpuCount: 1, vramMb: 23095, ramMb: 65536 },
      verdict: "yes",
      bestQuantization: { name: "F16", bitsPerWeight: 16.0005, vramMb: 15259, fitLevel: "comfortable", fitRatio: 1.16, compositeScore: 87, speedEstimate: "moderate" },
      allQuantizations: [
        { name: "Q4_K_M", vramNeeded: 4693, vramAvailable: 23095, fitLevel: "excellent", compositeScore: 88, speedEstimate: "fast" },
        { name: "Q5_K_M", vramNeeded: 5439, vramAvailable: 23095, fitLevel: "excellent", compositeScore: 89, speedEstimate: "fast" },
        { name: "Q8_0", vramNeeded: 8107, vramAvailable: 23095, fitLevel: "excellent", compositeScore: 89, speedEstimate: "moderate" },
        { name: "F16", vramNeeded: 15259, vramAvailable: 23095, fitLevel: "comfortable", compositeScore: 87, speedEstimate: "moderate" },
      ],
      pullCommand: "ollama pull llama3.1:8b",
      // The tag installs Q4_K_M, the one exact file, not the F16 best above.
      pullQuantization: "Q4_K_M",
      gpuOffload: { gpuLayers: "all", totalLayers: 32, estimatedVramUsedMb: 15259, reason: "full_fit", ollamaCommand: null },
    });
    expect(json.bestQuantization.memory).toEqual({
      contextTokens: 32768, weightsMb: 15259, weightsExact: false, kvCacheMb: 4096, kvExact: true,
      computeMb: expect.closeTo(615.1, 6), totalMb: expect.closeTo(19970.1, 6), budgetMb: 23095, budgetExact: false,
    });
    expect(json.allQuantizations.map((q: { memory: { weightsExact: boolean } }) => q.memory.weightsExact)).toEqual([true, false, false, false]);
  });

  it("csv: memory columns appended after the existing ones, which keep their values", async () => {
    await checkCommand("llama3.1:8b", { format: "csv", verbose: false });
    const [header, ...rows] = stdout.join("\n").split("\n");
    expect(header).toBe(
      "quantization,bitsPerWeight,vramNeeded,vramAvailable,fitLevel,compositeScore,speedEstimate,isRecommended," +
      "contextTokens,weightsMb,weightsExact,kvCacheMb,kvExact,computeMb,totalMb,budgetMb,budgetExact",
    );
    expect(rows.map((row) => row.split(",").slice(0, 8).join(","))).toEqual([
      "Q4_K_M,4.8944,4693,23095,excellent,88,fast,false",
      "Q5_K_M,5.7036,5439,23095,excellent,89,fast,false",
      "Q8_0,8.5008,8107,23095,excellent,89,moderate,false",
      "F16,16.0005,15259,23095,comfortable,87,moderate,true",
    ]);
    expect(rows[0].split(",").slice(8).join(",")).toBe("32768,4693,true,4096,true,615.1,9404.1,23095,false");
  });
});
