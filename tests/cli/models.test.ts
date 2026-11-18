import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import chalk from "chalk";
import { modelsCommand } from "../../src/cli/commands/models.js";
import { recommendationTable, comparisonTable } from "../../src/cli/ui/tables.js";
import { getRecommendations } from "../../src/analysis/recommender.js";
import { scoreModel } from "../../src/analysis/scorer.js";
import { getModelById } from "../../src/models/database.js";
import type { HardwareProfile } from "../../src/core/types.js";
import highEnd from "../fixtures/hardware-profiles/high-end-nvidia.json";

const HIGH_END = highEnd as HardwareProfile;

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

const baseOptions = { category: "all" as const, fits: false, live: false, installed: false, format: "json" };

describe("models --live --library while offline", () => {
  afterEach(() => { vi.unstubAllEnvs(); });

  it("says the live library is off and never contacts ollama.com", async () => {
    vi.stubEnv("LLMPULSE_OFFLINE", "1");
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("no network"));
    vi.spyOn(console, "error").mockImplementation((...args) => { stdout.push(args.join(" ")); });
    await modelsCommand({ ...baseOptions, live: true, library: true, format: "table" });
    expect(stdout.join("\n")).toContain("live library is off (LLMPULSE_OFFLINE or offline config)");
    expect(fetchSpy.mock.calls.map(([url]) => String(url)).filter((url) => url.includes("ollama.com"))).toEqual([]);
  });

  it("keeps json stdout parseable: the note goes to stderr", async () => {
    vi.stubEnv("LLMPULSE_OFFLINE", "1");
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("no network"));
    const stderr: string[] = [];
    vi.spyOn(console, "error").mockImplementation((...args) => { stderr.push(args.join(" ")); });
    await modelsCommand({ ...baseOptions, live: true, library: true, format: "json" });
    expect(() => JSON.parse(stdout.join("\n"))).not.toThrow();
    expect(stderr.join("\n")).toContain("live library is off");
  });
});

describe("models --search", () => {
  it("also applies --category in curated mode", async () => {
    await modelsCommand({ ...baseOptions, search: "llama", category: "coding" });
    const models: { id: string; categories: string[] }[] = JSON.parse(stdout.join("\n"));
    expect(models.length).toBeGreaterThan(0);
    for (const m of models) expect(m.categories).toContain("coding");
  });

  it("without --category returns every search match", async () => {
    await modelsCommand({ ...baseOptions, search: "llama" });
    const all: unknown[] = JSON.parse(stdout.join("\n"));
    stdout = [];
    await modelsCommand({ ...baseOptions, search: "llama", category: "coding" });
    expect(all.length).toBeGreaterThan(JSON.parse(stdout.join("\n")).length);
  });
});

describe("tables without colour", () => {
  it("models table emits no ANSI escape codes", async () => {
    await modelsCommand({ ...baseOptions, format: "table" });
    expect(stdout.join("\n")).not.toContain("[");
  });

  it("recommendation and comparison tables emit no ANSI escape codes", () => {
    const recs = getRecommendations(HIGH_END, { category: "all", top: 3, onlyFitting: true });
    expect(recommendationTable(recs, HIGH_END, new Set())).not.toContain("[");
    expect(comparisonTable(recs.map((r) => r.score), HIGH_END, 0)).not.toContain("[");
  });
});

describe("memory labels", () => {
  it("models: per-quant sizes are weights, est. unless they are the file `ollama pull` installs", async () => {
    await modelsCommand({ ...baseOptions, format: "table" });
    const out = stdout.join("\n");
    expect(out).toContain("Weights");
    expect(out).not.toContain("VRAM");
    // Llama 3.2 1B lists Q4_K_M first, but its pulled file is Q8_0: 1e9 x 4.8944 / 8 / 2^20 = 583 MiB is an estimate.
    expect(out).toMatch(/Llama 3\.2 1B .*Q4_K_M {2,}583 MiB est\./);
    expect(out).toMatch(/Llama 3\.1 8B .*Q4_K_M {2,}4\.6 GiB {2,}—/);
  });

  // llama3.1:8b on the 24576 MiB fixture GPU at Ollama's default 32768 tokens: Q4_K_M (pulled file) 9340.1 MiB,
  // Q8_0 (estimated weights) 12754.1 MiB; usable 23095 MiB.
  it("scan and compare tables: need at Ollama's default context, est. where estimated, speed called rough", () => {
    const llama = getModelById("llama-3.1-8b")!;
    const score = (name: string) => scoreModel(llama, llama.quantizations.find((q) => q.name === name)!, HIGH_END);
    const recs = [score("Q4_K_M"), score("Q8_0")].map((s, i) => ({ rank: i + 1, score: s, pullCommand: null }));
    for (const table of [recommendationTable(recs, HIGH_END), comparisonTable(recs.map((r) => r.score), HIGH_END, 0)]) {
      expect(table).toMatch(/9\.2 GiB {2,}/); // 4693 + 4096 + 615.1 = 9404.1 MiB
      expect(table).toContain("12.5 GiB est.");
      expect(table).toContain("Need: weights + KV cache + compute (≈) at Ollama's default context");
      expect(table).toContain("Speed: rough (decode is bandwidth-bound); measure: llm-pulse benchmark");
      expect(table).not.toMatch(/VRAM|\bGB\b/);
    }
    expect(comparisonTable(recs.map((r) => r.score), HIGH_END, 0)).toContain("Usable memory: 22.6 GiB");
  });
});

describe("models weightsExact", () => {
  it("json: true only when the listed quant is exact", async () => {
    await modelsCommand({ ...baseOptions, search: "llama" });
    const models: { id: string; quantization: string; weightsExact: boolean }[] = JSON.parse(stdout.join("\n"));
    expect(models.some((m) => m.weightsExact)).toBe(true);
    for (const m of models) {
      const quant = getModelById(m.id)!.quantizations.find((q) => q.name === m.quantization)!;
      expect(m.weightsExact).toBe(quant.exact === true);
    }
  });

  it("csv: weightsExact is the last column", async () => {
    await modelsCommand({ ...baseOptions, search: "llama", format: "csv" });
    expect(stdout.join("\n").split("\n")[0]).toBe("id,name,provider,parametersBillion,categories,quantization,vramMb,ollamaTag,weightsExact");
  });
});
