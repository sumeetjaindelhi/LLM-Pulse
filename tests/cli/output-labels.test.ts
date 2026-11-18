import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import chalk from "chalk";
import type { HardwareProfile, ModelEntry } from "../../src/core/types.js";
import highEnd from "../fixtures/hardware-profiles/high-end-nvidia.json";

const state = vi.hoisted(() => ({ hardware: null as unknown, library: null as unknown, curated: null as unknown }));

vi.mock("../../src/hardware/index.js", async () => ({ detectHardware: async () => state.hardware }));
vi.mock("../../src/runtimes/ollama.js", () => ({ withExactFacts: async (_host: string, model: unknown) => model }));
// The library row stands for a catalog entry whose context window is a guess; every other id resolves to the curated data.
vi.mock("../../src/models/database.js", async (importOriginal) => {
  const real = await importOriginal<typeof import("../../src/models/database.js")>();
  return {
    ...real,
    resolveModel: (arg: string) => (arg === "lib" ? state.library : real.resolveModel(arg)),
    filterByCategory: () => [state.library],
  };
});

import { checkCommand } from "../../src/cli/commands/check.js";
import { compareCommand } from "../../src/cli/commands/compare.js";
import { modelsCommand } from "../../src/cli/commands/models.js";
import { quantAdviceCommand } from "../../src/cli/commands/quant-advice.js";
import { optimizeCommand } from "../../src/cli/commands/optimize.js";

const HIGH_END = highEnd as HardwareProfile;
const TWO_GPUS: HardwareProfile = { ...HIGH_END, gpus: [HIGH_END.gpus[0], HIGH_END.gpus[0]] };

const libraryRow: ModelEntry = {
  id: "lib",
  name: "Library Row",
  provider: "Test",
  parametersBillion: 8,
  contextWindow: 8192,
  contextWindowEstimated: true,
  categories: ["general"],
  qualityTier: "strong",
  qualityScore: 70,
  quantizations: [{ name: "default", bitsPerWeight: 5, vramMb: 5000, qualityRetention: 1 }],
  ollamaTag: "lib:8b",
  releaseDate: "2025-06",
  source: "library",
};

let stdout: string[];
const originalChalkLevel = chalk.level;

beforeEach(() => {
  stdout = [];
  chalk.level = 0;
  state.hardware = HIGH_END;
  state.library = libraryRow;
  vi.spyOn(console, "log").mockImplementation((...args) => { stdout.push(args.join(" ")); });
});

afterEach(() => {
  vi.restoreAllMocks();
  chalk.level = originalChalkLevel;
});

describe("a guessed context window", () => {
  it("check: table says est., json carries the flag, and no pulled file is named", async () => {
    await checkCommand("lib", { format: "table", verbose: false });
    expect(stdout.join("\n")).toMatch(/Ctx\s+8K tokens est\./);
    stdout = [];
    await checkCommand("lib", { format: "json", verbose: false });
    const json = JSON.parse(stdout.join("\n"));
    expect(json.model.contextWindowEstimated).toBe(true);
    expect(json.pullQuantization).toBeNull();
  });

  it("optimize: json carries the flag, true for a library row and false for a curated one", async () => {
    await optimizeCommand("lib", { format: "json", verbose: false });
    expect(JSON.parse(stdout.join("\n")).model.contextWindowEstimated).toBe(true);
    stdout = [];
    await optimizeCommand("llama3.1:8b", { format: "json", verbose: false });
    expect(JSON.parse(stdout.join("\n")).model.contextWindowEstimated).toBe(false);
  });

  it("compare: table says est.; json and csv carry the flag", async () => {
    await compareCommand(["lib", "llama3.1:8b"], { format: "table", category: "all", top: 3 });
    expect(stdout.join("\n")).toMatch(/Context\s+8K est\./);
    stdout = [];
    await compareCommand(["lib", "llama3.1:8b"], { format: "json", category: "all", top: 3 });
    expect(JSON.parse(stdout.join("\n")).models.map((m: { contextWindowEstimated: boolean }) => m.contextWindowEstimated)).toEqual([true, false]);
    stdout = [];
    await compareCommand(["lib", "llama3.1:8b"], { format: "csv", category: "all", top: 3 });
    const [header, ...rows] = stdout.join("\n").split("\n");
    expect(header.endsWith(",contextWindowEstimated")).toBe(true);
    expect(rows.map((r) => r.split(",").at(-1))).toEqual(["true", "false"]);
  });

  it("models: the Context column says est.", async () => {
    await modelsCommand({ category: "all", fits: false, live: false, installed: false, format: "table" });
    expect(stdout.join("\n")).toContain("8K est.");
  });
});

describe("a budget that sums several GPUs", () => {
  // Each card: 24576 - 457 (estimate:) - 1024 = 23095 MiB; two cards 46190 MiB = 45.1 GiB.
  it("check names every GPU and gives the count in json", async () => {
    state.hardware = TWO_GPUS;
    await checkCommand("llama3.1:8b", { format: "table", verbose: false });
    expect(stdout.join("\n")).toContain("2× GeForce RTX 4090 · 45.1 GiB usable");
    stdout = [];
    await checkCommand("llama3.1:8b", { format: "json", verbose: false });
    expect(JSON.parse(stdout.join("\n")).hardware).toMatchObject({ gpuCount: 2, vramMb: 46190 });
  });

  it("quant-advice json carries the count and the quant the tag installs", async () => {
    state.hardware = TWO_GPUS;
    await quantAdviceCommand("llama3.1:8b", { format: "json", verbose: false });
    const json = JSON.parse(stdout.join("\n"));
    expect(json.hardware.gpuCount).toBe(2);
    expect(json.pullQuantization).toBe("Q4_K_M");
  });
});
