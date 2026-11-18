import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import chalk from "chalk";
import stringWidth from "string-width";
import type { HardwareProfile } from "../../src/core/types.js";
import highEnd from "../fixtures/hardware-profiles/high-end-nvidia.json";
import windowsNvidia from "../fixtures/hardware-profiles/windows-nvidia.json";
import appleM2 from "../fixtures/hardware-profiles/apple-m2.json";
import amdRdna3 from "../fixtures/hardware-profiles/amd-rdna3.json";
import cpuOnly from "../fixtures/hardware-profiles/cpu-only.json";

const state = vi.hoisted(() => ({ hardware: null as unknown, installed: [] as { name: string }[] }));

vi.mock("../../src/hardware/index.js", async () => ({ detectHardware: async () => state.hardware }));
vi.mock("../../src/runtimes/ollama.js", () => ({ withExactFacts: async (_host: string, model: unknown) => model }));
vi.mock("../../src/runtimes/index.js", () => ({
  detectAllRuntimes: async () => state.installed.length === 0 ? [] : [
    { name: "Ollama", status: "running", version: "0.12.0", path: null, models: state.installed.map((m) => m.name) },
  ],
}));
vi.mock("../../src/models/ollama-models.js", () => ({ fetchOllamaModels: async () => state.installed }));
// Two library-only cards with long tags; the catalog is read only where a test turns offline mode off.
vi.mock("../../src/models/library-cache.js", () => ({
  readLibraryCatalogSync: () => ({
    fetchedAt: Date.UTC(2026, 10, 18),
    cards: [
      { slug: "nemotron-3.5-super", description: "x", parameterSizes: ["120b"], capabilities: [], pulls: 50000, updated: "2026-10-01" },
      { slug: "nemotron-cascade-2", description: "x", parameterSizes: ["30b"], capabilities: [], pulls: 50000, updated: "2026-10-01" },
    ],
  }),
  readLibraryStatus: () => null,
}));
// Return the box's content unwrapped: boxen's own wrapping would hide an over-long line.
vi.mock("ora", () => ({ default: () => ({ start() { return this; }, succeed() {}, stop() {}, text: "" }) }));
vi.mock("../../src/cli/ui/boxes.js", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../src/cli/ui/boxes.js")>()),
  titleBox: (content: string) => content,
}));

import { checkCommand } from "../../src/cli/commands/check.js";
import { quantAdviceCommand } from "../../src/cli/commands/quant-advice.js";
import { optimizeCommand } from "../../src/cli/commands/optimize.js";
import { contextFitCommand } from "../../src/cli/commands/context-fit.js";
import { scanCommand } from "../../src/cli/commands/scan.js";
import { compareCommand } from "../../src/cli/commands/compare.js";
import { getRecommendations } from "../../src/analysis/recommender.js";
import { getCuratedModels, resetModelCache } from "../../src/models/database.js";

// boxen at 80 columns leaves 72 for content; one longer line switches it to wrap mode and strips every indent.
const MAX_COLUMNS = 72;

function threeGpuModels(): HardwareProfile {
  const base = highEnd as HardwareProfile;
  const gpu = (model: string) => ({ ...base.gpus[0], model, vramMb: 24564 });
  const gpus = [gpu("NVIDIA GeForce RTX 4090"), gpu("NVIDIA GeForce RTX 3060"), gpu("NVIDIA GeForce RTX 3090")];
  return { ...base, gpus, primaryGpu: gpus[0] };
}

const hardwares: Record<string, HardwareProfile> = {
  nvidia: highEnd as HardwareProfile,
  windows: windowsNvidia as HardwareProfile,
  apple: appleM2 as HardwareProfile,
  amd: amdRdna3 as HardwareProfile,
  cpu: cpuOnly as HardwareProfile,
  trio: threeGpuModels(),
  small: { ...(highEnd as HardwareProfile), gpus: [{ ...(highEnd as HardwareProfile).gpus[0], vramMb: 6144 }], primaryGpu: { ...(highEnd as HardwareProfile).primaryGpu!, vramMb: 6144 } },
};

let stdout: string[];
const originalChalkLevel = chalk.level;

beforeEach(() => {
  stdout = [];
  state.installed = [];
  chalk.level = 0;
  vi.spyOn(console, "log").mockImplementation((...args) => { stdout.push(args.join(" ")); });
});

afterEach(() => {
  vi.restoreAllMocks();
  chalk.level = originalChalkLevel;
});

function tooWide(): string[] {
  // Untrimmed, as boxen measures: trailing spaces count.
  return stdout.join("\n").split("\n").filter((line) => stringWidth(line) > MAX_COLUMNS);
}

describe("box content stays within 72 columns", () => {
  const ids = getCuratedModels().map((m) => m.ollamaTag ?? m.id);

  it.each(Object.keys(hardwares))("check, quant-advice, optimize and context-fit on %s, every curated model", async (name) => {
    state.hardware = hardwares[name];
    const offenders = new Set<string>();
    for (const id of ids) {
      for (const run of [
        () => checkCommand(id, { format: "table", verbose: true }),
        () => quantAdviceCommand(id, { format: "table", verbose: true }),
        () => optimizeCommand(id, { format: "table", verbose: true }),
        () => contextFitCommand(id, { promptTokens: 100_000, format: "table" }),
        () => contextFitCommand(id, { promptTokens: 3_000, format: "table" }),
      ]) {
        stdout = [];
        await run();
        for (const line of tooWide()) offenders.add(`${id}: ${line.trim()}`);
      }
    }
    expect([...offenders].slice(0, 10)).toEqual([]);
  }, 60_000);
});

describe("box content stays within 72 columns: library tags, scan and compare", () => {
  beforeEach(() => {
    vi.stubEnv("LLMPULSE_OFFLINE", "");
    resetModelCache();
  });
  afterEach(() => {
    vi.unstubAllEnvs();
    resetModelCache();
  });

  it.each(Object.keys(hardwares))("check, quant-advice, optimize and context-fit on %s, long library tags", async (name) => {
    state.hardware = hardwares[name];
    const offenders = new Set<string>();
    for (const id of ["nemotron-3.5-super:120b", "nemotron-cascade-2:30b"]) {
      for (const run of [
        () => checkCommand(id, { format: "table", verbose: true }),
        () => quantAdviceCommand(id, { format: "table", verbose: true }),
        () => optimizeCommand(id, { format: "table", verbose: true }),
        () => contextFitCommand(id, { promptTokens: 100_000, format: "table" }),
      ]) {
        stdout = [];
        await run();
        expect(stdout.join("\n")).toContain("nemotron"); // resolved, not the "model not found" screen
        expect(stdout.join("\n")).not.toContain("not found");
        for (const line of tooWide()) offenders.add(`${id}: ${line.trim()}`);
      }
    }
    expect([...offenders].slice(0, 10)).toEqual([]);
  });

  it.each(Object.keys(hardwares))("scan on %s, top 5 and 10, with and without installed models", async (name) => {
    state.hardware = hardwares[name];
    for (const top of [5, 10]) {
      for (const installed of [false, true]) {
        stdout = [];
        state.installed = installed ? getCuratedModels().map((m) => ({ name: m.ollamaTag ?? m.id })) : [];
        await scanCommand({ format: "table", category: "all", top });
        expect(tooWide()).toEqual([]);
      }
    }
  });

  it("scan marks installed rows with '●' and keeps every word of a wrapped name", async () => {
    state.hardware = hardwares.nvidia;
    state.installed = getCuratedModels().map((m) => ({ name: m.ollamaTag ?? m.id }));
    await scanCommand({ format: "table", category: "all", top: 10 });
    const out = stdout.join("\n");
    const recommendations = getRecommendations(hardwares.nvidia, { category: "all", top: 10, onlyFitting: true });
    expect(recommendations.some((r) => r.score.model.name.includes(" "))).toBe(true);
    for (const { score } of recommendations) {
      const [first, ...rest] = score.model.name.split(" ");
      expect(out).toContain(`● ${first}`);
      for (const word of rest) expect(out).toContain(word);
    }
  });

  it.each(Object.keys(hardwares))("compare of three models on %s", async (name) => {
    state.hardware = hardwares[name];
    await compareCommand(["llama3.1:8b", "qwen2.5-coder:14b", "mistral-small:22b"], { format: "table", category: "all", top: 3 });
    expect(stdout.join("\n")).toContain("Model Comparison");
    expect(tooWide()).toEqual([]);
  });

  it.each(Object.keys(hardwares))("compare of two long-named curated models on %s", async (name) => {
    state.hardware = hardwares[name];
    await compareCommand(["deepseek-coder-v2:16b", "mistral-small:24b"], { format: "table", category: "all", top: 3 });
    expect(stdout.join("\n")).toContain("Model Comparison");
    expect(tooWide()).toEqual([]);
  });
});
