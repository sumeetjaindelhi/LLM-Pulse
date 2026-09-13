import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import chalk from "chalk";
import { modelsCommand } from "../../src/cli/commands/models.js";
import { recommendationTable, comparisonTable } from "../../src/cli/ui/tables.js";
import { getRecommendations } from "../../src/analysis/recommender.js";
import type { HardwareProfile } from "../../src/core/types.js";
import highEnd from "../fixtures/hardware-profiles/high-end-nvidia.json";

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
    const recs = getRecommendations(highEnd as HardwareProfile, { category: "all", top: 3, onlyFitting: true });
    expect(recommendationTable(recs, new Set())).not.toContain("[");
    expect(comparisonTable(recs.map((r) => r.score), 24000, 0)).not.toContain("[");
  });
});
