import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import chalk from "chalk";
import type { HardwareProfile } from "../../src/core/types.js";
import highEnd from "../fixtures/hardware-profiles/high-end-nvidia.json";
import appleM2 from "../fixtures/hardware-profiles/apple-m2.json";
import cpuOnly from "../fixtures/hardware-profiles/cpu-only.json";
import amdRdna3 from "../fixtures/hardware-profiles/amd-rdna3.json";

const state = vi.hoisted(() => ({ hardware: null as unknown }));
vi.mock("../../src/hardware/index.js", async () => ({ detectHardware: async () => state.hardware }));
vi.mock("../../src/runtimes/ollama.js", () => ({ withExactFacts: async (_host: string, model: unknown) => model }));
vi.mock("../../src/runtimes/index.js", () => ({ detectAllRuntimes: async () => [] }));
vi.mock("../../src/models/ollama-models.js", () => ({ fetchOllamaModels: async () => [] }));
vi.mock("ora", () => ({ default: () => ({ start() { return this; }, succeed() {}, stop() {}, text: "" }) }));

import { checkCommand } from "../../src/cli/commands/check.js";
import { optimizeCommand } from "../../src/cli/commands/optimize.js";
import { contextFitCommand } from "../../src/cli/commands/context-fit.js";
import { quantAdviceCommand } from "../../src/cli/commands/quant-advice.js";
import { renderMarkdownReport } from "../../src/cli/commands/report.js";
import { runDiagnostics } from "../../src/analysis/doctor.js";
import { budgetExact } from "../../src/analysis/scorer.js";
import { memoryBreakdown } from "../../src/analysis/context-capacity.js";
import { resolveModel } from "../../src/models/database.js";

const NVIDIA = highEnd as HardwareProfile;
const APPLE = appleM2 as HardwareProfile;
const CPU = cpuOnly as HardwareProfile;
const AMD = amdRdna3 as HardwareProfile;

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
const out = () => stdout.join("\n");

describe("budgetExact: the usable figure includes llm-pulse's 457 MiB estimate except on Metal and CPU-only", () => {
  it.each([[NVIDIA, false], [AMD, false], [APPLE, true], [CPU, true]] as const)("hardware %#", (hardware, expected) => {
    expect(budgetExact(hardware)).toBe(expected);
    const model = resolveModel("llama3.1:8b")!;
    expect(memoryBreakdown(model, model.quantizations[0], hardware).budgetExact).toBe(expected);
  });
});

describe("usable figures carry est. when not exact", () => {
  const runs: [string, () => Promise<void>][] = [
    ["check", () => checkCommand("llama3.1:8b", { format: "table", verbose: true })],
    ["optimize", () => optimizeCommand("llama3.1:8b", { format: "table", verbose: true })],
    ["context-fit", () => contextFitCommand("llama3.1:8b", { promptTokens: 3000, format: "table" })],
  ];
  it.each(runs)("%s on a discrete GPU", async (_name, run) => {
    state.hardware = NVIDIA;
    await run();
    expect(out()).toContain("22.6 GiB usable est.");
  });
  it.each(runs)("%s on Apple Metal", async (_name, run) => {
    state.hardware = APPLE;
    await run();
    expect(out()).toMatch(/GiB usable/);
    expect(out()).not.toContain("usable est.");
  });

  it("check's quant table marks Usable and explains the compute (≈) in Need", async () => {
    state.hardware = NVIDIA;
    await checkCommand("llama3.1:8b", { format: "table", verbose: false });
    expect(out()).toMatch(/Usable est\./);
    expect(out()).toContain("compute (≈)");
  });

  it("quant-advice", async () => {
    state.hardware = NVIDIA;
    await quantAdviceCommand("llama3.1:8b", { format: "table", verbose: false });
    expect(out()).toContain("Usable memory: 22.6 GiB est.");
    stdout = [];
    state.hardware = APPLE;
    await quantAdviceCommand("llama3.1:8b", { format: "table", verbose: false });
    expect(out()).not.toMatch(/Usable memory: [\d.]+ GiB est\./);
  });

  it("report", () => {
    const row = (hardware: HardwareProfile) => renderMarkdownReport({
      version: "1.2.0", generatedAt: new Date("2026-09-13T12:00:00Z"), platform: "linux/x64", hardware, runtimes: [],
      installedTags: new Set(), recommendations: [], health: runDiagnostics(hardware, []), availableVramMb: 23083,
    });
    expect(row(NVIDIA)).toContain("| Usable for inference | 22.5 GiB est. |");
    expect(row(CPU)).toContain("| Usable for inference | 22.5 GiB |");
  });
});
