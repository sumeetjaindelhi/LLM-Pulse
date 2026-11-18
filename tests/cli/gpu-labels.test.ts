import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import chalk from "chalk";
import type { GpuInfo, HardwareProfile } from "../../src/core/types.js";
import highEnd from "../fixtures/hardware-profiles/high-end-nvidia.json";
import cpuOnly from "../fixtures/hardware-profiles/cpu-only.json";

const state = vi.hoisted(() => ({ hardware: null as unknown }));
vi.mock("../../src/hardware/index.js", async () => ({ detectHardware: async () => state.hardware }));
vi.mock("../../src/runtimes/ollama.js", () => ({ withExactFacts: async (_host: string, model: unknown) => model }));
vi.mock("../../src/runtimes/index.js", () => ({ detectAllRuntimes: async () => [] }));
vi.mock("../../src/models/ollama-models.js", () => ({ fetchOllamaModels: async () => [] }));
// The box's content unwrapped: boxen's own wrapping would hide an over-long line.
vi.mock("../../src/cli/ui/boxes.js", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../src/cli/ui/boxes.js")>()),
  titleBox: (content: string) => content,
}));
vi.mock("ora", () => ({ default: () => ({ start() { return this; }, succeed() {}, stop() {}, text: "" }) }));

import { checkCommand } from "../../src/cli/commands/check.js";
import { optimizeCommand } from "../../src/cli/commands/optimize.js";
import { contextFitCommand } from "../../src/cli/commands/context-fit.js";
import { quantAdviceCommand } from "../../src/cli/commands/quant-advice.js";
import { scanCommand } from "../../src/cli/commands/scan.js";
import { renderMarkdownReport } from "../../src/cli/commands/report.js";
import { runDiagnostics } from "../../src/analysis/doctor.js";

const HIGH_END = highEnd as HardwareProfile;
const rtx = (model: string, vramMb: number): GpuInfo => ({ ...HIGH_END.gpus[0], model, vramMb });
const rtx4090 = rtx("NVIDIA GeForce RTX 4090", 24564);
const rtx3060 = rtx("NVIDIA GeForce RTX 3060", 12288);
const MIXED: HardwareProfile = { ...HIGH_END, gpus: [rtx4090, rtx3060], primaryGpu: rtx4090 };

// An AMD APU Ollama does not schedule on (not gfx1151, OLLAMA_IGPU_ENABLE unset): CPU-only budget.
const apu: GpuInfo = {
  vendor: "AMD", model: "AMD Radeon 780M", vramMb: 32768, driverVersion: "", acceleratorVersion: null, acceleratorType: "rocm",
  utilizationPercent: null, temperatureCelsius: null, vramUsedMb: null, unifiedMemory: true, gfxTarget: "gfx1103",
};
const DROPPED_APU: HardwareProfile = { ...(cpuOnly as HardwareProfile), gpus: [apu], primaryGpu: apu };
const NOT_USED = "not used by Ollama";

let stdout: string[];
const originalChalkLevel = chalk.level;

beforeEach(() => {
  stdout = [];
  chalk.level = 0;
  vi.stubEnv("OLLAMA_IGPU_ENABLE", "");
  vi.spyOn(console, "log").mockImplementation((...args) => { stdout.push(args.join(" ")); });
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
  chalk.level = originalChalkLevel;
});

const out = () => stdout.join("\n");

describe("B1: GPUs of different models are named one by one", () => {
  // Budgets: 4090 24564 - 457 - 1024 = 23083; 3060 12288 - 457 - 1024 = 10807; sum 33890 MiB = 33.1 GiB.
  it.each([
    ["check", () => checkCommand("llama3.1:8b", { format: "table", verbose: false })],
    ["optimize", () => optimizeCommand("llama3.1:8b", { format: "table", verbose: false })],
    ["context-fit", () => contextFitCommand("llama3.1:8b", { promptTokens: 1000, format: "table" })],
  ])("%s: 'A + B', usable memory on a sub-line when the row would pass 72 columns", async (_name, run) => {
    state.hardware = MIXED;
    await run();
    expect(out()).toContain("NVIDIA GeForce RTX 4090 + NVIDIA GeForce RTX 3060");
    expect(out()).toContain("33.1 GiB usable");
    expect(out()).not.toContain("2×");
    expect(stdout.join("\n").split("\n").filter((l) => [...l.trimEnd()].length > 72)).toEqual([]);
  });

  it("keeps 'N× model' when every GPU is the same model", async () => {
    state.hardware = { ...HIGH_END, gpus: [HIGH_END.gpus[0], HIGH_END.gpus[0]] };
    await checkCommand("llama3.1:8b", { format: "table", verbose: false });
    expect(out()).toContain("2× GeForce RTX 4090 · 45.1 GiB usable");
  });

  it("report: GPU row lists both models and their summed VRAM (24564 + 12288 = 36852 MiB = 36.0 GiB)", () => {
    const md = renderMarkdownReport({
      version: "1.2.0", generatedAt: new Date("2026-09-13T12:00:00Z"), platform: "linux/x64", hardware: MIXED, runtimes: [],
      installedTags: new Set(), recommendations: [], health: runDiagnostics(MIXED, []), availableVramMb: 33890,
    });
    expect(md).toContain("| GPU | NVIDIA GeForce RTX 4090 + NVIDIA GeForce RTX 3060 · 36.0 GiB VRAM");
  });

  it("scan: the GPU line lists both models once each, without a doubled vendor", async () => {
    state.hardware = MIXED;
    await scanCommand({ format: "table", category: "all", top: 1 });
    expect(out()).toContain("NVIDIA GeForce RTX 4090 + NVIDIA GeForce RTX 3060");
    expect(out()).not.toContain("NVIDIA NVIDIA");
    expect(out()).not.toContain("2×");
  });
});

describe("B5: scan labels several identical GPUs", () => {
  const TWO: HardwareProfile = { ...HIGH_END, gpus: [HIGH_END.gpus[0], HIGH_END.gpus[0]] };

  it("table: '2× NVIDIA GeForce RTX 4090' with the summed VRAM (2 x 24576 = 48.0 GiB)", async () => {
    state.hardware = TWO;
    await scanCommand({ format: "table", category: "all", top: 1 });
    expect(out()).toContain("2× NVIDIA GeForce RTX 4090");
    expect(out()).toContain("48.0 GiB VRAM");
  });

  it("json: gpuCount is the number of budget GPUs", async () => {
    state.hardware = TWO;
    await scanCommand({ format: "json", category: "all", top: 1 });
    expect(JSON.parse(out()).hardware.gpuCount).toBe(2);
  });
});

describe("B2: a GPU Ollama does not use is not shown as usable memory", () => {
  it("check: the CPU branch, no unified-memory note, and why the GPU is ignored", async () => {
    state.hardware = DROPPED_APU;
    await checkCommand("llama3.1:8b", { format: "table", verbose: false });
    expect(out()).toContain("None — CPU inference");
    expect(out()).toContain(`AMD Radeon 780M ${NOT_USED}`);
    expect(out()).toContain("OLLAMA_IGPU_ENABLE=1 enables it");
    expect(out()).not.toContain("unified memory");
  });

  it("optimize: no 'Ollama offloads all layers' next to a CPU-only budget", async () => {
    state.hardware = DROPPED_APU;
    await optimizeCommand("llama3.1:8b", { format: "table", verbose: false });
    expect(out()).not.toContain("offloads all layers");
    expect(out()).toContain(NOT_USED);
  });

  it("quant-advice: no 'shared with system RAM' note", async () => {
    state.hardware = DROPPED_APU;
    await quantAdviceCommand("llama3.1:8b", { format: "table", verbose: false });
    expect(out()).not.toContain("unified memory");
  });

  it("context-fit: names the ignored GPU", async () => {
    state.hardware = DROPPED_APU;
    await contextFitCommand("llama3.1:8b", { promptTokens: 1000, format: "table" });
    expect(out()).toContain(NOT_USED);
  });

  it("scan: no 32.0 GiB VRAM line", async () => {
    state.hardware = DROPPED_APU;
    await scanCommand({ format: "table", category: "all", top: 1 });
    expect(out()).toContain("None — CPU inference");
    expect(out()).toContain(`AMD Radeon 780M ${NOT_USED} (integrated;`);
    expect(out()).not.toContain("32.0 GiB VRAM");
  });

  it("report: the GPU row says so instead of a VRAM figure", () => {
    const md = renderMarkdownReport({
      version: "1.2.0", generatedAt: new Date("2026-09-13T12:00:00Z"), platform: "linux/x64", hardware: DROPPED_APU, runtimes: [],
      installedTags: new Set(), recommendations: [], health: runDiagnostics(DROPPED_APU, []), availableVramMb: 40000,
    });
    expect(md).toContain(`| GPU | AMD Radeon 780M ${NOT_USED}`);
    expect(md).not.toContain("32.0 GiB VRAM");
  });

  it("doctor: no passing VRAM check for it", () => {
    const checks = runDiagnostics(DROPPED_APU, []).checks;
    expect(checks.filter((c) => c.severity === "pass" && c.message.includes("VRAM"))).toEqual([]);
    expect(checks.find((c) => c.message.includes(`AMD Radeon 780M ${NOT_USED} (integrated; OLLAMA_IGPU_ENABLE=1 enables it)`))?.severity).toBe("warning");
  });
});

describe("B5: other gaps", () => {
  it("doctor suggestions give sizes in GiB", () => {
    const small: HardwareProfile = { ...HIGH_END, gpus: [{ ...HIGH_END.gpus[0], vramMb: 2048 }], primaryGpu: { ...HIGH_END.gpus[0], vramMb: 2048 }, disk: { ...HIGH_END.disk, freeGb: 5 } };
    const suggestions = runDiagnostics(small, []).checks.map((c) => c.suggestion ?? "").join("\n");
    expect(suggestions).toContain("4+ GiB VRAM recommended");
    expect(suggestions).toContain("A 7B Q4 model needs ~4 GiB");
  });

  it("quant-advice lists quants by weights: mistral-small:22b Q4_0 (about 4.9 bits) before Q4_K_M", async () => {
    state.hardware = HIGH_END;
    await quantAdviceCommand("mistral-small:22b", { format: "json", verbose: false });
    const rows = JSON.parse(out()).quantizations as { name: string; vramMb: number }[];
    const names = rows.map((r) => r.name);
    expect(names.indexOf("Q4_0")).toBeLessThan(names.indexOf("Q4_K_M"));
    expect(rows.map((r) => r.vramMb)).toEqual([...rows.map((r) => r.vramMb)].sort((a, b) => a - b));
  });
});
