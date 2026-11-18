import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import chalk from "chalk";

const hw = vi.hoisted(() => ({ current: null as unknown }));

vi.mock("../../src/hardware/index.js", async () => ({
  detectHardware: async () =>
    hw.current ?? (await import("../fixtures/hardware-profiles/cpu-only.json")).default,
}));

// No live Ollama or registry in unit tests: exact-fact lookups pass the model through unless a test says otherwise.
const withExactFacts = vi.hoisted(() => vi.fn(async (_host: string, model: ModelEntry) => model));
vi.mock("../../src/runtimes/ollama.js", () => ({ withExactFacts }));

import { optimizeCommand } from "../../src/cli/commands/optimize.js";
import type { HardwareProfile, ModelEntry } from "../../src/core/types.js";
import highEnd from "../fixtures/hardware-profiles/high-end-nvidia.json";
import { EXACT_Q8_MB, withExactQ8 } from "../fixtures/exact-facts.js";

function gpuWithGib(gib: number): HardwareProfile {
  const base = highEnd as HardwareProfile;
  const gpu = { ...base.primaryGpu!, vramMb: gib * 1024 };
  return { ...base, gpus: [gpu], primaryGpu: gpu };
}

let stdout: string[];
const originalChalkLevel = chalk.level;

beforeEach(() => {
  stdout = [];
  chalk.level = 0; // theme colours off, as in a pipe, so table lines match as printed
  vi.spyOn(console, "log").mockImplementation((...args) => { stdout.push(args.join(" ")); });
  process.exitCode = undefined;
  hw.current = null;
});

afterEach(() => {
  vi.restoreAllMocks();
  process.exitCode = undefined;
  chalk.level = originalChalkLevel;
});

describe("optimize", () => {
  it("json: no fitting quantization prints the error payload and exits 1", async () => {
    await optimizeCommand("llama-4-scout", { format: "json", verbose: false });
    expect(JSON.parse(stdout.join("\n")).error).toBe("No quantization fits");
    expect(process.exitCode).toBe(1);
  });

  it("table: the no-fit message names usable memory", async () => {
    await optimizeCommand("llama-4-scout", { format: "table", verbose: false });
    expect(stdout.join("\n")).toMatch(/your [\d.]+ GiB of usable memory\./);
  });

  it("leaves the exit code alone on success", async () => {
    await optimizeCommand("llama3.1:8b", { format: "json", verbose: false });
    expect(JSON.parse(stdout.join("\n")).parameters).toBeDefined();
    expect(process.exitCode).toBeUndefined();
  });

  it("json: --quant is matched on the model with exact facts", async () => {
    hw.current = gpuWithGib(24);
    withExactFacts.mockImplementationOnce(withExactQ8);
    await optimizeCommand("llama3.1:8b", { quant: "Q8_0", format: "json", verbose: false });
    expect(JSON.parse(stdout.join("\n")).quant.vramMb).toBe(EXACT_Q8_MB);
  });

  it("table: renders Modelfile, /set parameter and server-wide lines", async () => {
    await optimizeCommand("llama3.1:8b", { format: "table", verbose: false });
    const out = stdout.join("\n");
    expect(out).toMatch(/FROM llama3\.1:8b\s/);
    expect(out).toMatch(/PARAMETER num_ctx \d+/);
    expect(out).toMatch(/PARAMETER num_thread \d+/);
    expect(out).toMatch(/\/set parameter num_ctx \d+/);
    expect(out).toMatch(/\/set parameter num_thread \d+/);
    expect(out).not.toMatch(/(PARAMETER|set parameter) num_(gpu|batch)/);
    const numCtx = /PARAMETER num_ctx (\d+)/.exec(out)![1];
    expect(out).toContain(`OLLAMA_CONTEXT_LENGTH=${numCtx} ollama serve`);
    expect(out).toMatch(/default context.*4,?096/i);
  });

  it("json: modelfile and additive default-context field", async () => {
    // CPU box, 10000 MiB available. The tag's file Q4_K_M (4693 MiB): need(4096) = 4693 + 512 + 275.55 = 5480.6 <= 0.8 x 10000.
    // num_ctx: need(32768) = 4693 + 4096 + 551.1 = 9340.1 fits; above 32768 the batch is 2048, so
    // c <= (10000 - 4693 - 1102.2) / 0.125 = 33638, below the next tier -> 32768.
    await optimizeCommand("llama3.1:8b", { format: "json", verbose: false });
    const json = JSON.parse(stdout.join("\n"));
    expect(json.modelfile).toBe("FROM llama3.1:8b\nPARAMETER num_ctx 32768\nPARAMETER num_thread 6");
    expect(json.ollamaDefaultContext).toBe(4096);
  });

  it.each([
    [8, 4096],
    [32, 32768],
    // Rule: Ollama clamps the tier (262144 at >= 47 GiB) to the training context; llama3.1:8b trains on 131072.
    [64, 131072],
  ])("json: default context for a %i GB GPU is %i", async (gib, expected) => {
    hw.current = gpuWithGib(gib);
    await optimizeCommand("llama3.1:8b", { format: "json", verbose: false });
    expect(JSON.parse(stdout.join("\n")).ollamaDefaultContext).toBe(expected);
  });

  // 24 GiB GPU: budget 23095 MiB (22.6 GiB), default context 32768. The tag's file Q4_K_M (4693 MiB, exact):
  // need(32768) = 4693 + 4096 + 551.1 = 9340.1 <= 0.8 x 23095 -> excellent. num_ctx 131072, the training context:
  // KV 32 x 8 x 256 x 2 B x 131072 = 16384 MiB, compute 128256 x 2048 x 4 x 1.1 / 2^20 = 1102.2 MiB, total 22179.2 <= 23095;
  // all 32 layers fit: (23095 - 1102.2) / ((4693 + 16384) / 32) = 33.4.
  it("json: tunes the tag's file, with memory at num_ctx", async () => {
    hw.current = gpuWithGib(24);
    await optimizeCommand("llama3.1:8b", { format: "json", verbose: false });
    const { memory, ...json } = JSON.parse(stdout.join("\n"));
    expect(json).toEqual({
      model: { id: "llama-3.1-8b", name: "Llama 3.1 8B", provider: "Meta", parametersBillion: 8, contextWindow: 131072, contextWindowEstimated: false, ollamaTag: "llama3.1:8b" },
      hardware: { gpu: "GeForce RTX 4090", gpuCount: 1, vramMb: 23095, cpuCores: 16 },
      // llama.cpp's measured Q4_K_M file bits per weight (tools/quantize/README.md); retention exp(-mean KLD 0.031273).
      quant: { name: "Q4_K_M", bitsPerWeight: 4.8944, vramMb: 4693, qualityRetention: Math.exp(-0.031273), qualityMeasured: true, forced: false },
      fitLevel: "excellent",
      parameters: {
        num_ctx: { value: 131072, note: "capped at native context" },
        num_gpu: { value: "all", note: "full 32/32 layers on GPU" },
        num_thread: { value: 16, note: "physical cores (no SMT)" },
        // ctx 131072 > 32768 -> automatic batch 2048 (O server/sched.go:900-909); not in the Modelfile.
        num_batch: { value: 2048, note: "automatic — Ollama uses up to this" },
      },
      modelfile: "FROM llama3.1:8b\nPARAMETER num_ctx 131072\nPARAMETER num_thread 16",
      ollamaDefaultContext: 32768,
      kvQ8MaxContext: null,
    });
    expect(memory).toEqual({
      contextTokens: 131072, weightsMb: 4693, weightsExact: true, kvCacheMb: 16384, kvExact: true,
      // compute 1102.2 + KQ mask 2 B x 131072 x 2048 (512) = 1614.2; total 4693 + 16384 + 1614.2 = 22691.2
      computeMb: expect.closeTo(1614.2, 6), totalMb: expect.closeTo(22691.2, 6), budgetMb: 23095, budgetExact: false,
    });
  });

  it("table: memory breakdown at num_ctx", async () => {
    hw.current = gpuWithGib(24);
    await optimizeCommand("llama3.1:8b", { format: "table", verbose: false });
    const out = stdout.join("\n");
    expect(out).toContain(`22.2 GiB at ${(131072).toLocaleString()} tokens`);
    expect(out).toContain("of 22.6 GiB usable");
    expect(out).toContain("weights 4.6 GiB · KV 16.0 GiB · compute ≈1.6 GiB");
  });

  // FROM llama3.1:8b and `ollama run llama3.1:8b` load the Q4_K_M file: parameters tuned for F16 get a placeholder in
  // their place, which fails if pasted, and where F16's tag is listed.
  it("table: a forced quant the tag does not install gets a placeholder for its tag", async () => {
    hw.current = gpuWithGib(24);
    await optimizeCommand("llama3.1:8b", { format: "table", verbose: false, quant: "F16" });
    let out = stdout.join("\n");
    expect(out).toContain("→ ollama pull llama3.1:8b  installs Q4_K_M (4.6 GiB)");
    expect(out).toContain("For F16, pick its tag at");
    expect(out).toContain("https://ollama.com/library/llama3.1/tags");
    expect(out).toContain("FROM <F16 tag>");
    expect(out).toContain("ollama run <F16 tag>");
    expect(out).not.toContain("FROM llama3.1:8b");
    expect(out).toContain("weights 14.9 GiB est.");

    stdout = [];
    await optimizeCommand("llama3.1:8b", { format: "table", verbose: false });
    out = stdout.join("\n");
    expect(out).toContain("installs Q4_K_M (4.6 GiB)");
    expect(out).not.toContain("pick its tag");
    expect(out).toContain("FROM llama3.1:8b");
    expect(out).toContain("ollama run llama3.1:8b");
  });

  // F16 (8e9 x 16.0005 / 8 / 2^20 = 15259 MiB est.) on 24 GiB: need(32768) = 15259 + 4096 + 551.1 = 19906.1 <= 23095
  // (comfortable: no num_batch line); above 32768, (23095 - 15259 - 1102.2) / 0.125 = 53870 tokens, below the next tier.
  it("json: the modelfile names a placeholder for a forced quant the tag does not install", async () => {
    hw.current = gpuWithGib(24);
    await optimizeCommand("llama3.1:8b", { format: "json", verbose: false, quant: "F16" });
    expect(JSON.parse(stdout.join("\n")).modelfile).toBe("FROM <F16 tag>\nPARAMETER num_ctx 32768\nPARAMETER num_thread 16");
  });

  // 12 GiB GPU: the f16 KV cache caps num_ctx below the window (see the q8_0 test below); the cap is usable memory.
  it("table: labels the num_ctx limit usable memory", async () => {
    hw.current = gpuWithGib(12);
    await optimizeCommand("llama3.1:8b", { format: "table", verbose: false });
    const out = stdout.join("\n");
    expect(out).toContain("Usable memory limits num_ctx");
    expect(out).not.toContain("VRAM limits");
  });

  // 16 GiB GPU: budget 16384 - 1481 = 14903 MiB; gemma3:27b's tag file Q4_K_M (16607 MiB) fits only partly ("barely":
  // need(4096) = 16607 + 944 KV + 563.2 compute = 18114.2, 0.75 x 18114.2 <= 14903) and Q8_0 not at all, so the
  // Modelfile pins num_batch 256, which turns off Ollama's automatic batch and runs -b 256 -ub 256 (O server/routes.go:214-224,
  // llm/llama_server.go:630-631). At num_ctx 2048, 16 KV heads x (128 + 128) x 2 B = 8192 B per token and layer:
  // (10 full layers x 2048 + 52 sliding layers x (1024 + 256) slots, L src/llama-kv-cache-iswa.cpp:73) x 8192 B = 680 MiB;
  // compute 262145 x 256 x 4 x 1.1 / 2^20 = 281.6 MiB + KQ masks 2 B x 256 tokens x (2048 + 1280 SWA cells) = 1.625 MiB = 283.2.
  it("json: memory at the num_batch the Modelfile pins", async () => {
    hw.current = gpuWithGib(16);
    await optimizeCommand("gemma3:27b", { format: "json", verbose: false });
    const json = JSON.parse(stdout.join("\n"));
    expect(json.modelfile).toContain("PARAMETER num_batch 256");
    expect(json.memory).toMatchObject({ contextTokens: 2048, kvCacheMb: 680, kvExact: true, computeMb: expect.closeTo(283.2, 1) });
  });

  it("json: reports the q8_0 ceiling when VRAM limits num_ctx below the model window", async () => {
    // 12 GB: budget 12288 - 1481 = 10807. The tag's file Q4_K_M (4693 MiB); above 32768 tokens the batch is 2048:
    // f16 KV (0.125 MiB/token) holds (10807 - 4693 - 1102.2)/0.125 = 40094 -> 39936 tokens -> tier 32768;
    // q8_0 KV (0.0664 MiB/token) holds 75264.
    hw.current = gpuWithGib(12);
    await optimizeCommand("llama3.1:8b", { format: "json", verbose: false });
    const json = JSON.parse(stdout.join("\n"));
    expect(json.parameters.num_ctx.value).toBeLessThan(json.model.contextWindow);
    expect(json.kvQ8MaxContext).toBeGreaterThan(json.parameters.num_ctx.value);
  });
});
