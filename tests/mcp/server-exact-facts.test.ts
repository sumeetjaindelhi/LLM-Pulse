import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { ModelEntry } from "../../src/core/types.js";
import { EXACT_Q8_MB, withExactQ8 } from "../fixtures/exact-facts.js";

type ToolHandler = (args: Record<string, unknown>) => Promise<{ content: { text: string }[] }>;
const tools = vi.hoisted(() => new Map<string, ToolHandler>());
const descriptions = vi.hoisted(() => new Map<string, string>());
vi.mock("@modelcontextprotocol/sdk/server/mcp.js", () => ({
  McpServer: class {
    tool(name: string, description: string, _schema: unknown, handler: ToolHandler) {
      tools.set(name, handler);
      descriptions.set(name, description);
    }
  },
}));
vi.mock("@modelcontextprotocol/sdk/server/stdio.js", () => ({ StdioServerTransport: class {} }));
vi.mock("../../src/hardware/index.js", async () => ({
  detectHardware: async () => (await import("../fixtures/hardware-profiles/high-end-nvidia.json")).default,
}));
vi.mock("../../src/runtimes/index.js", () => ({ detectAllRuntimes: async () => [] }));

// No live Ollama or registry in unit tests: exact-fact lookups pass the model through unless a test says otherwise.
const withExactFacts = vi.hoisted(() => vi.fn(async (_host: string, model: ModelEntry) => model));
vi.mock("../../src/runtimes/ollama.js", () => ({ withExactFacts }));

await import("../../src/mcp/server.js");
const { checkCommand } = await import("../../src/cli/commands/check.js");

async function call(tool: string, args: Record<string, unknown>) {
  const result = await tools.get(tool)!(args);
  return JSON.parse(result.content[0].text);
}

describe("MCP tools use the model with exact facts", () => {
  beforeEach(() => {
    withExactFacts.mockReset();
  });

  it("check looks them up on the given localhost host and matches quant on the result", async () => {
    withExactFacts.mockImplementationOnce(withExactQ8);
    const out = await call("check", { model: "llama3.1:8b", quant: "Q8_0", host: "http://localhost:11500" });
    expect(withExactFacts).toHaveBeenCalledWith("http://localhost:11500", expect.objectContaining({ id: "llama-3.1-8b" }));
    expect(out.bestQuantization.vramMb).toBe(EXACT_Q8_MB);
  });

  // MCP models lists a library row's sole quant as "default"; exact facts rename it to the file's type.
  it("check accepts the listed placeholder name of a library row's renamed quant", async () => {
    withExactFacts.mockImplementationOnce(async (_host, model) => ({
      ...model,
      source: "library",
      quantizations: [{ name: "F16", bitsPerWeight: 16.0005, vramMb: 258, qualityRetention: 1, exact: true }],
    }));
    const out = await call("check", { model: "llama3.1:8b", quant: "default" });
    expect(out.bestQuantization?.name).toBe("F16");
  });

  it("context-fit-check matches quant on the result", async () => {
    withExactFacts.mockImplementationOnce(withExactQ8);
    const out = await call("context-fit-check", { model: "llama3.1:8b", promptTokens: 1000, responseTokens: 0, quant: "Q8_0" });
    expect(out.quant.vramMb).toBe(EXACT_Q8_MB);
  });
});

describe("MCP memory provenance", () => {
  beforeEach(() => {
    withExactFacts.mockImplementation(async (_host, model) => model);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("check returns exactly what `check --format json` prints", async () => {
    const out = await call("check", { model: "llama3.1:8b" });
    const printed: string[] = [];
    vi.spyOn(console, "log").mockImplementation((...args) => { printed.push(args.join(" ")); });
    await checkCommand("llama3.1:8b", { format: "json", verbose: false });
    expect(out).toEqual(JSON.parse(printed.join("\n")));
    expect(out.bestQuantization.memory.contextTokens).toBe(32768);
  });

  it("scan and recommend carry each pick's memory at Ollama's default context", async () => {
    for (const tool of ["scan", "recommend"]) {
      const out = await call(tool, { category: "all", top: 3, onlyFitting: true });
      expect(out.recommendations).toHaveLength(3);
      for (const rec of out.recommendations) {
        expect(rec.memory.budgetMb).toBe(23095);
        expect(rec.memory.weightsMb).toBe(rec.vramMb);
        expect(rec.memory.totalMb).toBeGreaterThan(rec.vramMb);
      }
    }
  });

  it("scan and recommend say how many GPUs the budget sums; recommend flags a guessed window", async () => {
    expect((await call("scan", { category: "all", top: 1 })).hardware.gpuCount).toBe(1);
    const out = await call("recommend", { category: "all", top: 1, onlyFitting: true });
    expect(out.hardware.gpuCount).toBe(1);
    expect(out.recommendations[0].model.contextWindowEstimated).toBe(false);
  });

  it("recommend's description says how it ranks", () => {
    expect(descriptions.get("recommend")).toMatch(/fit first.*one model per family.*embedding and ternary.*excluded/s);
  });

  it("context-fit-check carries the memory at the workload's needed tokens", async () => {
    const out = await call("context-fit-check", { model: "llama3.1:8b", promptTokens: 1000, responseTokens: 512 });
    expect(out.memory.contextTokens).toBe(1512);
    expect(out.memory.weightsMb).toBe(out.quant.vramMb);
  });
});
