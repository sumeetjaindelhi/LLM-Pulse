import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { CatalogHealth, HardwareProfile } from "../../src/core/types.js";
import highEnd from "../fixtures/hardware-profiles/high-end-nvidia.json";

const health = vi.hoisted(() => ({ current: null as unknown }));
vi.mock("../../src/models/database.js", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../src/models/database.js")>()),
  getCatalogHealth: () => health.current as CatalogHealth,
}));
vi.mock("../../src/hardware/index.js", () => ({ detectHardware: async () => highEnd as HardwareProfile }));
vi.mock("../../src/runtimes/index.js", () => ({ detectAllRuntimes: async () => [] }));

const { scanCommand } = await import("../../src/cli/commands/scan.js");

async function scanOutput(format: "json" | "csv"): Promise<string> {
  const out: string[] = [];
  vi.spyOn(console, "log").mockImplementation((...args) => { out.push(args.join(" ")); });
  await scanCommand({ format, category: "all", top: 3 });
  return out.join("\n");
}

async function scanJson(): Promise<Record<string, unknown>> {
  return JSON.parse(await scanOutput("json"));
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("scan --format json catalog", () => {
  it("reports a live catalog with an ISO fetch time", async () => {
    const fetchedAt = Date.UTC(2026, 10, 18, 12);
    health.current = { source: "live", fetchedAt, liveModels: 87, lastError: null, lastSuccessAt: fetchedAt };
    expect((await scanJson()).catalog).toEqual({ source: "live", fetchedAt: "2026-11-18T12:00:00.000Z", liveModels: 87 });
  });

  it("reports the bundled list with a null fetch time", async () => {
    health.current = { source: "bundled", fetchedAt: null, liveModels: 0, lastError: "HTTP 503", lastSuccessAt: null };
    expect((await scanJson()).catalog).toEqual({ source: "bundled", fetchedAt: null, liveModels: 0 });
  });
});

describe("scan memory per recommendation", () => {
  beforeEach(() => {
    health.current = { source: "bundled", fetchedAt: null, liveModels: 0, lastError: null, lastSuccessAt: null };
  });

  it("json: memory at Ollama's default context; every pre-existing field keeps its value", async () => {
    const recs = (await scanJson()).recommendations as Record<string, unknown>[];
    // scan --format json recommendations before `memory` was added.
    expect(recs.map(({ memory: _memory, ...rec }) => rec)).toEqual([
      { rank: 1, model: "Qwen 2.5 Coder 14B", quantization: "Q4_K_M", fitLevel: "excellent", vramMb: 8572, compositeScore: 91, pullCommand: "ollama pull qwen2.5-coder:14b" },
      { rank: 2, model: "Qwen 3 14B", quantization: "Q4_K_M", fitLevel: "excellent", vramMb: 8846, compositeScore: 91, pullCommand: "ollama pull qwen3:14b" },
      { rank: 3, model: "Gemma 2 27B", quantization: "Q4_0", fitLevel: "excellent", vramMb: 14904, compositeScore: 90, pullCommand: "ollama pull gemma2:27b" },
    ]);
    // Rule: Ollama's 32768 default on this GPU is capped by the training context (Gemma 2 27B: 8192).
    expect(recs.map((rec) => (rec.memory as { contextTokens: number }).contextTokens)).toEqual([32768, 32768, 8192]);
  });

  it("csv: memory columns appended after the existing ones, which keep their values", async () => {
    const [header, ...rows] = (await scanOutput("csv")).split("\n");
    expect(header).toBe(
      "rank,model,quantization,fitLevel,vramMb,compositeScore,pullCommand," +
      "contextTokens,weightsMb,weightsExact,kvCacheMb,kvExact,computeMb,totalMb,budgetMb,budgetExact",
    );
    expect(rows.map((row) => row.split(",").slice(0, 7).join(","))).toEqual([
      "1,Qwen 2.5 Coder 14B,Q4_K_M,excellent,8572,91,ollama pull qwen2.5-coder:14b",
      "2,Qwen 3 14B,Q4_K_M,excellent,8846,91,ollama pull qwen3:14b",
      "3,Gemma 2 27B,Q4_0,excellent,14904,90,ollama pull gemma2:27b",
    ]);
  });
});
