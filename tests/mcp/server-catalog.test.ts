import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { ModelEntry } from "../../src/core/types.js";

type ToolHandler = (args: Record<string, unknown>) => Promise<{ content: { text: string }[] }>;
const tools = vi.hoisted(() => new Map<string, ToolHandler>());
vi.mock("@modelcontextprotocol/sdk/server/mcp.js", () => ({
  McpServer: class {
    tool(name: string, _description: string, _schema: unknown, handler: ToolHandler) {
      tools.set(name, handler);
    }
    async connect() {}
  },
}));
vi.mock("@modelcontextprotocol/sdk/server/stdio.js", () => ({ StdioServerTransport: class {} }));

const refresh = vi.hoisted(() => vi.fn(async () => {}));
vi.mock("../../src/models/library-cache.js", () => ({ refreshLibraryCatalogIfStale: refresh }));
vi.mock("../../src/core/config.js", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../src/core/config.js")>()),
  // Tests must not read the developer's real ~/.llmpulserc.
  isOffline: () => process.env.LLMPULSE_OFFLINE === "1",
}));

const rows = vi.hoisted(() => ({ list: [] as ModelEntry[] }));
vi.mock("../../src/models/database.js", () => ({
  searchModels: () => rows.list,
  filterByCategory: () => rows.list,
  resolveModel: () => null,
  getCatalogHealth: () => ({ source: "bundled", fetchedAt: null, liveModels: 0, lastError: null, lastSuccessAt: null, offline: true }),
}));

const { startServer } = await import("../../src/mcp/server.js");

function model(i: number): ModelEntry {
  return {
    id: `m${i}`, name: `M${i}`, provider: "Test", parametersBillion: 8, contextWindow: 8192, categories: ["general"],
    qualityTier: "good", qualityScore: 60, quantizations: [{ name: "default", bitsPerWeight: 5, vramMb: 1000, qualityRetention: 0.9 }],
    ollamaTag: `m${i}:8b`, releaseDate: "2026-01",
  };
}

async function callModels(args: Record<string, unknown> = {}): Promise<{ count: number; total: number; models: unknown[] }> {
  const result = await tools.get("models")!({ category: "all", fits: false, limit: 100, ...args });
  return JSON.parse(result.content[0].text);
}

describe("MCP models tool limit", () => {
  beforeEach(() => {
    rows.list = Array.from({ length: 150 }, (_, i) => model(i));
  });

  it("returns the first 100 rows by default and the full total", async () => {
    const out = await callModels();
    expect(out.count).toBe(100);
    expect(out.models).toHaveLength(100);
    expect(out.total).toBe(150);
  });

  it("honours a smaller limit", async () => {
    const out = await callModels({ limit: 5 });
    expect(out.models).toHaveLength(5);
    expect(out.total).toBe(150);
  });

  it("returns every row when the limit exceeds the total", async () => {
    rows.list = rows.list.slice(0, 10);
    const out = await callModels();
    expect(out).toMatchObject({ count: 10, total: 10 });
  });
});

describe("startServer catalog refresh", () => {
  beforeEach(() => {
    refresh.mockClear();
    vi.useFakeTimers();
    vi.stubEnv("LLMPULSE_OFFLINE", "");
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllEnvs();
  });

  it("refreshes at startup without awaiting it, then hourly", async () => {
    let finishRefresh: () => void = () => {};
    refresh.mockImplementationOnce(() => new Promise<void>((resolve) => { finishRefresh = resolve; }));
    await startServer();
    expect(refresh).toHaveBeenCalledTimes(1);

    await vi.advanceTimersByTimeAsync(60 * 60 * 1000);
    expect(refresh).toHaveBeenCalledTimes(2);
    finishRefresh();
  });

  it("never fetches when offline", async () => {
    vi.stubEnv("LLMPULSE_OFFLINE", "1");
    await startServer();
    await vi.advanceTimersByTimeAsync(2 * 60 * 60 * 1000);
    expect(refresh).not.toHaveBeenCalled();
  });
});
