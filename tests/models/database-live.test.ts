import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { parseLibraryHtml } from "../../src/models/ollama-library.js";
import type { LibrarySnapshot, LibraryStatus } from "../../src/models/library-cache.js";

const snapshot = vi.hoisted(() => ({ current: null as unknown }));
const readLibraryCatalogSync = vi.hoisted(() => vi.fn(() => snapshot.current as LibrarySnapshot | null));
const readLibraryStatus = vi.hoisted(() => vi.fn((): LibraryStatus | null => null));
vi.mock("../../src/models/library-cache.js", () => ({ readLibraryCatalogSync, readLibraryStatus }));
vi.mock("../../src/core/config.js", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../src/core/config.js")>()),
  // Tests must not read the developer's real ~/.llmpulserc.
  isOffline: () => process.env.LLMPULSE_OFFLINE === "1",
}));

const { getAllModels, resolveModel, searchModels, getModelByTag, resetModelCache, getCatalogHealth } = await import("../../src/models/database.js");

const __dirname = dirname(fileURLToPath(import.meta.url));
const FETCHED_AT = Date.UTC(2026, 10, 18);
const cards = parseLibraryHtml(
  readFileSync(join(__dirname, "../fixtures/ollama-library/library-newest.html"), "utf-8"),
  FETCHED_AT,
);

function setCatalog(next: LibrarySnapshot | null): void {
  snapshot.current = next;
}

function whileOffline<T>(read: () => T): T {
  vi.stubEnv("LLMPULSE_OFFLINE", "1");
  resetModelCache();
  try {
    return read();
  } finally {
    vi.stubEnv("LLMPULSE_OFFLINE", "");
    resetModelCache();
  }
}

beforeEach(() => {
  setCatalog({ cards, fetchedAt: FETCHED_AT });
  readLibraryCatalogSync.mockClear();
  vi.stubEnv("LLMPULSE_OFFLINE", "");
  resetModelCache();
});

afterEach(() => {
  vi.unstubAllEnvs();
  resetModelCache();
});

describe("getAllModels", () => {
  it("returns only curated rows, marked as such, when offline, without reading the catalog", () => {
    vi.stubEnv("LLMPULSE_OFFLINE", "1");
    resetModelCache();
    const rows = getAllModels();
    expect(rows.length).toBeGreaterThan(0);
    expect(rows.every((m) => m.source === "curated")).toBe(true);
    expect(readLibraryCatalogSync).not.toHaveBeenCalled();
  });

  it("returns curated rows only when no catalog is cached", () => {
    setCatalog(null);
    expect(getAllModels().every((m) => m.source === "curated")).toBe(true);
  });

  it("appends library rows after the curated rows when a catalog is cached", () => {
    const curatedCount = whileOffline(() => getAllModels().length);
    const rows = getAllModels();
    expect(rows.length).toBeGreaterThan(curatedCount);
    expect(rows.slice(0, curatedCount).every((m) => m.source === "curated")).toBe(true);
    expect(rows.slice(curatedCount).every((m) => m.source === "library")).toBe(true);
  });

  it("memoizes by catalog identity and re-merges when the cards change", () => {
    const first = getAllModels();
    expect(getAllModels()).toBe(first);
    setCatalog({ cards: [...cards], fetchedAt: FETCHED_AT });
    expect(getAllModels()).not.toBe(first);
  });

  it("resetModelCache forces a rebuild", () => {
    const first = getAllModels();
    resetModelCache();
    expect(getAllModels()).not.toBe(first);
  });
});

describe("resolveModel with a live catalog", () => {
  it("resolves every curated id, name, tag and tag slug to the same row as the bundled list", () => {
    const queries = whileOffline(() =>
      getAllModels().flatMap((m) => [m.id, m.name, ...(m.ollamaTag ? [m.ollamaTag, m.ollamaTag.split(":")[0]] : [])]),
    );
    const resolveIds = () => queries.map((q) => resolveModel(q)?.id ?? null);

    const bundled = whileOffline(resolveIds);
    const live = resolveIds();
    expect(queries.length).toBeGreaterThan(100);
    expect(live).toEqual(bundled);
  });

  // Library cards whose names contain a curated model's name, which makes the search ambiguous.
  it.each([
    ["yi", "yi-lightning", "9b"],
    ["smollm2", "smollm2-plus", "3b"],
    ["starcoder2", "starcoder2-next", "15b"],
    ["deepseek-coder-v2", "deepseek-coder-v2-next", "30b"],
  ])("returns the curated row for %s when a library row also matches", (query, slug, size) => {
    const lookalike = { slug, description: `${slug} model`, parameterSizes: [size], capabilities: [], updated: "2026-10-01", pulls: 5000 };
    setCatalog({ cards: [lookalike, ...cards], fetchedAt: FETCHED_AT });
    resetModelCache();

    const hits = searchModels(query);
    expect(hits.some((m) => m.source === "library")).toBe(true);
    expect(hits.filter((m) => m.source === "curated")).toHaveLength(1);
    expect(resolveModel(query)?.source).toBe("curated");
  });

  it("searchModels matches an ollama tag, so a bare family name suggests its sizes", () => {
    const tags = searchModels("llama3.1").map((m) => m.ollamaTag);
    expect(tags).toEqual(expect.arrayContaining(["llama3.1:8b", "llama3.1:70b", "llama3.1:405b"]));
  });

  it("still returns null when a search hits several curated rows", () => {
    expect(resolveModel("Llama")).toBeNull();
  });

  it("resolves a library row by its exact tag", () => {
    const row = getAllModels().find((m) => m.source === "library" && m.ollamaTag?.includes(":"));
    expect(row).toBeDefined();
    expect(getModelByTag(row!.ollamaTag!)?.id).toBe(row!.id);
    expect(resolveModel(row!.ollamaTag!)?.source).toBe("library");
  });
});

describe("getMergedModels with a live catalog", () => {
  it("lists curated rows only, so models --live output is unchanged", async () => {
    vi.doMock("../../src/models/ollama-models.js", () => ({ fetchOllamaModels: async () => [] }));
    const { getMergedModels } = await import("../../src/models/merged-models.js");
    const merged = await getMergedModels();
    vi.doUnmock("../../src/models/ollama-models.js");

    expect(merged.length).toBeGreaterThan(0);
    expect(merged.every((m) => m.entry?.source === "curated")).toBe(true);
  });
});

describe("getCatalogHealth", () => {
  afterEach(() => {
    readLibraryStatus.mockReturnValue(null);
  });

  it("reports the live catalog with its row count and fetch time", () => {
    readLibraryStatus.mockReturnValue({ lastAttemptAt: FETCHED_AT, lastSuccessAt: FETCHED_AT, lastError: null, cards: cards.length });
    const libraryRows = getAllModels().filter((m) => m.source === "library").length;
    expect(getCatalogHealth()).toEqual({
      source: "live",
      fetchedAt: FETCHED_AT,
      liveModels: libraryRows,
      lastError: null,
      lastSuccessAt: FETCHED_AT,
      offline: false,
    });
  });

  it("reports the bundled list when no catalog is cached, keeping the fetch error", () => {
    setCatalog(null);
    readLibraryStatus.mockReturnValue({ lastAttemptAt: FETCHED_AT, lastSuccessAt: null, lastError: "HTTP 503", cards: 0 });
    expect(getCatalogHealth()).toEqual({ source: "bundled", fetchedAt: null, liveModels: 0, lastError: "HTTP 503", lastSuccessAt: null, offline: false });
  });

  it("reports the bundled list, with no error, when offline", () => {
    vi.stubEnv("LLMPULSE_OFFLINE", "1");
    readLibraryStatus.mockReturnValue({ lastAttemptAt: FETCHED_AT, lastSuccessAt: null, lastError: "HTTP 503", cards: 0 });
    expect(getCatalogHealth()).toEqual({ source: "bundled", fetchedAt: null, liveModels: 0, lastError: null, lastSuccessAt: null, offline: true });
  });
});
