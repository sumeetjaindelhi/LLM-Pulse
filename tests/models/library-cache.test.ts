import { describe, it, expect, vi, beforeEach, afterEach, afterAll } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// core/cache derives its root from homedir() when it is first imported, so the
// throwaway home must be set before the dynamic imports below. Setting it later
// (e.g. in beforeEach) leaves the root relative to cwd and writes into the repo.
const home = vi.hoisted(() => ({ dir: "" }));
vi.mock("node:os", async (importOriginal) => {
  const actual = await importOriginal<typeof import("node:os")>();
  return { ...actual, homedir: () => home.dir };
});

home.dir = mkdtempSync(join(tmpdir(), "llmpulse-library-cache-"));
const { getLibraryCatalog, resetLibraryCache } = await import("../../src/models/library-cache.js");
const { writeCache } = await import("../../src/core/cache.js");

const catalog = [
  { slug: "llama3.1", description: "Llama 3.1", parameterSizes: ["8b"], capabilities: ["tools"] },
];

describe("getLibraryCatalog", () => {
  beforeEach(() => {
    resetLibraryCache();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  afterAll(() => {
    rmSync(home.dir, { recursive: true, force: true });
  });

  it("keeps the stale catalog when a refresh fetch fails", async () => {
    writeCache("ollama-library", catalog, { ttlMs: 60_000, version: 1 });
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));

    expect(await getLibraryCatalog({ refresh: true })).toEqual(catalog);
  });
});
