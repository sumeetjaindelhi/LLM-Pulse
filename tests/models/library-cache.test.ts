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
const {
  getLibraryCatalog,
  resetLibraryCache,
  readLibraryCatalogSync,
  refreshLibraryCatalogIfStale,
  readLibraryStatus,
} = await import("../../src/models/library-cache.js");
const { writeCache } = await import("../../src/core/cache.js");

const catalog = [
  { slug: "llama3.1", description: "Llama 3.1", parameterSizes: ["8b"], capabilities: ["tools"] },
];

const HOUR_MS = 60 * 60 * 1000;
const DAY_MS = 24 * HOUR_MS;
const NOW = Date.UTC(2026, 10, 18, 12);

function cardHtml(slug: string): string {
  return `<li><a href="/library/${slug}"><p class="max-w-lg">${slug} model</p></a></li>`;
}

function okFetch(...slugs: string[]) {
  return vi.fn().mockImplementation(async () => ({
    ok: true,
    url: "https://ollama.com/library?sort=newest",
    headers: new Headers(),
    text: async () => slugs.map(cardHtml).join(""),
  }));
}

describe("getLibraryCatalog", () => {
  beforeEach(() => {
    resetLibraryCache();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  afterAll(() => {
    rmSync(home.dir, { recursive: true, force: true });
  });

  it("keeps the stale catalog when a refresh fetch fails", async () => {
    writeCache("ollama-library", catalog, { ttlMs: 60_000, version: 2 });
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));

    expect(await getLibraryCatalog({ refresh: true })).toEqual(catalog);
  });
  it("makes no request while offline and returns what the cache holds, or []", async () => {
    process.env.LLMPULSE_OFFLINE = "1";
    try {
      const fetchMock = okFetch("qwen3.8");
      vi.stubGlobal("fetch", fetchMock);
      expect(await getLibraryCatalog({ refresh: true })).toEqual([]);
      writeCache("ollama-library", catalog, { ttlMs: 60_000, version: 2 });
      expect(await getLibraryCatalog({ refresh: true })).toEqual(catalog);
      expect(fetchMock).not.toHaveBeenCalled();
    } finally {
      delete process.env.LLMPULSE_OFFLINE;
    }
  });

  it("ignores a v1 cache file and refetches", async () => {
    writeCache("ollama-library", catalog, { ttlMs: 60_000, version: 1 });
    expect(readLibraryCatalogSync()).toBeNull();
    const fetchMock = okFetch("qwen3.8");
    vi.stubGlobal("fetch", fetchMock);

    const cards = await getLibraryCatalog();

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(cards.map((c) => c.slug)).toEqual(["qwen3.8"]);
  });

  it("reads the disk cache at any age and keeps one array identity", () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(NOW - 30 * DAY_MS);
    writeCache("ollama-library", catalog, { ttlMs: DAY_MS, version: 2 });
    vi.setSystemTime(NOW);

    const first = readLibraryCatalogSync();
    expect(first!.cards).toEqual(catalog);
    expect(first!.fetchedAt).toBe(NOW - 30 * DAY_MS);
    expect(readLibraryCatalogSync()!.cards).toBe(first!.cards);
  });

  it("does not refetch while the memo is inside the TTL", async () => {
    const fetchMock = okFetch("qwen3.8");
    vi.stubGlobal("fetch", fetchMock);
    await refreshLibraryCatalogIfStale({ timeoutMs: 5000 });
    await refreshLibraryCatalogIfStale({ timeoutMs: 5000 });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("refetches once the memo is older than 24 hours", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(NOW);
    const fetchMock = okFetch("qwen3.8");
    vi.stubGlobal("fetch", fetchMock);
    await refreshLibraryCatalogIfStale({ timeoutMs: 5000 });

    vi.setSystemTime(NOW + DAY_MS + 1);
    fetchMock.mockImplementation(async () => ({
      ok: true,
      url: "",
      headers: new Headers(),
      text: async () => cardHtml("gemma4"),
    }));
    await refreshLibraryCatalogIfStale({ timeoutMs: 5000 });

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(readLibraryCatalogSync()!.cards.map((c) => c.slug)).toEqual(["gemma4"]);
    expect(readLibraryCatalogSync()!.fetchedAt).toBe(NOW + DAY_MS + 1);
  });

  it("shares one in-flight fetch between concurrent callers", async () => {
    const fetchMock = okFetch("qwen3.8");
    vi.stubGlobal("fetch", fetchMock);
    await Promise.all([
      refreshLibraryCatalogIfStale({ timeoutMs: 5000 }),
      refreshLibraryCatalogIfStale({ timeoutMs: 5000 }),
      getLibraryCatalog({ refresh: true }),
    ]);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("backs off an implicit refresh for an hour after a failure", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(NOW);
    const fetchMock = vi.fn().mockRejectedValue(new Error("offline"));
    vi.stubGlobal("fetch", fetchMock);

    await refreshLibraryCatalogIfStale({ timeoutMs: 5000 });
    vi.setSystemTime(NOW + HOUR_MS - 1);
    await refreshLibraryCatalogIfStale({ timeoutMs: 5000 });
    expect(fetchMock).toHaveBeenCalledTimes(1);

    vi.setSystemTime(NOW + HOUR_MS);
    await refreshLibraryCatalogIfStale({ timeoutMs: 5000 });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("lets an explicit refresh bypass the backoff", async () => {
    const failing = vi.fn().mockRejectedValue(new Error("offline"));
    vi.stubGlobal("fetch", failing);
    await refreshLibraryCatalogIfStale({ timeoutMs: 5000 });

    const fetchMock = okFetch("qwen3.8");
    vi.stubGlobal("fetch", fetchMock);
    const cards = await getLibraryCatalog({ refresh: true });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(cards.map((c) => c.slug)).toEqual(["qwen3.8"]);
  });

  it("treats getLibraryCatalog on a stale cache as explicit: bypasses backoff", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));
    await refreshLibraryCatalogIfStale({ timeoutMs: 5000 });

    const fetchMock = okFetch("qwen3.8");
    vi.stubGlobal("fetch", fetchMock);
    const cards = await getLibraryCatalog();

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(cards.map((c) => c.slug)).toEqual(["qwen3.8"]);
  });

  it("does not refetch in getLibraryCatalog while the cache is fresh", async () => {
    vi.stubGlobal("fetch", okFetch("qwen3.8"));
    await getLibraryCatalog();
    const fetchMock = okFetch("gemma4");
    vi.stubGlobal("fetch", fetchMock);

    expect((await getLibraryCatalog()).map((c) => c.slug)).toEqual(["qwen3.8"]);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  describe("timeouts", () => {
    // Rejects only when the request is aborted, like a hung connection.
    function hangingFetch() {
      return vi.fn().mockImplementation(
        (_url: string, init: { signal: AbortSignal }) =>
          new Promise((_resolve, reject) => {
            init.signal.addEventListener("abort", () => reject(new Error("aborted")));
          }),
      );
    }

    async function settlesAfter(run: () => Promise<unknown>, ms: number): Promise<boolean> {
      let settled = false;
      const pending = run().then(() => {
        settled = true;
      });
      await vi.advanceTimersByTimeAsync(ms - 1);
      const early = settled;
      await vi.advanceTimersByTimeAsync(1);
      await pending;
      return !early && settled;
    }

    it.each([
      ["an implicit refresh", () => refreshLibraryCatalogIfStale(), 5000],
      ["an explicit refresh", () => getLibraryCatalog({ refresh: true }), 10_000],
      ["a custom timeout", () => refreshLibraryCatalogIfStale({ timeoutMs: 1234 }), 1234],
    ])("aborts %s at the expected time", async (_name, run, ms) => {
      vi.useFakeTimers();
      vi.stubGlobal("fetch", hangingFetch());
      expect(await settlesAfter(run, ms)).toBe(true);
    });
  });

  it("keeps the other process's success when this attempt later fails", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(NOW);
    let rejectFetch!: (err: Error) => void;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(() => new Promise((_resolve, reject) => (rejectFetch = reject))),
    );
    const attempt = refreshLibraryCatalogIfStale({ timeoutMs: 5000 });

    writeCache("ollama-library-status", { lastAttemptAt: NOW, lastSuccessAt: NOW, lastError: null, cards: 2 }, {
      ttlMs: DAY_MS,
      version: 1,
    });
    rejectFetch(new Error("offline"));
    await attempt;

    expect(readLibraryStatus()?.lastSuccessAt).toBe(NOW);
  });

  it("names the network cause in lastError", async () => {
    const error = new Error("fetch failed", { cause: new Error("connect ECONNREFUSED 127.0.0.1:1") });
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(error));
    await refreshLibraryCatalogIfStale({ timeoutMs: 5000 });
    expect(readLibraryStatus()?.lastError).toBe("fetch failed: connect ECONNREFUSED 127.0.0.1:1");
  });

  it("stamps memo, status and disk with the time the fetch finished", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(NOW);
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(async () => {
        vi.setSystemTime(NOW + 4000);
        return { ok: true, url: "", headers: new Headers(), text: async () => cardHtml("qwen3.8") };
      }),
    );
    await refreshLibraryCatalogIfStale({ timeoutMs: 5000 });

    expect(readLibraryCatalogSync()!.fetchedAt).toBe(NOW + 4000);
    expect(readLibraryStatus()).toMatchObject({ lastAttemptAt: NOW, lastSuccessAt: NOW + 4000 });
  });

  it("ignores a disk catalog above the 2000-card cap", () => {
    const oversized = Array.from({ length: 2001 }, (_, i) => ({ ...catalog[0], slug: `m${i}` }));
    writeCache("ollama-library", oversized, { ttlMs: DAY_MS, version: 2 });
    expect(readLibraryCatalogSync()).toBeNull();
  });

  it("never throws from refreshLibraryCatalogIfStale", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));
    await expect(refreshLibraryCatalogIfStale({ timeoutMs: 5000 })).resolves.toBeUndefined();
  });

  it("records attempts in the status record", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(NOW);
    expect(readLibraryStatus()).toBeNull();

    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));
    await refreshLibraryCatalogIfStale({ timeoutMs: 5000 });
    expect(readLibraryStatus()).toEqual({
      lastAttemptAt: NOW,
      lastSuccessAt: null,
      lastError: "offline",
      cards: 0,
    });

    vi.setSystemTime(NOW + HOUR_MS);
    vi.stubGlobal("fetch", okFetch("qwen3.8", "gemma4"));
    await getLibraryCatalog({ refresh: true });
    expect(readLibraryStatus()).toEqual({
      lastAttemptAt: NOW + HOUR_MS,
      lastSuccessAt: NOW + HOUR_MS,
      lastError: null,
      cards: 2,
    });
  });

  it("keeps lastSuccessAt when a later attempt fails", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(NOW);
    vi.stubGlobal("fetch", okFetch("qwen3.8"));
    await getLibraryCatalog({ refresh: true });

    vi.setSystemTime(NOW + 2 * DAY_MS);
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));
    await refreshLibraryCatalogIfStale({ timeoutMs: 5000 });

    expect(readLibraryStatus()).toMatchObject({
      lastAttemptAt: NOW + 2 * DAY_MS,
      lastSuccessAt: NOW,
      lastError: "offline",
      cards: 1,
    });
  });
});
