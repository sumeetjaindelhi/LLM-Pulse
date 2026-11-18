import { describe, it, expect, vi, afterEach } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { parseLibraryHtml, fetchOllamaLibrary } from "../../src/models/ollama-library.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
// Real cards trimmed from https://ollama.com/library?sort=newest, original order kept.
const FIXTURE = join(__dirname, "../fixtures/ollama-library/library-newest.html");
const FETCHED_AT = Date.UTC(2026, 10, 18);

const html = readFileSync(FIXTURE, "utf-8");
const models = parseLibraryHtml(html, FETCHED_AT);
const bySlug = (slug: string) => models.find((m) => m.slug === slug)!;

function card(slug: string, extra = "", description = "d"): string {
  return `<li><a href="/library/${slug}"><p class="max-w-lg">${description}</p>${extra}</a></li>`;
}
const pullsSpan = (text: string) =>
  `<span >${text}</span><span class="hidden sm:flex">&nbsp;Pulls</span>`;
const updatedAttr = (stamp: string) => `<span title="${stamp}"></span>`;

describe("parseLibraryHtml on the newest-first page", () => {
  it("keeps page order, which is creation order", () => {
    expect(models.map((m) => m.slug)).toEqual([
      "ripple-preview",
      "kimi-k3.1-pro",
      "bonsai-2",
      "pplx-decider",
      "embeddinggemma-2",
      "nemotron-3.5-super",
      "qwen3.8",
      "lfm2.5",
      "granite4.1-guardian",
      "nemotron-cascade-2",
      "gemma4",
      "nemotron-3-super",
      "glm-4.7-flash",
      "qwen3-coder",
      "qwen3",
      "llama4",
      "gemma3",
      "llama3.1",
      "mixtral",
      "mistral",
    ]);
  });

  it("converts K/M suffixed pull counts", () => {
    expect(bySlug("llama3.1").pulls).toBe(121_200_000);
    expect(bySlug("qwen3.8").pulls).toBe(6_800_000);
    expect(bySlug("kimi-k3.1-pro").pulls).toBe(23_400);
  });

  it("reads the updated date from the UTC title", () => {
    expect(bySlug("qwen3.8").updated).toBe("2026-11-08");
    expect(bySlug("llama3.1").updated).toBe("2024-11-30");
    expect(bySlug("qwen3-coder").updated).toBe("2025-09-23");
  });

  it("flags cloud cards, with or without local sizes", () => {
    expect(bySlug("kimi-k3.1-pro").cloud).toBe(true);
    expect(bySlug("kimi-k3.1-pro").parameterSizes).toEqual([]);
    expect(bySlug("gemma4").cloud).toBe(true);
    expect(bySlug("gemma4").parameterSizes).toEqual(["e2b", "e4b", "12b", "26b", "31b"]);
    expect(bySlug("qwen3.8").cloud).toBeUndefined();
  });

  it("keeps size-less cards and descriptions that carry the sizes", () => {
    expect(bySlug("glm-4.7-flash").parameterSizes).toEqual([]);
    expect(bySlug("glm-4.7-flash").description).toMatch(/30B class/);
    expect(bySlug("nemotron-cascade-2").description).toMatch(/3B activated/);
    expect(bySlug("nemotron-3-super").description).toMatch(/activating just 12B/);
  });

  it("keeps embedding and decision cards", () => {
    expect(bySlug("embeddinggemma-2").parameterSizes).toEqual(["270m", "440m", "570m", "740m"]);
    expect(bySlug("pplx-decider").description).toMatch(/decision model/);
  });
});

describe("parseLibraryHtml field rules", () => {
  it.each([
    ["121.2M", 121_200_000],
    ["116.7K", 116_700],
    ["4,355", 4355],
    ["1.2B", 1_200_000_000],
    ["12", 12],
  ])("parses pulls %s", (text, expected) => {
    expect(parseLibraryHtml(card("m", pullsSpan(text)), FETCHED_AT)[0].pulls).toBe(expected);
  });

  it("drops malformed or out-of-range pulls", () => {
    expect(parseLibraryHtml(card("m", pullsSpan("1.2.3M")), FETCHED_AT)[0].pulls).toBeUndefined();
    expect(parseLibraryHtml(card("m", pullsSpan("2000B")), FETCHED_AT)[0].pulls).toBeUndefined();
    expect(parseLibraryHtml(card("m"), FETCHED_AT)[0].pulls).toBeUndefined();
  });

  it("converts the UTC title to YYYY-MM-DD", () => {
    const parsed = parseLibraryHtml(card("m", updatedAttr("Feb 3, 2024 1:05 AM UTC")), FETCHED_AT);
    expect(parsed[0].updated).toBe("2024-02-03");
  });

  it("rejects dates before 2023, in the future, or impossible", () => {
    for (const stamp of [
      "Dec 31, 2022 1:05 AM UTC",
      "Nov 20, 2026 1:05 AM UTC",
      "Feb 31, 2024 1:05 AM UTC",
    ]) {
      expect(parseLibraryHtml(card("m", updatedAttr(stamp)), FETCHED_AT)[0].updated).toBeUndefined();
    }
  });

  it("accepts a date up to one day after fetchedAt", () => {
    const parsed = parseLibraryHtml(card("m", updatedAttr("Nov 19, 2026 1:05 AM UTC")), FETCHED_AT);
    expect(parsed[0].updated).toBe("2026-11-19");
  });

  it("flags cloud only for a text-cyan-500 span containing cloud", () => {
    const cyan = (cls: string, text: string) => `<span class="${cls}">${text}</span>`;
    expect(parseLibraryHtml(card("m", cyan("bg-cyan-50 text-cyan-500", "cloud")), FETCHED_AT)[0].cloud).toBe(true);
    expect(parseLibraryHtml(card("m", cyan("text-cyan-500", "other")), FETCHED_AT)[0].cloud).toBeUndefined();
    expect(parseLibraryHtml(card("m", cyan("text-blue-600", "cloud")), FETCHED_AT)[0].cloud).toBeUndefined();
    expect(parseLibraryHtml(card("m", cyan("not-text-cyan-500", "cloud")), FETCHED_AT)[0].cloud).toBeUndefined();
  });

  it("strips control characters from size and capability badges", () => {
    const span = (cls: string, text: string) => `<span class="${cls}">${text}</span>`;
    const [parsed] = parseLibraryHtml(
      card("m", span("text-blue-600", "7b\u001b[2J") + span("text-indigo-600", "\u001b]0;pwn\u0007tools")),
      FETCHED_AT,
    );
    expect(parsed.parameterSizes).toEqual(["7b[2J"]);
    expect(parsed.capabilities).toEqual(["]0;pwntools"]);
  });

  it("replaces control characters in descriptions and collapses whitespace", () => {
    const parsed = parseLibraryHtml(card("m", "", "a\u0000b\u0085c\u007fd\n\n  e"), FETCHED_AT);
    expect(parsed[0].description).toBe("a b c d e");
  });

  it("caps descriptions at 2048 characters", () => {
    const parsed = parseLibraryHtml(card("m", "", "x".repeat(5000)), FETCHED_AT);
    expect(parsed[0].description).toHaveLength(2048);
  });

  it("rejects slugs with a slash or uppercase", () => {
    expect(parseLibraryHtml(card("user/model"), FETCHED_AT)).toEqual([]);
    expect(parseLibraryHtml(card("Llama3"), FETCHED_AT)).toEqual([]);
    expect(parseLibraryHtml(card("-lead"), FETCHED_AT)).toEqual([]);
    expect(parseLibraryHtml(card("a".repeat(65)), FETCHED_AT)).toEqual([]);
    expect(parseLibraryHtml(card("a".repeat(64)), FETCHED_AT)).toHaveLength(1);
  });

  it("caps the number of cards at 2000", () => {
    const many = Array.from({ length: 2100 }, (_, i) => card(`m${i}`)).join("");
    expect(parseLibraryHtml(many, FETCHED_AT)).toHaveLength(2000);
  });
});

describe("parseLibraryHtml hostile input timing", () => {
  it("handles a huge pulls-like digit run quickly", () => {
    const hostile = card("m", `<span >${"1".repeat(40_000)}</span><span>&nbsp;Pulls</span>`);
    const start = performance.now();
    parseLibraryHtml(hostile, FETCHED_AT);
    expect(performance.now() - start).toBeLessThan(500);
  });

  it("handles many unterminated cyan spans quickly", () => {
    const hostile = card("m", '<span class="text-cyan-500" '.repeat(2_000));
    const start = performance.now();
    parseLibraryHtml(hostile, FETCHED_AT);
    expect(performance.now() - start).toBeLessThan(500);
  });

  it("parses a page of many distinct unterminated-span cards quickly", () => {
    const page = Array.from({ length: 80 }, (_, i) => card(`m${i}`, "<span ".repeat(10_000))).join("");
    const start = performance.now();
    parseLibraryHtml(page, FETCHED_AT);
    expect(performance.now() - start).toBeLessThan(3000);
  });

  it("handles repeated whitespace between pulls spans quickly", () => {
    const hostile = card("m", `<span >5</span>${" ".repeat(40_000)}x`);
    const start = performance.now();
    parseLibraryHtml(hostile, FETCHED_AT);
    expect(performance.now() - start).toBeLessThan(500);
  });
});

describe("fetchOllamaLibrary", () => {
  const originalFetch = globalThis.fetch;
  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.useRealTimers();
  });

  const okResponse = (url: string) => ({
    ok: true,
    url,
    headers: new Headers(),
    text: async () => html,
  });

  it("requests the newest-first listing", async () => {
    const fetchMock = vi.fn().mockResolvedValue(okResponse("https://ollama.com/library?sort=newest"));
    globalThis.fetch = fetchMock as unknown as typeof fetch;
    const result = await fetchOllamaLibrary();
    expect(fetchMock.mock.calls[0][0]).toBe("https://ollama.com/library?sort=newest");
    expect(result.length).toBe(20);
  });

  it("rejects a redirect to another host", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(okResponse("https://evil.example/library")) as unknown as typeof fetch;
    await expect(fetchOllamaLibrary()).rejects.toThrow(/host/);
  });

  it("accepts a response with an empty final url", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue(okResponse("")) as unknown as typeof fetch;
    await expect(fetchOllamaLibrary()).resolves.toHaveLength(20);
  });

  it("aborts the request after timeoutMs", async () => {
    vi.useFakeTimers();
    globalThis.fetch = vi.fn(
      (_url: unknown, init?: RequestInit) =>
        new Promise((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => reject(new Error("aborted")));
        }),
    ) as unknown as typeof fetch;
    const pending = expect(fetchOllamaLibrary({ timeoutMs: 5000 })).rejects.toThrow(/aborted/);
    await vi.advanceTimersByTimeAsync(5000);
    await pending;
  });
});
