import { describe, it, expect, vi, afterEach } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { parseLibraryHtml, fetchOllamaLibrary } from "../../src/models/ollama-library.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
// Trimmed snapshot of https://ollama.com/library (2026-09-13): page head and
// footer kept verbatim, model list cut down to nine representative cards.
const FIXTURE = join(__dirname, "../fixtures/ollama-library/library.html");

describe("parseLibraryHtml", () => {
  const html = readFileSync(FIXTURE, "utf-8");
  const models = parseLibraryHtml(html);

  it("parses every model card in the fixture", () => {
    expect(models.map((m) => m.slug).sort()).toEqual([
      "deepseek-r1",
      "gemma3",
      "llama3.1",
      "llama3.2",
      "llama4",
      "mistral",
      "nomic-embed-text",
      "qwen2.5vl",
      "qwen3",
    ]);
  });

  it("captures descriptions for known models", () => {
    const llama = models.find((m) => m.slug === "llama3.1");
    expect(llama).toBeDefined();
    expect(llama!.description).toMatch(/llama 3\.1/i);
    expect(llama!.description.length).toBeGreaterThan(20);
  });

  it("parses parameter sizes as short tokens", () => {
    const llama = models.find((m) => m.slug === "llama3.1");
    expect(llama!.parameterSizes).toEqual(["8b", "70b", "405b"]);
    expect(models.find((m) => m.slug === "llama4")!.parameterSizes).toEqual(["16x17b", "128x17b"]);
  });

  it("parses capability tags", () => {
    expect(models.find((m) => m.slug === "deepseek-r1")!.capabilities).toEqual(["tools", "thinking"]);
    expect(models.find((m) => m.slug === "llama4")!.capabilities).toEqual(["vision", "tools"]);
  });

  it("returns no sizes for a model card without size badges", () => {
    const embed = models.find((m) => m.slug === "nomic-embed-text");
    expect(embed!.capabilities).toEqual(["embedding"]);
    expect(embed!.parameterSizes).toEqual([]);
  });

  it("dedupes slugs even if HTML contains repeated cards", () => {
    const doubled = html + html;
    const parsed = parseLibraryHtml(doubled);
    const uniqueSlugs = new Set(parsed.map((m) => m.slug));
    expect(parsed.length).toBe(uniqueSlugs.size);
  });

  it("returns empty array on empty HTML", () => {
    expect(parseLibraryHtml("")).toEqual([]);
    expect(parseLibraryHtml("<html><body>no models here</body></html>")).toEqual([]);
  });

  it("skips model blocks with invalid slugs", () => {
    const bad = `
      <li class="flex items-baseline">
        <a href="/library/<script>alert(1)</script>">
          <p class="max-w-lg">bad</p>
        </a>
      </li>
    `;
    expect(parseLibraryHtml(bad)).toEqual([]);
  });

  it("scans many unclosed <li tags in linear time", () => {
    const hostile = "<li>".repeat(200_000);
    const start = performance.now();
    expect(parseLibraryHtml(hostile)).toEqual([]);
    expect(performance.now() - start).toBeLessThan(1000);
  });

  it("skips oversized <li> blocks instead of running quadratic regexes over them", () => {
    const hostile = `<li><a href="/library/x">${"<span ".repeat(20_000)}</li>`;
    const start = performance.now();
    expect(parseLibraryHtml(hostile)).toEqual([]);
    expect(performance.now() - start).toBeLessThan(500);
  });
});

describe("fetchOllamaLibrary", () => {
  const originalFetch = globalThis.fetch;
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it("rejects a response whose Content-Length exceeds the cap without buffering it", async () => {
    const text = vi.fn();
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers({ "content-length": String(6 * 1024 * 1024) }),
      text,
    }) as unknown as typeof fetch;

    await expect(fetchOllamaLibrary()).rejects.toThrow(/exceeds/);
    expect(text).not.toHaveBeenCalled();
  });
});
