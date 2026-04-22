import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { parseLibraryHtml } from "../../src/models/ollama-library.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const FIXTURE = join(__dirname, "../fixtures/ollama-library/library.html");

describe("parseLibraryHtml", () => {
  const html = readFileSync(FIXTURE, "utf-8");
  const models = parseLibraryHtml(html);

  it("parses a substantial number of models from the fixture", () => {
    expect(models.length).toBeGreaterThanOrEqual(50);
  });

  it("extracts well-known slugs", () => {
    const slugs = models.map((m) => m.slug);
    expect(slugs).toContain("llama3.1");
    expect(slugs).toContain("deepseek-r1");
    expect(slugs).toContain("llama3.2");
  });

  it("captures descriptions for known models", () => {
    const llama = models.find((m) => m.slug === "llama3.1");
    expect(llama).toBeDefined();
    expect(llama!.description).toMatch(/llama 3\.1/i);
    expect(llama!.description.length).toBeGreaterThan(20);
  });

  it("parses parameter sizes as short tokens", () => {
    const llama = models.find((m) => m.slug === "llama3.1");
    expect(llama!.parameterSizes).toContain("8b");
    expect(llama!.parameterSizes).toContain("70b");
    expect(llama!.parameterSizes).toContain("405b");
  });

  it("parses capability tags", () => {
    const ds = models.find((m) => m.slug === "deepseek-r1");
    expect(ds!.capabilities).toContain("tools");
    expect(ds!.capabilities).toContain("thinking");
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
      <li x-test-model class="...">
        <a href="/library/<script>alert(1)</script>">
          <p class="max-w-lg">bad</p>
        </a>
      </li>
    `;
    expect(parseLibraryHtml(bad)).toEqual([]);
  });
});
