import { describe, it, expect, vi, afterEach } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import type { LibraryCatalogModel, ModelEntry } from "../../src/core/types.js";
import { mergeLiveCatalog, qualityTierFor } from "../../src/models/library-entries.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const CURATED_FILE = join(__dirname, "../../data/models.json");
const FETCHED_AT = new Date("2026-11-18T00:00:00Z");

const bundled: ModelEntry[] = JSON.parse(readFileSync(CURATED_FILE, "utf-8"));

function card(slug: string, overrides: Partial<LibraryCatalogModel> = {}): LibraryCatalogModel {
  return {
    slug,
    description: "",
    parameterSizes: ["8b"],
    capabilities: [],
    updated: "2026-10-20",
    ...overrides,
  };
}

function curatedRow(overrides: Partial<ModelEntry> = {}): ModelEntry {
  return {
    id: "curated-x",
    name: "Curated X",
    provider: "Test",
    parametersBillion: 8,
    contextWindow: 131072,
    categories: ["general"],
    qualityTier: "strong",
    qualityScore: 80,
    quantizations: [{ name: "Q4_K_M", bitsPerWeight: 4.83, vramMb: 5000, qualityRetention: 0.92 }],
    ollamaTag: "curatedx:8b",
    releaseDate: "2026-10",
    ...overrides,
  };
}

const merge = (cards: LibraryCatalogModel[], curated: ModelEntry[] = [], at = FETCHED_AT) =>
  mergeLiveCatalog(curated, cards, at).filter((m) => m.source === "library");
const byId = (rows: ModelEntry[], id: string) => rows.find((r) => r.id === id);
// Rule: library rows hold weights only (params x bpw / 8, MiB); KV and compute are added by the memory model.
const expectedVram = (billions: number, bpw: number) => Math.round((billions * 1e9 * bpw) / 8 / 2 ** 20);
// Rule: retention is Q4_K_M's measured exp(-meanKLD) = exp(-0.031273).
const Q4_K_M_RETENTION = Math.exp(-0.031273);

describe("size rules", () => {
  it("builds one fully populated row per size badge", () => {
    const rows = merge([card("qwen3.8", { parameterSizes: ["8b", "27b"], capabilities: ["tools"] })]);
    expect(rows.map((r) => r.id)).toEqual(["qwen3.8:8b", "qwen3.8:27b"]);
    const row = rows[1];
    expect(row).toMatchObject({
      name: "qwen3.8 27B",
      provider: "Ollama library",
      source: "library",
      parametersBillion: 27,
      ollamaTag: "qwen3.8:27b",
      releaseDate: "2026-10",
      contextWindow: 32768,
      contextWindowEstimated: true,
    });
    expect(row.activeParametersBillion).toBeUndefined();
    expect(row.kvMbPer1kTokens).toBeUndefined();
    expect(row.quantizations).toEqual([
      { name: "default", bitsPerWeight: 5, vramMb: expectedVram(27, 5), qualityRetention: Q4_K_M_RETENTION },
    ]);
  });

  it("converts m sizes to billions and skips sizes under 0.1B", () => {
    const rows = merge([card("tiny", { parameterSizes: ["70m", "270m", "1.5b"] })]);
    expect(rows.map((r) => r.id)).toEqual(["tiny:270m", "tiny:1.5b"]);
    expect(rows[0].parametersBillion).toBe(0.27);
    expect(rows[0].name).toBe("tiny 270M");
  });

  it("never synthesizes from eNb or AxBb badges", () => {
    const rows = merge([card("odd", { parameterSizes: ["e2b", "e4b", "8x7b", "16x17b", "8b"] })]);
    expect(rows.map((r) => r.id)).toEqual(["odd:8b"]);
  });

  it("uses an older context window for pre-2025 releases", () => {
    const rows = merge([card("old", { updated: "2024-12-31" })]);
    expect(rows[0]).toMatchObject({ releaseDate: "2024-12", contextWindow: 8192 });
  });

  it("takes the total from the description for size-less local cards", () => {
    const rows = merge([
      card("laguna", { parameterSizes: [], description: "A 33B parameters model for agents." }),
    ]);
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({ id: "laguna", ollamaTag: "laguna", name: "laguna", parametersBillion: 33 });
  });

  it("skips size-less cards with no total in the description", () => {
    expect(merge([card("mystery", { parameterSizes: [], description: "Fast and small." })])).toEqual([]);
  });
});

describe("skip rules", () => {
  it("skips cloud-only cards but keeps cloud cards that have local sizes", () => {
    const rows = merge([
      card("kimi", { parameterSizes: [], cloud: true, description: "A 1000B parameters model." }),
      card("gemma4", { parameterSizes: ["e4b", "31b"], cloud: true }),
    ]);
    expect(rows.map((r) => r.id)).toEqual(["gemma4:31b"]);
  });

  it("skips embedding and decision capabilities", () => {
    const rows = merge([
      card("embedder", { capabilities: ["embedding"] }),
      card("chooser", { capabilities: ["tools", "decision"] }),
      card("chatty", { capabilities: ["tools"] }),
    ]);
    expect(rows.map((r) => r.id)).toEqual(["chatty:8b"]);
  });

  it.each(["llama-guard3", "shieldgemma", "deepseek-ocr", "medgemma", "translategemma"])(
    "skips slug %s",
    (slug) => {
      expect(merge([card(slug)])).toEqual([]);
    },
  );

  it.each([
    "A safety classifier for prompts.",
    "Built for content moderation.",
    "text-to-speech engine",
    "speech recognition model",
    "Fine-tuned for Medical Q&A",
    "clinical notes",
  ])("skips description %s", (description) => {
    expect(merge([card("special", { description })])).toEqual([]);
  });

  it("keeps words that only contain a banned stem", () => {
    expect(merge([card("remedy", { description: "Helps with immediate tasks." })])).toHaveLength(1);
  });

  it("skips non-curated cards released more than 24 months before REF", () => {
    const cards = [
      card("fresh", { updated: "2026-10-20" }),
      card("edge", { updated: "2024-10-05" }),
      card("stale", { updated: "2024-09-30" }),
    ];
    expect(merge(cards).map((r) => r.id)).toEqual(["fresh:8b", "edge:8b"]);
  });

  it("keeps stale cards whose slug matches a curated row", () => {
    const cards = [card("fresh"), card("curatedx", { updated: "2023-01-10", parameterSizes: ["13b"] })];
    expect(merge(cards, [curatedRow()]).map((r) => r.id)).toEqual(["fresh:8b", "curatedx:13b"]);
  });
});

describe("mixture of experts", () => {
  it("reads total and active from an AB-AxB description and propagates the ratio to siblings", () => {
    const rows = merge([
      card("lfm2.5", { parameterSizes: ["8b", "24b"], description: "Hybrid 8B-A1B model." }),
    ]);
    expect(byId(rows, "lfm2.5:8b")!.activeParametersBillion).toBe(1);
    expect(byId(rows, "lfm2.5:24b")!.activeParametersBillion).toBe(3);
  });

  it("reads an 'N B activated' count and applies it to the largest size", () => {
    const rows = merge([
      card("cascade", { parameterSizes: ["30b"], description: "A 30B MoE with 3B activated per token." }),
    ]);
    expect(rows[0].activeParametersBillion).toBe(3);
  });

  it("binds a bare active count to the largest size and scales the siblings", () => {
    const rows = merge([card("m", { parameterSizes: ["4b", "30b"], description: "A 30B MoE with 3B active." })]);
    expect([byId(rows, "m:4b")!.activeParametersBillion, byId(rows, "m:30b")!.activeParametersBillion]).toEqual([
      0.4, 3,
    ]);
  });

  it("ignores an active count that is not smaller than the total", () => {
    const [row] = merge([card("m", { parameterSizes: ["8b"], description: "12B active." })]);
    expect(row.activeParametersBillion).toBeUndefined();
  });

  it("reads 'N B active' and 'activating just N B' counts", () => {
    const rows = merge([
      card("a", { parameterSizes: ["120b"], description: "Only 12B active." }),
      card("b", { parameterSizes: ["120b"], description: "Open model activating just 12B parameters." }),
    ]);
    expect(rows.map((r) => r.activeParametersBillion)).toEqual([12, 12]);
  });

  it("defaults to a total/8 active count for MoE cards without a stated count", () => {
    const rows = merge([
      card("moe", { parameterSizes: ["16b", "64b"], description: "A mixture-of-experts model." }),
      card("moe2", { parameterSizes: ["40b"], description: "Sparse MoE design." }),
    ]);
    expect(rows.map((r) => r.activeParametersBillion)).toEqual([2, 8, 5]);
  });

  it("leaves dense cards without an active count", () => {
    expect(merge([card("dense", { description: "A dense model." })])[0].activeParametersBillion).toBeUndefined();
  });

  it("scores an MoE row between its active and total sizes", () => {
    const [dense, moe, small] = merge([
      card("dense", { parameterSizes: ["32b"] }),
      card("moe", { parameterSizes: ["32b"], description: "A MoE model." }),
      card("small", { parameterSizes: ["4b"] }),
    ]);
    expect(moe.qualityScore).toBeLessThan(dense.qualityScore);
    expect(moe.qualityScore).toBeGreaterThan(small.qualityScore);
  });

  it("weights total^0.75 against active^0.25", () => {
    const [row] = merge([card("m", { parameterSizes: ["30b"], description: "A 30B MoE with 3B active." })]);
    expect(row.qualityScore).toBe(89);
  });
});

describe("ternary", () => {
  it.each(["a ternary model", "1.58-bit weights", "BitNet style"])("sizes %s at 2 bits per weight", (description) => {
    const [row] = merge([card("bonsai", { parameterSizes: ["8b"], description })]);
    expect(row.quantizations[0]).toMatchObject({ name: "ternary", bitsPerWeight: 2, vramMb: expectedVram(8, 2) });
  });

  it("scales effective parameters by 2/5", () => {
    const [plain, ternary] = merge([
      card("plain", { parameterSizes: ["20b"] }),
      card("tern", { parameterSizes: ["20b"], description: "ternary" }),
    ]);
    const [reference] = merge([card("ref", { parameterSizes: ["8b"] })]);
    expect(ternary.qualityScore).toBeLessThan(plain.qualityScore);
    expect(ternary.qualityScore).toBe(reference.qualityScore);
  });
});

describe("release estimate", () => {
  it("uses the running minimum of updated in page order", () => {
    const rows = merge([
      card("a", { updated: "2026-10-20" }),
      card("b", { updated: "2026-07-02" }),
      card("c", { updated: "2026-10-25" }),
      card("d", { updated: undefined }),
    ]);
    expect(rows.map((r) => [r.id, r.releaseDate])).toEqual([
      ["a:8b", "2026-10"],
      ["b:8b", "2026-07"],
      ["c:8b", "2026-07"],
    ]);
  });

  const anchorSlugs = ["w", "x", "y", "z", "v"];
  const anchorCards = anchorSlugs.map((slug) => card(slug, { updated: "2026-10-20", parameterSizes: ["4b"] }));
  const anchors = (releaseDate: string) =>
    anchorSlugs.map((slug) => curatedRow({ id: `cur-${slug}`, ollamaTag: `${slug}:4b`, releaseDate }));
  const cards = [card("early", { updated: "2026-01-10" }), ...anchorCards, card("late", { updated: "2026-11-10" })];

  it("keeps the running minimum when anchors agree", () => {
    const rows = merge(cards, anchors("2026-01"));
    expect(byId(rows, "late:8b")!.releaseDate).toBe("2026-01");
  });

  it("falls back to raw updated dates when under 70% of anchors agree", () => {
    const rows = merge(cards, anchors("2026-10"));
    expect(byId(rows, "late:8b")!.releaseDate).toBe("2026-11");
    expect(byId(rows, "early:8b")!.releaseDate).toBe("2026-01");
  });

  it("does not trip the guard with fewer than 5 anchors", () => {
    const rows = merge(cards, anchors("2026-10").slice(0, 4));
    expect(byId(rows, "late:8b")!.releaseDate).toBe("2026-01");
  });

  it("passes the guard at exactly 70% of anchor slugs agreeing", () => {
    const slugs = Array.from({ length: 10 }, (_, i) => `s${i}`);
    const tenCards = [
      card("early", { updated: "2026-01-10" }),
      ...slugs.map((slug) => card(slug, { updated: "2026-10-20", parameterSizes: ["4b"] })),
      card("late", { updated: "2026-11-10" }),
    ];
    const tenAnchors = (agreeing: number) =>
      slugs.map((slug, i) =>
        curatedRow({ id: `cur-${slug}`, ollamaTag: `${slug}:4b`, releaseDate: i < agreeing ? "2026-01" : "2025-06" }),
      );
    expect(byId(merge(tenCards, tenAnchors(7)), "late:8b")!.releaseDate).toBe("2026-01");
    expect(byId(merge(tenCards, tenAnchors(6)), "late:8b")!.releaseDate).toBe("2026-11");
  });

  it("counts one anchor per curated slug, so one multi-size card cannot trip the guard", () => {
    const sizes = ["1b", "2b", "3b", "4b", "5b", "6b"];
    const family = card("family", { updated: "2025-10-10", parameterSizes: sizes });
    const rows = merge(
      [card("newest", { updated: "2026-11-01" }), family, card("repushed", { updated: "2026-10-20" })],
      sizes.map((size) => curatedRow({ id: `cur-${size}`, ollamaTag: `family:${size}`, releaseDate: "2024-01" })),
    );
    expect(byId(rows, "repushed:8b")!.releaseDate).toBe("2025-10");
  });

  it("does not let multi-size families outvote single-size slugs", () => {
    const singles = ["p", "q", "r"];
    const families = ["f", "g", "h"];
    const cardsForVote = [
      card("early", { updated: "2026-01-10" }),
      ...singles.map((slug) => card(slug, { updated: "2026-10-20", parameterSizes: ["4b"] })),
      ...families.map((slug) => card(slug, { updated: "2026-10-20", parameterSizes: ["1b", "2b", "3b", "4b"] })),
      card("late", { updated: "2026-11-10" }),
    ];
    const curatedVote = [
      ...singles.map((slug) => curatedRow({ id: `cur-${slug}`, ollamaTag: `${slug}:4b`, releaseDate: "2025-06" })),
      ...families.flatMap((slug) =>
        ["1b", "2b", "3b", "4b"].map((size) =>
          curatedRow({ id: `cur-${slug}-${size}`, ollamaTag: `${slug}:${size}`, releaseDate: "2026-01" }),
        ),
      ),
    ];
    expect(byId(merge(cardsForVote, curatedVote), "late:8b")!.releaseDate).toBe("2026-11");
  });

  it("caps REF at the fetch month so future dates cannot shift the window", () => {
    const at = new Date("2026-03-10T00:00:00Z");
    const rows = merge([card("future", { updated: "2026-09-01" }), card("aging", { updated: "2024-04-10" })], [], at);
    expect(rows.map((r) => r.id)).toEqual(["future:8b", "aging:8b"]);
  });
});

afterEach(() => vi.useRealTimers());

describe("quality score", () => {
  it("matches the fitted formula for a fresh 8B row", () => {
    const [row] = merge([card("fresh")]);
    expect(row.qualityScore).toBe(80);
    expect(row.qualityTier).toBe("strong");
  });

  it("penalizes age beyond a 3 month grace", () => {
    const rows = merge([card("new", { updated: "2026-10-20" }), card("aged", { updated: "2026-02-10" })]);
    // aged: running minimum 2026-02, REF 2026-10: 8 months old, 5 past grace.
    expect(byId(rows, "new:8b")!.qualityScore).toBe(80);
    expect(byId(rows, "aged:8b")!.qualityScore).toBe(75);
  });

  it("caps effective size at 98B", () => {
    const [big, bigger] = merge([card("big", { parameterSizes: ["98b"] }), card("bigger", { parameterSizes: ["400b"] })]);
    expect(bigger.qualityScore).toBe(big.qualityScore);
  });

  it("clamps the top of the scale at 100", () => {
    const [row] = merge([card("big-coder", { parameterSizes: ["120b"], pulls: 1e9 }), card("filler", { pulls: 1 })]);
    expect(row.qualityScore).toBe(100);
  });

  it("gives a model released in 2025-01 a long context window", () => {
    const [row] = merge([card("edge", { updated: "2025-01-15" })]);
    expect(row.contextWindow).toBe(32768);
  });

  it("clamps scores to 0..100", () => {
    const [tiny] = merge([card("tiny", { parameterSizes: ["0.1b"], updated: "2025-01-10" })]);
    expect(tiny.qualityScore).toBeGreaterThanOrEqual(0);
    expect(tiny.qualityScore).toBeLessThanOrEqual(100);
  });

  describe("popularity", () => {
    const pulled = (pulls: Array<number | undefined>) =>
      merge(pulls.map((p, i) => card(`m${i}`, { pulls: p }))).map((r) => r.qualityScore);

    it("is zero at the median velocity and clamps at plus or minus 4", () => {
      expect(pulled([1e3, 1e5, 1e7])).toEqual([76, 80, 84]);
    });

    it("scales as 2.5 per decade of velocity", () => {
      expect(pulled([1e4, 1e4, 1e4 * 10 ** 0.8])).toEqual([80, 80, 82]);
    });

    it("divides pulls by days since release", () => {
      const rows = merge([
        card("a", { updated: "2026-11-11", pulls: 1e5 }),
        card("b", { updated: "2026-09-19", pulls: 1e5 }),
        card("c", { updated: "2026-07-21", pulls: 1e5 }),
      ]);
      expect(rows.map((r) => r.qualityScore)).toEqual([82, 80, 78]);
    });

    it("floors the release span at 7 days", () => {
      const rows = merge([
        card("a", { updated: "2026-11-16", pulls: 1e5 }),
        card("b", { updated: "2026-11-11", pulls: 1e5 }),
      ]);
      expect(rows[0].qualityScore).toBe(rows[1].qualityScore);
    });

    it("stays finite when most rows have zero pulls", () => {
      const rows = merge([card("a", { pulls: 0 }), card("b", { pulls: 0 }), card("c", { pulls: 5000 })]);
      expect(rows.every((r) => Number.isFinite(r.qualityScore))).toBe(true);
    });

    it("is zero when pulls are unknown", () => {
      expect(pulled([1e3, undefined, 1e7])[1]).toBe(80);
    });

    it("applies to curated rows through the card with the same slug", () => {
      const row = (pulls: number | undefined) =>
        mergeLiveCatalog(
          [curatedRow()],
          [card("a", { pulls: 1e3 }), card("b", { pulls: 1e3 }), card("curatedx", { pulls })],
          FETCHED_AT,
        )[0].qualityScore;
      expect(row(1e9)).toBe(row(undefined) + 4);
    });
  });

  it("adds 4 for coders", () => {
    const [plain, coder] = merge([card("plain"), card("qwen-coder")]);
    expect(coder.qualityScore).toBe(plain.qualityScore + 4);
  });

  it.each([
    [91, "frontier"],
    [90, "strong"],
    [75, "strong"],
    [74, "good"],
    [60, "good"],
    [59, "lightweight"],
  ])("maps score %i to tier %s", (score, tier) => {
    expect(qualityTierFor(score)).toBe(tier);
  });

  it("sets the tier from the score on live rows", () => {
    const [row] = merge([card("large", { parameterSizes: ["32b"] })]);
    expect(row.qualityTier).toBe(qualityTierFor(row.qualityScore));
  });
});

describe("categories", () => {
  it("gives coders coding, plus reasoning when they think", () => {
    const rows = merge([
      card("qwen3-coder", { capabilities: ["tools"] }),
      card("devstral", { capabilities: ["thinking"] }),
      card("starcoder9"),
      card("codestral"),
    ]);
    expect(rows.map((r) => r.categories)).toEqual([
      ["coding"],
      ["coding", "reasoning"],
      ["coding"],
      ["coding"],
    ]);
  });

  it("derives general rows from capabilities and the description", () => {
    const rows = merge([
      card("plain"),
      card("agent", { capabilities: ["tools", "thinking"], description: "A Multilingual agent." }),
    ]);
    expect(rows[0].categories).toEqual(["general"]);
    expect(rows[1].categories).toEqual(["general", "coding", "reasoning", "multilingual"]);
  });
});

describe("dedupe against curated rows", () => {
  const qwen3 = curatedRow({
    id: "qwen-3-32b",
    ollamaTag: "qwen3:32b",
    parametersBillion: 32,
    releaseDate: "2025-04",
  });

  it("drops an exact tag match but keeps qwen3:30b next to curated qwen3:32b", () => {
    const rows = merge([card("qwen3", { parameterSizes: ["30b", "32b"] })], [qwen3]);
    expect(rows.map((r) => r.id)).toEqual(["qwen3:30b"]);
  });

  it("drops same-slug rows whose params match exactly (phi3:mini vs phi3:3.8b)", () => {
    const phi = curatedRow({ id: "phi-3-mini", ollamaTag: "phi3:mini", parametersBillion: 3.8 });
    const rows = merge([card("phi3", { parameterSizes: ["3.8b", "4b", "14b"] })], [phi]);
    expect(rows.map((r) => r.id)).toEqual(["phi3:4b", "phi3:14b"]);
  });

  it("matches params within 0.05 only, never by a 10% band", () => {
    const row = curatedRow({ ollamaTag: "foo:10b", parametersBillion: 10, id: "foo-10" });
    const rows = merge([card("foo", { parameterSizes: ["9.5b", "10.04b", "10.1b"] })], [row]);
    expect(rows.map((r) => r.id)).toEqual(["foo:9.5b", "foo:10.1b"]);
  });

  it("drops a size-less live row equal to a curated bare tag", () => {
    const tiny = curatedRow({ id: "tinyllama-1.1b", ollamaTag: "tinyllama", parametersBillion: 1.1 });
    const rows = merge([card("tinyllama", { parameterSizes: [], description: "A 1.1B model." })], [tiny]);
    expect(rows).toEqual([]);
  });

  it("never emits an id equal to a curated id or tag", () => {
    const clash = curatedRow({ id: "mistral", ollamaTag: "other:7b", parametersBillion: 7 });
    const rows = merge([card("mistral", { parameterSizes: [], description: "A 7B parameters model." })], [clash]);
    expect(rows).toEqual([]);
    const live = mergeLiveCatalog(bundled, bundledCards(), FETCHED_AT).filter((m) => m.source === "library");
    const taken = new Set(bundled.flatMap((c) => [c.id, c.ollamaTag ?? c.id]));
    expect(live.filter((r) => taken.has(r.id))).toEqual([]);
  });
});

function bundledCards(): LibraryCatalogModel[] {
  const slugs = [...new Set(bundled.map((c) => c.ollamaTag?.split(":")[0] ?? c.id))];
  return slugs.map((slug) => {
    const sizes = bundled
      .filter((c) => (c.ollamaTag?.split(":")[0] ?? c.id) === slug && c.ollamaTag?.includes(":"))
      .map((c) => c.ollamaTag!.split(":")[1]);
    return card(slug, { parameterSizes: [...sizes, "13b"], updated: "2026-10-01" });
  });
}

describe("output order", () => {
  it("returns recomputed curated rows first, then library rows in page order", () => {
    const curated = [curatedRow({ id: "c1", ollamaTag: "c1:8b" }), curatedRow({ id: "c2", ollamaTag: "c2:8b" })];
    const out = mergeLiveCatalog(
      curated,
      [card("zeta", { parameterSizes: ["4b", "9b"] }), card("alpha", { parameterSizes: ["2b"] })],
      FETCHED_AT,
    );
    expect(out.map((r) => r.id)).toEqual(["c1", "c2", "zeta:4b", "zeta:9b", "alpha:2b"]);
    expect(out.map((r) => r.source)).toEqual(["curated", "curated", "library", "library", "library"]);
  });

  it("keeps curated facts exact and does not mutate the input", () => {
    const row = curatedRow();
    const snapshot = structuredClone(row);
    const [out] = mergeLiveCatalog([row], [], FETCHED_AT);
    expect(row).toEqual(snapshot);
    expect({ ...out, qualityScore: 0, qualityTier: "good", source: undefined }).toEqual({
      ...snapshot,
      qualityScore: 0,
      qualityTier: "good",
      source: undefined,
    });
  });

  it("is deterministic", () => {
    const cards = [card("a", { pulls: 5e5 }), card("b", { pulls: 2e3 })];
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-11-18T00:00:00Z"));
    const first = mergeLiveCatalog(bundled, cards, FETCHED_AT);
    vi.setSystemTime(new Date("2027-06-01T00:00:00Z"));
    expect(mergeLiveCatalog(bundled, cards, FETCHED_AT)).toEqual(first);
  });
});

describe("calibration against the bundled models", () => {
  it("reproduces at least 80% of the hand-assigned tiers from size and date alone", () => {
    const refit = mergeLiveCatalog(bundled, [], new Date("2025-04-15T00:00:00Z"));
    const matches = bundled.filter((row, i) => refit[i].qualityTier === row.qualityTier).length;
    expect(matches / bundled.length).toBeGreaterThanOrEqual(0.8);
  });
});
