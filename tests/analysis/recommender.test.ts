import { describe, it, expect, vi, beforeEach } from "vitest";
import type { GpuInfo, HardwareProfile, ModelEntry, QuantizationVariant } from "../../src/core/types.js";
import cpuOnly from "../fixtures/hardware-profiles/cpu-only.json";

const catalog = vi.hoisted(() => ({ models: [] as ModelEntry[] }));
vi.mock("../../src/models/database.js", () => ({
  filterByCategory: () => catalog.models,
}));

const { getRecommendations } = await import("../../src/analysis/recommender.js");

const POOL_MB = 10240;

// 8B model: KV is 150 MB per 1k tokens.
function row(id: string, vramMb: number, over: Partial<ModelEntry> = {}, quants?: QuantizationVariant[]): ModelEntry {
  return {
    id,
    name: id,
    provider: "Test",
    parametersBillion: 8,
    contextWindow: 32768,
    categories: ["general"],
    qualityTier: "strong",
    qualityScore: 80,
    quantizations: quants ?? [{ name: "Q4_K_M", bitsPerWeight: 4.83, vramMb, qualityRetention: 0.9 }],
    ollamaTag: `${id}:8b`,
    releaseDate: "2025-06",
    ...over,
  };
}

function gpu(over: Partial<GpuInfo> = {}): GpuInfo {
  return {
    vendor: "NVIDIA", model: "Test GPU", vramMb: POOL_MB, driverVersion: "1", acceleratorVersion: "12",
    acceleratorType: "cuda", utilizationPercent: null, temperatureCelsius: null, vramUsedMb: null, ...over,
  };
}

function hardware(primaryGpu: GpuInfo | null): HardwareProfile {
  const base = cpuOnly as HardwareProfile;
  return { ...base, gpus: primaryGpu ? [primaryGpu] : [], primaryGpu };
}

const discrete = hardware(gpu());
// Every machine below has the same 8759 MiB budget: discrete 10240 - 457 - 1024; Metal 10295 - 512 - 1024; CPU-only RAM 8759.
const BUDGET_MB = 8759;
const unified = hardware(gpu({ vendor: "Apple", acceleratorType: "metal", unifiedMemory: true, vramMb: BUDGET_MB + 1536 }));
const cpuPool = { ...hardware(null), memory: { ...(cpuOnly as HardwareProfile).memory, availableMb: BUDGET_MB } };

const ids = (hw: HardwareProfile, opts = {}) => getRecommendations(hw, { top: 20, ...opts }).map((r) => r.score.model.id);

beforeEach(() => {
  catalog.models = [];
});

describe("getRecommendations ranking", () => {
  // Weights in MiB; need at the 4096 default = weights + 614.4 KV + 563.2 compute = weights + 1177.6; budget 8759.
  // 5000: excellent (6177.6 <= 0.8 x 8759). 7000: comfortable (8177.6 <= 8759).
  // 9500: barely (10677.6 > budget, >= 0.75 x need). 20000: cannot_run.
  const mixed = () => [
    row("cannot", 20000, { qualityScore: 99, qualityTier: "frontier" }),
    row("barely", 9500, { qualityScore: 95, qualityTier: "frontier" }),
    row("comfortable", 7000, { qualityScore: 90, qualityTier: "frontier" }),
    row("roomy", 5000, { qualityScore: 40, qualityTier: "lightweight" }),
  ];

  // Rule: rank is the fit level (excellent/comfortable 0, tight 1, barely 2, cannot_run 3), then composite.
  it("ranks by fit level, then composite: comfortable before barely before cannot_run", () => {
    catalog.models = mixed();
    expect(ids(discrete)).toEqual(["comfortable", "roomy", "barely", "cannot"]);
  });

  it("ranks tight above barely even when barely scores higher", () => {
    catalog.models = [
      row("barely-hi", 22500, { qualityScore: 99, qualityTier: "frontier" }),
      row("tight-lo", 19000, { qualityScore: 40, qualityTier: "lightweight" }),
    ];
    // 24 GB card: default context 32768 (KV about 4.7 GiB), budget 23095. 19000 MiB fits only at a reduced context; 22500 overflows into offload.
    const big = hardware(gpu({ vramMb: 24576 }));
    const levels = getRecommendations(big, { top: 2 }).map((r) => [r.score.model.id, r.score.fitLevel]);
    expect(levels).toEqual([["tight-lo", "tight"], ["barely-hi", "barely"]]);
  });

  it("excludes embedding rows by id or tag", () => {
    catalog.models = [
      row("nomic-embed-text", 500, { ollamaTag: "nomic-embed-text:latest" }),
      row("vectors", 500, { ollamaTag: "mxbai-Embed-large:335m" }),
      row("chat", 5000),
    ];
    expect(ids(discrete)).toEqual(["chat"]);
  });

  it("excludes rows whose only quant is ternary, keeps ternary alongside others", () => {
    const quant = (name: string): QuantizationVariant => ({ name, bitsPerWeight: 2, vramMb: 2000, qualityRetention: 0.9 });
    catalog.models = [
      row("bitnet", 2000, { ollamaTag: "bitnet:2b" }, [quant("ternary")]),
      row("plain", 2000, { ollamaTag: "plain:2b" }, [quant("default")]),
      row("mixed", 2000, { ollamaTag: "mixed:2b" }, [quant("ternary"), quant("Q4_K_M")]),
    ];
    expect(ids(discrete).sort()).toEqual(["mixed", "plain"]);
  });

  it("ranks a comfortable row above a tight row with a higher composite", () => {
    catalog.models = [
      row("tight-hi", 19000, { qualityScore: 99, qualityTier: "frontier" }),
      row("comfy-lo", 5000, { qualityScore: 40, qualityTier: "lightweight" }),
    ];
    const big = hardware(gpu({ vramMb: 24576 }));
    expect(ids(big)).toEqual(["comfy-lo", "tight-hi"]);
  });

  // Rule: at 48 GB Ollama defaults to 262144 tokens, so large models read "tight" there; a tight row that still holds
  // 32768 tokens (Ollama's default below 48 GiB) ranks with the comfortable ones.
  it("treats a tight row with 32768 tokens of room as comfortable", () => {
    catalog.models = [
      row("big", 40000, { qualityScore: 99, qualityTier: "frontier", contextWindow: 262144 }),
      row("small", 5000, { qualityScore: 40, qualityTier: "lightweight", contextWindow: 262144 }),
    ];
    const recs = getRecommendations(hardware(gpu({ vramMb: 49140 })), { top: 2 });
    expect(recs.map((r) => [r.score.model.id, r.score.fitLevel])).toEqual([["big", "tight"], ["small", "comfortable"]]);
  });

  it("drops barely rows when there is no discrete GPU to offload to", () => {
    catalog.models = mixed();
    expect(ids(unified)).toEqual(["comfortable", "roomy", "cannot"]);
    // Rule: CPU-only has no "barely" level (below the 4096 minimum context it is cannot_run), so that row is not dropped, it ranks last.
    // Order is composite score: cannot (78) ranks before barely (76).
    expect(ids(cpuPool)).toEqual(["comfortable", "roomy", "cannot", "barely"]);
  });

  it("onlyFitting removes cannot_run", () => {
    catalog.models = mixed();
    expect(ids(discrete, { onlyFitting: true })).toEqual(["comfortable", "roomy", "barely"]);
  });

  it("breaks composite ties by quality score, then by id", () => {
    const retention = (r: number): QuantizationVariant[] => [{ name: "Q4_K_M", bitsPerWeight: 4.83, vramMb: 5000, qualityRetention: r }];
    catalog.models = [
      row("a-low", 5000, { qualityScore: 76, ollamaTag: "alpha:8b" }, retention(0.99)),
      row("z-high", 5000, { qualityScore: 80, ollamaTag: "zeta:8b" }, retention(0.91)),
      row("m-twin", 5000, { qualityScore: 80, ollamaTag: "mu:8b" }, retention(0.91)),
    ];
    const recs = getRecommendations(discrete, { top: 3 });
    expect(new Set(recs.map((r) => r.score.compositeScore)).size).toBe(1);
    expect(recs.map((r) => r.score.model.id)).toEqual(["m-twin", "z-high", "a-low"]);
  });
});

describe("getRecommendations default quant", () => {
  const quants = (...names: string[]): QuantizationVariant[] =>
    names.map((name, i) => ({ name, bitsPerWeight: 4 + i, vramMb: 4000 + i * 500, qualityRetention: 0.85 + i * 0.07 }));

  it("scores Q4_K_M for curated rows even when a higher-scoring quant exists", () => {
    catalog.models = [row("curated", 0, {}, quants("Q8_0", "Q4_K_M", "Q2_K"))];
    expect(getRecommendations(discrete)[0].score.quantization.name).toBe("Q4_K_M");
  });

  it("falls back to the first quant when Q4_K_M is absent", () => {
    catalog.models = [row("odd", 0, {}, quants("Q5_K_M", "Q8_0"))];
    expect(getRecommendations(discrete)[0].score.quantization.name).toBe("Q5_K_M");
  });

  it("scores the single default quant of a library row", () => {
    catalog.models = [row("live", 0, { source: "library" }, quants("default"))];
    expect(getRecommendations(discrete)[0].score.quantization.name).toBe("default");
  });
});

describe("getRecommendations families", () => {
  it("keeps the best row per family and treats coders as their own family", () => {
    catalog.models = [
      row("qwen3-8b", 5000, { ollamaTag: "qwen3:8b", qualityScore: 70 }),
      row("qwen38-27b", 5000, { ollamaTag: "qwen3.8:27b", qualityScore: 90 }),
      row("qwen3-coder", 5000, { ollamaTag: "qwen3-coder:30b", qualityScore: 80 }),
      row("qwen25-coder", 5000, { ollamaTag: "qwen2.5-coder:7b", qualityScore: 60 }),
      row("untagged-a", 5000, { ollamaTag: null, qualityScore: 50 }),
    ];
    expect(ids(discrete)).toEqual(["qwen38-27b", "qwen3-coder", "untagged-a"]);
  });

  it("falls back to the id as the family when the tag slug has no letter prefix", () => {
    catalog.models = [
      row("one", 5000, { ollamaTag: "123:8b", qualityScore: 70 }),
      row("two", 5000, { ollamaTag: "456:8b", qualityScore: 60 }),
    ];
    expect(ids(discrete)).toEqual(["one", "two"]);
  });
});
