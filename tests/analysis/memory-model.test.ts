import { describe, it, expect } from "vitest";
import {
  assessFit,
  autoBatch,
  budgetMb,
  computeBufferMb,
  kqMaskBytes,
  kvCacheBytes,
  layersThatFit,
  maxGpuContext,
  ollamaDefaultContext,
  padContext,
  resolveSliding,
  snapToContextTier,
  speedClass,
  type ArchFacts,
} from "../../src/analysis/memory-model.js";

const MIB = 1024 * 1024;

const llama31_8b: ArchFacts = {
  architecture: "llama",
  layers: 32,
  headCountKv: 8,
  keyLength: 128,
  valueLength: 128,
};

describe("padContext / autoBatch", () => {
  it("pads to a multiple of 256", () => {
    expect(padContext(1)).toBe(256);
    expect(padContext(256)).toBe(256);
    expect(padContext(257)).toBe(512);
    expect(padContext(5000)).toBe(5120);
  });

  it("picks the Ollama batch tier by context", () => {
    expect(autoBatch(4096)).toBe(512);
    expect(autoBatch(4097)).toBe(1024);
    expect(autoBatch(32768)).toBe(1024);
    expect(autoBatch(32769)).toBe(2048);
  });
});

describe("kvCacheBytes: full attention", () => {
  it("llama3.1-8B: 32 x 8 x (128+128) x 2 B = 131072 B/token", () => {
    expect(kvCacheBytes(llama31_8b, 256, "f16").bytes).toBe(131072 * 256);
  });

  it("pads ctx 5000 to 5120 -> 131072 x 5120 = 640 MiB", () => {
    const kv = kvCacheBytes(llama31_8b, 5000, "f16");
    expect(kv.bytes).toBe(131072 * 5120);
    expect(kv.bytes / MIB).toBe(640);
    expect(kv.exact).toBe(true);
  });

  it("q8_0 row = n/32 x 34 B: per layer 2 x 1088 B, x 32 = 69632 B/token", () => {
    // n = 128 x 8 = 1024 -> 1024/32 x 34 = 1088 for K and again for V
    expect(kvCacheBytes(llama31_8b, 256, "q8_0").bytes).toBe(69632 * 256);
  });

  it("q4_0 row = n/32 x 18 B: per layer 2 x 576 B, x 32 = 36864 B/token", () => {
    expect(kvCacheBytes(llama31_8b, 256, "q4_0").bytes).toBe(36864 * 256);
  });

  it("per-layer head counts sum per layer", () => {
    // 2 layers: kv 4 and 8, 64+64 dims, f16: (4 + 8) x 128 x 2 = 3072 B/token
    const facts: ArchFacts = { architecture: "x", layers: 2, headCountKv: [4, 8], keyLength: 64, valueLength: 64 };
    expect(kvCacheBytes(facts, 256, "f16").bytes).toBe(3072 * 256);
  });
});

describe("kvCacheBytes: MLA", () => {
  it("deepseek2: 61 layers x 576 x 2 B = 70272 B/token, no V cache", () => {
    const facts: ArchFacts = {
      architecture: "deepseek2",
      layers: 61,
      headCountKv: 1,
      keyLength: 576,
      valueLength: 512,
      mla: true,
    };
    // 70272 x 4096 = 287834112
    expect(kvCacheBytes(facts, 4096, "f16").bytes).toBe(287_834_112);
  });
});

describe("kvCacheBytes: hybrid / recurrent", () => {
  const hybrid: ArchFacts = {
    architecture: "qwen3next",
    layers: 8,
    headCountKv: [0, 0, 0, 4, 0, 0, 0, 4],
    keyLength: 128,
    valueLength: 128,
  };

  it("layers with 0 kv heads hold no KV: 2 x 4 x 256 x 2 = 4096 B/token", () => {
    expect(kvCacheBytes(hybrid, 256, "f16").bytes).toBe(4096 * 256);
  });

  it("adds constant recurrent state on top", () => {
    expect(kvCacheBytes({ ...hybrid, recurrentStateBytes: 1000 }, 256, "f16").bytes).toBe(4096 * 256 + 1000);
  });

  it("all-recurrent model has only state", () => {
    const mamba: ArchFacts = { ...hybrid, headCountKv: 0, recurrentStateBytes: 5000 };
    expect(kvCacheBytes(mamba, 8192, "f16").bytes).toBe(5000);
  });
});

describe("resolveSliding: layer mask", () => {
  const base = { layers: 12, headCountKv: 4, keyLength: 64, valueLength: 64 } as const;

  it("gemma3 N=6 (not dense-first): il % 6 < 5 is SWA, every 6th layer is full", () => {
    const { mask, windowTokens } = resolveSliding({ ...base, architecture: "gemma3", slidingWindow: 1024 });
    expect(windowTokens).toBe(1024);
    expect(mask).toEqual([true, true, true, true, true, false, true, true, true, true, true, false]);
  });

  it("gpt-oss N=2: even layers SWA, odd layers full", () => {
    const { mask } = resolveSliding({ ...base, architecture: "gpt-oss", slidingWindow: 128 });
    expect(mask.slice(0, 4)).toEqual([true, false, true, false]);
  });

  it("dense-first N=4 (laguna): il % 4 != 0 is SWA", () => {
    const { mask } = resolveSliding({ ...base, layers: 8, architecture: "laguna", slidingWindow: 512 });
    expect(mask).toEqual([false, true, true, true, false, true, true, true]);
  });

  it("explicit boolean[] pattern wins over the arch default", () => {
    const pattern = [true, false, false, true];
    const { mask } = resolveSliding({ ...base, layers: 4, architecture: "gemma3", slidingWindow: 64, slidingWindowPattern: pattern });
    expect(mask).toEqual(pattern);
  });

  it("scalar pattern on an unknown arch uses the not-dense-first rule", () => {
    const { mask, reason } = resolveSliding({ ...base, layers: 4, architecture: "newarch", slidingWindow: 64, slidingWindowPattern: 2 });
    expect(mask).toEqual([true, false, true, false]);
    expect(reason).toBeUndefined();
  });

  it("pattern 0 makes every layer SWA (lfm2)", () => {
    const { mask } = resolveSliding({ ...base, layers: 3, architecture: "lfm2", slidingWindow: 256 });
    expect(mask).toEqual([true, true, true]);
  });

  it("'only if n_swa > 0' archs are full attention without a window, still exact", () => {
    const r = resolveSliding({ ...base, architecture: "gemma3" });
    expect(r.mask.every((m) => !m)).toBe(true);
    expect(r.reason).toBeUndefined();
  });

  it("exaone4 is SWA only at 64 layers, default window 4096", () => {
    expect(resolveSliding({ ...base, layers: 32, architecture: "exaone4" }).mask.some(Boolean)).toBe(false);
    const r = resolveSliding({ ...base, layers: 64, architecture: "exaone4" });
    expect(r.windowTokens).toBe(4096);
    expect(r.mask[0]).toBe(true);
    expect(r.mask[3]).toBe(false);
  });

  it("llama4 chunked attention defaults to window 8192, N=4", () => {
    const r = resolveSliding({ ...base, layers: 4, architecture: "llama4" });
    expect(r.windowTokens).toBe(8192);
    expect(r.mask).toEqual([true, true, true, false]);
  });

  it("llama4: explicit sliding_window 0 means no chunking; any window > 0 is forced to 8192", () => {
    const off = resolveSliding({ ...base, layers: 4, architecture: "llama4", slidingWindow: 0 });
    expect(off.mask.some(Boolean)).toBe(false);
    expect(off.reason).toBeUndefined();
    expect(resolveSliding({ ...base, layers: 4, architecture: "llama4", slidingWindow: 2048 }).windowTokens).toBe(8192);
  });

  it("smallthinker: SWA only when the GGUF has a window > 0, then forced to 4096", () => {
    const none = resolveSliding({ ...base, architecture: "smallthinker" });
    expect(none.mask.some(Boolean)).toBe(false);
    expect(none.reason).toBeUndefined();
    const on = resolveSliding({ ...base, layers: 8, architecture: "smallthinker", slidingWindow: 2048 });
    expect(on.windowTokens).toBe(4096);
    expect(on.mask).toEqual([false, true, true, true, false, true, true, true]);
  });

  it("laguna and lfm2 without a window are full attention, exact", () => {
    for (const architecture of ["laguna", "lfm2"]) {
      const r = resolveSliding({ ...base, architecture });
      expect(r.mask.some(Boolean)).toBe(false);
      expect(r.reason).toBeUndefined();
    }
  });

  it("gemma2 defaults to window 4096 when the GGUF omits it", () => {
    const r = resolveSliding({ ...base, layers: 4, architecture: "gemma2" });
    expect(r.windowTokens).toBe(4096);
    expect(r.mask).toEqual([true, false, true, false]);
  });

  it("phi3 has no sliding window even when the metadata lists one", () => {
    const r = resolveSliding({ ...base, architecture: "phi3", slidingWindow: 2047 });
    expect(r.mask.some(Boolean)).toBe(false);
    expect(r.reason).toBeUndefined();
  });

  it("unknown arch with a window and no pattern: full attention, inexact", () => {
    const r = resolveSliding({ ...base, architecture: "newarch", slidingWindow: 4096 });
    expect(r.mask.some(Boolean)).toBe(false);
    expect(r.reason).toBe("unknown sliding-window layout");
  });

  it("known SWA arch with no window size at all is inexact", () => {
    const r = resolveSliding({ ...base, architecture: "gpt-oss" });
    expect(r.mask.some(Boolean)).toBe(false);
    expect(r.reason).toBe("missing sliding-window size");
  });
});

describe("kvCacheBytes: SWA", () => {
  it("gemma3 N=6, 62 layers, 16 kv x 128/128, window 1024, ctx 8192", () => {
    const facts: ArchFacts = {
      architecture: "gemma3",
      layers: 62,
      headCountKv: 16,
      keyLength: 128,
      valueLength: 128,
      slidingWindow: 1024,
    };
    // per layer per slot: 16 x 256 x 2 = 8192 B. Full layers: il % 6 == 5 -> 10 layers.
    // ctx 8192 -> batch 1024; SWA slots = PAD(min(8192, 1024 + 1024), 256) = 2048; 52 SWA layers.
    // 10 x 8192 x 8192 + 52 x 8192 x 2048 = 671088640 + 872415232
    const kv = kvCacheBytes(facts, 8192, "f16");
    expect(kv.bytes).toBe(1_543_503_872);
    expect(kv.exact).toBe(true);
  });

  it("gpt-oss N=2, 24 layers, 8 kv x 64/64, window 128, ctx 4096", () => {
    const facts: ArchFacts = {
      architecture: "gpt-oss",
      layers: 24,
      headCountKv: 8,
      keyLength: 64,
      valueLength: 64,
      slidingWindow: 128,
    };
    // per layer per slot: 8 x 128 x 2 = 2048 B. 12 full x 4096 slots; 12 SWA x PAD(min(4096, 128+512), 256) = 768 slots.
    // 12 x 2048 x 4096 + 12 x 2048 x 768 = 100663296 + 18874368
    expect(kvCacheBytes(facts, 4096, "f16").bytes).toBe(119_537_664);
  });

  it("SWA slots never exceed the padded context", () => {
    const facts: ArchFacts = { ...llama31_8b, architecture: "gpt-oss", layers: 2, slidingWindow: 100_000 };
    // window exceeds the context, so the SWA layer is capped at 256 slots like the full one: 2 x (8 x 256 x 2) x 256
    expect(kvCacheBytes(facts, 256, "f16").bytes).toBe(2 * 4096 * 256);
  });

  it("reports inexact with the reason for unknown layouts", () => {
    const kv = kvCacheBytes({ ...llama31_8b, architecture: "newarch", slidingWindow: 4096 }, 256, "f16");
    expect(kv.exact).toBe(false);
    expect(kv.reason).toBe("unknown sliding-window layout");
    expect(kv.bytes).toBe(131072 * 256);
  });
});

describe("kvCacheBytes: per-layer shapes", () => {
  it("issue 21497 gemma4-26B-A4B, q8_0, ctx 100096: SWA layers use key_length_swa", () => {
    // full layers (il % 6 == 5): 5 x kv 2 x (512+512) = 2048 elements x 34/32 B = 2176 B/slot each
    //   5 x 2176 x 100096 = 1_089_044_480 B = 1038.59 MiB (log 1038.59)
    // SWA layers: 25 x kv 8 x (256+256) = 4096 elements x 34/32 = 4352 B/slot each
    //   slots = PAD(min(100096, 4096 + 512), 256) = 4608; 25 x 4352 x 4608 = 501_350_400 B = 478.12 MiB
    // total 1516.71 MiB (log 1516.71)
    const facts: ArchFacts = {
      architecture: "gemma4",
      layers: 30,
      headCountKv: Array.from({ length: 30 }, (_, il) => (il % 6 === 5 ? 2 : 8)),
      keyLength: 512,
      valueLength: 512,
      keyLengthSwa: 256,
      valueLengthSwa: 256,
      slidingWindow: 4096,
      slidingWindowPattern: Array.from({ length: 30 }, (_, il) => il % 6 !== 5),
    };
    const kv = kvCacheBytes(facts, 100096, "q8_0", 512);
    expect(kv.bytes).toBe(1_089_044_480 + 501_350_400);
    expect(kv.bytes / MIB).toBeCloseTo(1516.71, 1);
    expect(kv.exact).toBe(true);
  });

  it("gemma3n: only the first 20 layers own KV (30 layers, kv 2, 256/256, window 512, ctx 32768)", () => {
    // per slot 2 x 512 x 2 B = 2048 B. SWA layers il % 5 < 4: full layers 4, 9, 14, 19 -> 4 full, 16 SWA among KV owners.
    // full: 4 x 2048 x 32768 = 268_435_456; SWA slots = PAD(min(32768, 512 + 1024), 256) = 1536,
    // 16 x 2048 x 1536 = 50_331_648; total 318_767_104 B = 304 MiB
    const facts: ArchFacts = {
      architecture: "gemma3n",
      layers: 30,
      headCountKv: 2,
      keyLength: 256,
      valueLength: 256,
      slidingWindow: 512,
    };
    expect(kvCacheBytes(facts, 32768, "f16").bytes).toBe(318_767_104);
  });

  it("kvLayers limits KV to a leading prefix of layers (shared-KV and nextn layers own none)", () => {
    const facts: ArchFacts = { ...llama31_8b, kvLayers: 30 };
    expect(kvCacheBytes(facts, 256, "f16").bytes).toBe(30 * 4096 * 256);
  });

  it("issue 19892 qwen35: recurrentLayers mask leaves 16 of 64 layers with KV", () => {
    // full_attention_interval 4: layer il is recurrent unless (il + 1) % 4 == 0.
    // 16 x kv 4 x (256+256) x 2 B = 65536 B/slot x 1024 slots = 64 MiB (log: 64.00 MiB, 16 layers)
    const facts: ArchFacts = {
      architecture: "qwen35",
      layers: 64,
      headCountKv: 4,
      keyLength: 256,
      valueLength: 256,
      recurrentLayers: Array.from({ length: 64 }, (_, il) => (il + 1) % 4 !== 0),
    };
    const kv = kvCacheBytes(facts, 1024, "f16");
    expect(kv.bytes / MIB).toBe(64);
    expect(kv.exact).toBe(false);
    expect(kv.reason).toBe("missing recurrent state size");
    expect(kvCacheBytes({ ...facts, recurrentStateBytes: 1000 }, 1024, "f16").exact).toBe(true);
  });

  it("an inexactReason on the facts makes the estimate inexact", () => {
    const kv = kvCacheBytes({ ...llama31_8b, inexactReason: "MTP drafting" }, 256, "f16");
    expect(kv.exact).toBe(false);
    expect(kv.reason).toBe("MTP drafting");
    expect(kv.bytes).toBe(32 * 4096 * 256);
  });

  it("per-layer head array shorter than layers is inexact", () => {
    const kv = kvCacheBytes({ ...llama31_8b, headCountKv: [8, 8] }, 256, "f16");
    expect(kv.exact).toBe(false);
    expect(kv.reason).toBe("per-layer array length mismatch");
  });
});

describe("kqMaskBytes: F16 [n_kv, n_tokens] per attention-mask input (L src/llama-graph.cpp:32-47)", () => {
  const tiny: ArchFacts = { architecture: "llama", layers: 2, headCountKv: 1, keyLength: 64, valueLength: 64 };

  it("full attention: 2 B x padded ctx 4096 x 512 tokens = 4 MiB", () => {
    expect(kqMaskBytes(tiny, 4096, 512)).toBe(4 * MIB);
  });

  it("pads the context like the KV cache: ctx 5000 -> 5120 x 512 x 2 B = 5 MiB", () => {
    expect(kqMaskBytes(tiny, 5000, 512)).toBe(5 * MIB);
  });

  it("n_tokens is the smaller of context and batch: ctx 256, batch 512 -> 2 B x 256 x 256 = 128 KiB", () => {
    expect(kqMaskBytes(tiny, 256, 512)).toBe(128 * 1024);
  });

  it("iSWA adds a second mask of the SWA cache size", () => {
    // gemma3 period 6, 6 layers: layers 0-4 SWA, layer 5 full. ctx 8192, batch 1024, window 1024.
    // base mask 2 x 8192 x 1024 = 16 MiB; SWA cells = PAD(min(8192, 1024 + 1024), 256) = 2048 -> 2 x 2048 x 1024 = 4 MiB.
    const gemma: ArchFacts = { ...tiny, architecture: "gemma3", layers: 6, slidingWindow: 1024 };
    expect(kqMaskBytes(gemma, 8192, 1024)).toBe(20 * MIB);
  });

  it("an all-SWA model has only the SWA mask", () => {
    // lfm2 pattern 0 = every layer; window 512, ctx 4096, batch 512: cells PAD(min(4096, 1024), 256) = 1024 -> 2 x 1024 x 512 = 1 MiB.
    expect(kqMaskBytes({ ...tiny, architecture: "lfm2", slidingWindow: 512 }, 4096, 512)).toBe(1 * MIB);
  });

  it("a model with no attention layer has no mask", () => {
    expect(kqMaskBytes({ ...tiny, headCountKv: 0 }, 4096, 512)).toBe(0);
  });

  it("without architecture facts: one full mask on the padded context", () => {
    expect(kqMaskBytes(undefined, 4096, 512)).toBe(4 * MIB);
  });
});

describe("computeBufferMb vs llama.cpp logs (within 10%)", () => {
  // formula: vocab x ubatch x 4 B / 2^20 x 1.10
  const logs = [
    { issue: "12171 Qwen2 14B", vocab: 152064, ubatch: 512, loggedMiB: 307.0 },
    { issue: "18996 gpt-oss", vocab: 201088, ubatch: 512, loggedMiB: 398.38 },
    { issue: "21879 gemma3", vocab: 262208, ubatch: 512, loggedMiB: 522.62 },
    { issue: "21497 gemma4", vocab: 262144, ubatch: 512, loggedMiB: 521.91 },
    { issue: "19892 Qwen3.5", vocab: 248320, ubatch: 128, loggedMiB: 123.75 },
  ];
  it.each(logs)("$issue", ({ vocab, ubatch, loggedMiB }) => {
    const est = computeBufferMb(vocab, ubatch);
    expect(Math.abs(est - loggedMiB) / loggedMiB).toBeLessThanOrEqual(0.1);
  });

  it("hand value: 262144 x 512 x 4 / 2^20 = 512 MiB, x 1.10 = 563.2", () => {
    expect(computeBufferMb(262144, 512)).toBeCloseTo(563.2, 6);
  });

  it("unknown vocab defaults to 262144", () => {
    expect(computeBufferMb(undefined, 512)).toBeCloseTo(563.2, 6);
  });
});

describe("ollamaDefaultContext", () => {
  const GIB = 1024;
  it("tiers at 23 and 47 GiB total VRAM", () => {
    expect(ollamaDefaultContext(47 * GIB)).toBe(262144);
    expect(ollamaDefaultContext(47 * GIB - 1)).toBe(32768);
    expect(ollamaDefaultContext(23 * GIB)).toBe(32768);
    expect(ollamaDefaultContext(23 * GIB - 1)).toBe(4096);
    expect(ollamaDefaultContext(0)).toBe(4096);
  });

  it("clamps to the training context", () => {
    expect(ollamaDefaultContext(47 * GIB, 8192)).toBe(8192);
    expect(ollamaDefaultContext(0, 2048)).toBe(2048);
  });

  it("ignores a missing (0) training context, as Ollama does", () => {
    expect(ollamaDefaultContext(24 * GIB, 0)).toBe(32768);
  });
});

describe("budgetMb", () => {
  it("discrete GPU: vram - 457 - 1024", () => {
    expect(budgetMb({ kind: "discrete-gpu", vramMb: 24576 })).toBe(23095);
  });
  it("Apple Metal: working set - 1024 (fit margin only; free = working set - allocated ~ 0 before load)", () => {
    expect(budgetMb({ kind: "apple-metal", vramMb: 16384 })).toBe(15360);
  });
  it("non-Apple unified: available - 457 - 1024", () => {
    expect(budgetMb({ kind: "unified", availableMb: 100_000 })).toBe(98519);
  });
  it("AMD APU: min(GTT, available) - 457 - 1024", () => {
    expect(budgetMb({ kind: "amd-apu", gttMb: 32768, availableMb: 20000 })).toBe(18519);
    expect(budgetMb({ kind: "amd-apu", gttMb: 10000, availableMb: 20000 })).toBe(8519);
  });
  it("CPU-only: available system memory, no deductions", () => {
    expect(budgetMb({ kind: "cpu-only", availableMb: 8000 })).toBe(8000);
  });
  it("floors at 0", () => {
    expect(budgetMb({ kind: "discrete-gpu", vramMb: 1000 })).toBe(0);
  });
});

describe("assessFit boundaries", () => {
  // need(c) = c / 4 MB: c_default 32768 -> 8192, c_min 4096 -> 1024
  const need = (c: number) => c / 4;
  const run = (budget: number, cpuOnly = false) =>
    assessFit({ budgetMb: budget, cpuOnly, defaultContext: 32768, need });

  it("excellent at need <= 0.80 x budget", () => {
    expect(run(10240).level).toBe("excellent"); // 0.8 x 10240 = 8192
    expect(run(10239).level).toBe("comfortable");
  });

  it("comfortable at need <= budget", () => {
    expect(run(8192).level).toBe("comfortable");
    expect(run(8192).fitRatio).toBe(1);
  });

  it("tight below need(default) down to need(c_min)", () => {
    const r = run(8191);
    expect(r.level).toBe("tight");
    expect(r.fitRatio).toBeCloseTo(8191 / 8192, 10);
    expect(run(1024).level).toBe("tight");
  });

  it("barely between 75% and 100% of need(c_min)", () => {
    expect(run(1023).level).toBe("barely"); // 0.75 x 1024 = 768
    expect(run(768).level).toBe("barely");
  });

  it("cannot_run below 75% of need(c_min)", () => {
    expect(run(767).level).toBe("cannot_run");
    expect(run(0).level).toBe("cannot_run");
  });

  it("CPU-only never reports barely", () => {
    expect(run(1023, true).level).toBe("cannot_run");
    expect(run(1024, true).level).toBe("tight");
  });

  it("c_min shrinks to the default context when the model trains below 4096", () => {
    // need(2048) = 512 > budget 500, so the level is decided by need(c_min): 500 >= 0.75 x 512 = 384 -> barely.
    // With c_min fixed at 4096, need(4096) = 1024 and 500 < 0.75 x 1024 = 768 would give cannot_run.
    expect(assessFit({ budgetMb: 500, cpuOnly: false, defaultContext: 2048, need }).level).toBe("barely");
    expect(assessFit({ budgetMb: 600, cpuOnly: false, defaultContext: 2048, need }).level).toBe("comfortable");
  });

  it("maxContext honours the training-context ceiling", () => {
    const r = assessFit({ budgetMb: 100_000, cpuOnly: false, defaultContext: 8192, need, trainContext: 8192 });
    expect(r.maxContext).toBe(8192);
  });

  it("maxContext reaches a known training context above 262144, and stops at 262144 when it is unknown", () => {
    const roomy = { budgetMb: 1_000_000, cpuOnly: false, defaultContext: 262144, need };
    expect(assessFit({ ...roomy, trainContext: 1_048_576 }).maxContext).toBe(1_048_576);
    expect(assessFit(roomy).maxContext).toBe(262144);
  });
});

describe("maxGpuContext", () => {
  const need = (c: number) => c / 4;
  it("largest 256-multiple with need <= budget", () => {
    // c <= 32764 -> 127 x 256 = 32512
    expect(maxGpuContext(8191, need, 262144)).toBe(32512);
  });
  it("caps at the ceiling", () => {
    expect(maxGpuContext(1_000_000, need, 32768)).toBe(32768);
  });
  it("is 0 when even 256 tokens do not fit", () => {
    expect(maxGpuContext(10, need, 262144)).toBe(0);
  });
});

describe("snapToContextTier", () => {
  it("rounds down to a display tier", () => {
    expect(snapToContextTier(32512)).toBe(16384);
    expect(snapToContextTier(32768)).toBe(32768);
    expect(snapToContextTier(300_000)).toBe(262144);
  });
  it("leaves values below the smallest tier unchanged", () => {
    expect(snapToContextTier(1000)).toBe(1000);
  });
});

describe("speedClass", () => {
  const GB = 1024;
  it("cannot_run and barely are slow", () => {
    expect(speedClass({ level: "cannot_run", cpuOnly: false, weightsMb: 1 })).toBe("slow");
    expect(speedClass({ level: "barely", cpuOnly: false, weightsMb: 1 })).toBe("slow");
  });
  it("GPU-resident thresholds: 6 GB fast, 24 GB moderate", () => {
    expect(speedClass({ level: "excellent", cpuOnly: false, weightsMb: 6 * GB })).toBe("fast");
    expect(speedClass({ level: "excellent", cpuOnly: false, weightsMb: 6 * GB + 1 })).toBe("moderate");
    expect(speedClass({ level: "comfortable", cpuOnly: false, weightsMb: 24 * GB })).toBe("moderate");
    expect(speedClass({ level: "tight", cpuOnly: false, weightsMb: 24 * GB + 1 })).toBe("slow");
  });
  it("CPU-only: moderate up to 2 GB active, else slow", () => {
    expect(speedClass({ level: "comfortable", cpuOnly: true, weightsMb: 2 * GB })).toBe("moderate");
    expect(speedClass({ level: "comfortable", cpuOnly: true, weightsMb: 2 * GB + 1 })).toBe("slow");
  });
  it("MoE uses active bytes: 40 GB weights x 3B/30B active = 4 GB -> fast", () => {
    expect(
      speedClass({ level: "excellent", cpuOnly: false, weightsMb: 40 * GB, activeParams: 3, totalParams: 30 }),
    ).toBe("fast");
  });
});

describe("layersThatFit", () => {
  // Rule: each layer carries its weights and its KV share (L src/llama-kv-cache.cpp:216-221); the compute buffer is fixed.
  it("floor((budget - compute) / ((weights + kv) / layers))", () => {
    // (10000 - 500) / ((8000 + 1000) / 32 = 281.25) = 33.8 -> clamped to 32 layers
    expect(layersThatFit({ budgetMb: 10000, kvMb: 1000, computeMb: 500, weightsMb: 8000, layers: 32 })).toBe(32);
    // (5000 - 500) / 281.25 = 16
    expect(layersThatFit({ budgetMb: 5000, kvMb: 1000, computeMb: 500, weightsMb: 8000, layers: 32 })).toBe(16);
  });
  it("is 0 when the compute buffer alone exceeds the budget", () => {
    expect(layersThatFit({ budgetMb: 400, kvMb: 900, computeMb: 500, weightsMb: 8000, layers: 32 })).toBe(0);
  });
});
