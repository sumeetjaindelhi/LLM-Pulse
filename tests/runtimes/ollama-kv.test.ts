import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import type { ArchFacts, ModelEntry } from "../../src/core/types.js";
import { kvCacheBytes } from "../../src/analysis/memory-model.js";
import { fetchInstalledArch, fetchLoadedModels, fetchLoadedModelsWithKv, withExactFacts, withInstalledKv } from "../../src/runtimes/ollama.js";

// The registry client has its own tests (tests/models/ollama-registry.test.ts); here it answers whatever a test sets.
const fetchRegistryFacts = vi.hoisted(() => vi.fn());
vi.mock("../../src/models/ollama-registry.js", () => ({ fetchRegistryFacts }));

const HOST = "http://127.0.0.1:11434";

function llamaInfo(over: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    "general.architecture": "llama",
    "llama.block_count": 32,
    "llama.attention.head_count": 32,
    "llama.attention.head_count_kv": 8,
    "llama.attention.key_length": 128,
    "llama.attention.value_length": 128,
    "llama.embedding_length": 4096,
    "llama.context_length": 131072,
    ...over,
  };
}

// f16 KV bytes at a 1024-token context through the shared formula (no sliding window for these architectures).
const kvAt1k = (arch: ArchFacts | null) => kvCacheBytes(arch!, 1024, "f16").bytes;

describe("ollama KV helpers", () => {
  const originalFetch = globalThis.fetch;
  let fetchMock: ReturnType<typeof vi.fn>;

  const respondWith = (model_info: Record<string, unknown>) =>
    fetchMock.mockResolvedValue({ ok: true, json: async () => ({ details: { format: "gguf" }, model_info }) });

  beforeEach(() => {
    fetchMock = vi.fn();
    globalThis.fetch = fetchMock as unknown as typeof fetch;
  });
  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  describe("fetchInstalledArch", () => {
    it("reads llama-style facts (128 MiB of f16 KV per 1024 tokens) and POSTs with redirect error", async () => {
      respondWith(llamaInfo());
      const arch = await fetchInstalledArch(HOST, "llama3.1:8b");
      expect(arch).toEqual({
        architecture: "llama",
        layers: 32,
        headCountKv: 8,
        keyLength: 128,
        valueLength: 128,
        contextLength: 131072,
      });
      // 32 layers x 8 heads x (128 + 128) x 2 B x 1024 tokens
      expect(kvAt1k(arch)).toBe(32 * 8 * 256 * 2 * 1024);
      const [url, init] = fetchMock.mock.calls[0];
      expect(url).toBe(`${HOST}/api/show`);
      expect(init.method).toBe("POST");
      expect(init.redirect).toBe("error");
      expect(init.signal).toBeInstanceOf(AbortSignal);
      expect(JSON.parse(init.body)).toEqual({ model: "llama3.1:8b" });
    });

    it("returns null for a safetensors model: its model_info is built from config.json, not GGUF metadata", async () => {
      fetchMock.mockResolvedValue({ ok: true, json: async () => ({ details: { format: "safetensors" }, model_info: llamaInfo() }) });
      expect(await fetchInstalledArch(HOST, "lfm3:1.7b")).toBeNull();
      fetchMock.mockResolvedValue({ ok: true, json: async () => ({ model_info: llamaInfo() }) });
      expect(await fetchInstalledArch(HOST, "lfm3:1.7b")).toBeNull();
    });

    it("sums a per-layer head_count_kv array", async () => {
      respondWith(llamaInfo({
        "llama.block_count": 4,
        "llama.attention.head_count_kv": [8, 8, 0, 8],
      }));
      // 24 kv heads x 256 x 2 B x 1024 tokens
      expect(kvAt1k(await fetchInstalledArch(HOST, "x"))).toBe(24 * 256 * 2 * 1024);
    });

    it("falls back to embedding_length / head_count when key/value length are absent", async () => {
      const info = llamaInfo();
      delete info["llama.attention.key_length"];
      delete info["llama.attention.value_length"];
      respondWith(info);
      expect(kvAt1k(await fetchInstalledArch(HOST, "x"))).toBe(32 * 8 * 256 * 2 * 1024);
    });

    // Rule: MLA needs key_length_mla and value_length_mla and one latent head (L src/llama-hparams.cpp:303-308); it has no V cache
    // (L src/llama-kv-cache.cpp:231): 32 layers x 1 head x 576 x 2 B x 1024 tokens.
    it("sizes MLA architectures from the K cache alone and flags arch.mla", async () => {
      respondWith(llamaInfo({
        "llama.attention.head_count_kv": 1,
        "llama.attention.key_length": 576,
        "llama.attention.key_length_mla": 192,
        "llama.attention.value_length_mla": 128,
      }));
      const arch = await fetchInstalledArch(HOST, "x");
      expect(kvAt1k(arch)).toBe(32 * 576 * 2 * 1024);
      expect(arch?.mla).toBe(true);
    });

    // Rule: kv_lora_rank without *_mla keys is old-format deepseek2, which caches K and V (L src/llama-hparams.cpp:303-308).
    it("sizes old-format deepseek2 (kv_lora_rank only) with both K and V", async () => {
      respondWith(llamaInfo({ "llama.attention.kv_lora_rank": 512, "llama.attention.key_length": 192, "llama.attention.value_length": 128 }));
      const arch = await fetchInstalledArch(HOST, "x");
      expect(kvAt1k(arch)).toBe(32 * 8 * (192 + 128) * 2 * 1024);
      expect(arch).not.toHaveProperty("mla");
    });

    it("skips recurrent layers and nextn layers (MTP is off by default)", async () => {
      respondWith(llamaInfo({
        "llama.block_count": 8,
        "llama.attention.head_count_kv": 8,
        "llama.full_attention_interval": 4,
        "llama.nextn_predict_layers": 1,
      }));
      // layers 0-6 are recurrent unless (il + 1) % 4 == 0 -> layer 3 attends; layer 7 is the nextn layer and owns no KV
      // (kvLayers = 8 - 1 = 7): 1 attention layer.
      expect(kvAt1k(await fetchInstalledArch(HOST, "x"))).toBe(1 * 8 * 256 * 2 * 1024);
    });

    it("returns null when context_length is missing", async () => {
      const info = llamaInfo();
      delete info["llama.context_length"];
      respondWith(info);
      expect(await fetchInstalledArch(HOST, "x")).toBeNull();
    });

    it("returns null on HTTP error", async () => {
      fetchMock.mockResolvedValue({ ok: false, status: 404, json: async () => ({}) });
      expect(await fetchInstalledArch(HOST, "x")).toBeNull();
    });

    it("returns null when the request throws", async () => {
      fetchMock.mockRejectedValue(new Error("ECONNREFUSED"));
      expect(await fetchInstalledArch(HOST, "x")).toBeNull();
    });

    it("returns null for garbage numbers", async () => {
      for (const bad of [
        { "llama.block_count": -1 },
        { "llama.block_count": 0 },
        { "llama.block_count": 1e9 },
        { "llama.attention.head_count_kv": "eight" },
        { "llama.attention.head_count_kv": [8, 8] },
        { "llama.attention.key_length": 0 },
        { "llama.attention.value_length": null },
        { "llama.context_length": Number.MAX_SAFE_INTEGER },
        { "general.architecture": 7 },
      ]) {
        respondWith(llamaInfo(bad));
        expect(await fetchInstalledArch(HOST, "x"), JSON.stringify(bad)).toBeNull();
      }
    });
  });

  describe("fetchLoadedModels", () => {
    it("returns name and context length of loaded models", async () => {
      fetchMock.mockResolvedValue({
        ok: true,
        json: async () => ({ models: [{ name: "llama3.1:8b", context_length: 4096 }, { name: "old:1b" }] }),
      });
      expect(await fetchLoadedModels(HOST)).toEqual([
        { name: "llama3.1:8b", contextLength: 4096 },
        { name: "old:1b", contextLength: null },
      ]);
      expect(fetchMock.mock.calls[0][0]).toBe(`${HOST}/api/ps`);
      expect(fetchMock.mock.calls[0][1].redirect).toBe("error");
    });

    it("returns [] on failure", async () => {
      fetchMock.mockRejectedValue(new Error("down"));
      expect(await fetchLoadedModels(HOST)).toEqual([]);
      fetchMock.mockResolvedValue({ ok: false });
      expect(await fetchLoadedModels(HOST)).toEqual([]);
    });

    it("carries the quantization level and drops non-positive or fractional context lengths", async () => {
      fetchMock.mockResolvedValue({
        ok: true,
        json: async () => ({ models: [
          { name: "a:1b", context_length: 4096, details: { quantization_level: "Q4_K_M" } },
          { name: "b:1b", context_length: 0 },
          { name: "c:1b", context_length: 4096.5 },
        ] }),
      });
      expect(await fetchLoadedModels(HOST)).toEqual([
        { name: "a:1b", contextLength: 4096, quantizationLevel: "Q4_K_M" },
        { name: "b:1b", contextLength: null },
        { name: "c:1b", contextLength: null },
      ]);
    });
  });

  describe("fetchLoadedModelsWithKv", () => {
    it("adds the architecture facts from /api/show to each loaded model", async () => {
      fetchMock.mockImplementation(async (url: string) => ({
        ok: true,
        json: async () => (url.endsWith("/api/ps")
          ? { models: [{ name: "llama3.1:8b", context_length: 4096 }] }
          : { details: { format: "gguf" }, model_info: llamaInfo() }),
      }));
      expect(await fetchLoadedModelsWithKv(HOST)).toEqual([
        { name: "llama3.1:8b", contextLength: 4096, arch: expect.objectContaining({ architecture: "llama", layers: 32 }) },
      ]);
    });
  });

  describe("withInstalledKv", () => {
    const base = { id: "m", name: "M", provider: "p", parametersBillion: 8, contextWindow: 8192, ollamaTag: "m:8b" } as unknown as ModelEntry;

    it("returns the same object without calling Ollama when the model has no tag", async () => {
      const untagged = { ...base, ollamaTag: undefined };
      expect(await withInstalledKv(HOST, untagged)).toBe(untagged);
      expect(fetchMock).not.toHaveBeenCalled();
    });

    // Rule: Ollama caps num_ctx at the file's training context (O llm/server.go:111-114), so it replaces the listed window.
    it("replaces the listed window with the file's training context", async () => {
      respondWith(llamaInfo());
      expect((await withInstalledKv(HOST, base)).contextWindow).toBe(131072);
    });

    it("replaces an estimated window with the exact context length and clears the flag", async () => {
      respondWith(llamaInfo());
      const estimated = { ...base, contextWindow: 32768, contextWindowEstimated: true };
      const result = await withInstalledKv(HOST, estimated);
      expect(result.contextWindow).toBe(131072);
      expect(result).not.toHaveProperty("contextWindowEstimated");
    });

    it("keeps an estimated window when Ollama has no profile", async () => {
      fetchMock.mockResolvedValue({ ok: false });
      const estimated = { ...base, contextWindow: 32768, contextWindowEstimated: true };
      expect(await withInstalledKv(HOST, estimated)).toBe(estimated);
    });

    it("returns the model unchanged when Ollama has no profile", async () => {
      fetchMock.mockResolvedValue({ ok: false });
      expect(await withInstalledKv(HOST, base)).toBe(base);
    });

    it("attaches the exact arch facts", async () => {
      respondWith(llamaInfo());
      expect((await withInstalledKv(HOST, base)).arch).toMatchObject({ architecture: "llama", layers: 32, headCountKv: 8 });
    });

    // /api/show leaves out the token array, so most installed files report no vocabulary size (qwen2, qwen3, phi3 have no
    // <arch>.vocab_size key); the same tag's size from the registry header stays, and one /api/show does report wins.
    it("keeps the row's vocabulary size when the installed facts have none", async () => {
      const known = { ...base, arch: { architecture: "llama", layers: 32, headCountKv: 8, keyLength: 128, valueLength: 128, vocabSize: 151936 } } as ModelEntry;
      respondWith(llamaInfo());
      expect((await withInstalledKv(HOST, known)).arch?.vocabSize).toBe(151936);
      respondWith(llamaInfo({ "llama.vocab_size": 128256 }));
      expect((await withInstalledKv(HOST, known)).arch?.vocabSize).toBe(128256);
    });

    describe("exact installed file size", () => {
      const withQuants = {
        ...base,
        quantizations: [
          { name: "Q4_K_M", bitsPerWeight: 4.9, vramMb: 4000, qualityRetention: 0.97 },
          { name: "Q8_0", bitsPerWeight: 8.5, vramMb: 8000, qualityRetention: 0.99 },
        ],
      } as ModelEntry;
      const FILE_BYTES = 4_920_000_000;

      const serve = (tags: unknown) =>
        fetchMock.mockImplementation(async (url: string) => {
          if (url.endsWith("/api/tags")) {
            if (tags instanceof Error) throw tags;
            return { ok: true, json: async () => tags };
          }
          return { ok: true, json: async () => ({ details: { format: "gguf" }, model_info: llamaInfo() }) };
        });
      const tagList = (name: string, quantization_level = "Q4_K_M") => ({
        models: [{ name, size: FILE_BYTES, details: { quantization_level } }],
      });

      it("sets the matching quant's vramMb to the file size in MiB and marks it exact", async () => {
        serve(tagList("m:8b"));
        const result = await withInstalledKv(HOST, withQuants);
        expect(result.quantizations[0]).toMatchObject({ name: "Q4_K_M", vramMb: Math.round(FILE_BYTES / 2 ** 20), exact: true });
        expect(result.quantizations[1]).toEqual(withQuants.quantizations[1]);
        expect(fetchMock.mock.calls.some(([u]) => u === `${HOST}/api/tags`)).toBe(true);
      });

      // The tag identifies the file, so a library row's single quant (named "default") is the installed one: it takes the
      // file's type as its name, as the registry path does, with that type's bpw and exp(-mean KLD) from src/models/quant-facts.ts.
      it("turns a library row's sole quant into the installed file", async () => {
        serve(tagList("qwen3:0.6b"));
        const library = {
          ...base,
          source: "library",
          ollamaTag: "qwen3:0.6b",
          quantizations: [{ name: "default", bitsPerWeight: 5, vramMb: 358, qualityRetention: 0.9692 }],
        } as ModelEntry;
        expect((await withInstalledKv(HOST, library)).quantizations).toEqual([
          { name: "Q4_K_M", bitsPerWeight: 4.8944, vramMb: Math.round(FILE_BYTES / 2 ** 20), qualityRetention: Math.exp(-0.031273), exact: true },
        ]);
        const curated = { ...library, source: "curated" } as ModelEntry;
        expect((await withInstalledKv(HOST, curated)).quantizations).toEqual(curated.quantizations);
        // The type becomes a quant name on screen, so one that is not a plain file type name is ignored.
        serve(tagList("qwen3:0.6b", "\u001b[2JQ4_K_M"));
        expect((await withInstalledKv(HOST, library)).quantizations).toEqual(library.quantizations);
      });

      // At most one quant is exact: the file `ollama pull <tag>` installs. Another local file under the tag (a re-pushed
      // tag, a custom model) leaves the row's quants alone.
      it("keeps a row's exact quant when the installed file is another quant", async () => {
        serve(tagList("m:8b", "Q4_K_M"));
        const pulledQ8 = { ...withQuants, quantizations: [withQuants.quantizations[0], { ...withQuants.quantizations[1], exact: true }] } as ModelEntry;
        expect((await withInstalledKv(HOST, pulledQ8)).quantizations).toEqual(pulledQ8.quantizations);
      });

      it("matches a bare tag to :latest", async () => {
        serve(tagList("m:latest"));
        const result = await withInstalledKv(HOST, { ...withQuants, ollamaTag: "m" });
        expect(result.quantizations[0].exact).toBe(true);
      });

      it("leaves quants alone when the tag is not installed or the quant is unknown", async () => {
        serve(tagList("other:8b"));
        expect((await withInstalledKv(HOST, withQuants)).quantizations).toEqual(withQuants.quantizations);
        serve(tagList("m:8b", "IQ2_XS"));
        expect((await withInstalledKv(HOST, withQuants)).quantizations).toEqual(withQuants.quantizations);
        serve({ models: [{ name: "m:8b", size: FILE_BYTES }] });
        expect((await withInstalledKv(HOST, withQuants)).quantizations).toEqual(withQuants.quantizations);
      });

      it("still applies the profile when /api/tags fails, and the size when /api/show fails", async () => {
        serve(new Error("ECONNRESET"));
        expect((await withInstalledKv(HOST, withQuants)).arch).toMatchObject({ architecture: "llama" });

        fetchMock.mockImplementation(async (url: string) =>
          url.endsWith("/api/tags") ? { ok: true, json: async () => tagList("m:8b") } : { ok: false });
        const sizeOnly = await withInstalledKv(HOST, withQuants);
        expect(sizeOnly.quantizations[0].exact).toBe(true);
        expect(sizeOnly).not.toHaveProperty("arch");
      });

      it("ignores sizes below 1 MiB or beyond 2^50 bytes", async () => {
        for (const size of [0.5, 1e300, 2 ** 50 + 1]) {
          serve({ models: [{ name: "m:8b", size, details: { quantization_level: "Q4_K_M" } }] });
          expect((await withInstalledKv(HOST, withQuants)).quantizations, String(size)).toEqual(withQuants.quantizations);
        }
      });

      it("ignores a non-positive size", async () => {
        serve({ models: [{ name: "m:8b", size: 0, details: { quantization_level: "Q4_K_M" } }] });
        expect((await withInstalledKv(HOST, withQuants)).quantizations).toEqual(withQuants.quantizations);
      });
    });
  });

  describe("withExactFacts", () => {
    const ARCH: ArchFacts = { architecture: "llama", layers: 32, headCountKv: 8, keyLength: 128, valueLength: 128, contextLength: 131072 };
    const WEIGHTS = 4_920_000_000; // 4692 MiB
    const facts = (over: Record<string, unknown> = {}) => ({ weightsBytes: WEIGHTS, fileType: "Q4_K_M", arch: ARCH, ...over });
    const row = (over: Partial<ModelEntry> = {}): ModelEntry => ({
      id: "m", name: "M", provider: "p", parametersBillion: 8, contextWindow: 8192, categories: ["general"],
      qualityTier: "good", qualityScore: 60, ollamaTag: "m:8b", releaseDate: "2025-06", source: "curated",
      quantizations: [
        { name: "Q4_K_M", bitsPerWeight: 4.8944, vramMb: 4668, qualityRetention: 0.97 },
        { name: "Q8_0", bitsPerWeight: 8.5008, vramMb: 8107, qualityRetention: 0.99 },
      ],
      ...over,
    });
    // A live-catalog row (src/models/library-entries.ts): one placeholder quant at the 5 bpw estimate, a guessed window.
    const library = row({
      source: "library",
      contextWindow: 32768,
      contextWindowEstimated: true,
      quantizations: [{ name: "default", bitsPerWeight: 5, vramMb: 4768, qualityRetention: 0.9692 }],
    });

    beforeEach(() => {
      fetchMock.mockResolvedValue({ ok: false }); // not installed
      fetchRegistryFacts.mockReset().mockResolvedValue(null);
    });

    it("asks the registry nothing for a row with arch facts and an exact quant, or for an installed model", async () => {
      const curated = row({ arch: ARCH, quantizations: [{ ...row().quantizations[0], exact: true }] });
      expect(await withExactFacts(HOST, curated)).toBe(curated);

      fetchMock.mockImplementation(async (url: string) => ({
        ok: true,
        json: async () => (url.endsWith("/api/tags")
          ? { models: [{ name: "m:8b", size: WEIGHTS, details: { quantization_level: "Q4_K_M" } }] }
          : { details: { format: "gguf" }, model_info: llamaInfo() }),
      }));
      expect((await withExactFacts(HOST, library)).quantizations[0]).toMatchObject({ vramMb: 4692, exact: true });
      expect(fetchRegistryFacts).not.toHaveBeenCalled();
    });

    // Curated llama3.2-vision: an exact quant and no arch, because mllama has no llama.cpp arch facts in any source.
    it("asks the registry nothing for a row with an exact quant but no arch facts", async () => {
      const curated = row({ quantizations: [{ ...row().quantizations[0], exact: true }] });
      expect(await withExactFacts(HOST, curated)).toBe(curated);
      expect(fetchRegistryFacts).not.toHaveBeenCalled();
    });

    it("gives a library row the file's name and size when the registry has no arch facts for it", async () => {
      const { arch: _, ...noArch } = facts();
      fetchRegistryFacts.mockResolvedValue(noArch);
      const result = await withExactFacts(HOST, library);
      expect(result).not.toHaveProperty("arch");
      expect(result).toMatchObject({ contextWindow: 32768, contextWindowEstimated: true });
      expect(result.quantizations).toEqual([
        { name: "Q4_K_M", bitsPerWeight: 4.8944, vramMb: 4692, qualityRetention: Math.exp(-0.031273), exact: true },
      ]);
    });

    it("gives a library row the registry's arch, training context and file: name, size, bpw and retention", async () => {
      fetchRegistryFacts.mockResolvedValue(facts({ projectorBytes: 851_000_000 }));
      const result = await withExactFacts(HOST, library);
      expect(fetchRegistryFacts).toHaveBeenCalledWith("m:8b");
      expect(result.arch).toEqual(ARCH);
      expect(result.contextWindow).toBe(131072);
      expect(result).not.toHaveProperty("contextWindowEstimated");
      // (4,920,000,000 + 851,000,000) B / 2^20 = 5503.7 MiB; Q4_K_M file bpw and exp(-mean KLD) as in src/models/quant-facts.ts.
      expect(result.quantizations).toEqual([
        { name: "Q4_K_M", bitsPerWeight: 4.8944, vramMb: 5504, qualityRetention: Math.exp(-0.031273), exact: true },
      ]);
    });

    it("counts the draft layer in the file size, as /api/tags does", async () => {
      fetchRegistryFacts.mockResolvedValue(facts({ projectorBytes: 851_000_000, draftBytes: 288_800_000 }));
      // (4,920,000,000 + 851,000,000 + 288,800,000) B / 2^20 = 6,059,800,000 / 1,048,576 = 5779.08 MiB
      const result = await withExactFacts(HOST, library);
      expect(result.quantizations[0].vramMb).toBe(5779);
    });

    // Ollama loads the draft model by default (DraftNumPredict 4, O api/types.go:1364, O server/routes.go:168-170); its KV
    // and compute are not modelled, so the KV figure is an estimate.
    it("marks the KV inexact when the file carries a draft model, whether the arch is the row's or the registry's", async () => {
      fetchRegistryFacts.mockResolvedValue(facts({ draftBytes: 288_800_000 }));
      for (const base of [library, { ...library, arch: ARCH }]) {
        const arch = (await withExactFacts(HOST, base)).arch!;
        expect(arch.inexactReason).toMatch(/draft/);
        expect(kvCacheBytes(arch, 1024, "f16").exact).toBe(false);
      }
      fetchRegistryFacts.mockResolvedValue(facts());
      expect(kvCacheBytes((await withExactFacts(HOST, library)).arch!, 1024, "f16").exact).toBe(true);
    });

    it("keeps a library row's own bpw and retention where quant-facts has no figure for the file type", async () => {
      // TQ1_0 is not in the table; MXFP4 has a nominal bpw only (4.25 + 0.4 estimated excess) and no published quality.
      for (const [fileType, bitsPerWeight] of [["TQ1_0", 5], ["MXFP4", 4.65]] as const) {
        fetchRegistryFacts.mockResolvedValue(facts({ fileType }));
        expect((await withExactFacts(HOST, library)).quantizations, fileType).toEqual([
          { name: fileType, bitsPerWeight, vramMb: 4692, qualityRetention: 0.9692, exact: true },
        ]);
      }
    });

    it("marks the quant named like the file (any case) exact on a row that has none, and leaves the others", async () => {
      const curated = row();
      fetchRegistryFacts.mockResolvedValue(facts({ fileType: "q8_0" }));
      const result = await withExactFacts(HOST, curated);
      expect(result.arch).toEqual(ARCH);
      expect(result.contextWindow).toBe(131072);
      expect(result.quantizations).toEqual([curated.quantizations[0], { ...curated.quantizations[1], vramMb: 4692, exact: true }]);

      fetchRegistryFacts.mockResolvedValue(facts({ fileType: "IQ2_XS" }));
      expect((await withExactFacts(HOST, curated)).quantizations).toEqual(curated.quantizations);
    });

    it("keeps the installed arch facts over the registry's", async () => {
      // /api/show answers and /api/tags does not, so only the file size is still an estimate.
      fetchMock.mockImplementation(async (url: string) =>
        url.endsWith("/api/show") ? { ok: true, json: async () => ({ details: { format: "gguf" }, model_info: llamaInfo() }) } : { ok: false });
      fetchRegistryFacts.mockResolvedValue(facts({ arch: { ...ARCH, contextLength: 8192 } }));
      const result = await withExactFacts(HOST, row());
      expect(result.arch?.contextLength).toBe(131072);
      expect(result.contextWindow).toBe(131072);
      expect(result.quantizations[0]).toMatchObject({ name: "Q4_K_M", vramMb: 4692, exact: true });
    });

    it("keeps the listed window when the registry's arch facts have no training context", async () => {
      const { contextLength: _, ...noContext } = ARCH;
      fetchRegistryFacts.mockResolvedValue(facts({ arch: noContext }));
      const result = await withExactFacts(HOST, library);
      expect(result.arch).toEqual(noContext);
      expect(result).toMatchObject({ contextWindow: 32768, contextWindowEstimated: true });
    });

    it("returns the model unchanged when the registry has no facts, and asks nothing for a model without a tag", async () => {
      expect(await withExactFacts(HOST, library)).toBe(library);
      const untagged = row({ ollamaTag: null });
      expect(await withExactFacts(HOST, untagged)).toBe(untagged);
      expect(fetchRegistryFacts).toHaveBeenCalledTimes(1);
    });
  });
});
