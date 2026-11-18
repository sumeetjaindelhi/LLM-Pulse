import { describe, it, expect } from "vitest";
import { getCuratedModels } from "../../src/models/database.js";
import { kvCacheBytes } from "../../src/analysis/memory-model.js";

// data/models.json carries facts read from registry.ollama.ai: the default tag's file (manifest model + projector
// layers) and the GGUF header's architecture. These tests pin the invariants the memory model relies on.
const curated = getCuratedModels();
const byTag = (tag: string) => curated.find((m) => m.ollamaTag === tag)!;

describe("curated registry facts", () => {
  it("lists only quants Ollama publishes a tag for", () => {
    // ollama.com/library/<slug>/tags (2026-11-18): these sizes have q4_K_M, q8_0 and fp16 tags (gemma3 also qat, bf16),
    // no q5_K_M.
    const tags = ["deepseek-r1:7b", "deepseek-r1:14b", "deepseek-r1:32b", "phi4:14b", "gemma3:12b", "gemma3:27b",
      "qwen3:8b", "qwen3:14b", "qwen3:32b", "mistral-small:24b", "qwq:32b", "llama4:scout"];
    for (const tag of tags) {
      expect(byTag(tag).quantizations.map((q) => q.name), tag).toEqual(["Q4_K_M", "Q8_0"]);
    }
  });

  it("marks exactly one quant per model as the file its tag installs", () => {
    for (const model of curated) {
      expect(model.quantizations.filter((q) => q.exact), model.id).toHaveLength(1);
    }
  });

  it("gives every llama.cpp architecture exact KV sizing and its GGUF training context as the window", () => {
    // llama3.2-vision is arch mllama, which neither llama.cpp a300a9c nor Ollama b8db8d5's compat layer knows.
    for (const model of curated.filter((m) => m.ollamaTag !== "llama3.2-vision:11b")) {
      expect(model.arch, model.id).toBeDefined();
      expect(model.contextWindow, model.id).toBe(model.arch!.contextLength);
      expect(kvCacheBytes(model.arch!, 32768, "f16").exact, model.id).toBe(true);
    }
  });

  it("matches the registry for spot-checked models", () => {
    // gemma3:27b: Q4_K_M model layer 16555661472 B + projector 857739168 B = 16606.7 MiB; GGUF gemma3.* keys.
    expect(byTag("gemma3:27b").quantizations.find((q) => q.exact)).toMatchObject({ name: "Q4_K_M", vramMb: 16607 });
    expect(byTag("gemma3:27b").arch).toMatchObject({ layers: 62, headCountKv: 16, keyLength: 128, slidingWindow: 1024, contextLength: 131072 });
    // gemma2:9b's default tag is a Q4_0 file (config blob file_type), not Q4_K_M.
    expect(byTag("gemma2:9b").quantizations.find((q) => q.exact)).toMatchObject({ name: "Q4_0", vramMb: 5191 });
    // qwen2.5's GGUF trains on 32768 tokens (qwen2.context_length); 131072 needs YaRN, which Ollama does not enable.
    expect(byTag("qwen2.5:14b").contextWindow).toBe(32768);
  });
});
