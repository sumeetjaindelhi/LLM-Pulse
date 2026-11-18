import { describe, it, expect } from "vitest";
import { archFactsFromMetadata } from "../../src/models/arch-facts.js";

function llama(over: Record<string, unknown> = {}): Record<string, unknown> {
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

describe("archFactsFromMetadata", () => {
  it("reads a llama-style model from a record and from a Map", () => {
    const expected = {
      architecture: "llama",
      layers: 32,
      headCountKv: 8,
      keyLength: 128,
      valueLength: 128,
      contextLength: 131072,
    };
    expect(archFactsFromMetadata(llama())).toEqual(expected);
    expect(archFactsFromMetadata(new Map(Object.entries(llama())))).toEqual(expected);
  });

  it("falls back to embedding_length / head_count for head sizes and to head_count for kv heads", () => {
    const kv = llama();
    for (const k of ["key_length", "value_length", "head_count_kv"]) delete kv[`llama.attention.${k}`];
    const facts = archFactsFromMetadata(kv);
    expect(facts).toMatchObject({ headCountKv: 32, keyLength: 128, valueLength: 128 });
  });

  // Rule: mllama is not in L src/llama-arch.cpp and has no O llama/compat handler, so llama.cpp's KV formula does not apply.
  it("returns null for an architecture llama.cpp does not run (llama3.2-vision's mllama)", () => {
    const kv = Object.fromEntries(Object.entries(llama()).map(([k, v]) => [k.replace(/^llama\./, "mllama."), v]));
    expect(archFactsFromMetadata({ ...kv, "general.architecture": "mllama" })).toBeNull();
  });

  it("keeps a per-layer head_count_kv array, zeros included", () => {
    const facts = archFactsFromMetadata(llama({ "llama.block_count": 4, "llama.attention.head_count_kv": [8, 8, 0, 8] }));
    expect(facts?.headCountKv).toEqual([8, 8, 0, 8]);
  });

  it("takes vocab from <arch>.vocab_size, else from the parsed token count", () => {
    expect(archFactsFromMetadata(llama({ "llama.vocab_size": 128256 }), 1)?.vocabSize).toBe(128256);
    expect(archFactsFromMetadata(llama(), 128256)?.vocabSize).toBe(128256);
    expect(archFactsFromMetadata(llama())).not.toHaveProperty("vocabSize");
    expect(archFactsFromMetadata(llama(), 1e12)).not.toHaveProperty("vocabSize");
  });

  it("reads sliding-window keys with a scalar pattern", () => {
    const facts = archFactsFromMetadata({
      "general.architecture": "gemma3",
      "gemma3.block_count": 62,
      "gemma3.attention.head_count": 32,
      "gemma3.attention.head_count_kv": 16,
      "gemma3.attention.key_length": 128,
      "gemma3.attention.value_length": 128,
      "gemma3.attention.key_length_swa": 64,
      "gemma3.attention.value_length_swa": 64,
      "gemma3.attention.sliding_window": 1024,
      "gemma3.attention.sliding_window_pattern": 6,
    });
    expect(facts).toMatchObject({
      keyLengthSwa: 64,
      valueLengthSwa: 64,
      slidingWindow: 1024,
      slidingWindowPattern: 6,
    });
  });

  it("accepts a bool-array sliding-window pattern only when its length equals the layer count", () => {
    const pattern = [true, false, true, false];
    const kv = (p: unknown) => llama({ "llama.block_count": 4, "llama.attention.head_count_kv": 8, "llama.attention.sliding_window_pattern": p });
    expect(archFactsFromMetadata(kv(pattern))?.slidingWindowPattern).toEqual(pattern);
    expect(archFactsFromMetadata(kv([true, false]))).not.toHaveProperty("slidingWindowPattern");
    expect(archFactsFromMetadata(kv(["x", "y", "z", "w"]))).not.toHaveProperty("slidingWindowPattern");
  });

  // Rule: MLA needs both key_length_mla and value_length_mla (L src/llama-hparams.cpp:303-308); kv_lora_rank alone
  // is an old-format deepseek2 (or minicpm3/plm) that caches K and V.
  describe("MLA", () => {
    const mla = { "llama.attention.key_length_mla": 192, "llama.attention.value_length_mla": 128, "llama.attention.head_count_kv": 1 };

    it("flags MLA only from non-zero key_length_mla and value_length_mla", () => {
      expect(archFactsFromMetadata(llama(mla))?.mla).toBe(true);
      expect(archFactsFromMetadata(llama({ ...mla, "llama.attention.key_length_mla": 0, "llama.attention.value_length_mla": 0 }))).not.toHaveProperty("mla");
      expect(archFactsFromMetadata(llama({ ...mla, "llama.attention.value_length_mla": undefined }))).not.toHaveProperty("mla");
    });

    it("does not flag old-format deepseek2 that has kv_lora_rank but no *_mla keys", () => {
      const facts = archFactsFromMetadata(llama({ "llama.attention.kv_lora_rank": 512, "llama.attention.head_count_kv": 16 }));
      expect(facts).not.toHaveProperty("mla");
      expect(facts).toMatchObject({ headCountKv: 16, keyLength: 128, valueLength: 128 });
    });

    // Rule: llama.cpp MLA GGUFs store one latent KV head (L conversion/deepseek.py:331-333,356-360); Ollama's glm4moelite
    // keeps per-head dims that its compat layer rewrites at load time, so the stored layout is unknown.
    it("returns null for MLA metadata whose kv head count is not 1", () => {
      expect(archFactsFromMetadata(llama({ ...mla, "llama.attention.head_count_kv": 20 }))).toBeNull();
      expect(archFactsFromMetadata(llama({ ...mla, "llama.block_count": 4, "llama.attention.head_count_kv": [1, 1, 20, 1] }))).toBeNull();
    });

    it("returns null for MLA without key_length (the embedding fallback would be wrong)", () => {
      const kv = llama(mla);
      delete kv["llama.attention.key_length"];
      expect(archFactsFromMetadata(kv)).toBeNull();
    });
  });

  it("keeps an explicit sliding_window of 0 (llama4 turns chunked SWA off, L src/models/llama4.cpp:8-11)", () => {
    expect(archFactsFromMetadata(llama({ "llama.attention.sliding_window": 0 }))?.slidingWindow).toBe(0);
  });

  describe("recurrent layers", () => {
    const qwen = (over: Record<string, unknown> = {}) => llama({
      "general.architecture": "qwen35",
      "qwen35.block_count": 8,
      "qwen35.attention.head_count": 16,
      "qwen35.attention.head_count_kv": 2,
      "qwen35.attention.key_length": 256,
      "qwen35.attention.value_length": 256,
      "llama.block_count": undefined,
      ...over,
    });

    it("marks layer il recurrent unless (il + 1) % full_attention_interval == 0", () => {
      const facts = archFactsFromMetadata(qwen({ "qwen35.full_attention_interval": 4 }));
      expect(facts?.recurrentLayers).toEqual([true, true, true, false, true, true, true, false]);
    });

    it("prefers the explicit attention.recurrent_layers array", () => {
      const mask = [false, true, false, true, false, true, false, true];
      const facts = archFactsFromMetadata(qwen({ "qwen35.attention.recurrent_layers": mask, "qwen35.full_attention_interval": 4 }));
      expect(facts?.recurrentLayers).toEqual(mask);
    });

    // Rule: llama.cpp defaults full_attn_interval to 4 (qwen3next/qwen35/qwen35moe/qwen4exp) or 8 (minimax-01)
    // when the key is absent (L src/models/qwen3next.cpp:17-23, qwen35.cpp:17-23, qwen35moe.cpp:20-26, qwen4exp.cpp:152-158, minimax-01.cpp:12-17).
    it("applies the architecture's default interval when full_attention_interval is absent", () => {
      const withArch = (arch: string) => llama({
        "general.architecture": arch,
        [`${arch}.block_count`]: 8,
        [`${arch}.attention.head_count_kv`]: 2,
        [`${arch}.attention.key_length`]: 256,
        [`${arch}.attention.value_length`]: 256,
        "llama.block_count": undefined,
      });
      for (const arch of ["qwen3next", "qwen35", "qwen35moe", "qwen4exp"]) {
        expect(archFactsFromMetadata(withArch(arch))?.recurrentLayers, arch).toEqual([true, true, true, false, true, true, true, false]);
      }
      expect(archFactsFromMetadata(withArch("minimax-01"))?.recurrentLayers).toEqual([true, true, true, true, true, true, true, false]);
      expect(archFactsFromMetadata(llama({ "llama.block_count": 8 }))).not.toHaveProperty("recurrentLayers");
    });

    // Rule: the mask covers il < n_layer - nextn_predict_layers (L src/models/qwen35.cpp:21).
    it("never marks nextn prediction layers recurrent", () => {
      const facts = archFactsFromMetadata(qwen({ "qwen35.block_count": 9, "qwen35.full_attention_interval": 4, "qwen35.nextn_predict_layers": 1 }));
      expect(facts?.recurrentLayers).toEqual([true, true, true, false, true, true, true, false, false]);
    });

    it("omits recurrentLayers when every layer is full attention or the interval is invalid", () => {
      expect(archFactsFromMetadata(qwen({ "qwen35.full_attention_interval": 1 }))).not.toHaveProperty("recurrentLayers");
      expect(archFactsFromMetadata(qwen({ "qwen35.full_attention_interval": 0 }))).not.toHaveProperty("recurrentLayers");
      expect(archFactsFromMetadata(qwen({ "qwen35.attention.recurrent_layers": [true] }))).not.toHaveProperty("recurrentLayers");
    });

    it("sizes the constant F32 state: 4 x ((d_conv-1)(d_inner + 2 n_group d_state) + d_state d_inner) x recurrent layers", () => {
      const ssm = {
        "qwen35.full_attention_interval": 4,
        "qwen35.ssm.conv_kernel": 4,
        "qwen35.ssm.inner_size": 4096,
        "qwen35.ssm.state_size": 128,
        "qwen35.ssm.group_count": 8,
      };
      // 4 x (3 x (4096 + 2048) + 524288) = 2170880 per layer; 6 of 8 layers recurrent.
      expect(archFactsFromMetadata(qwen(ssm))?.recurrentStateBytes).toBe(2170880 * 6);
    });

    it("counts layers with zero kv heads as recurrent and treats a missing group_count as 0", () => {
      const facts = archFactsFromMetadata(qwen({
        "qwen35.full_attention_interval": 1, // explicit: every layer attends, so only zero-head layers are recurrent
        "qwen35.attention.head_count_kv": [0, 0, 2, 0, 0, 2, 0, 0],
        "qwen35.ssm.conv_kernel": 4,
        "qwen35.ssm.inner_size": 1000,
        "qwen35.ssm.state_size": 16,
      }));
      // 6 zero-head layers x 4 x (3 x 1000 + 16 x 1000)
      expect(facts?.recurrentStateBytes).toBe(6 * 4 * 19000);
    });

    // Rule: nemotron_h / nemotron_h_moe layers are recurrent only when n_head_kv == 0 and n_ff == 0 (L src/models/nemotron-h.cpp:13-16).
    it("counts a zero-head nemotron_h layer as recurrent only when its feed_forward_length is 0", () => {
      const nemotron = (ff: unknown) => llama({
        "general.architecture": "nemotron_h_moe",
        "nemotron_h_moe.block_count": 4,
        "nemotron_h_moe.attention.head_count_kv": [0, 0, 2, 0],
        "nemotron_h_moe.attention.key_length": 128,
        "nemotron_h_moe.attention.value_length": 128,
        "nemotron_h_moe.feed_forward_length": ff,
        "nemotron_h_moe.ssm.conv_kernel": 4,
        "nemotron_h_moe.ssm.inner_size": 1000,
        "nemotron_h_moe.ssm.state_size": 16,
        "llama.block_count": undefined,
      });
      // layers 0 and 3 have ff 0 -> recurrent; layer 1 is an MoE layer without heads.
      expect(archFactsFromMetadata(nemotron([0, 1024, 0, 0]))?.recurrentStateBytes).toBe(2 * 4 * 19000);
      expect(archFactsFromMetadata(nemotron(undefined))).not.toHaveProperty("recurrentStateBytes");
    });

    it("omits the state size when a key is missing or invalid, or no layer is recurrent", () => {
      const partial = { "qwen35.full_attention_interval": 4, "qwen35.ssm.conv_kernel": 4, "qwen35.ssm.inner_size": 4096 };
      expect(archFactsFromMetadata(qwen(partial))).not.toHaveProperty("recurrentStateBytes");
      const noRecurrent = { "qwen35.full_attention_interval": 1, "qwen35.ssm.conv_kernel": 4, "qwen35.ssm.inner_size": 4096, "qwen35.ssm.state_size": 128 };
      expect(archFactsFromMetadata(qwen(noRecurrent))).not.toHaveProperty("recurrentStateBytes");
      expect(archFactsFromMetadata(qwen({ ...partial, "qwen35.ssm.state_size": -1 }))).not.toHaveProperty("recurrentStateBytes");
    });
  });

  describe("kvLayers", () => {
    it("subtracts attention.shared_kv_layers", () => {
      const facts = archFactsFromMetadata(llama({ "llama.block_count": 35, "llama.attention.shared_kv_layers": 20 }));
      expect(facts?.kvLayers).toBe(15);
    });
    // O server/routes.go:168-170 zeroes DraftNumPredict without a draft model unless draft_num_predict is set, and
    // O llm/llama_server.go:869-871 then adds no speculative args: MTP is off by default, so the nextn layers own no KV
    // (L src/llama-model.cpp:3466-3472).
    it("excludes nextn_predict_layers from KV: MTP is off while draft_num_predict is unset", () => {
      const facts = archFactsFromMetadata(llama({ "llama.block_count": 48, "llama.nextn_predict_layers": 1 }));
      expect(facts?.kvLayers).toBe(47);
      expect(facts).not.toHaveProperty("inexactReason");
    });
    it("also subtracts shared_kv_layers on top of nextn layers", () => {
      // 35 - 20 shared - 1 nextn = 14
      expect(archFactsFromMetadata(llama({ "llama.block_count": 35, "llama.attention.shared_kv_layers": 20, "llama.nextn_predict_layers": 1 }))?.kvLayers).toBe(14);
    });
    it("ignores counts that leave no KV layer", () => {
      expect(archFactsFromMetadata(llama({ "llama.attention.shared_kv_layers": 32 }))).not.toHaveProperty("kvLayers");
      expect(archFactsFromMetadata(llama({ "llama.attention.shared_kv_layers": 0 }))).not.toHaveProperty("kvLayers");
    });
  });

  // Parallel hybrids: every layer has KV and SSM state (L src/models/falcon-h1.cpp:14, L src/llama-model.cpp:3352-3354).
  describe("parallel hybrids", () => {
    it.each(["falcon-h1", "inkling2"])("%s is not exact: the SSM state is not modelled", (architecture) => {
      const facts = archFactsFromMetadata({
        "general.architecture": architecture,
        [`${architecture}.block_count`]: 44,
        [`${architecture}.attention.head_count`]: 8,
        [`${architecture}.attention.head_count_kv`]: 2,
        [`${architecture}.attention.key_length`]: 128,
        [`${architecture}.attention.value_length`]: 128,
      });
      expect(facts?.inexactReason).toMatch(/parallel hybrid/);
    });
    it("plain transformers carry no inexactReason", () => {
      expect(archFactsFromMetadata(llama())).not.toHaveProperty("inexactReason");
    });
  });

  describe("validation", () => {
    it("returns null for missing or malformed required facts", () => {
      for (const bad of [
        { "general.architecture": 7 },
        { "general.architecture": "" },
        { "llama.block_count": -1 },
        { "llama.block_count": 0 },
        { "llama.block_count": 1e9 },
        { "llama.block_count": 1.5 },
        { "llama.attention.head_count_kv": "eight" },
        { "llama.attention.head_count_kv": 0 },
        { "llama.attention.head_count_kv": [8, 8] },
        { "llama.attention.head_count_kv": [0, 0, 0, 0] },
        { "llama.attention.head_count_kv": [-1, 8, 8, 8] },
        { "llama.attention.key_length": 0 },
        { "llama.attention.key_length": 1e9 },
        { "llama.attention.value_length": null },
        { "llama.embedding_length": 4097, "llama.attention.key_length": undefined },
      ]) {
        const kv = llama({ "llama.block_count": 4, ...bad });
        expect(archFactsFromMetadata(kv), JSON.stringify(bad)).toBeNull();
      }
      const noArch = llama();
      delete noArch["general.architecture"];
      expect(archFactsFromMetadata(noArch)).toBeNull();
      expect(archFactsFromMetadata({})).toBeNull();
    });

    it("omits optional facts that are out of range instead of failing", () => {
      const facts = archFactsFromMetadata(llama({
        "llama.context_length": Number.MAX_SAFE_INTEGER,
        "llama.attention.sliding_window": -5,
        "llama.attention.key_length_swa": 0,
        "llama.vocab_size": "big",
      }));
      expect(facts).toEqual({ architecture: "llama", layers: 32, headCountKv: 8, keyLength: 128, valueLength: 128 });
    });

    it("rejects non-finite numbers", () => {
      expect(archFactsFromMetadata(llama({ "llama.attention.key_length": Infinity }))).toBeNull();
      expect(archFactsFromMetadata(llama({ "llama.block_count": NaN }))).toBeNull();
    });
  });
});
