import type { ArchFacts } from "../core/types.js";

// Definition: sanity bounds that reject garbage metadata; every real model is far inside them.
const MAX_LAYERS = 1024;
const MAX_HEADS = 4096;
const MAX_HEAD_DIM = 65536;
const MAX_CONTEXT_LENGTH = 2 ** 24;
const MAX_VOCAB_SIZE = 2 ** 24;
const MAX_SSM_DIM = 2 ** 20;
const MAX_SSM_GROUPS = 1024;
const MAX_FFN_LENGTH = 2 ** 24;
const BYTES_PER_F32 = 4; // recurrent state is F32 (L src/llama-model.cpp:3133-3134, recurrent_type_r/s)

// Interval of full-attention layers when the GGUF has no full_attention_interval key.
const DEFAULT_FULL_ATTENTION_INTERVAL = new Map([
  ["qwen3next", 4], // L src/models/qwen3next.cpp:17-23
  ["qwen35", 4], // L src/models/qwen35.cpp:17-23
  ["qwen35moe", 4], // L src/models/qwen35moe.cpp:20-26
  ["qwen4exp", 4], // L src/models/qwen4exp.cpp:152-158
  ["minimax-01", 8], // L src/models/minimax-01.cpp:12-17
]);

// Ollama-published architectures that are not llama.cpp architectures (absent from L src/llama-arch.cpp, and with no
// O llama/compat handler), so llama.cpp's KV formula does not describe them: callers fall back to estimates.
const NOT_LLAMA_CPP_ARCHITECTURES = new Set(["mllama"]);

// Parallel hybrids: every layer holds both KV and SSM state (L src/models/falcon-h1.cpp:14, L src/llama-model.cpp:3352-3354,
// L src/llama-arch.cpp:1402); the state is not modelled.
const PARALLEL_HYBRID_ARCHITECTURES = new Set(["falcon-h1", "inkling2"]);

// Zero-head layers of these architectures are recurrent only when their feed_forward_length is also 0 (L src/models/nemotron-h.cpp:13-16).
const NEMOTRON_H_ARCHITECTURES = new Set(["nemotron_h", "nemotron_h_moe"]);

type Lookup = (name: string) => unknown;

function boundedInt(value: unknown, min: number, max: number): number | undefined {
  return typeof value === "number" && Number.isInteger(value) && value >= min && value <= max ? value : undefined;
}

function boolArray(value: unknown, length: number): boolean[] | undefined {
  return Array.isArray(value) && value.length === length && value.every((v) => typeof v === "boolean")
    ? (value as boolean[])
    : undefined;
}

/** Scalar for every layer, or one entry per layer; absent means MHA (L src/llama-model.cpp:1599-1602: n_head_kv defaults to n_head). */
function headCountKv(raw: unknown, layers: number, headCount: number | undefined): number | number[] | undefined {
  if (raw === undefined) return headCount;
  if (Array.isArray(raw)) {
    if (raw.length !== layers) return undefined;
    const perLayer = raw.map((v) => boundedInt(v, 0, MAX_HEADS));
    return perLayer.every((v): v is number => v !== undefined) && perLayer.some((v) => v! > 0) ? perLayer : undefined;
  }
  return boundedInt(raw, 1, MAX_HEADS);
}

/**
 * Layer il is recurrent unless (il + 1) % interval == 0, and never past layers - nextn (L src/models/qwen35.cpp:17-21).
 * An explicit per-layer array wins; an absent interval falls back to the architecture's default.
 */
function recurrentLayers(get: Lookup, architecture: string, layers: number, nextn: number): boolean[] | undefined {
  const explicit = get("attention.recurrent_layers");
  const rawInterval = get("full_attention_interval");
  const interval = rawInterval === undefined ? DEFAULT_FULL_ATTENTION_INTERVAL.get(architecture) : boundedInt(rawInterval, 1, MAX_LAYERS);
  const mask = explicit !== undefined
    ? boolArray(explicit, layers)
    : interval === undefined ? undefined : Array.from({ length: layers }, (_, il) => il < layers - nextn && (il + 1) % interval !== 0);
  return mask?.some(Boolean) ? mask : undefined;
}

/** F32 state of every recurrent layer (L src/llama-hparams.cpp:208-263). */
function recurrentStateBytes(get: Lookup, architecture: string, layers: number, heads: number | number[], recurrent: boolean[] | undefined): number | undefined {
  const dConv = boundedInt(get("ssm.conv_kernel"), 1, MAX_SSM_DIM);
  const dInner = boundedInt(get("ssm.inner_size"), 1, MAX_SSM_DIM);
  const dState = boundedInt(get("ssm.state_size"), 1, MAX_SSM_DIM);
  const nGroup = get("ssm.group_count") === undefined ? 0 : boundedInt(get("ssm.group_count"), 0, MAX_SSM_GROUPS);
  if (dConv === undefined || dInner === undefined || dState === undefined || nGroup === undefined) return undefined;

  const rawFfn = get("feed_forward_length");
  const ffn = Array.isArray(rawFfn) && rawFfn.length === layers ? rawFfn.map((v) => boundedInt(v, 0, MAX_FFN_LENGTH)) : undefined;
  const needsFfn = NEMOTRON_H_ARCHITECTURES.has(architecture);
  if (needsFfn && !ffn?.every((v) => v !== undefined)) return undefined;

  let recurrentLayerCount = 0;
  for (let il = 0; il < layers; il++) {
    const noKvHeads = Array.isArray(heads) && heads[il] === 0 && (!needsFfn || ffn?.[il] === 0);
    if (recurrent?.[il] || noKvHeads) recurrentLayerCount++;
  }
  if (recurrentLayerCount === 0) return undefined;
  const perLayer = (dConv - 1) * (dInner + 2 * nGroup * dState) + dState * dInner;
  return BYTES_PER_F32 * perLayer * recurrentLayerCount;
}

/**
 * Architecture facts from GGUF metadata, using llama.cpp's key names. `kv` is a parsed GGUF header (Map) or Ollama's
 * /api/show `model_info` (Record). `tokenCount` is the tokenizer array length when `<arch>.vocab_size` is absent.
 * null when a required fact is missing or malformed; malformed optional facts are omitted so callers fall back to estimates.
 */
export function archFactsFromMetadata(kv: Map<string, unknown> | Record<string, unknown>, tokenCount?: number): ArchFacts | null {
  const read = (key: string): unknown =>
    kv instanceof Map ? kv.get(key) : Object.hasOwn(kv, key) ? kv[key] : undefined;
  const architecture = read("general.architecture");
  if (typeof architecture !== "string" || architecture === "" || NOT_LLAMA_CPP_ARCHITECTURES.has(architecture)) return null;
  const get: Lookup = (name) => read(`${architecture}.${name}`);

  const layers = boundedInt(get("block_count"), 1, MAX_LAYERS);
  if (layers === undefined) return null;
  const headCount = boundedInt(get("attention.head_count"), 1, MAX_HEADS);
  const heads = headCountKv(get("attention.head_count_kv"), layers, headCount);
  if (heads === undefined) return null;

  const mlaKeyLength = boundedInt(get("attention.key_length_mla"), 1, MAX_HEAD_DIM);
  const mlaValueLength = boundedInt(get("attention.value_length_mla"), 1, MAX_HEAD_DIM);
  // MLA needs both dims (L src/llama-hparams.cpp:303-308); kv_lora_rank alone is old-format deepseek2, which caches K and V.
  const mla = mlaKeyLength !== undefined && mlaValueLength !== undefined;
  // llama.cpp's MLA GGUFs hold one latent KV head (L conversion/deepseek.py:331-333,356-360). Ollama's own MLA layouts
  // (glm4moelite) are rewritten at load time, so their stored dims do not describe the cache: caller falls back to estimates.
  if (mla && [heads].flat().some((h) => h > 1)) return null;

  const embedding = get("embedding_length");
  const headDim = (name: string): number | undefined => {
    const raw = get(`attention.${name}`);
    if (raw !== undefined) return boundedInt(raw, 1, MAX_HEAD_DIM);
    // The embedding split is wrong for MLA, whose cache rows are kv_lora_rank + rope wide.
    if (mla || headCount === undefined || typeof embedding !== "number") return undefined;
    return boundedInt(embedding / headCount, 1, MAX_HEAD_DIM);
  };
  const keyLength = headDim("key_length");
  const valueLength = headDim("value_length");
  if (keyLength === undefined || valueLength === undefined) return null;

  const facts: ArchFacts = { architecture, layers, headCountKv: heads, keyLength, valueLength };
  const set = <K extends keyof ArchFacts>(key: K, value: ArchFacts[K] | undefined) => {
    if (value !== undefined) facts[key] = value as ArchFacts[K];
  };

  set("keyLengthSwa", boundedInt(get("attention.key_length_swa"), 1, MAX_HEAD_DIM));
  set("valueLengthSwa", boundedInt(get("attention.value_length_swa"), 1, MAX_HEAD_DIM));
  set("slidingWindow", boundedInt(get("attention.sliding_window"), 0, MAX_CONTEXT_LENGTH)); // 0 = chunked SWA off (L src/models/llama4.cpp:8-11)
  const pattern = get("attention.sliding_window_pattern");
  set("slidingWindowPattern", Array.isArray(pattern) ? boolArray(pattern, layers) : boundedInt(pattern, 0, MAX_LAYERS));
  if (mla) facts.mla = true;

  // Only the first layers own KV: gemma4 shares the rest (L src/models/gemma4.cpp:9-17), and nextn prediction layers own
  // none (L src/llama-model.cpp:3466-3472). That assumes draft_num_predict is unset: O server/routes.go:168-170 then zeroes
  // DraftNumPredict for a model without a draft, and O llm/llama_server.go:869-871 adds no speculative args, so MTP is off.
  const nextn = boundedInt(get("nextn_predict_layers"), 1, layers - 1);
  const sharedKvLayers = boundedInt(get("attention.shared_kv_layers"), 1, layers - 1);
  const kvLayers = layers - (sharedKvLayers ?? 0) - (nextn ?? 0);
  if (kvLayers < layers && kvLayers >= 1) facts.kvLayers = kvLayers;
  if (PARALLEL_HYBRID_ARCHITECTURES.has(architecture)) facts.inexactReason = "parallel hybrid: SSM state is not modelled";

  const recurrent = recurrentLayers(get, architecture, layers, nextn ?? 0);
  set("recurrentLayers", recurrent);
  set("recurrentStateBytes", recurrentStateBytes(get, architecture, layers, heads, recurrent));

  set("contextLength", boundedInt(get("context_length"), 1, MAX_CONTEXT_LENGTH));
  set("vocabSize", boundedInt(get("vocab_size"), 1, MAX_VOCAB_SIZE) ?? boundedInt(tokenCount, 1, MAX_VOCAB_SIZE));
  return facts;
}
