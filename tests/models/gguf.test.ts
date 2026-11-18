import { describe, it, expect } from "vitest";
import { parseGgufMetadata, GGUF_MAX_INPUT_BYTES } from "../../src/models/gguf.js";

const enc = new TextEncoder();

function concat(...parts: Uint8Array[]): Uint8Array {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}
function u32(n: number): Uint8Array {
  const b = new Uint8Array(4);
  new DataView(b.buffer).setUint32(0, n, true);
  return b;
}
function u64(n: number | bigint): Uint8Array {
  const b = new Uint8Array(8);
  new DataView(b.buffer).setBigUint64(0, BigInt(n), true);
  return b;
}
function f32(n: number): Uint8Array {
  const b = new Uint8Array(4);
  new DataView(b.buffer).setFloat32(0, n, true);
  return b;
}
const str = (s: string) => concat(u64(enc.encode(s).length), enc.encode(s));
const entry = (key: string, type: number, value: Uint8Array) => concat(str(key), u32(type), value);

const kvStr = (key: string, v: string) => entry(key, 8, str(v));
const kvU32 = (key: string, v: number) => entry(key, 4, u32(v));
const kvBool = (key: string, v: boolean) => entry(key, 7, Uint8Array.of(v ? 1 : 0));
const kvF32 = (key: string, v: number) => entry(key, 6, f32(v));
const kvArrayU32 = (key: string, values: number[]) =>
  entry(key, 9, concat(u32(4), u64(values.length), ...values.map(u32)));
const kvArrayBool = (key: string, values: boolean[]) =>
  entry(key, 9, concat(u32(7), u64(values.length), Uint8Array.from(values.map((v) => (v ? 1 : 0)))));
const kvArrayStr = (key: string, values: string[]) =>
  entry(key, 9, concat(u32(8), u64(values.length), ...values.map(str)));
// Header plus a few elements: the parser must stop at the header and never read the elements.
const kvArrayStrHeader = (key: string, count: number) => entry(key, 9, concat(u32(8), u64(count), str("a")));

function gguf(entries: Uint8Array[], opts: { version?: number; declared?: number | bigint } = {}): Uint8Array {
  return concat(
    u32(0x46554747),
    u32(opts.version ?? 3),
    u64(0),
    u64(opts.declared ?? entries.length),
    ...entries,
  );
}

describe("parseGgufMetadata", () => {
  it("reads scalars, arch and short numeric/bool arrays", () => {
    const meta = parseGgufMetadata(gguf([
      kvStr("general.architecture", "llama"),
      kvU32("llama.block_count", 32),
      kvBool("llama.flag", true),
      kvF32("llama.rope.freq_base", 500000),
      kvArrayU32("llama.attention.head_count_kv", [8, 8, 0]),
      kvArrayBool("llama.attention.sliding_window_pattern", [true, false]),
    ]));
    expect(meta?.architecture).toBe("llama");
    expect(meta?.truncated).toBe(false);
    expect(meta?.kv.get("llama.block_count")).toBe(32);
    expect(meta?.kv.get("llama.flag")).toBe(true);
    expect(meta?.kv.get("llama.rope.freq_base")).toBe(500000);
    expect(meta?.kv.get("llama.attention.head_count_kv")).toEqual([8, 8, 0]);
    expect(meta?.kv.get("llama.attention.sliding_window_pattern")).toEqual([true, false]);
  });

  it("accepts version 2 and rejects version 1, bad magic and empty input", () => {
    const entries = [kvStr("general.architecture", "llama")];
    expect(parseGgufMetadata(gguf(entries, { version: 2 }))?.architecture).toBe("llama");
    expect(parseGgufMetadata(gguf(entries, { version: 1 }))).toBeNull();
    expect(parseGgufMetadata(gguf(entries, { version: 4 }))).toBeNull();
    expect(parseGgufMetadata(enc.encode("NOTGGUF-------------------------"))).toBeNull();
    expect(parseGgufMetadata(new Uint8Array(0))).toBeNull();
    expect(parseGgufMetadata(u32(0x46554747))).toBeNull();
  });

  it("returns an empty truncated result when the header is cut after the version", () => {
    const meta = parseGgufMetadata(concat(u32(0x46554747), u32(3), u64(0)));
    expect(meta).toEqual({ kv: new Map(), truncated: true });
  });

  it("stops at the first tokenizer array and records the token count as vocabSize", () => {
    const meta = parseGgufMetadata(gguf([
      kvStr("general.architecture", "llama"),
      kvStr("tokenizer.ggml.model", "gpt2"),
      kvArrayStrHeader("tokenizer.ggml.tokens", 128256),
      kvArrayU32("tokenizer.ggml.scores", [1, 2]),
      kvU32("after.tokens", 1),
    ]));
    expect(meta?.vocabSize).toBe(128256);
    expect(meta?.kv.get("tokenizer.ggml.model")).toBe("gpt2");
    expect(meta?.kv.has("after.tokens")).toBe(false);
    expect(meta?.kv.has("tokenizer.ggml.scores")).toBe(false);
    expect(meta?.truncated).toBe(false);
  });

  it("leaves vocabSize unset when another tokenizer array comes first", () => {
    const meta = parseGgufMetadata(gguf([kvArrayStrHeader("tokenizer.ggml.scores", 10), kvU32("x", 1)]));
    expect(meta?.vocabSize).toBeUndefined();
    expect(meta?.kv.size).toBe(0);
  });

  it("skips string arrays and arrays over 4096 elements, then keeps reading", () => {
    const big = Array.from({ length: 4097 }, (_, i) => i);
    const meta = parseGgufMetadata(gguf([
      kvArrayStr("general.tags", ["a", "bb", ""]),
      kvArrayU32("big", big),
      kvArrayU32("exact", big.slice(0, 4096)),
      kvU32("llama.block_count", 7),
    ]));
    expect(meta?.kv.has("general.tags")).toBe(false);
    expect(meta?.kv.has("big")).toBe(false);
    expect((meta?.kv.get("exact") as number[]).length).toBe(4096);
    expect(meta?.kv.get("llama.block_count")).toBe(7);
    expect(meta?.truncated).toBe(false);
  });

  it("skips a string value longer than 64 KiB and keeps reading", () => {
    const meta = parseGgufMetadata(gguf([
      kvStr("general.description", "x".repeat(64 * 1024 + 1)),
      kvU32("llama.block_count", 7),
    ]));
    expect(meta?.kv.has("general.description")).toBe(false);
    expect(meta?.kv.get("llama.block_count")).toBe(7);
  });

  it("keeps entries read before the bytes end and flags truncation", () => {
    const full = gguf([kvU32("llama.block_count", 32), kvStr("general.architecture", "llama")]);
    const meta = parseGgufMetadata(full.subarray(0, full.length - 3));
    expect(meta?.kv.get("llama.block_count")).toBe(32);
    expect(meta?.truncated).toBe(true);
  });

  describe("hostile input", () => {
    it("flags a kv count above the 4096 cap after reading that many entries", () => {
      const entries = Array.from({ length: 4097 }, (_, i) => kvU32(`k${i}`, i));
      const meta = parseGgufMetadata(gguf(entries.slice(0, 4096), { declared: 5000 }));
      expect(meta?.kv.size).toBe(4096);
      expect(meta?.truncated).toBe(true);
    });

    it("flags a declared kv count larger than the data", () => {
      expect(parseGgufMetadata(gguf([kvU32("a", 1)], { declared: 3 }))?.truncated).toBe(true);
    });

    it("flags a kv count beyond 2^53", () => {
      expect(parseGgufMetadata(gguf([kvU32("a", 1)], { declared: 2n ** 60n }))?.truncated).toBe(true);
    });

    it("flags a key longer than 64 KiB", () => {
      const meta = parseGgufMetadata(gguf([kvU32("k".repeat(64 * 1024 + 1), 1), kvU32("after", 2)]));
      expect(meta?.kv.size).toBe(0);
      expect(meta?.truncated).toBe(true);
    });

    it("flags a string length that runs past the buffer", () => {
      const lying = entry("general.name", 8, u64(10_000_000));
      const meta = parseGgufMetadata(gguf([kvU32("a", 1), lying]));
      expect(meta?.kv.get("a")).toBe(1);
      expect(meta?.truncated).toBe(true);
    });

    it("flags array counts the buffer cannot hold", () => {
      for (const elementType of [4, 8]) {
        const lying = entry("x", 9, concat(u32(elementType), u64(2n ** 40n)));
        expect(parseGgufMetadata(gguf([lying]))?.truncated, `type ${elementType}`).toBe(true);
      }
    });

    it("flags nested arrays and unknown value types", () => {
      expect(parseGgufMetadata(gguf([entry("x", 9, concat(u32(9), u64(1)))]))?.truncated).toBe(true);
      expect(parseGgufMetadata(gguf([entry("x", 99, new Uint8Array(8))]))?.truncated).toBe(true);
    });

    it("ignores bytes past 256 KiB", () => {
      const padding = kvStr("general.padding", "p".repeat(60 * 1024));
      const entries = [kvStr("general.architecture", "llama"), ...Array.from({ length: 5 }, () => padding)];
      entries.push(kvU32("late", 1));
      const bytes = gguf(entries, { declared: entries.length });
      expect(bytes.length).toBeGreaterThan(GGUF_MAX_INPUT_BYTES);
      const meta = parseGgufMetadata(bytes);
      expect(meta?.architecture).toBe("llama");
      expect(meta?.kv.has("late")).toBe(false);
      expect(meta?.truncated).toBe(true);
    });

    it("never throws on random bytes behind a valid header", () => {
      let seed = 12345;
      const next = () => (seed = (Math.imul(seed, 1103515245) + 12345) >>> 0);
      for (let round = 0; round < 300; round++) {
        const body = Uint8Array.from({ length: 200 }, () => next() >>> 24);
        const meta = parseGgufMetadata(concat(u32(0x46554747), u32(3), u64(0), u64(40), body));
        expect(meta).not.toBeNull();
      }
    });
  });
});
