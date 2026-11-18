// Bounded reader for the GGUF header and metadata key/values.
// Format: ggml docs/gguf.md (github.com/ggml-org/ggml). Little-endian, versions 2 and 3 (64-bit counts).

export type GgufValue = number | boolean | string | Array<number | boolean>;

export interface GgufMetadata {
  architecture?: string;
  kv: Map<string, GgufValue>;
  /** Element count of `tokenizer.ggml.tokens`, read without decoding the array. */
  vocabSize?: number;
  /** True when the input ended, a cap was hit or the bytes were malformed before the metadata was fully read. */
  truncated: boolean;
}

// Definition: caps that bound memory and work on hostile input. Real headers before the tokenizer arrays are far smaller.
export const GGUF_MAX_INPUT_BYTES = 256 * 1024;
const MAX_STRING_BYTES = 64 * 1024;
const MAX_KV_COUNT = 4096;
const MAX_KEPT_ARRAY_ELEMENTS = 4096;

const MAGIC = 0x46554747; // "GGUF" read as a little-endian u32
const SUPPORTED_VERSIONS = new Set([2, 3]);
const TOKENIZER_PREFIX = "tokenizer.";
const TOKENS_KEY = "tokenizer.ggml.tokens";

// gguf_metadata_value_type ids (ggml docs/gguf.md).
const TYPE_BOOL = 7;
const TYPE_STRING = 8;
const TYPE_ARRAY = 9;
// Byte width of each fixed-size type id: u8 i8 u16 i16 u32 i32 f32 bool, then (10..12) u64 i64 f64.
const FIXED_WIDTH: Record<number, number> = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };

const utf8 = new TextDecoder();

/** Thrown internally when the bytes end early or are malformed; never escapes parseGgufMetadata. */
class Truncated extends Error {}

class Reader {
  private pos = 0;
  private readonly view: DataView;
  constructor(private readonly bytes: Uint8Array) {
    this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  }
  get remaining(): number {
    return this.bytes.length - this.pos;
  }
  private take(n: number): number {
    if (n > this.remaining) throw new Truncated();
    const at = this.pos;
    this.pos += n;
    return at;
  }
  skip(n: number): void {
    this.take(n);
  }
  u32(): number {
    return this.view.getUint32(this.take(4), true);
  }
  /** A u64 as a safe integer; a larger value cannot be a real count or length. */
  u64(): number {
    const value = this.view.getBigUint64(this.take(8), true);
    if (value > BigInt(Number.MAX_SAFE_INTEGER)) throw new Truncated();
    return Number(value);
  }
  /** The string, or undefined when it is longer than `maxBytes` (its bytes are still consumed). */
  string(maxBytes: number): string | undefined {
    const length = this.u64();
    const at = this.take(length);
    return length > maxBytes ? undefined : utf8.decode(this.bytes.subarray(at, at + length));
  }
  /** Scalar of fixed-width type `type` (never string/array). */
  scalar(type: number): number | boolean {
    const at = this.take(FIXED_WIDTH[type]);
    switch (type) {
      case 0: return this.view.getUint8(at);
      case 1: return this.view.getInt8(at);
      case 2: return this.view.getUint16(at, true);
      case 3: return this.view.getInt16(at, true);
      case 4: return this.view.getUint32(at, true);
      case 5: return this.view.getInt32(at, true);
      case 6: return this.view.getFloat32(at, true);
      case TYPE_BOOL: return this.view.getUint8(at) !== 0;
      case 10: return Number(this.view.getBigUint64(at, true));
      case 11: return Number(this.view.getBigInt64(at, true));
      default: return this.view.getFloat64(at, true);
    }
  }
}

function skipStringArray(reader: Reader, count: number): void {
  // Every element carries at least its 8-byte length, so a larger count cannot fit the buffer.
  if (count > reader.remaining / 8) throw new Truncated();
  for (let i = 0; i < count; i++) reader.skip(reader.u64());
}

/** Reads an array value: kept when numeric/bool and short, otherwise skipped (undefined). */
function readArray(reader: Reader, elementType: number, count: number): Array<number | boolean> | undefined {
  if (elementType === TYPE_STRING) {
    skipStringArray(reader, count);
    return undefined;
  }
  const width = FIXED_WIDTH[elementType];
  if (width === undefined) throw new Truncated(); // nested arrays or an unknown type
  if (count > reader.remaining / width) throw new Truncated();
  if (count > MAX_KEPT_ARRAY_ELEMENTS) {
    reader.skip(count * width);
    return undefined;
  }
  return Array.from({ length: count }, () => reader.scalar(elementType));
}

function readValue(reader: Reader, type: number): GgufValue | undefined {
  if (type === TYPE_STRING) return reader.string(MAX_STRING_BYTES);
  if (type === TYPE_ARRAY) return readArray(reader, reader.u32(), reader.u64());
  if (FIXED_WIDTH[type] === undefined) throw new Truncated();
  return reader.scalar(type);
}

/**
 * Reads GGUF metadata from the start of a file, stopping at the first `tokenizer.` array (the token list is large and
 * only its length is needed). Never throws: malformed or short input yields `truncated: true` with whatever was read.
 * Returns null when the bytes are not GGUF v2/v3.
 */
export function parseGgufMetadata(input: Uint8Array): GgufMetadata | null {
  const bytes = input.length > GGUF_MAX_INPUT_BYTES ? input.subarray(0, GGUF_MAX_INPUT_BYTES) : input;
  const reader = new Reader(bytes);
  const result: GgufMetadata = { kv: new Map(), truncated: false };
  try {
    if (reader.u32() !== MAGIC) return null;
    if (!SUPPORTED_VERSIONS.has(reader.u32())) return null;
  } catch {
    return null;
  }

  try {
    reader.skip(8); // tensor_count
    const declared = reader.u64();
    for (let i = 0; i < Math.min(declared, MAX_KV_COUNT); i++) {
      const key = reader.string(MAX_STRING_BYTES);
      if (key === undefined) throw new Truncated();
      const type = reader.u32();
      if (type === TYPE_ARRAY && key.startsWith(TOKENIZER_PREFIX)) {
        reader.u32(); // element type
        if (key === TOKENS_KEY) result.vocabSize = reader.u64();
        return finish(result, false);
      }
      const value = readValue(reader, type);
      if (value !== undefined) result.kv.set(key, value);
    }
    return finish(result, declared > MAX_KV_COUNT);
  } catch (error) {
    if (!(error instanceof Truncated)) throw error;
    return finish(result, true);
  }
}

function finish(result: GgufMetadata, truncated: boolean): GgufMetadata {
  const architecture = result.kv.get("general.architecture");
  if (typeof architecture === "string") result.architecture = architecture;
  result.truncated = truncated;
  return result;
}
