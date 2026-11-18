import { describe, it, expect, vi, beforeEach, afterEach, afterAll } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// core/cache derives its root from homedir() at import time, so the throwaway home is set before the dynamic import.
const home = vi.hoisted(() => ({ dir: "" }));
vi.mock("node:os", async (importOriginal) => {
  const actual = await importOriginal<typeof import("node:os")>();
  return { ...actual, homedir: () => home.dir };
});
home.dir = mkdtempSync(join(tmpdir(), "llmpulse-registry-"));
const { fetchRegistryFacts } = await import("../../src/models/ollama-registry.js");

const enc = new TextEncoder();
const concat = (...parts: Uint8Array[]) => Buffer.concat(parts);
const u32 = (n: number) => { const b = Buffer.alloc(4); b.writeUInt32LE(n); return b; };
const u64 = (n: number) => { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(n)); return b; };
const str = (s: string) => concat(u64(enc.encode(s).length), enc.encode(s));
const entry = (key: string, type: number, value: Uint8Array) => concat(str(key), u32(type), value);
const kvU32 = (key: string, v: number) => entry(key, 4, u32(v));

function ggufBlob(over: { arch?: string; omitBlockCount?: boolean } = {}): Buffer {
  const arch = over.arch ?? "llama";
  const entries = [
    entry("general.architecture", 8, str(arch)),
    ...(over.omitBlockCount ? [] : [kvU32(`${arch}.block_count`, 32)]),
    kvU32(`${arch}.attention.head_count`, 32),
    kvU32(`${arch}.attention.head_count_kv`, 8),
    kvU32(`${arch}.attention.key_length`, 128),
    kvU32(`${arch}.attention.value_length`, 128),
    kvU32(`${arch}.context_length`, 8192),
    entry("tokenizer.ggml.tokens", 9, concat(u32(8), u64(128256))),
  ];
  return concat(u32(0x46554747), u32(3), u64(0), u64(entries.length), ...entries);
}

const digest = (c: string) => `sha256:${c.repeat(64)}`;
// Facts are cached by digest, so each test gets its own model and config digests.
let MODEL = digest("a");
let CONFIG = digest("b");
const PROJECTOR = digest("c");
const REGISTRY = "https://registry.ollama.ai/v2/library";

function manifestJson(over: { projector?: boolean; draft?: number; draftDigest?: string; model?: boolean; modelDigest?: string } = {}): string {
  return JSON.stringify({
    schemaVersion: 2,
    config: { mediaType: "application/vnd.docker.container.image.v1+json", digest: CONFIG, size: 400 },
    layers: [
      ...(over.model === false ? [] : [{ mediaType: "application/vnd.ollama.image.model", digest: MODEL, size: 4_920_000_000 }]),
      ...(over.projector ? [{ mediaType: "application/vnd.ollama.image.projector", digest: PROJECTOR, size: 851_000_000 }] : []),
      ...(over.draft ? [{ mediaType: "application/vnd.ollama.image.draft", digest: over.draftDigest ?? digest("e"), size: over.draft }] : []),
      { mediaType: "application/vnd.ollama.image.params", digest: digest("d"), size: 100 },
    ],
  });
}
const CONFIG_JSON = JSON.stringify({ model_format: "gguf", file_type: "Q4_K_M", model_type: "8.0B" });

const json = (body: string) => new Response(body, { status: 200 });
const redirect = (location: string) => new Response(null, { status: 307, headers: { location } });
const R2_URL = "https://abc123.r2.cloudflarestorage.com/blob?sig=1";

interface Routes {
  manifest?: () => Response;
  config?: () => Response;
  blob?: (url: string, init: RequestInit) => Response;
  /** Bodies of manifest-list children, by digest; they are fetched as blobs (O server/images.go pullSelectedManifest). */
  children?: Record<string, string>;
}

describe("fetchRegistryFacts", () => {
  const originalFetch = globalThis.fetch;
  let fetchMock: ReturnType<typeof vi.fn>;
  let n = 0;
  let tag: string;

  function serve(routes: Routes = {}): void {
    fetchMock.mockImplementation(async (input: string | URL, init: RequestInit = {}) => {
      const url = String(input);
      if (url.startsWith("https://abc123.r2")) return routes.blob?.(url, init) ?? new Response(ggufBlob(), { status: 206 });
      if (url.includes("/manifests/")) return routes.manifest?.() ?? json(manifestJson());
      if (url.includes(`/blobs/${encodeURIComponent(CONFIG)}`)) return routes.config?.() ?? json(CONFIG_JSON);
      const child = Object.entries(routes.children ?? {}).find(([d]) => url.endsWith(`/blobs/${encodeURIComponent(d)}`));
      if (child) return json(child[1]);
      if (url.includes("/blobs/")) return routes.blob?.(url, init) ?? new Response(ggufBlob(), { status: 206 });
      throw new Error(`unexpected ${url}`);
    });
  }

  beforeEach(() => {
    delete process.env.LLMPULSE_OFFLINE;
    tag = `model${++n}:tag${n}`; // fresh cache keys per test
    MODEL = `sha256:${(2 * n).toString(16).padStart(64, "0")}`;
    CONFIG = `sha256:${(2 * n + 1).toString(16).padStart(64, "0")}`;
    fetchMock = vi.fn();
    globalThis.fetch = fetchMock as unknown as typeof fetch;
  });
  afterEach(() => {
    globalThis.fetch = originalFetch;
    delete process.env.LLMPULSE_OFFLINE;
  });
  afterAll(() => rmSync(home.dir, { recursive: true, force: true }));

  const urls = () => fetchMock.mock.calls.map(([u]) => String(u));

  it("returns exact sizes, file type and GGUF-derived arch facts", async () => {
    serve();
    const facts = await fetchRegistryFacts(tag);
    expect(facts).toEqual({
      weightsBytes: 4_920_000_000,
      fileType: "Q4_K_M",
      arch: { architecture: "llama", layers: 32, headCountKv: 8, keyLength: 128, valueLength: 128, contextLength: 8192, vocabSize: 128256 },
    });
  });

  it("builds registry URLs, asks for the Docker manifest, ranges the blob and never auto-follows redirects", async () => {
    serve();
    const [slug, version] = tag.split(":");
    await fetchRegistryFacts(tag);
    expect(urls()).toEqual([
      `${REGISTRY}/${slug}/manifests/${version}`,
      `${REGISTRY}/${slug}/blobs/${encodeURIComponent(CONFIG)}`,
      `${REGISTRY}/${slug}/blobs/${encodeURIComponent(MODEL)}`,
    ]);
    const [, manifestInit] = fetchMock.mock.calls[0];
    expect(new Headers(manifestInit.headers).get("accept")).toBe(
      "application/vnd.ollama.manifest.list.v2+json, application/vnd.docker.distribution.manifest.v2+json",
    );
    const [, blobInit] = fetchMock.mock.calls[2];
    expect(new Headers(blobInit.headers).get("range")).toBe("bytes=0-262143");
    for (const [, init] of fetchMock.mock.calls) {
      expect(init.redirect).toBe("manual");
      expect(init.signal).toBeInstanceOf(AbortSignal);
    }
  });

  it("defaults a bare slug to the latest tag", async () => {
    serve();
    await fetchRegistryFacts(`bare${n}`);
    expect(urls()[0]).toBe(`${REGISTRY}/bare${n}/manifests/latest`);
  });

  it("includes the projector layer size when the manifest has one", async () => {
    serve({ manifest: () => json(manifestJson({ projector: true })) });
    expect((await fetchRegistryFacts(tag))?.projectorBytes).toBe(851_000_000);
  });

  it("sums the draft layer into draftBytes", async () => {
    serve({ manifest: () => json(manifestJson({ draft: 288_800_000 })) });
    expect((await fetchRegistryFacts(tag))?.draftBytes).toBe(288_800_000);
  });

  describe("manifest lists", () => {
    const MLX = digest("7");
    const LLAMACPP = digest("8");
    const list = JSON.stringify({
      schemaVersion: 2,
      mediaType: "application/vnd.ollama.manifest.list.v2+json",
      manifests: [
        { mediaType: "application/vnd.docker.distribution.manifest.v2+json", digest: MLX, runner: "mlx", format: "safetensors" },
        { mediaType: "application/vnd.docker.distribution.manifest.v2+json", digest: LLAMACPP, runner: "llamacpp", format: "gguf" },
      ],
    });
    const tensorChild = JSON.stringify({
      schemaVersion: 2,
      config: { mediaType: "application/vnd.docker.container.image.v1+json", digest: CONFIG, size: 400 },
      layers: [{ mediaType: "application/vnd.ollama.image.tensor", digest: digest("9"), size: 1_680_000_000 }],
    });
    const originalPlatform = Object.getOwnPropertyDescriptor(process, "platform")!;
    const originalArch = Object.getOwnPropertyDescriptor(process, "arch")!;
    const runOn = (platform: string, arch: string) => {
      Object.defineProperty(process, "platform", { value: platform });
      Object.defineProperty(process, "arch", { value: arch });
    };
    afterEach(() => {
      Object.defineProperty(process, "platform", originalPlatform);
      Object.defineProperty(process, "arch", originalArch);
    });

    // O manifest/manifest.go:853-860: darwin+arm64 prefers mlx, every other platform llamacpp.
    it("takes the llamacpp child off Apple silicon and fetches it by digest", async () => {
      runOn("linux", "x64");
      serve({ manifest: () => json(list), children: { [MLX]: tensorChild, [LLAMACPP]: manifestJson() } });
      expect((await fetchRegistryFacts(tag))?.weightsBytes).toBe(4_920_000_000);
      const [slug] = tag.split(":");
      expect(urls()).toContain(`${REGISTRY}/${slug}/blobs/${encodeURIComponent(LLAMACPP)}`);
      expect(urls()).not.toContain(`${REGISTRY}/${slug}/blobs/${encodeURIComponent(MLX)}`);
    });

    it("takes the mlx child on darwin+arm64, which holds no GGUF model layer, so the estimate stays", async () => {
      runOn("darwin", "arm64");
      serve({ manifest: () => json(list), children: { [MLX]: tensorChild, [LLAMACPP]: manifestJson() } });
      expect(await fetchRegistryFacts(tag)).toBeNull();
      expect(urls().some((u) => u.includes(encodeURIComponent(LLAMACPP)))).toBe(false);
    });

    it("keeps one manifest cache entry per platform", async () => {
      serve({ manifest: () => json(list), children: { [MLX]: tensorChild, [LLAMACPP]: manifestJson() } });
      runOn("linux", "x64");
      expect(await fetchRegistryFacts(tag)).not.toBeNull();
      runOn("darwin", "arm64");
      expect(await fetchRegistryFacts(tag)).toBeNull();
    });

    it("returns null when the list has no child for any runner", async () => {
      serve({ manifest: () => json(JSON.stringify({ manifests: [{ digest: MLX, runner: "other" }] })) });
      expect(await fetchRegistryFacts(tag)).toBeNull();
    });
  });

  it("follows a 307 from the registry to r2 and keeps the Range header", async () => {
    serve({
      blob: (url, init) => url.startsWith("https://abc123.r2")
        ? (expect(new Headers(init.headers).get("range")).toBe("bytes=0-262143"), new Response(ggufBlob(), { status: 206 }))
        : redirect(R2_URL),
    });
    expect((await fetchRegistryFacts(tag))?.arch.layers).toBe(32);
    expect(urls().at(-1)).toBe(R2_URL);
  });

  it("refuses redirects to other hosts, plain http, other ports and lookalike domains", async () => {
    for (const target of [
      "https://evil.example.com/blob",
      "http://abc123.r2.cloudflarestorage.com/blob",
      "https://abc123.r2.cloudflarestorage.com.evil.example/blob",
      "https://abc123.r2.cloudflarestorage.com:8443/blob",
      "https://r2.cloudflarestorage.com/blob",
      "https://169.254.169.254/latest",
    ]) {
      fetchMock.mockClear();
      serve({ blob: () => redirect(target) });
      expect(await fetchRegistryFacts(`${tag}-${target.length}`), target).toBeNull();
      expect(urls().some((u) => u.startsWith(target)), target).toBe(false);
    }
  });

  it("refuses a manifest redirected to r2 and a redirect without a location", async () => {
    serve({ manifest: () => redirect(R2_URL) });
    expect(await fetchRegistryFacts(tag)).toBeNull();
    expect(urls()).not.toContain(R2_URL);
    serve({ blob: () => new Response(null, { status: 307 }) });
    expect(await fetchRegistryFacts(`${tag}x`)).toBeNull();
  });

  it("gives up on a redirect loop after the initial request and 3 redirects", async () => {
    fetchMock.mockImplementation(async (input: string | URL) => {
      const url = String(input);
      if (url.includes("/manifests/")) return json(manifestJson());
      if (url.includes(`/blobs/${encodeURIComponent(CONFIG)}`)) return json(CONFIG_JSON);
      return redirect("https://registry.ollama.ai/loop");
    });
    expect(await fetchRegistryFacts(tag)).toBeNull();
    expect(fetchMock).toHaveBeenCalledTimes(2 + 4);
  });

  it("refuses a redirect whose Location carries credentials", async () => {
    const target = "https://user:pass@registry.ollama.ai/blob";
    serve({ blob: () => redirect(target) });
    expect(await fetchRegistryFacts(tag)).toBeNull();
    expect(urls().some((u) => u.includes("user:pass"))).toBe(false);
  });

  it("shares one 5 s budget: every request carries the same abort signal", async () => {
    serve();
    await fetchRegistryFacts(tag);
    const signals = fetchMock.mock.calls.map(([, init]) => init.signal);
    expect(signals).toHaveLength(3);
    expect(new Set(signals).size).toBe(1);
  });

  // Overwrites the facts cached for the current MODEL and CONFIG digests.
  async function cacheFacts(data: unknown): Promise<void> {
    const { writeCache } = await import("../../src/core/cache.js");
    const { createHash } = await import("node:crypto");
    const key = `ollama-facts-${createHash("sha256").update([MODEL, CONFIG].join("\n")).digest("hex").slice(0, 32)}`;
    writeCache(key, data, { ttlMs: 60_000 });
  }

  it("ignores cached facts that exceed the producer's bounds", async () => {
    serve();
    const first = await fetchRegistryFacts(tag);
    await cacheFacts({ ...first, arch: { ...first!.arch, layers: 2 ** 40 } });
    expect(await fetchRegistryFacts(tag)).toEqual(first);
  });

  // The file type becomes a quant name on screen, so a cached one must pass the config blob's own check.
  it("ignores a cached file type the config blob schema would refuse", async () => {
    serve();
    const first = await fetchRegistryFacts(tag);
    await cacheFacts({ ...first, fileType: "\u001b[2JQ4_K_M" });
    expect(await fetchRegistryFacts(tag)).toEqual(first);
  });

  it("keeps inexactReason through the cache, so a cached hybrid or draft model is not exact again", async () => {
    serve();
    const first = await fetchRegistryFacts(tag);
    const inexact = { ...first, arch: { ...first!.arch, inexactReason: "parallel hybrid: SSM state is not modelled" } };
    await cacheFacts(inexact);
    fetchMock.mockClear();
    expect(await fetchRegistryFacts(tag)).toEqual(inexact);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("still reads facts cached with the former top-level contextLength and vocabSize", async () => {
    serve();
    const first = await fetchRegistryFacts(tag);
    await cacheFacts({ ...first, contextLength: 8192, vocabSize: 128256 });
    fetchMock.mockClear();
    expect(await fetchRegistryFacts(tag)).toEqual(first);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("cancels the body of a non-2xx response so the socket is released", async () => {
    const cancel = vi.fn(async () => undefined);
    serve({ manifest: () => ({ ok: false, status: 404, body: { cancel }, headers: new Headers() } as unknown as Response) });
    expect(await fetchRegistryFacts(tag)).toBeNull();
    expect(cancel).toHaveBeenCalled();
  });

  it("stops reading at 256 KiB when the server ignores Range and answers 200", async () => {
    const header = ggufBlob();
    const chunk = 64 * 1024;
    let enqueued = 0;
    const stream = new ReadableStream<Uint8Array>({
      pull(controller) {
        // First chunk carries the real header; the "file" then continues for 16 MiB.
        controller.enqueue(enqueued === 0 ? concat(header, Buffer.alloc(chunk - header.length)) : new Uint8Array(chunk));
        enqueued += chunk;
        if (enqueued >= 16 * 1024 * 1024) controller.close();
      },
    });
    serve({ blob: () => new Response(stream, { status: 200 }) });
    expect((await fetchRegistryFacts(tag))?.arch.architecture).toBe("llama");
    expect(enqueued).toBeLessThanOrEqual(256 * 1024 + 2 * chunk);
  });

  it("returns null for malformed or oversized JSON bodies", async () => {
    serve({ manifest: () => json("{not json") });
    expect(await fetchRegistryFacts(`${tag}a`)).toBeNull();
    serve({ manifest: () => json(JSON.stringify({ config: { digest: "sha256:short" }, layers: [] })) });
    expect(await fetchRegistryFacts(`${tag}b`)).toBeNull();
    serve({ manifest: () => json(manifestJson() + " ".repeat(64 * 1024)) });
    expect(await fetchRegistryFacts(`${tag}c`)).toBeNull();
    serve({ config: () => json("{not json") });
    expect(await fetchRegistryFacts(`${tag}d`)).toBeNull();
    serve({ config: () => json(JSON.stringify({ file_type: "Q4_K_M", pad: "x".repeat(64 * 1024) })) });
    expect(await fetchRegistryFacts(`${tag}e`)).toBeNull();
    serve({ config: () => json(JSON.stringify({ file_type: "../../etc" })) });
    expect(await fetchRegistryFacts(`${tag}f`)).toBeNull();
  });

  it("returns null when the manifest has no model layer, the blob is not GGUF or ends before its arch keys, or the server errors", async () => {
    serve({ manifest: () => json(manifestJson({ model: false })) });
    expect(await fetchRegistryFacts(`${tag}a`)).toBeNull();
    serve({ blob: () => new Response(enc.encode("not a gguf file at all, sorry"), { status: 206 }) });
    expect(await fetchRegistryFacts(`${tag}b`)).toBeNull();
    serve({ blob: () => new Response(ggufBlob().subarray(0, 64), { status: 206 }) });
    expect(await fetchRegistryFacts(`${tag}c`)).toBeNull();
    serve({ manifest: () => new Response("nope", { status: 404 }) });
    expect(await fetchRegistryFacts(`${tag}d`)).toBeNull();
    serve({ blob: () => new Response("nope", { status: 416 }) });
    expect(await fetchRegistryFacts(`${tag}e`)).toBeNull();
  });

  // mllama (llama3.2-vision) is no llama.cpp architecture, and a header can lack required keys: the file's size and type
  // still hold, and caching them by digest spares every later run the config and GGUF reads.
  it("keeps the size and type of a GGUF without llama.cpp arch facts, and caches them", async () => {
    for (const [i, blob] of [ggufBlob({ arch: "mllama" }), ggufBlob({ omitBlockCount: true })].entries()) {
      serve({ blob: () => new Response(blob, { status: 206 }) });
      fetchMock.mockClear();
      expect(await fetchRegistryFacts(`${tag}${i}`)).toEqual({ weightsBytes: 4_920_000_000, fileType: "Q4_K_M" });
      expect(fetchMock).toHaveBeenCalledTimes(3);
      expect(await fetchRegistryFacts(`${tag}${i}`)).toEqual({ weightsBytes: 4_920_000_000, fileType: "Q4_K_M" });
      expect(fetchMock).toHaveBeenCalledTimes(3);
      MODEL = digest(String(i + 1)); // the second blob gets its own facts cache key
    }
  });

  it("returns null when the network fails", async () => {
    fetchMock.mockRejectedValue(new Error("ENOTFOUND"));
    expect(await fetchRegistryFacts(tag)).toBeNull();
  });

  it("rejects slugs and tags that are not plain registry names, without fetching", async () => {
    for (const bad of ["", "../x:1", "a b:1", "a:b:c", "a/b:1", "Upper:1", `${"x".repeat(65)}:1`, "m:" + "t".repeat(129), "m:-bad", "m:%2e", "m:1?x=1", "m:1#f"]) {
      expect(await fetchRegistryFacts(bad), bad).toBeNull();
    }
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("makes no request when offline", async () => {
    process.env.LLMPULSE_OFFLINE = "1";
    serve();
    expect(await fetchRegistryFacts(tag)).toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("caches the manifest by tag and the facts by digest", async () => {
    serve();
    const first = await fetchRegistryFacts(tag);
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(await fetchRegistryFacts(tag)).toEqual(first);
    expect(fetchMock).toHaveBeenCalledTimes(3);

    // A different tag pointing at the same digests refetches only its manifest.
    expect(await fetchRegistryFacts(`${tag}-alias`)).toEqual(first);
    expect(fetchMock).toHaveBeenCalledTimes(4);
  });

  it("keys the facts by the draft digests too: another draft layer under the same model and config refetches", async () => {
    serve({ manifest: () => json(manifestJson({ draft: 288_800_000 })) });
    await fetchRegistryFacts(tag);
    expect(fetchMock).toHaveBeenCalledTimes(3); // manifest, config, blob
    serve({ manifest: () => json(manifestJson({ draft: 300_000_000, draftDigest: digest("f") })) });
    expect((await fetchRegistryFacts(`${tag}-b`))?.draftBytes).toBe(300_000_000);
    expect(fetchMock).toHaveBeenCalledTimes(6); // manifest, config and blob again: the cached facts are not reused
  });

  it("does not cache failures", async () => {
    serve({ blob: () => new Response("nope", { status: 500 }) });
    expect(await fetchRegistryFacts(tag)).toBeNull();
    serve();
    expect(await fetchRegistryFacts(tag)).not.toBeNull();
  });
});
