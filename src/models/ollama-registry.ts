import { createHash } from "node:crypto";
import { z } from "zod";
import { OllamaConfigBlobSchema, OllamaManifestListSchema, OllamaManifestSchema, RegistryFactsSchema } from "../core/api-schemas.js";
import { readCache, writeCache } from "../core/cache.js";
import { isOffline } from "../core/config.js";
import type { ArchFacts } from "../core/types.js";
import { archFactsFromMetadata } from "./arch-facts.js";
import { GGUF_MAX_INPUT_BYTES, parseGgufMetadata } from "./gguf.js";

/** Exact facts about an Ollama library model, read from registry.ollama.ai without downloading the weights. */
export interface RegistryFacts {
  weightsBytes: number;
  projectorBytes?: number;
  /** Speculative-decoding draft model; Ollama loads it by default (O server/images.go:830-835, routes.go:168-170). */
  draftBytes?: number;
  /** Quantization name from the config blob, e.g. "Q4_K_M". */
  fileType: string;
  /** Absent when the GGUF header holds no llama.cpp architecture facts (mllama, a missing required key). */
  arch?: ArchFacts;
}

const REGISTRY_HOST = "registry.ollama.ai";
const CDN_HOST_SUFFIX = ".r2.cloudflarestorage.com"; // where the registry redirects blob downloads
const REGISTRY_BASE = `https://${REGISTRY_HOST}/v2/library`;
const MANIFEST_MEDIA_TYPE = "application/vnd.docker.distribution.manifest.v2+json"; // O manifest/manifest.go:32
const MANIFEST_ACCEPT = `application/vnd.ollama.manifest.list.v2+json, ${MANIFEST_MEDIA_TYPE}`; // O server/images.go:1659
const MODEL_LAYER = "application/vnd.ollama.image.model"; // O manifest/layer.go:26
const PROJECTOR_LAYER = "application/vnd.ollama.image.projector"; // O manifest/layer.go:27
const DRAFT_LAYER = "application/vnd.ollama.image.draft"; // O manifest/layer.go:28

/** Runners Ollama tries in order when it pulls a manifest list without --runner (O manifest/manifest.go:853-860). */
function runnerPreferences(): string[] {
  return process.platform === "darwin" && process.arch === "arm64" ? ["mlx", "llamacpp", "ggml"] : ["llamacpp", "ggml", "mlx"];
}

// Definition: names the library actually uses; anything else never reaches a URL.
const SLUG_RE = /^[a-z0-9][a-z0-9._-]{0,63}$/;
const TAG_RE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;

const TIMEOUT_MS = 5000; // definition: total budget for one lookup, shared by every request
const MAX_JSON_BYTES = 64 * 1024; // definition: manifests and config blobs are well under 4 KiB in practice
const MAX_REDIRECTS = 3; // definition: the registry needs one hop to the CDN
const MANIFEST_TTL_MS = 24 * 60 * 60 * 1000; // definition: a tag can be re-pushed, so manifests are re-read daily
const FACTS_TTL_MS = 365 * 24 * 60 * 60 * 1000; // definition: facts are immutable per digest; this only bounds disk staleness

const cacheKey = (prefix: string, ...parts: string[]): string =>
  `${prefix}-${createHash("sha256").update(parts.join("\n")).digest("hex").slice(0, 32)}`;

function isTrustedUrl(url: URL, allowCdn: boolean): boolean {
  if (url.protocol !== "https:" || url.port !== "" || url.username !== "" || url.password !== "") return false;
  return url.hostname === REGISTRY_HOST || (allowCdn && url.hostname.endsWith(CDN_HOST_SUFFIX));
}

/** GET that follows redirects by hand, only to trusted hosts. null on a refused redirect, loop or non-2xx status. */
async function trustedGet(url: string, headers: Record<string, string>, allowCdn: boolean, signal: AbortSignal): Promise<Response | null> {
  let current = new URL(url);
  for (let hop = 0; hop <= MAX_REDIRECTS; hop++) {
    if (!isTrustedUrl(current, allowCdn)) return null;
    const response = await fetch(current.href, { headers, signal, redirect: "manual" });
    if (response.status < 300 || response.status >= 400) {
      if (response.ok) return response;
      await response.body?.cancel(); // an unread body keeps the socket, and the process, alive until the timeout
      return null;
    }
    await response.body?.cancel();
    const location = response.headers.get("location");
    if (!location) return null;
    current = new URL(location, current);
  }
  return null;
}

/** Reads at most `maxBytes` of the body and cancels the stream beyond that, whatever the server sends. */
async function readCapped(response: Response, maxBytes: number): Promise<{ bytes: Uint8Array; overflow: boolean }> {
  const reader = response.body?.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  while (reader) {
    const { done, value } = await reader.read();
    if (done) break;
    if (total + value.length > maxBytes) {
      chunks.push(value.subarray(0, maxBytes - total));
      await reader.cancel();
      return { bytes: Buffer.concat(chunks), overflow: true };
    }
    chunks.push(value);
    total += value.length;
  }
  return { bytes: Buffer.concat(chunks), overflow: false };
}

async function getJson<S extends z.ZodType>(url: string, headers: Record<string, string>, allowCdn: boolean, schema: S, signal: AbortSignal): Promise<z.infer<S> | null> {
  const response = await trustedGet(url, headers, allowCdn, signal);
  if (!response) return null;
  const { bytes, overflow } = await readCapped(response, MAX_JSON_BYTES);
  if (overflow) return null;
  const parsed = schema.safeParse(JSON.parse(new TextDecoder().decode(bytes)));
  return parsed.success ? parsed.data : null;
}

function parseTag(tag: string): { slug: string; version: string } | null {
  const [slug, version = "latest", ...rest] = tag.split(":");
  return rest.length === 0 && SLUG_RE.test(slug) && TAG_RE.test(version) ? { slug, version } : null;
}

/**
 * Exact weights size, quantization and GGUF architecture facts for a library tag such as "llama3.1:8b".
 * Reads the manifest, the small config blob and the first 256 KiB of the model blob (the GGUF header).
 * null when offline, the tag is not a plain library name, or any step fails or returns malformed data.
 */
export async function fetchRegistryFacts(tag: string): Promise<RegistryFacts | null> {
  if (isOffline()) return null;
  const name = parseTag(tag);
  if (!name) return null;
  const { slug, version } = name;
  const signal = AbortSignal.timeout(TIMEOUT_MS);
  const blobUrl = (digest: string) => `${REGISTRY_BASE}/${encodeURIComponent(slug)}/blobs/${encodeURIComponent(digest)}`;

  try {
    // The child a list resolves to depends on the platform, so it is part of the key.
    const manifestKey = cacheKey("ollama-manifest", runnerPreferences()[0], slug, version);
    let manifest = readCache(manifestKey, OllamaManifestSchema, { ttlMs: MANIFEST_TTL_MS })?.data ?? null;
    if (!manifest) {
      const fetched = await getJson(
        `${REGISTRY_BASE}/${encodeURIComponent(slug)}/manifests/${encodeURIComponent(version)}`,
        { Accept: MANIFEST_ACCEPT },
        false,
        z.union([OllamaManifestSchema, OllamaManifestListSchema]),
        signal,
      );
      if (!fetched) return null;
      if ("manifests" in fetched) {
        const children = fetched.manifests.filter((c) => !c.mediaType || c.mediaType === MANIFEST_MEDIA_TYPE);
        const runner = runnerPreferences().find((r) => children.some((c) => c.runner?.trim().toLowerCase() === r));
        const child = children.find((c) => c.runner?.trim().toLowerCase() === runner);
        if (!child) return null;
        // A child is stored as a blob, not under a manifest name (O server/images.go pullSelectedManifest).
        manifest = await getJson(blobUrl(child.digest), {}, true, OllamaManifestSchema, signal);
      } else {
        manifest = fetched;
      }
      if (!manifest) return null;
      writeCache(manifestKey, manifest, { ttlMs: MANIFEST_TTL_MS });
    }

    const model = manifest.layers.find((l) => l.mediaType === MODEL_LAYER);
    if (!model || model.size === 0) return null;
    const projector = manifest.layers.find((l) => l.mediaType === PROJECTOR_LAYER);
    const drafts = manifest.layers.filter((l) => l.mediaType === DRAFT_LAYER);
    const draftBytes = drafts.reduce((sum, l) => sum + l.size, 0);

    const factsKey = cacheKey("ollama-facts", model.digest, manifest.config.digest, ...drafts.map((l) => l.digest));
    const cached = readCache(factsKey, RegistryFactsSchema, { ttlMs: FACTS_TTL_MS })?.data;
    if (cached) return cached;

    const config = await getJson(blobUrl(manifest.config.digest), {}, true, OllamaConfigBlobSchema, signal);
    if (!config) return null;

    const blob = await trustedGet(blobUrl(model.digest), { Range: `bytes=0-${GGUF_MAX_INPUT_BYTES - 1}` }, true, signal);
    if (!blob) return null;
    // A server that ignores Range sends the whole file; only the header is read.
    const header = parseGgufMetadata((await readCapped(blob, GGUF_MAX_INPUT_BYTES)).bytes);
    const arch = header && archFactsFromMetadata(header.kv, header.vocabSize);
    // A whole header without llama.cpp arch facts (mllama) is a fact of the file, so its size and type are kept and cached;
    // one cut short may only be a failed read.
    if (!header || (!arch && header.truncated)) return null;

    const facts: RegistryFacts = {
      weightsBytes: model.size,
      ...(projector && { projectorBytes: projector.size }),
      ...(draftBytes > 0 && { draftBytes }),
      fileType: config.file_type,
      ...(arch && { arch }),
    };
    writeCache(factsKey, facts, { ttlMs: FACTS_TTL_MS });
    return facts;
  } catch {
    return null;
  }
}
