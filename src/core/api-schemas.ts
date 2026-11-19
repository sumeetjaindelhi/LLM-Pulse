import { z } from "zod";

// Length-capped string (max 256 chars). Content is not checked.
const safeString = z.string().max(256);

/**
 * Zod validator for URLs that MUST point to localhost. Used to constrain the
 * `host` parameter on MCP tool calls — without this, a malicious/compromised
 * MCP client could instruct the server to fetch arbitrary URLs (including
 * cloud metadata endpoints like http://169.254.169.254 or internal services),
 * turning every MCP tool into an SSRF primitive.
 *
 * The CLI's --host flag is NOT constrained by this schema because the CLI's
 * trust boundary is the user themselves — they can curl internal URLs from
 * their own shell regardless. The MCP surface is where outside instructions
 * (from whatever client has connected) reach the fetch layer.
 */
export const LocalhostUrl = z.url().refine(
  (url) => {
    try {
      const u = new URL(url);
      const host = u.hostname;
      return (
        host === "localhost" ||
        host === "127.0.0.1" ||
        host === "::1" ||
        host === "[::1]"
      );
    } catch {
      return false;
    }
  },
  { error: "host must point to localhost (localhost, 127.0.0.1, or ::1)" },
);

// Ollama lists cloud-hosted models (`ollama run kimi-k3:cloud`) beside local ones and marks them with
// remote_host / remote_model. They run on ollama.com, not this machine, so they never count as installed.
const cloudMarkers = {
  remote_host: safeString.optional(),
  remote_model: safeString.optional(),
};

function dropCloudModels<T extends { remote_host?: string; remote_model?: string }>(models: T[]): T[] {
  return models.filter((m) => m.remote_host === undefined && m.remote_model === undefined);
}

// Ollama /api/version
export const OllamaVersionSchema = z.object({
  version: safeString,
});

// Ollama /api/tags
export const OllamaTagsSchema = z.object({
  models: z.array(z.object({
    name: safeString,
    size: z.number(),
    details: z.object({
      parameter_size: safeString,
      quantization_level: safeString,
      family: safeString,
    }).partial().default({}),
    ...cloudMarkers,
  })).default([]).transform(dropCloudModels),
});

// Ollama /api/ps
export const OllamaPsSchema = z.object({
  models: z.array(z.object({
    name: safeString,
    size: z.number().optional(),
    details: z.object({
      quantization_level: safeString.optional(),
      parameter_size: safeString.optional(),
    }).optional(),
    size_vram: z.number().optional(),
    context_length: z.number().optional(),
  })).default([]),
});

// Ollama /api/show — only model_info is read; its values are validated per key.
export const OllamaShowSchema = z.object({
  details: z.object({ format: z.string().optional() }).optional(),
  model_info: z.record(z.string(), z.unknown()),
});

// LM Studio /v1/models
export const LmStudioModelsSchema = z.object({
  data: z.array(z.object({
    id: safeString,
  })).default([]),
});

// Ollama benchmark streaming line
export const BenchmarkLineSchema = z.object({
  response: z.string().optional(),
  done: z.boolean().optional(),
});

// Ollama /api/tags (for pickModel in benchmark)
export const OllamaPickModelSchema = z.object({
  models: z.array(z.object({
    name: safeString,
    size: z.number(),
    ...cloudMarkers,
  })).default([]).transform(dropCloudModels),
});

// Ollama registry manifest (Docker distribution manifest v2). Digests end up in URLs and cache names, so they are strict.
const Sha256Digest = z.string().regex(/^sha256:[a-f0-9]{64}$/);
const MAX_BLOB_BYTES = 2 ** 50; // definition: ~1 PiB, far above any model file

export const OllamaManifestSchema = z.object({
  config: z.object({ digest: Sha256Digest }),
  layers: z.array(z.object({
    mediaType: safeString,
    size: z.number().int().nonnegative().max(MAX_BLOB_BYTES),
    digest: Sha256Digest,
  })).max(64),
});

// Ollama manifest list: children are bare references (digest and runner), fetched separately.
export const OllamaManifestListSchema = z.object({
  manifests: z.array(z.object({
    mediaType: safeString.optional(),
    digest: Sha256Digest,
    runner: safeString.optional(),
  })).max(16),
});

// A GGUF file type name such as "Q4_K_M", as the registry config blob and Ollama /api/tags give it; shown as a quant name.
export const FileType = z.string().regex(/^[A-Za-z0-9_.-]{1,32}$/);
// Ollama registry config blob: only the quantization name is read.
export const OllamaConfigBlobSchema = z.object({
  file_type: FileType,
});

// Definition: the producer's bounds (src/models/arch-facts.ts), so a tampered cache file cannot reach kvCacheBytes.
const Bounded = (min: number, max: number) => z.number().int().min(min).max(max);
const Layers = Bounded(1, 1024);
const Heads = Bounded(0, 4096);
const Dim = Bounded(1, 65536);
export const ArchFactsSchema = z.object({
  architecture: safeString,
  layers: Layers,
  headCountKv: z.union([Heads, z.array(Heads).max(1024)]),
  keyLength: Dim,
  valueLength: Dim,
  keyLengthSwa: Dim.optional(),
  valueLengthSwa: Dim.optional(),
  recurrentLayers: z.array(z.boolean()).max(1024).optional(),
  kvLayers: Layers.optional(),
  slidingWindow: Bounded(0, 2 ** 24).optional(),
  slidingWindowPattern: z.union([Bounded(0, 1024), z.array(z.boolean()).max(1024)]).optional(),
  mla: z.boolean().optional(),
  recurrentStateBytes: Bounded(0, MAX_BLOB_BYTES).optional(),
  vocabSize: Bounded(1, 2 ** 24).optional(),
  contextLength: Bounded(1, 2 ** 24).optional(),
  inexactReason: safeString.optional(),
}).refine(
  (a) => [a.headCountKv, a.recurrentLayers, a.slidingWindowPattern].every((v) => !Array.isArray(v) || v.length === a.layers),
  { error: "per-layer arrays must have one entry per layer" },
);

// Exact model facts from the registry (src/models/ollama-registry.ts), as cached on disk. Entries from older versions
// also carry top-level contextLength and vocabSize; z.object strips them, so those entries stay readable.
export const RegistryFactsSchema = z.object({
  weightsBytes: Bounded(0, MAX_BLOB_BYTES),
  projectorBytes: Bounded(0, MAX_BLOB_BYTES).optional(),
  draftBytes: Bounded(0, MAX_BLOB_BYTES).optional(),
  fileType: FileType,
  arch: ArchFactsSchema.optional(),
});
