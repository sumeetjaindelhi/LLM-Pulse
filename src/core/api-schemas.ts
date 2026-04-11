import { z } from "zod";

// Safe string: printable, reasonable length, no control chars
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
export const LocalhostUrl = z.string().url().refine(
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
  { message: "host must point to localhost (localhost, 127.0.0.1, or ::1)" },
);

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
  })).default([]),
});

// Ollama /api/ps
export const OllamaPsSchema = z.object({
  models: z.array(z.object({
    name: safeString,
    size: z.number().optional(),
    details: z.object({
      tokens_per_second: z.number().optional(),
      quantization_level: safeString.optional(),
      parameter_size: safeString.optional(),
    }).optional(),
    size_vram: z.number().optional(),
  })).default([]),
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
  })).default([]),
});
