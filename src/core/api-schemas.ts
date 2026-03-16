import { z } from "zod";

// Safe string: printable, reasonable length, no control chars
const safeString = z.string().max(256);

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
