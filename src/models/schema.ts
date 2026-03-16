import { z } from "zod";

export const QuantizationVariantSchema = z.object({
  name: z.string(),
  bitsPerWeight: z.number(),
  vramMb: z.number(),
  qualityRetention: z.number().min(0).max(1),
});

export const ModelEntrySchema = z.object({
  id: z.string(),
  name: z.string(),
  provider: z.string(),
  parametersBillion: z.number(),
  contextWindow: z.number(),
  categories: z.array(
    z.enum(["general", "coding", "reasoning", "creative", "multilingual"]),
  ),
  qualityTier: z.enum(["frontier", "strong", "good", "lightweight"]),
  qualityScore: z.number().min(0).max(100),
  quantizations: z.array(QuantizationVariantSchema).min(1),
  ollamaTag: z.string().regex(/^[a-zA-Z0-9:._\/-]+$/).nullable(),
  releaseDate: z.string(),
});

export const ModelDatabaseSchema = z.array(ModelEntrySchema);
