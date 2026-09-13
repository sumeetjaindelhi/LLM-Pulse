import { describe, it, expect } from "vitest";
import { ModelDatabaseSchema, ModelEntrySchema } from "../../src/models/schema.js";
import curatedModels from "../../data/models.json";

const baseEntry = {
  id: "test-model",
  name: "Test Model",
  provider: "Test",
  parametersBillion: 3.8,
  contextWindow: 128000,
  categories: ["general"],
  qualityTier: "good",
  qualityScore: 68,
  quantizations: [{ name: "Q4_K_M", bitsPerWeight: 4.83, vramMb: 2500, qualityRetention: 0.92 }],
  ollamaTag: "test:3.8b",
  releaseDate: "2024-04",
};

describe("ModelEntrySchema kvMbPer1kTokens", () => {
  it("is optional", () => {
    expect(ModelEntrySchema.safeParse(baseEntry).success).toBe(true);
  });

  it("accepts a positive number", () => {
    expect(ModelEntrySchema.safeParse({ ...baseEntry, kvMbPer1kTokens: 375 }).success).toBe(true);
  });

  it("rejects zero, negative, and non-numeric values", () => {
    expect(ModelEntrySchema.safeParse({ ...baseEntry, kvMbPer1kTokens: 0 }).success).toBe(false);
    expect(ModelEntrySchema.safeParse({ ...baseEntry, kvMbPer1kTokens: -1 }).success).toBe(false);
    expect(ModelEntrySchema.safeParse({ ...baseEntry, kvMbPer1kTokens: "375" }).success).toBe(false);
  });

  it("validates the curated database", () => {
    expect(ModelDatabaseSchema.safeParse(curatedModels).success).toBe(true);
  });
});

describe("curated ollamaTag values", () => {
  const tagOf = (id: string) => curatedModels.find((m) => m.id === id)?.ollamaTag;

  it("uses tags that exist in the Ollama library", () => {
    expect(tagOf("llama-4-scout")).toBe("llama4:scout");
    expect(tagOf("command-r-7b")).toBe("command-r7b");
  });
});
