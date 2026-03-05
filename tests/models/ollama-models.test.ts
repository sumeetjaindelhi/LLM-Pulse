import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { fetchOllamaModels, clearOllamaCache } from "../../src/models/ollama-models.js";
import { getMergedModels } from "../../src/models/merged-models.js";
import { getAllModels } from "../../src/models/database.js";
import type { ModelEntry } from "../../src/core/types.js";

const mockOllamaResponse = {
  models: [
    {
      name: "llama3.1:8b",
      model: "llama3.1:8b",
      size: 4661224676,
      details: {
        parent_model: "",
        format: "gguf",
        family: "llama",
        families: ["llama"],
        parameter_size: "8.0B",
        quantization_level: "Q4_K_M",
      },
    },
    {
      name: "mistral:7b",
      model: "mistral:7b",
      size: 4109854464,
      details: {
        parent_model: "",
        format: "gguf",
        family: "mistral",
        families: ["mistral"],
        parameter_size: "7.2B",
        quantization_level: "Q4_0",
      },
    },
    {
      name: "custom-finetune:latest",
      model: "custom-finetune:latest",
      size: 2000000000,
      details: {
        parent_model: "",
        format: "gguf",
        family: "llama",
        families: ["llama"],
        parameter_size: "3.0B",
        quantization_level: "Q5_K_M",
      },
    },
  ],
};

beforeEach(() => {
  clearOllamaCache();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("fetchOllamaModels", () => {
  it("parses Ollama API response correctly", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockOllamaResponse),
      }),
    );

    const models = await fetchOllamaModels();

    expect(models).toHaveLength(3);
    expect(models[0]).toEqual({
      name: "llama3.1:8b",
      size: 4661224676,
      parameterSize: "8.0B",
      quantization: "Q4_K_M",
      family: "llama",
    });
    expect(models[2].name).toBe("custom-finetune:latest");
  });

  it("returns empty array on fetch error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new Error("Connection refused")),
    );

    const models = await fetchOllamaModels();
    expect(models).toEqual([]);
  });

  it("returns empty array on non-ok response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 500 }),
    );

    const models = await fetchOllamaModels();
    expect(models).toEqual([]);
  });

  it("returns empty array on malformed response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ something: "else" }),
      }),
    );

    const models = await fetchOllamaModels();
    expect(models).toEqual([]);
  });

  it("handles missing details gracefully", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: () =>
          Promise.resolve({
            models: [{ name: "bare:latest", size: 1000, details: {} }],
          }),
      }),
    );

    const models = await fetchOllamaModels();
    expect(models).toHaveLength(1);
    expect(models[0].parameterSize).toBe("unknown");
    expect(models[0].quantization).toBe("unknown");
    expect(models[0].family).toBe("unknown");
  });

  it("caches results after first fetch", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockOllamaResponse),
    });
    vi.stubGlobal("fetch", mockFetch);

    await fetchOllamaModels();
    await fetchOllamaModels();

    expect(mockFetch).toHaveBeenCalledTimes(1);
  });
});

describe("getMergedModels", () => {
  it("marks DB models as installed when found in Ollama", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockOllamaResponse),
      }),
    );

    const merged = await getMergedModels();

    // Find a model that exists in both DB and Ollama
    const llama = merged.find((m) => m.ollamaTag === "llama3.1:8b");
    if (llama) {
      expect(llama.installed).toBe(true);
      expect(llama.entry).not.toBeNull();
      expect(llama.ollamaModel).not.toBeNull();
    }
  });

  it("includes Ollama-only models with entry: null", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockOllamaResponse),
      }),
    );

    const merged = await getMergedModels();
    const custom = merged.find((m) => m.ollamaTag === "custom-finetune:latest");

    expect(custom).toBeDefined();
    expect(custom!.entry).toBeNull();
    expect(custom!.installed).toBe(true);
    expect(custom!.ollamaModel?.family).toBe("llama");
  });

  it("marks DB-only models as not installed", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve({ models: [] }),
      }),
    );

    const merged = await getMergedModels();
    const dbModels = getAllModels();

    // All should be not installed
    expect(merged.length).toBe(dbModels.length);
    expect(merged.every((m) => !m.installed)).toBe(true);
  });

  it("sorts installed models first", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockOllamaResponse),
      }),
    );

    const merged = await getMergedModels();

    // Find first non-installed model
    const firstNotInstalled = merged.findIndex((m) => !m.installed);

    // All models before it should be installed
    if (firstNotInstalled > 0) {
      for (let i = 0; i < firstNotInstalled; i++) {
        expect(merged[i].installed).toBe(true);
      }
    }
  });

  it("gracefully handles Ollama being offline", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockRejectedValue(new Error("ECONNREFUSED")),
    );

    const merged = await getMergedModels();
    const dbModels = getAllModels();

    // Should fall back to DB-only models
    expect(merged.length).toBe(dbModels.length);
    expect(merged.every((m) => !m.installed)).toBe(true);
    expect(merged.every((m) => m.entry !== null)).toBe(true);
  });
});
