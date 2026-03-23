import { OLLAMA_API_URL } from "../core/constants.js";
import { OllamaTagsSchema } from "../core/api-schemas.js";
import type { OllamaModel } from "../core/types.js";

let cachedModels: OllamaModel[] | null = null;

export async function fetchOllamaModels(host?: string): Promise<OllamaModel[]> {
  const baseUrl = host || OLLAMA_API_URL;
  const isCustomHost = !!host && host !== OLLAMA_API_URL;

  // Bypass cache for non-default host
  if (!isCustomHost && cachedModels) return cachedModels;

  try {
    const response = await fetch(`${baseUrl}/api/tags`, {
      signal: AbortSignal.timeout(3000),
    });

    if (!response.ok) return [];

    const data = OllamaTagsSchema.parse(await response.json());

    const models = data.models.map((m) => ({
      name: m.name,
      size: m.size,
      parameterSize: m.details?.parameter_size ?? "unknown",
      quantization: m.details?.quantization_level ?? "unknown",
      family: m.details?.family ?? "unknown",
    }));

    if (!isCustomHost) cachedModels = models;
    return models;
  } catch {
    return [];
  }
}

export function clearOllamaCache(): void {
  cachedModels = null;
}
