import { OLLAMA_API_URL } from "../core/constants.js";
import type { OllamaModel } from "../core/types.js";

interface OllamaTagsResponse {
  models: Array<{
    name: string;
    size: number;
    details: {
      parameter_size: string;
      quantization_level: string;
      family: string;
    };
  }>;
}

let cachedModels: OllamaModel[] | null = null;

export async function fetchOllamaModels(): Promise<OllamaModel[]> {
  if (cachedModels) return cachedModels;

  try {
    const response = await fetch(`${OLLAMA_API_URL}/api/tags`, {
      signal: AbortSignal.timeout(3000),
    });

    if (!response.ok) return [];

    const data = (await response.json()) as OllamaTagsResponse;

    if (!data.models || !Array.isArray(data.models)) return [];

    cachedModels = data.models.map((m) => ({
      name: m.name,
      size: m.size,
      parameterSize: m.details?.parameter_size ?? "unknown",
      quantization: m.details?.quantization_level ?? "unknown",
      family: m.details?.family ?? "unknown",
    }));

    return cachedModels;
  } catch {
    return [];
  }
}

export function clearOllamaCache(): void {
  cachedModels = null;
}
