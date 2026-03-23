import { OllamaPickModelSchema } from "../../core/api-schemas.js";

/** Pick the smallest installed Ollama model (for quick benchmarks / profiling). */
export async function pickOllamaModel(baseUrl: string): Promise<string | null> {
  try {
    const res = await fetch(`${baseUrl}/api/tags`, {
      signal: AbortSignal.timeout(3000),
    });
    if (!res.ok) return null;
    const data = OllamaPickModelSchema.parse(await res.json());
    if (data.models.length === 0) return null;

    // Pick smallest model for quick operations
    const sorted = [...data.models].sort((a, b) => a.size - b.size);
    return sorted[0].name;
  } catch {
    return null;
  }
}
