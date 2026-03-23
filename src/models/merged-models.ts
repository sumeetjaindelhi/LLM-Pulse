import { getAllModels } from "./database.js";
import { fetchOllamaModels } from "./ollama-models.js";
import type { MergedModel, OllamaModel } from "../core/types.js";

export async function getMergedModels(ollamaHost?: string): Promise<MergedModel[]> {
  const [dbModels, ollamaModels] = await Promise.all([
    Promise.resolve(getAllModels()),
    fetchOllamaModels(ollamaHost),
  ]);

  // Index Ollama models by name for fast lookup
  const ollamaByName = new Map<string, OllamaModel>();
  for (const om of ollamaModels) {
    ollamaByName.set(om.name, om);
  }

  const merged: MergedModel[] = [];
  const matchedOllamaTags = new Set<string>();

  // Match DB models against Ollama
  for (const entry of dbModels) {
    const tag = entry.ollamaTag;
    const ollamaModel = tag ? ollamaByName.get(tag) ?? null : null;
    const installed = ollamaModel !== null;

    if (tag && installed) matchedOllamaTags.add(tag);

    merged.push({
      entry,
      ollamaModel,
      installed,
      ollamaTag: tag,
    });
  }

  // Add Ollama-only models (not in DB)
  for (const om of ollamaModels) {
    if (!matchedOllamaTags.has(om.name)) {
      merged.push({
        entry: null,
        ollamaModel: om,
        installed: true,
        ollamaTag: om.name,
      });
    }
  }

  // Sort: installed first, then by name
  merged.sort((a, b) => {
    if (a.installed !== b.installed) return a.installed ? -1 : 1;
    const nameA = a.entry?.name ?? a.ollamaModel?.name ?? "";
    const nameB = b.entry?.name ?? b.ollamaModel?.name ?? "";
    return nameA.localeCompare(nameB);
  });

  return merged;
}
