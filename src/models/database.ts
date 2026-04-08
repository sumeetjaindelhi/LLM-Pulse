import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";
import { ModelDatabaseSchema } from "./schema.js";
import type { ModelEntry, ModelCategory } from "../core/types.js";

export function resolveModel(query: string): ModelEntry | null {
  // 1. Exact ID match
  const byId = getModelById(query);
  if (byId) return byId;

  // 2. Exact Ollama tag match
  const byTag = getModelByTag(query);
  if (byTag) return byTag;

  // 3. Search — only if exactly 1 result
  const results = searchModels(query);
  if (results.length === 1) return results[0];

  return null;
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

let cachedModels: ModelEntry[] | null = null;

function findDataFile(): string {
  // Walk up from current file to find data/models.json
  let dir = __dirname;
  for (let i = 0; i < 5; i++) {
    const candidate = join(dir, "data/models.json");
    if (existsSync(candidate)) return candidate;
    dir = resolve(dir, "..");
  }
  throw new Error("Could not find data/models.json");
}

function loadModels(): ModelEntry[] {
  if (cachedModels) return cachedModels;

  const dataPath = findDataFile();
  const raw = JSON.parse(readFileSync(dataPath, "utf-8"));
  cachedModels = ModelDatabaseSchema.parse(raw);
  return cachedModels;
}

export function getAllModels(): ModelEntry[] {
  return loadModels();
}

export function searchModels(query: string): ModelEntry[] {
  const q = query.toLowerCase();
  return loadModels().filter(
    (m) =>
      m.name.toLowerCase().includes(q) ||
      m.id.toLowerCase().includes(q) ||
      m.provider.toLowerCase().includes(q),
  );
}

export function filterByCategory(category: ModelCategory): ModelEntry[] {
  return loadModels().filter((m) => m.categories.includes(category));
}

export function getModelById(id: string): ModelEntry | undefined {
  return loadModels().find((m) => m.id === id);
}

export function getModelByTag(tag: string): ModelEntry | undefined {
  return loadModels().find((m) => m.ollamaTag === tag);
}
