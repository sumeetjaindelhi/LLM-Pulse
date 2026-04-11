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

function findDataFile(): string | null {
  // Walk up from current file to find data/models.json
  let dir = __dirname;
  for (let i = 0; i < 5; i++) {
    const candidate = join(dir, "data/models.json");
    if (existsSync(candidate)) return candidate;
    dir = resolve(dir, "..");
  }
  return null;
}

/** Fatal exit helper — writes a clear user-facing message and exits with code 2.
 *
 * Why exit instead of return empty array: `loadModels()` is on the hot path of
 * every command (scan, doctor, models, check, compare, recommend). Returning
 * an empty array would cascade into "no models fit" warnings everywhere and
 * confuse the user far more than a single explicit error. A missing/corrupt
 * data file is an unrecoverable packaging error — fail fast, fail loud.
 */
function fatalLoadError(reason: string): never {
  process.stderr.write(`\nError: Could not load model database.\n`);
  process.stderr.write(`Reason: ${reason}\n`);
  process.stderr.write(`Fix: reinstall with \`npm install -g llm-pulse\`, or report at\n`);
  process.stderr.write(`     https://github.com/sumeetjaindelhi/LLM-Pulse/issues\n\n`);
  process.exit(2);
}

function loadModels(): ModelEntry[] {
  if (cachedModels) return cachedModels;

  const dataPath = findDataFile();
  if (!dataPath) {
    fatalLoadError("could not locate data/models.json relative to installed package");
  }

  let raw: unknown;
  try {
    const contents = readFileSync(dataPath, "utf-8");
    raw = JSON.parse(contents);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    fatalLoadError(`failed to read or parse JSON at ${dataPath} — ${msg}`);
  }

  const result = ModelDatabaseSchema.safeParse(raw);
  if (!result.success) {
    const firstIssue = result.error.issues[0];
    const path = firstIssue.path.join(".") || "<root>";
    fatalLoadError(`schema mismatch in ${dataPath} at "${path}": ${firstIssue.message}`);
  }

  cachedModels = result.data;
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
