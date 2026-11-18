import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";
import { ModelDatabaseSchema } from "./schema.js";
import { readLibraryCatalogSync, readLibraryStatus, type LibrarySnapshot } from "./library-cache.js";
import { DEFAULT_QUANT_NAME, TERNARY_QUANT_NAME, mergeLiveCatalog } from "./library-entries.js";
import { quantFileBpw, quantRetention } from "./quant-facts.js";
import { isOffline } from "../core/config.js";
import type { CatalogHealth, ModelEntry, ModelCategory, QuantizationVariant } from "../core/types.js";

export function resolveModel(query: string): ModelEntry | null {
  // 1. Exact ID match
  const byId = getModelById(query);
  if (byId) return byId;

  // 2. Exact Ollama tag match
  const byTag = getModelByTag(query);
  if (byTag) return byTag;

  // 3. Search. A curated name must not become ambiguous because the library grew,
  // and a library row must not answer a bare curated slug ("gemma3" is not "gemma3:270m").
  const results = searchModels(query);
  const curated = results.filter((m) => m.source === "curated");
  if (curated.length === 1) return curated[0];
  const sole = curated.length === 0 && results.length === 1 ? results[0] : null;
  const soleSlug = sole?.ollamaTag?.split(":")[0];
  const curatedSharesSlug = getCuratedModels().some((m) => m.ollamaTag?.split(":")[0] === soleSlug);
  return sole && !curatedSharesSlug ? sole : null;
}

/** The quant called `name`, in any case. A library row's sole quant also answers to its placeholder name, which scan and
 *  models still list after exact facts renamed it to its file type. */
export function findQuant(model: ModelEntry, name: string): QuantizationVariant | undefined {
  const wanted = name.toLowerCase();
  const libraryFile = model.source === "library" && model.quantizations.length === 1;
  if (libraryFile && [DEFAULT_QUANT_NAME, TERNARY_QUANT_NAME].includes(wanted)) return model.quantizations[0];
  return model.quantizations.find((q) => q.name.toLowerCase() === wanted);
}

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

let cachedCurated: ModelEntry[] | null = null;
let cachedMerged: { cards: LibrarySnapshot["cards"]; rows: ModelEntry[] } | null = null;

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

// Curated quants take bits per weight, weights size and quality retention from the cited quant table
// (quant-facts.ts), not the hand-set data values; an exact quant keeps its measured size, and a name the
// table does not know keeps its data values.
function normalizeQuant(quant: QuantizationVariant, parametersBillion: number): QuantizationVariant {
  const fileBpw = quantFileBpw(quant.name);
  if (!fileBpw) return quant;
  return {
    ...quant,
    bitsPerWeight: fileBpw.bpw,
    vramMb: quant.exact ? quant.vramMb : Math.round((parametersBillion * 1e9 * fileBpw.bpw) / 8 / 2 ** 20),
    qualityRetention: quantRetention(quant.name).retention,
  };
}

function loadCuratedModels(): ModelEntry[] {
  if (cachedCurated) return cachedCurated;

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

  cachedCurated = result.data.map((m) => ({
    ...m,
    quantizations: m.quantizations.map((q) => normalizeQuant(q, m.parametersBillion)),
    source: "curated" as const,
  }));
  return cachedCurated;
}

// Bundled rows plus, unless offline, the rows synthesized from the cached
// ollama.com catalog. Never touches the network; refreshing is the caller's job.
function loadModels(): ModelEntry[] {
  const curated = loadCuratedModels();
  if (isOffline()) return curated;
  const snapshot = readLibraryCatalogSync();
  if (!snapshot) return curated;
  if (cachedMerged?.cards !== snapshot.cards) {
    cachedMerged = {
      cards: snapshot.cards,
      rows: mergeLiveCatalog(curated, snapshot.cards, new Date(snapshot.fetchedAt)),
    };
  }
  return cachedMerged.rows;
}

export function resetModelCache(): void {
  cachedCurated = null;
  cachedMerged = null;
}

/** Where the model list came from, for doctor and scan. Offline hides fetch errors: nothing was attempted. */
export function getCatalogHealth(): CatalogHealth {
  const offline = isOffline();
  const snapshot = offline ? null : readLibraryCatalogSync();
  const status = offline ? null : readLibraryStatus();
  return {
    source: snapshot ? "live" : "bundled",
    fetchedAt: snapshot?.fetchedAt ?? null,
    liveModels: snapshot ? loadModels().filter((m) => m.source === "library").length : 0,
    lastError: status?.lastError ?? null,
    lastSuccessAt: status?.lastSuccessAt ?? null,
    offline,
  };
}

/** The bundled data/models.json rows only, with hand-set scores. */
export function getCuratedModels(): ModelEntry[] {
  return loadCuratedModels();
}

export function getAllModels(): ModelEntry[] {
  return loadModels();
}

export function searchModels(query: string, category: ModelCategory | "all" = "all"): ModelEntry[] {
  const q = query.toLowerCase();
  return filterByCategory(category).filter(
    (m) =>
      m.name.toLowerCase().includes(q) ||
      m.id.toLowerCase().includes(q) ||
      (m.ollamaTag?.toLowerCase().includes(q) ?? false) ||
      m.provider.toLowerCase().includes(q),
  );
}

export function filterByCategory(category: ModelCategory | "all"): ModelEntry[] {
  if (category === "all") return loadModels();
  return loadModels().filter((m) => m.categories.includes(category));
}

export function getModelById(id: string): ModelEntry | undefined {
  return loadModels().find((m) => m.id === id);
}

export function getModelByTag(tag: string): ModelEntry | undefined {
  return loadModels().find((m) => m.ollamaTag === tag);
}
