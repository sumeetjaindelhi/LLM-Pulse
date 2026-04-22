import {
  fetchOllamaLibrary,
  LibraryCatalogSchema,
  type LibraryModel,
} from "./ollama-library.js";
import { readCache, writeCache, clearCacheEntry } from "../core/cache.js";
import type { LibraryCatalogModel } from "../core/types.js";

const CACHE_NAME = "ollama-library";
const CACHE_VERSION = 1;
const TTL_MS = 24 * 60 * 60 * 1000; // 24 hours

export interface GetLibraryOptions {
  refresh?: boolean;
}

let memo: LibraryModel[] | null = null;

function toCatalog(models: LibraryModel[]): LibraryCatalogModel[] {
  return models.map((m) => ({
    slug: m.slug,
    description: m.description,
    parameterSizes: m.parameterSizes,
    capabilities: m.capabilities,
  }));
}

export async function getLibraryCatalog(
  opts: GetLibraryOptions = {},
): Promise<LibraryCatalogModel[]> {
  const refresh = opts.refresh ?? false;

  if (refresh) {
    clearCacheEntry(CACHE_NAME);
    memo = null;
  }

  if (memo && !refresh) return toCatalog(memo);

  if (!refresh) {
    const cached = readCache(CACHE_NAME, LibraryCatalogSchema, {
      ttlMs: TTL_MS,
      version: CACHE_VERSION,
    });
    if (cached) {
      memo = cached.data;
      return toCatalog(memo);
    }
  }

  try {
    const fresh = await fetchOllamaLibrary();
    memo = fresh;
    writeCache(CACHE_NAME, fresh, { ttlMs: TTL_MS, version: CACHE_VERSION });
    return toCatalog(fresh);
  } catch {
    // Fetch failed. If we have a stale cache, fall back to it rather than
    // returning nothing — a day-old catalog beats no catalog for an offline
    // user. The TTL check only applies to fresh fetches.
    const stale = readCache(CACHE_NAME, LibraryCatalogSchema, {
      ttlMs: Number.MAX_SAFE_INTEGER,
      version: CACHE_VERSION,
    });
    if (stale) {
      memo = stale.data;
      return toCatalog(memo);
    }
    return [];
  }
}

export function resetLibraryCache(): void {
  memo = null;
  clearCacheEntry(CACHE_NAME);
}
