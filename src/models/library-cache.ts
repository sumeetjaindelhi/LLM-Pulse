import { z } from "zod";
import {
  fetchOllamaLibrary,
  LibraryCatalogSchema,
  type LibraryModel,
} from "./ollama-library.js";
import { readCache, writeCache, clearCacheEntry } from "../core/cache.js";
import { isOffline } from "../core/config.js";
import type { LibraryCatalogModel } from "../core/types.js";

const CACHE_NAME = "ollama-library";
const STATUS_NAME = "ollama-library-status";
const CACHE_VERSION = 2;
const STATUS_VERSION = 1;
const TTL_MS = 24 * 60 * 60 * 1000; // 24 hours
const FAILURE_BACKOFF_MS = 60 * 60 * 1000; // implicit refreshes pause after a failure
const IMPLICIT_TIMEOUT_MS = 5_000;
const EXPLICIT_TIMEOUT_MS = 10_000;
// Disk entries are read at any age; the envelope's own expiry never applies.
const NO_EXPIRY_MS = Number.MAX_SAFE_INTEGER;

export interface GetLibraryOptions {
  refresh?: boolean;
}

export interface LibrarySnapshot {
  cards: LibraryCatalogModel[];
  fetchedAt: number;
}

const LibraryStatusSchema = z
  .object({
    lastAttemptAt: z.number().int().nonnegative(),
    lastSuccessAt: z.number().int().nonnegative().nullable(),
    lastError: z.string().max(512).nullable(),
    cards: z.number().int().nonnegative(),
  })
  .strict();

export type LibraryStatus = z.infer<typeof LibraryStatusSchema>;

let memo: LibrarySnapshot | null = null;
let inFlight: Promise<void> | null = null;

export function readLibraryStatus(): LibraryStatus | null {
  return readCache(STATUS_NAME, LibraryStatusSchema, { ttlMs: NO_EXPIRY_MS, version: STATUS_VERSION })?.data ?? null;
}

/** Cached catalog at any age, never touching the network. The memo keeps one array identity until a refresh replaces it. */
export function readLibraryCatalogSync(): LibrarySnapshot | null {
  if (memo) return memo;
  const cached = readCache(CACHE_NAME, LibraryCatalogSchema, { ttlMs: NO_EXPIRY_MS, version: CACHE_VERSION });
  if (!cached) return null;
  memo = { cards: cached.data, fetchedAt: Date.now() - cached.ageMs };
  return memo;
}

function isFresh(): boolean {
  const snapshot = readLibraryCatalogSync();
  return snapshot !== null && Date.now() - snapshot.fetchedAt < TTL_MS;
}

function isBackingOff(): boolean {
  const status = readLibraryStatus();
  return status !== null && status.lastError !== null && Date.now() - status.lastAttemptAt < FAILURE_BACKOFF_MS;
}

function describeError(err: unknown): string {
  if (!(err instanceof Error)) return String(err);
  // Node reports every network failure as "fetch failed"; the cause names it.
  const cause = err.cause instanceof Error ? `: ${err.cause.message}` : "";
  return `${err.message}${cause}`.slice(0, 512);
}

async function fetchAndStore(timeoutMs: number): Promise<void> {
  const lastAttemptAt = Date.now();
  try {
    const fresh: LibraryModel[] = await fetchOllamaLibrary({ timeoutMs });
    const fetchedAt = Date.now();
    writeCache(CACHE_NAME, fresh, { ttlMs: TTL_MS, version: CACHE_VERSION });
    memo = { cards: fresh, fetchedAt };
    writeStatus({ lastAttemptAt, lastSuccessAt: fetchedAt, lastError: null, cards: fresh.length });
  } catch (err) {
    // A stale catalog beats none, so the cache and memo stay as they are.
    // Read the status only now: another process may have succeeded meanwhile.
    writeStatus({
      lastAttemptAt,
      lastSuccessAt: readLibraryStatus()?.lastSuccessAt ?? null,
      lastError: describeError(err),
      cards: readLibraryCatalogSync()?.cards.length ?? 0,
    });
  }
}

function writeStatus(status: LibraryStatus): void {
  writeCache(STATUS_NAME, status, { ttlMs: NO_EXPIRY_MS, version: STATUS_VERSION });
}

// Never rejects. Concurrent callers share one fetch.
function refresh(explicit: boolean, timeoutMs: number): Promise<void> {
  if (isOffline() || inFlight) return inFlight ?? Promise.resolve();
  if (!explicit && (isFresh() || isBackingOff())) return Promise.resolve();
  inFlight = fetchAndStore(timeoutMs).finally(() => {
    inFlight = null;
  });
  return inFlight;
}

export function refreshLibraryCatalogIfStale(opts: { timeoutMs?: number } = {}): Promise<void> {
  return refresh(false, opts.timeoutMs ?? IMPLICIT_TIMEOUT_MS);
}

/**
 * Library paths are explicit requests: a stale or missing catalog is fetched
 * with the longer timeout even during the failure backoff. A fresh cache skips
 * the network unless `refresh: true` forces it.
 */
export async function getLibraryCatalog(
  opts: GetLibraryOptions = {},
): Promise<LibraryCatalogModel[]> {
  if (opts.refresh || !isFresh()) await refresh(true, EXPLICIT_TIMEOUT_MS);
  return readLibraryCatalogSync()?.cards ?? [];
}

export function resetLibraryCache(): void {
  memo = null;
  inFlight = null;
  clearCacheEntry(CACHE_NAME);
  clearCacheEntry(STATUS_NAME);
}
