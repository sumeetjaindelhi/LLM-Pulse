import { readFileSync, writeFileSync, mkdirSync, unlinkSync, realpathSync } from "node:fs";
import { homedir } from "node:os";
import { join, sep } from "node:path";
import { z } from "zod";

const DEFAULT_ROOT = join(homedir(), ".llmpulse", "cache");

// Namespaced so two different callers can't collide on filename; regex
// prevents path traversal if the name ever flows in from user-facing code
// (e.g. a future plugin system).
const NAME_RE = /^[a-zA-Z0-9_-]{1,64}$/;

const EnvelopeSchema = z.object({
  timestamp: z.number().int().nonnegative(),
  ttlMs: z.number().int().positive(),
  version: z.number().int().nonnegative().default(1),
  data: z.unknown(),
}).strict();

export interface CacheOptions {
  root?: string;
  ttlMs: number;
  version?: number;
}

export interface CacheReadResult<T> {
  data: T;
  ageMs: number;
}

function cachePath(name: string, root: string): string {
  if (!NAME_RE.test(name)) {
    throw new Error(`invalid cache name: ${name}`);
  }
  return join(root, `${name}.json`);
}

// Resolve symlinks on the file and root, then require the file's canonical
// path to live inside the root's canonical path. This blocks an attacker who
// symlinks ~/.llmpulse/cache/ollama-library.json → /etc/passwd from tricking
// us into reading an arbitrary file. If either path doesn't exist, realpath
// throws and the caller treats it as a missing entry.
function resolveInsideRoot(path: string, root: string): string | null {
  let canonical: string;
  let canonicalRoot: string;
  try {
    canonical = realpathSync(path);
    canonicalRoot = realpathSync(root);
  } catch {
    return null;
  }
  if (canonical === canonicalRoot) return null;
  if (!canonical.startsWith(canonicalRoot + sep)) return null;
  return canonical;
}

function isDangerousKey(key: string): boolean {
  return key === "__proto__" || key === "constructor" || key === "prototype";
}

function stripDangerousKeys(value: unknown): unknown {
  if (value === null || typeof value !== "object") return value;
  if (Array.isArray(value)) return value.map(stripDangerousKeys);
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(value)) {
    if (isDangerousKey(k)) continue;
    out[k] = stripDangerousKeys(v);
  }
  return out;
}

export function readCache<T>(
  name: string,
  schema: z.ZodType<T>,
  opts: CacheOptions,
): CacheReadResult<T> | null {
  const root = opts.root ?? DEFAULT_ROOT;
  const path = cachePath(name, root);

  const canonical = resolveInsideRoot(path, root);
  if (!canonical) return null;

  let envelope: z.infer<typeof EnvelopeSchema>;
  try {
    const raw = readFileSync(canonical, "utf-8");
    const sanitized = stripDangerousKeys(JSON.parse(raw));
    const parsed = EnvelopeSchema.safeParse(sanitized);
    if (!parsed.success) return null;
    envelope = parsed.data;
  } catch {
    return null;
  }

  const expectedVersion = opts.version ?? 1;
  if (envelope.version !== expectedVersion) return null;

  const ageMs = Date.now() - envelope.timestamp;
  if (ageMs > opts.ttlMs || ageMs < 0) return null;

  const payload = schema.safeParse(envelope.data);
  if (!payload.success) return null;

  return { data: payload.data, ageMs };
}

export function writeCache<T>(name: string, data: T, opts: CacheOptions): void {
  const root = opts.root ?? DEFAULT_ROOT;
  const path = cachePath(name, root);
  try {
    mkdirSync(root, { recursive: true });

    // If a pre-existing entry at this path is a symlink pointing outside our
    // cache dir, remove it before writing — otherwise writeFileSync would
    // follow the link and overwrite some other file.
    try {
      const existing = realpathSync(path);
      const canonicalRoot = realpathSync(root);
      if (!existing.startsWith(canonicalRoot + sep)) {
        unlinkSync(path);
      }
    } catch {
      // path didn't exist or couldn't be resolved — fine, writeFileSync will create it
    }

    const envelope = {
      timestamp: Date.now(),
      ttlMs: opts.ttlMs,
      version: opts.version ?? 1,
      data,
    };
    writeFileSync(path, JSON.stringify(envelope));
  } catch {
    // Cache writes are best-effort. If the filesystem rejects us (read-only
    // container, quota, perms), the feature still works — we just refetch
    // next time.
  }
}

export function clearCacheEntry(name: string, root?: string): void {
  const path = cachePath(name, root ?? DEFAULT_ROOT);
  try {
    unlinkSync(path);
  } catch {
    // ENOENT is the common case — nothing to clear. Other errors (perms) are
    // not actionable here; the next write will retry.
  }
}

export function cacheFilePath(name: string, root?: string): string {
  return cachePath(name, root ?? DEFAULT_ROOT);
}
