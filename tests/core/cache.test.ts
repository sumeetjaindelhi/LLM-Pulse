import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, rmSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { z } from "zod";
import {
  readCache,
  writeCache,
  clearCacheEntry,
  cacheFilePath,
} from "../../src/core/cache.js";

const PayloadSchema = z.array(z.object({ id: z.string(), n: z.number() }));
type Payload = z.infer<typeof PayloadSchema>;

describe("core/cache", () => {
  let root: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), "llmpulse-cache-test-"));
  });

  afterEach(() => {
    try {
      rmSync(root, { recursive: true, force: true });
    } catch {
      // OS will clean up /tmp eventually
    }
  });

  it("returns null when cache is missing", () => {
    const result = readCache<Payload>("missing", PayloadSchema, { root, ttlMs: 1000 });
    expect(result).toBeNull();
  });

  it("round-trips data through write then read", () => {
    const data: Payload = [
      { id: "a", n: 1 },
      { id: "b", n: 2 },
    ];
    writeCache("roundtrip", data, { root, ttlMs: 60_000 });
    const read = readCache<Payload>("roundtrip", PayloadSchema, { root, ttlMs: 60_000 });
    expect(read).not.toBeNull();
    expect(read!.data).toEqual(data);
    expect(read!.ageMs).toBeGreaterThanOrEqual(0);
    expect(read!.ageMs).toBeLessThan(5000);
  });

  it("returns null when entry is older than ttl", () => {
    writeCache("stale", [{ id: "x", n: 1 }], { root, ttlMs: 60_000 });
    const path = cacheFilePath("stale", root);
    const envelope = JSON.parse(readFileSync(path, "utf-8"));
    envelope.timestamp = Date.now() - 120_000; // 2 minutes ago
    writeFileSync(path, JSON.stringify(envelope));

    const read = readCache<Payload>("stale", PayloadSchema, { root, ttlMs: 60_000 });
    expect(read).toBeNull();
  });

  it("returns null when cached version doesn't match expected", () => {
    writeCache("versioned", [{ id: "a", n: 1 }], { root, ttlMs: 60_000, version: 1 });
    const read = readCache<Payload>("versioned", PayloadSchema, {
      root,
      ttlMs: 60_000,
      version: 2,
    });
    expect(read).toBeNull();
  });

  it("returns null when payload fails schema validation", () => {
    writeCache("bad", { not: "an array" } as unknown as Payload, { root, ttlMs: 60_000 });
    const read = readCache<Payload>("bad", PayloadSchema, { root, ttlMs: 60_000 });
    expect(read).toBeNull();
  });

  it("clears the entry", () => {
    writeCache("clearme", [{ id: "a", n: 1 }], { root, ttlMs: 60_000 });
    clearCacheEntry("clearme", root);
    const read = readCache<Payload>("clearme", PayloadSchema, { root, ttlMs: 60_000 });
    expect(read).toBeNull();
  });

  it("rejects names with path-traversal characters", () => {
    expect(() =>
      readCache<Payload>("../../evil", PayloadSchema, { root, ttlMs: 60_000 }),
    ).toThrow();
    expect(() =>
      writeCache("../../evil", [], { root, ttlMs: 60_000 }),
    ).toThrow();
  });
});
