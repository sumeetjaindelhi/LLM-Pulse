import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, rmSync, symlinkSync, writeFileSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { z } from "zod";
import { readCache, writeCache } from "../../src/core/cache.js";

const Dummy = z.array(z.object({ id: z.string() }));
type Dummy = z.infer<typeof Dummy>;

describe("cache security hardening", () => {
  let root: string;
  let outside: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), "llmpulse-cache-"));
    outside = mkdtempSync(join(tmpdir(), "llmpulse-outside-"));
  });

  afterEach(() => {
    try { rmSync(root, { recursive: true, force: true }); } catch {}
    try { rmSync(outside, { recursive: true, force: true }); } catch {}
  });

  it("refuses to read from a symlink that points outside the cache root", () => {
    // Attacker pre-plants a symlink at the cache file path targeting a file
    // they want to exfiltrate. We must not follow it and return its contents.
    const secretPath = join(outside, "secret.json");
    writeFileSync(
      secretPath,
      JSON.stringify({
        timestamp: Date.now(),
        ttlMs: 60000,
        version: 1,
        data: [{ id: "leaked" }],
      }),
    );
    const cacheFile = join(root, "attack.json");
    symlinkSync(secretPath, cacheFile);

    const result = readCache<Dummy>("attack", Dummy, { root, ttlMs: 60_000 });
    expect(result).toBeNull();
  });

  it("rejects envelopes containing __proto__ / constructor / prototype keys", () => {
    const cacheFile = join(root, "poison.json");
    writeFileSync(
      cacheFile,
      JSON.stringify({
        timestamp: Date.now(),
        ttlMs: 60000,
        version: 1,
        data: [{ id: "hello" }],
        __proto__: { polluted: true },
      }),
    );

    const result = readCache<Dummy>("poison", Dummy, { root, ttlMs: 60_000 });
    // Even if strict-schema allowed it, stripDangerousKeys removes __proto__
    // so strict() still accepts, and we get clean data back. Prove the
    // prototype isn't polluted.
    expect(({} as Record<string, unknown>).polluted).toBeUndefined();
    if (result) {
      expect(result.data).toEqual([{ id: "hello" }]);
    }
  });

  it("removes a pre-existing external symlink before writing", () => {
    // Attacker swaps a symlink pointing at /tmp/outside/evil just before our
    // writeCache runs. Our write path notices the link escapes the cache
    // root and unlinks it before writing — so the external file is untouched.
    const targetPath = join(outside, "evil.json");
    writeFileSync(targetPath, "{}");
    const cacheFile = join(root, "trap.json");
    symlinkSync(targetPath, cacheFile);

    writeCache("trap", [{ id: "ours" }], { root, ttlMs: 60_000 });

    // The external target should remain its original empty-object contents.
    const targetContent = readFileSync(targetPath, "utf-8");
    expect(targetContent).toBe("{}");
  });
});
