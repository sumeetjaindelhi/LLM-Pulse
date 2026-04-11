/**
 * Parse a commander option string into a positive integer, falling back to a
 * documented default on NaN/negative inputs and warning the user on stderr.
 *
 * Why we need this: commander.js hands every option value through as a string
 * (even when a default is declared). Callers were doing `parseInt(opts.x, 10)`
 * and using the result directly. If the user typed `--top abc`, `parseInt`
 * returns `NaN`, which then silently propagates:
 *   - `arr.slice(0, NaN)` → `[]`
 *   - `num_ctx: NaN` → server rejects with an opaque error
 * The user sees "0 recommendations" or a confusing API error with no hint that
 * their input was wrong.
 *
 * The `fallback` value passed here should be the same default that commander
 * uses when the flag is omitted — we're not inventing new defaults, we're
 * routing invalid inputs to the pre-existing default.
 */
export function parseIntSafe(value: string, fallback: number, optionName: string): number {
  const n = parseInt(value, 10);
  if (Number.isNaN(n) || n < 0) {
    process.stderr.write(
      `Warning: --${optionName} expected a positive integer, got "${value}". Using default ${fallback}.\n`,
    );
    return fallback;
  }
  return n;
}
