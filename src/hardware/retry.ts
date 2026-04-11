/**
 * Retry a promise-returning function with linear backoff, returning null if
 * every attempt fails. Used for one-shot detection subprocess calls (e.g.
 * `nvidia-smi`, `rocm-smi`) that can fail transiently when the GPU is under
 * heavy load — a retry catches the transient, but a sustained failure still
 * falls through to the existing null/no-GPU path.
 *
 * NOT used in live monitor polling paths (monitor.ts pollGpu, profile.ts
 * pollGpuSnapshot) because those have organic retry via the next tick — a
 * retry loop there would risk exceeding the poll interval and desynchronizing
 * the data stream.
 */
export async function retry<T>(
  fn: () => Promise<T>,
  opts: {
    attempts: number;
    delayMs: number;
    backoff: number;
    // Optional predicate. If it returns false for a thrown error, retry stops
    // immediately and returns null — useful for unrecoverable errors like
    // ENOENT (binary not found) where waiting won't help.
    shouldRetry?: (err: unknown) => boolean;
  },
): Promise<T | null> {
  let delay = opts.delayMs;
  for (let i = 0; i < opts.attempts; i++) {
    try {
      return await fn();
    } catch (err) {
      if (opts.shouldRetry && !opts.shouldRetry(err)) return null;
      if (i < opts.attempts - 1) {
        await new Promise((r) => setTimeout(r, delay));
        delay = Math.min(delay * opts.backoff, 2000);
      }
    }
  }
  return null;
}
