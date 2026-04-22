import { z } from "zod";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { homedir } from "node:os";
import { OLLAMA_API_URL, LMSTUDIO_API_URL } from "./constants.js";

// passthrough(): unknown keys are preserved rather than rejected. This gives
// forward/backward compatibility — a config written for a future version won't
// trip an "Invalid config" warning on an older llm-pulse binary. Trade-off:
// typos in known keys (e.g. `ollamHost`) are silently ignored. The tool's
// existing documented keys are the surface users should reach for.
const ConfigSchema = z.object({
  ollamaHost: z.string().url().optional(),
  lmstudioHost: z.string().url().optional(),
  defaultFormat: z.enum(["table", "json", "csv"]).optional(),
  defaultCategory: z.enum(["general", "coding", "reasoning", "creative", "multilingual", "all"]).optional(),
  defaultTop: z.number().int().min(1).max(50).optional(),
}).passthrough();

export type LlmPulseConfig = z.infer<typeof ConfigSchema>;

let cachedConfig: LlmPulseConfig | null = null;

const CONFIG_FILENAME = ".llmpulserc";

/** Load config from `.llmpulserc` (CWD first, then home dir). Returns `{}` on missing file or validation error. */
export function loadConfig(): LlmPulseConfig {
  // Try CWD first
  const cwdPath = resolve(process.cwd(), CONFIG_FILENAME);
  const homePath = resolve(homedir(), CONFIG_FILENAME);

  for (const path of [cwdPath, homePath]) {
    try {
      const raw = readFileSync(path, "utf-8");
      const parsed = JSON.parse(raw);
      const result = ConfigSchema.safeParse(parsed);
      if (result.success) {
        cachedConfig = result.data;
        return cachedConfig;
      }
      process.stderr.write(`Warning: Invalid config in ${path}: ${result.error.issues.map((i) => i.message).join(", ")}\n`);
      cachedConfig = {};
      return cachedConfig;
    } catch (err: unknown) {
      // ENOENT = file not found, keep trying next path
      if (err instanceof Error && "code" in err && (err as NodeJS.ErrnoException).code === "ENOENT") continue;
      // JSON parse error
      if (err instanceof SyntaxError) {
        process.stderr.write(`Warning: Malformed JSON in ${path}\n`);
        cachedConfig = {};
        return cachedConfig;
      }
      // Other errors, ignore
    }
  }

  cachedConfig = {};
  return cachedConfig;
}

/** Get cached config (loads if not yet loaded). */
export function getConfig(): LlmPulseConfig {
  if (cachedConfig === null) return loadConfig();
  return cachedConfig;
}

/** Strip trailing slashes so we never produce double-slashes when concatenating paths. */
function trimHost(url: string): string {
  return url.replace(/\/+$/, "");
}

/** Normalize a host candidate into a full URL string or return null if it
 *  doesn't look like one. Accepts both Ollama's env-var format (bare
 *  `host:port`, `0.0.0.0`) and full URLs (`http://host:port`). The bare-host
 *  form is promoted to `http://` since Ollama's API is HTTP by default.
 */
function normalizeHostCandidate(value: string | undefined): string | null {
  if (!value) return null;
  const trimmed = value.trim();
  if (trimmed.length === 0 || trimmed.length > 2048) return null;

  const withScheme = /^https?:\/\//i.test(trimmed) ? trimmed : `http://${trimmed}`;
  try {
    const u = new URL(withScheme);
    if (u.protocol !== "http:" && u.protocol !== "https:") return null;
    if (!u.hostname) return null;
    return u.toString();
  } catch {
    return null;
  }
}

/** Resolve Ollama host URL.
 *  Priority: CLI flag > config file > `OLLAMA_HOST` env var > default constant.
 *  The env var is the one Ollama itself uses, so we honor it rather than
 *  silently defaulting to 127.0.0.1 when the user has pointed their daemon
 *  somewhere else.
 */
export function resolveOllamaHost(cliHost?: string): string {
  if (cliHost) return trimHost(cliHost);
  const config = getConfig();
  if (config.ollamaHost) return trimHost(config.ollamaHost);
  const envNormalized = normalizeHostCandidate(process.env.OLLAMA_HOST);
  if (envNormalized) return trimHost(envNormalized);
  return OLLAMA_API_URL;
}

/** Resolve LM Studio host URL. Priority: CLI flag > config > default constant.
 *
 * Mirrors `resolveOllamaHost` — previously the `lmstudioHost` field in the
 * config schema was silently ignored because nothing ever consulted it. This
 * function makes the config field actually functional.
 */
export function resolveLmStudioHost(cliHost?: string): string {
  if (cliHost) return cliHost.replace(/\/+$/, "");
  const config = getConfig();
  if (config.lmstudioHost) return config.lmstudioHost.replace(/\/+$/, "");
  return LMSTUDIO_API_URL;
}
