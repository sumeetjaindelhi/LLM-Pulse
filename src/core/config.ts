import { z } from "zod";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { homedir } from "node:os";
import { OLLAMA_API_URL } from "./constants.js";

const ConfigSchema = z.object({
  ollamaHost: z.string().url().optional(),
  lmstudioHost: z.string().url().optional(),
  defaultFormat: z.enum(["table", "json", "csv"]).optional(),
  defaultCategory: z.enum(["general", "coding", "reasoning", "creative", "multilingual", "all"]).optional(),
  defaultTop: z.number().int().min(1).max(50).optional(),
}).strict();

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

/** Resolve Ollama host URL. Priority: CLI flag > config > default constant. */
export function resolveOllamaHost(cliHost?: string): string {
  if (cliHost) return cliHost.replace(/\/+$/, "");
  const config = getConfig();
  if (config.ollamaHost) return config.ollamaHost.replace(/\/+$/, "");
  return OLLAMA_API_URL;
}
