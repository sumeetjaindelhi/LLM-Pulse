import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { LMSTUDIO_PATH_HINTS } from "../core/constants.js";
import { resolveLmStudioHost } from "../core/config.js";
import { LmStudioModelsSchema } from "../core/api-schemas.js";
import type { RuntimeInfo } from "../core/types.js";

/**
 * Resolve a LM Studio path hint into an absolute filesystem path, or null if
 * the required environment variable is missing. Refuses to produce root-level
 * paths like `/LM Studio` when `LOCALAPPDATA` is unset — returns null so the
 * caller skips this hint instead of trying a bogus path.
 */
function resolveHint(hint: string): string | null {
  // Absolute POSIX paths and Windows drive paths pass through as-is.
  if (hint.startsWith("/") || /^[A-Za-z]:[\\/]/.test(hint)) return hint;
  // Otherwise: `ENVVAR/subpath` — read the env var (or homedir for HOME).
  const slashIdx = hint.indexOf("/");
  const envVar = slashIdx === -1 ? hint : hint.slice(0, slashIdx);
  const rest = slashIdx === -1 ? "" : hint.slice(slashIdx);
  let base: string | undefined;
  if (envVar === "HOME") base = homedir();
  else base = process.env[envVar];
  if (!base) return null;
  return `${base}${rest}`;
}

export async function detectLmStudio(host?: string): Promise<RuntimeInfo> {
  const baseUrl = resolveLmStudioHost(host);
  const info: RuntimeInfo = {
    name: "LM Studio",
    status: "not_found",
    version: null,
    path: null,
    models: [],
  };

  const platform = process.platform as "win32" | "darwin" | "linux";
  const hints = LMSTUDIO_PATH_HINTS[platform] ?? [];

  for (const hint of hints) {
    const resolved = resolveHint(hint);
    if (resolved && existsSync(resolved)) {
      info.status = "installed";
      info.path = resolved;
      break;
    }
  }

  // Check if LM Studio's local server is running
  if (info.status === "installed") {
    try {
      const response = await fetch(`${baseUrl}/v1/models`, {
        signal: AbortSignal.timeout(3000),
      });
      if (response.ok) {
        info.status = "running";
        const data = LmStudioModelsSchema.parse(await response.json());
        info.models = data.data.map((m) => m.id);
      }
    } catch {
      // Server not running
    }
  }

  return info;
}
