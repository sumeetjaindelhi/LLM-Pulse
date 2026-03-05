import { existsSync } from "node:fs";
import { LMSTUDIO_PATHS, LMSTUDIO_API_URL } from "../core/constants.js";
import type { RuntimeInfo } from "../core/types.js";

export async function detectLmStudio(): Promise<RuntimeInfo> {
  const info: RuntimeInfo = {
    name: "LM Studio",
    status: "not_found",
    version: null,
    path: null,
    models: [],
  };

  const platform = process.platform as "win32" | "darwin" | "linux";
  const paths = LMSTUDIO_PATHS[platform] ?? [];

  for (const p of paths) {
    if (p && existsSync(p)) {
      info.status = "installed";
      info.path = p;
      break;
    }
  }

  // Check if LM Studio's local server is running
  if (info.status === "installed") {
    try {
      const response = await fetch(`${LMSTUDIO_API_URL}/v1/models`, {
        signal: AbortSignal.timeout(3000),
      });
      if (response.ok) {
        info.status = "running";
        const data = (await response.json()) as { data: Array<{ id: string }> };
        info.models = data.data.map((m) => m.id);
      }
    } catch {
      // Server not running
    }
  }

  return info;
}
