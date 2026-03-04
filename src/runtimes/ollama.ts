import { execa } from "execa";
import { OLLAMA_API_URL } from "../core/constants.js";
import type { RuntimeInfo } from "../core/types.js";

export async function detectOllama(): Promise<RuntimeInfo> {
  const info: RuntimeInfo = {
    name: "Ollama",
    status: "not_found",
    version: null,
    path: null,
    models: [],
  };

  // Check if binary exists
  try {
    const whichCmd = process.platform === "win32" ? "where" : "which";
    const { stdout } = await execa(whichCmd, ["ollama"]);
    info.path = stdout.trim().split("\n")[0];
    info.status = "installed";
  } catch {
    return info;
  }

  // Check if API is running and get version
  try {
    const response = await fetch(`${OLLAMA_API_URL}/api/version`);
    if (response.ok) {
      const data = (await response.json()) as { version: string };
      info.version = data.version;
      info.status = "running";
    }
  } catch {
    // API not running, try getting version from CLI
    try {
      const { stdout } = await execa("ollama", ["--version"]);
      const match = stdout.match(/(\d+\.\d+\.\d+)/);
      if (match) info.version = match[1];
    } catch {
      // ignore
    }
    return info;
  }

  // Get installed models
  try {
    const response = await fetch(`${OLLAMA_API_URL}/api/tags`);
    if (response.ok) {
      const data = (await response.json()) as { models: Array<{ name: string }> };
      info.models = data.models.map((m) => m.name);
    }
  } catch {
    // ignore
  }

  return info;
}
