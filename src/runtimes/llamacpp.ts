import { execa } from "execa";
import type { RuntimeInfo } from "../core/types.js";

export async function detectLlamaCpp(): Promise<RuntimeInfo> {
  const info: RuntimeInfo = {
    name: "llama.cpp",
    status: "not_found",
    version: null,
    path: null,
    models: [],
  };

  // Check for various llama.cpp binary names
  const binaries = ["llama-server", "llama-cli", "llama-cpp", "main"];
  const whichCmd = process.platform === "win32" ? "where" : "which";

  for (const bin of binaries) {
    try {
      const { stdout } = await execa(whichCmd, [bin], { timeout: 5000 });
      info.path = stdout.trim().split("\n")[0];
      info.status = "installed";

      // Try to get version
      try {
        const { stdout: ver } = await execa(bin, ["--version"], { timeout: 5000 });
        const match = ver.match(/(\d+\.\d+[\.\d]*)/);
        if (match) info.version = match[1];
      } catch {
        // --version not supported on all builds
      }

      break;
    } catch {
      continue;
    }
  }

  return info;
}
