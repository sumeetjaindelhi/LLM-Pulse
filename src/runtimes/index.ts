import { detectOllama } from "./ollama.js";
import { detectLlamaCpp } from "./llamacpp.js";
import { detectLmStudio } from "./lmstudio.js";
import type { RuntimeInfo } from "../core/types.js";

export async function detectAllRuntimes(): Promise<RuntimeInfo[]> {
  const results = await Promise.allSettled([
    detectOllama(),
    detectLlamaCpp(),
    detectLmStudio(),
  ]);

  return results.map((r, i) => {
    if (r.status === "fulfilled") return r.value;
    // Fallback if detection throws
    const names = ["Ollama", "llama.cpp", "LM Studio"];
    return {
      name: names[i],
      status: "not_found" as const,
      version: null,
      path: null,
      models: [],
    };
  });
}
