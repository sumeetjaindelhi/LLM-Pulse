import { searchModels } from "../../models/database.js";
import { theme } from "./colors.js";

interface ModelNotFoundSuggestion {
  id: string;
  name: string;
  ollamaTag: string | null;
}

export interface ModelNotFoundPayload {
  error: "Model not found";
  query: string;
  suggestions: ModelNotFoundSuggestion[];
}

// Pure data shape consumed by silent/JSON output paths and the MCP server.
// Centralised here so a future schema change (extra fields, error code, etc.)
// only needs editing in one place.
export function modelNotFoundPayload(modelArg: string, limit = 5): ModelNotFoundPayload {
  const suggestions = searchModels(modelArg).slice(0, limit);
  return {
    error: "Model not found",
    query: modelArg,
    suggestions: suggestions.map((s) => ({ id: s.id, name: s.name, ollamaTag: s.ollamaTag })),
  };
}

interface RenderOptions {
  silent?: boolean;
  suggestionsLimit?: number;
  identityFormat?: "tag" | "id";
  showBrowseHint?: boolean;
}

// Logs the "model not found" error to stdout. In silent mode emits the JSON
// payload; otherwise renders the coloured "Did you mean / Browse all models"
// block. Variations exist because `compare` shows fewer suggestions and uses
// the model id (since users are comparing curated entries, not pulling tags).
export function renderModelNotFound(modelArg: string, options: RenderOptions = {}): void {
  const limit = options.suggestionsLimit ?? 5;
  const payload = modelNotFoundPayload(modelArg, limit);

  if (options.silent) {
    console.log(JSON.stringify(payload, null, 2));
    return;
  }

  const identityFormat = options.identityFormat ?? "tag";
  const showBrowseHint = options.showBrowseHint ?? true;

  console.log(`\n  ${theme.fail("✗")} Model not found: ${theme.value(modelArg)}`);
  if (payload.suggestions.length > 0) {
    console.log(`  ${theme.muted("Did you mean:")}`);
    for (const s of payload.suggestions) {
      const identity = identityFormat === "id"
        ? theme.muted(` (${s.id})`)
        : (s.ollamaTag ? theme.muted(` (${s.ollamaTag})`) : "");
      console.log(`    ${theme.muted("•")} ${s.name}${identity}`);
    }
  }
  if (showBrowseHint) {
    console.log(`  ${theme.muted("Browse all models:")} ${theme.command("llm-pulse models")}`);
  }
}
