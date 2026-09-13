import { z } from "zod";

const OLLAMA_LIBRARY_URL = "https://ollama.com/library";
const DEFAULT_TIMEOUT_MS = 10_000;
// Hard cap on HTML we'll parse — ollama.com/library is ~820 KB today, so 5 MB
// is ~6x headroom. Anything larger is almost certainly adversarial.
const MAX_HTML_BYTES = 5 * 1024 * 1024;
// Real model cards are a few KB; the card regexes are quadratic in block size.
const MAX_CARD_BYTES = 64_000;
const USER_AGENT = "llm-pulse (+https://github.com/sumeetjaindelhi/LLM-Pulse)";

export const LibraryModelSchema = z
  .object({
    slug: z.string().regex(/^[a-zA-Z0-9._\/-]+$/).max(128),
    description: z.string().max(2048),
    parameterSizes: z.array(z.string().max(16)).max(64),
    capabilities: z.array(z.string().max(32)).max(32),
  })
  .strict();

export const LibraryCatalogSchema = z.array(LibraryModelSchema);

export type LibraryModel = z.infer<typeof LibraryModelSchema>;

// ollama.com/library carries no semantic hooks on its model cards, so fields
// are anchored on the badge colour classes: indigo = capability, blue = size.
const LI_OPEN_RE = /<li\b/;
const SLUG_RE = /<a\s+href="\/library\/([a-zA-Z0-9._\/-]+)"/;
const DESCRIPTION_RE = /<p\s+class="max-w-lg[^"]*"[^>]*>([\s\S]*?)<\/p>/;
const CAPABILITY_RE = /<span[^>]*\btext-indigo-600\b[^>]*>([\s\S]*?)<\/span>/g;
const SIZE_RE = /<span[^>]*\btext-blue-600\b[^>]*>([\s\S]*?)<\/span>/g;

// Minimal HTML-entity decode for the subset Ollama's templating emits. One
// pass over the string + a lookup table beats chaining 7 `.replace()` calls —
// we parse ~250 model cards per refresh, so this matters in aggregate.
const ENTITY_RE = /&(?:amp|lt|gt|quot|#39|apos|nbsp);/g;
const ENTITY_MAP: Record<string, string> = {
  "&amp;": "&",
  "&lt;": "<",
  "&gt;": ">",
  "&quot;": '"',
  "&#39;": "'",
  "&apos;": "'",
  "&nbsp;": " ",
};
function decodeEntities(input: string): string {
  return input.replace(ENTITY_RE, (match) => ENTITY_MAP[match] ?? match);
}

function extractAll(re: RegExp, block: string): string[] {
  const matches = block.matchAll(re);
  const out: string[] = [];
  for (const m of matches) {
    out.push(decodeEntities(m[1].trim()));
  }
  return out;
}

function firstMatch(re: RegExp, block: string): string | null {
  const m = block.match(re);
  return m ? m[1] : null;
}

export function parseLibraryHtml(html: string): LibraryModel[] {
  if (html.length > MAX_HTML_BYTES) {
    throw new Error(
      `library HTML exceeds ${MAX_HTML_BYTES} bytes (${html.length}) — refusing to parse`,
    );
  }

  const results: LibraryModel[] = [];
  const seen = new Set<string>();

  // Splitting on "<li" bounds every block regex to a single card, and the card
  // size cap bounds the backtracking those regexes can do inside one card, so
  // hostile input stays cheap to parse.
  for (const chunk of html.split(LI_OPEN_RE).slice(1)) {
    const end = chunk.indexOf("</li>");
    if (end < 0 || end > MAX_CARD_BYTES) continue;
    const block = chunk.slice(0, end);

    const slug = firstMatch(SLUG_RE, block);
    if (!slug) continue;
    if (seen.has(slug)) continue;

    const descRaw = firstMatch(DESCRIPTION_RE, block);
    const description = descRaw ? decodeEntities(descRaw.replace(/\s+/g, " ").trim()) : "";

    const capabilities = extractAll(CAPABILITY_RE, block);
    const parameterSizes = extractAll(SIZE_RE, block);

    const candidate = { slug, description, parameterSizes, capabilities };
    const parsed = LibraryModelSchema.safeParse(candidate);
    if (!parsed.success) continue;

    seen.add(slug);
    results.push(parsed.data);
  }

  return results;
}

export interface FetchLibraryOptions {
  timeoutMs?: number;
  signal?: AbortSignal;
}

export async function fetchOllamaLibrary(opts: FetchLibraryOptions = {}): Promise<LibraryModel[]> {
  const timeoutMs = opts.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  if (opts.signal) {
    if (opts.signal.aborted) controller.abort();
    else opts.signal.addEventListener("abort", () => controller.abort(), { once: true });
  }

  try {
    const res = await fetch(OLLAMA_LIBRARY_URL, {
      headers: { "user-agent": USER_AGENT, accept: "text/html" },
      signal: controller.signal,
      redirect: "follow",
    });
    if (!res.ok) {
      throw new Error(`ollama.com/library returned HTTP ${res.status}`);
    }
    // Refuse an oversized body before buffering it. A missing or lying
    // Content-Length still hits the cap in parseLibraryHtml.
    const declaredBytes = Number(res.headers.get("content-length"));
    if (declaredBytes > MAX_HTML_BYTES) {
      throw new Error(`library HTML exceeds ${MAX_HTML_BYTES} bytes (${declaredBytes}) — refusing to download`);
    }
    const html = await res.text();
    const models = parseLibraryHtml(html);
    if (models.length === 0) {
      throw new Error("parsed 0 models from ollama.com/library — page layout may have changed");
    }
    return models;
  } finally {
    clearTimeout(timeout);
  }
}
