import { z } from "zod";

const OLLAMA_LIBRARY_URL = "https://ollama.com/library?sort=newest";
const LIBRARY_HOSTNAME = "ollama.com";
const DEFAULT_TIMEOUT_MS = 10_000;
// Hard cap on HTML we'll parse — ollama.com/library is ~820 KB today, so 5 MB
// is ~6x headroom. Anything larger is almost certainly adversarial.
const MAX_HTML_BYTES = 5 * 1024 * 1024;
// Real model cards are a few KB; the card regexes are quadratic in block size.
const MAX_CARD_BYTES = 64_000;
// The page lists ~270 cards today; the cap only stops a hostile page from
// inflating the cache.
const MAX_CARDS = 2000;
const MAX_DESCRIPTION_CHARS = 2048;
const MAX_PULLS = 1e12;
const DAY_MS = 24 * 60 * 60 * 1000;
const EARLIEST_UPDATED_MS = Date.UTC(2023, 0, 1);
const USER_AGENT = "llm-pulse (+https://github.com/sumeetjaindelhi/LLM-Pulse)";

export const LibraryModelSchema = z
  .object({
    slug: z.string().regex(/^[a-z0-9][a-z0-9._-]{0,63}$/),
    description: z.string().max(MAX_DESCRIPTION_CHARS),
    parameterSizes: z.array(z.string().max(16)).max(64),
    capabilities: z.array(z.string().max(32)).max(32),
    pulls: z.number().int().min(0).max(MAX_PULLS).optional(),
    updated: z.string().regex(/^\d{4}-\d{2}-\d{2}$/).optional(),
    cloud: z.boolean().optional(),
  })
  .strict();

export const LibraryCatalogSchema = z.array(LibraryModelSchema).max(MAX_CARDS);

export type LibraryModel = z.infer<typeof LibraryModelSchema>;

// ollama.com/library carries no semantic hooks on its model cards, so fields
// are anchored on the badge colour classes: indigo = capability, blue = size.
const LI_OPEN_RE = /<li\b/;
const SLUG_RE = /<a\s+href="\/library\/([a-zA-Z0-9._\/-]+)"/;
const DESCRIPTION_RE = /<p\s+class="max-w-lg[^"]*"[^>]*>([\s\S]*?)<\/p>/;
// Badge spans hold plain text and short attributes (longest seen: 121 and 24
// chars), so both scans are bounded; the class tokens are checked in code
// because a class-matching regex backtracks badly.
const SPAN_RE = /<span\b([^>]{0,512})>([^<]{0,512})</g;
const CLASS_ATTR_RE = /\sclass="([^"]*)"/;
const PULLS_RE = /<span\s*>([0-9][0-9.,]{0,15}[KMB]?)<\/span>\s*<span[^>]*>&nbsp;Pulls/;
const UPDATED_RE =
  /title="(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) (\d{1,2}), (\d{4}) \d{1,2}:\d{2} [AP]M UTC"/;
const CONTROL_CHARS_RE = /[\u0000-\u001f\u007f-\u009f]/g;

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const COUNT_MULTIPLIERS: Record<string, number> = { K: 1e3, M: 1e6, B: 1e9 };

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

function firstMatch(re: RegExp, block: string): string | null {
  const m = block.match(re);
  return m ? m[1] : null;
}

// "121.2M" -> 121200000, "4,355" -> 4355.
function parsePulls(raw: string): number | undefined {
  const m = raw.replace(/,/g, "").match(/^(\d+(?:\.\d+)?)([KMB]?)$/);
  if (!m) return undefined;
  const count = Math.round(Number(m[1]) * (COUNT_MULTIPLIERS[m[2]] ?? 1));
  return count <= MAX_PULLS ? count : undefined;
}

// Page titles are "Nov 8, 2026 12:42 AM UTC". Dates outside 2023..fetchedAt+1d
// (or impossible, like Feb 31) are dropped rather than trusted.
function parseUpdated(block: string, fetchedAt: number): string | undefined {
  const m = block.match(UPDATED_RE);
  if (!m) return undefined;
  const month = MONTHS.indexOf(m[1]);
  const day = Number(m[2]);
  const year = Number(m[3]);
  const ms = Date.UTC(year, month, day);
  if (new Date(ms).getUTCDate() !== day) return undefined;
  if (ms < EARLIEST_UPDATED_MS || ms > fetchedAt + DAY_MS) return undefined;
  return `${year}-${String(month + 1).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
}

interface Badges {
  capabilities: string[];
  parameterSizes: string[];
  cloud: boolean;
}

function scanBadges(block: string): Badges {
  const badges: Badges = { capabilities: [], parameterSizes: [], cloud: false };
  for (const [, attrs, text] of block.matchAll(SPAN_RE)) {
    const classes = attrs.match(CLASS_ATTR_RE)?.[1].split(/\s+/);
    if (!classes) continue;
    const label = decodeEntities(text.trim()).replace(CONTROL_CHARS_RE, "");
    if (classes.includes("text-indigo-600")) badges.capabilities.push(label);
    if (classes.includes("text-blue-600")) badges.parameterSizes.push(label);
    if (classes.includes("text-cyan-500") && /cloud/i.test(text)) badges.cloud = true;
  }
  return badges;
}

function parseDescription(block: string): string {
  const raw = firstMatch(DESCRIPTION_RE, block);
  if (!raw) return "";
  return decodeEntities(raw)
    .replace(CONTROL_CHARS_RE, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, MAX_DESCRIPTION_CHARS);
}

export function parseLibraryHtml(html: string, fetchedAt: number = Date.now()): LibraryModel[] {
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

    const pullsRaw = firstMatch(PULLS_RE, block);

    const badges = scanBadges(block);
    const candidate = {
      slug,
      description: parseDescription(block),
      parameterSizes: badges.parameterSizes,
      capabilities: badges.capabilities,
      pulls: pullsRaw ? parsePulls(pullsRaw) : undefined,
      updated: parseUpdated(block, fetchedAt),
      cloud: badges.cloud ? true : undefined,
    };
    const parsed = LibraryModelSchema.safeParse(candidate);
    if (!parsed.success) continue;

    seen.add(slug);
    results.push(parsed.data);
    if (results.length >= MAX_CARDS) break;
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
    // redirect: "follow" can land anywhere; only ollama.com pages are trusted.
    // Some fetch implementations leave `url` empty, which carries no signal.
    const finalHost = res.url ? new URL(res.url).hostname : LIBRARY_HOSTNAME;
    if (finalHost !== LIBRARY_HOSTNAME) {
      throw new Error(`ollama.com/library redirected to another host (${finalHost})`);
    }
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
