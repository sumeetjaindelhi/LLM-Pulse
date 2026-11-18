import type {
  LibraryCatalogModel,
  ModelCategory,
  ModelEntry,
  QualityTier,
} from "../core/types.js";
import { quantRetention } from "./quant-facts.js";

// Quality fit: q = intercept + linear*L + quadratic*L^2 - penalty*max(0, age - grace) + popularity + coder,
// with L = log2(effective B).
// Refit of the hand-assigned curated scores (R^2 0.965, rmse 3.47).
export const QUALITY_INTERCEPT = 37.8;
export const QUALITY_LINEAR = 18.234;
export const QUALITY_QUADRATIC = -1.378;
// Vertex of the fitted parabola; larger models would score lower, so size stops counting here.
export const QUALITY_PARAMS_CAP_B = 98;
// Fresh releases are not penalized while the rest of the field catches up.
export const AGE_GRACE_MONTHS = 3;
export const AGE_PENALTY_PER_MONTH = 1;
// Pull velocity per decade of difference from the median; clamped so hype cannot outrank size.
export const POPULARITY_POINTS_PER_DECADE = 2.5;
export const POPULARITY_CLAMP = 4;
// Brand-new cards have too few days for pulls/day to mean anything.
export const MIN_VELOCITY_DAYS = 7;
// Coder-tuned models beat same-size generalists on the coding category the fit does not see.
export const CODER_BONUS = 4;
// Tier cut-offs match the bands of the hand-assigned tiers in data/models.json.
export const TIER_FRONTIER_MIN = 91;
export const TIER_STRONG_MIN = 75;
export const TIER_GOOD_MIN = 60;

// ESTIMATE for what `ollama pull <tag>` fetches. Ollama default builds measured 4.90-5.28 bpw (registry model-layer bytes x 8 / published params
// for llama3.1:8b, qwen3:8b, gpt-oss:20b); sized by badge params, a row's weights are off by -9% to +2%.
export const DEFAULT_BITS_PER_WEIGHT = 5;
// BitNet-style ternary weights are packed at about 2 bits.
export const TERNARY_BITS_PER_WEIGHT = 2;
// Ternary models are modelled as roughly this fraction of an equally sized dense model.
export const TERNARY_QUALITY_SCALE = TERNARY_BITS_PER_WEIGHT / DEFAULT_BITS_PER_WEIGHT;
// Quant name of a natively ternary row; the recommender excludes such rows.
export const TERNARY_QUANT_NAME = "ternary";
// Quant name of any other row: the default build, whose file type is unknown until exact facts name it.
export const DEFAULT_QUANT_NAME = "default";
// Same slug and sizes this close (in billions) are one model printed two ways, e.g. phi3:mini vs phi3:3.8b.
export const DEDUPE_PARAMS_TOLERANCE_B = 0.05;
// Sparse mixtures are usually near 1/8 active; used when the card gives no count.
export const MOE_DEFAULT_ACTIVE_FRACTION = 1 / 8;
// Effective size of an MoE sits between active and total: total^0.75 * active^0.25.
export const MOE_TOTAL_EXPONENT = 0.75;
// Below this the model is a toy, not a recommendation.
export const MIN_PARAMS_B = 0.1;
// Ollama does not publish windows on the library page; stay conservative.
export const LONG_CONTEXT_TOKENS = 32768;
export const SHORT_CONTEXT_TOKENS = 8192;
export const LONG_CONTEXT_FROM = "2025-01";
// Older non-curated models are noise next to the current field.
export const WINDOW_MONTHS = 24;
// Running-minimum dating is checked against curated release dates before it is trusted.
export const GUARD_MIN_ANCHORS = 5;
export const GUARD_MIN_AGREEMENT = 0.7;
export const GUARD_TOLERANCE_MONTHS = 2;
// Two decimals keep ratio-derived active counts free of float noise.
const ACTIVE_ROUNDING_SCALE = 100;

const SIZE_BADGE_RE = /^(\d{1,4}(?:\.\d{1,3})?)([mb])$/;
const DESCRIPTION_TOTAL_RE = /\b(\d{1,4}(?:\.\d{1,2})?)\s?B(?=-A\d|\s(?:total|parameters?|MoE|mixture|model|class|open))/i;
const ACTIVE_PAIR_RE = /\b(\d{1,4}(?:\.\d{1,2})?)B-A(\d{1,3}(?:\.\d{1,2})?)B\b/i;
const ACTIVE_COUNT_RE = /\b(\d{1,3}(?:\.\d{1,2})?)\s?B\s(?:active|activated)\b/i;
const ACTIVATING_COUNT_RE = /\bactivating\s(?:just\s|only\s)?(\d{1,3}(?:\.\d{1,2})?)\s?B\b/i;
const MOE_MENTION_RE = /mixture[- ]of[- ]experts|\bMoE\b/i;
const TERNARY_RE = /\bternary\b|1\.58[- ]bit|\bbitnet\b/i;
const SKIP_SLUG_RE = /guard|shield|ocr|^med|translat/;
const SKIP_DESCRIPTION_RE =
  /\b(safety classifier|content moderation|text-to-speech|speech recognition|medical|clinical)\b/i;
const SKIPPED_CAPABILITIES = ["embedding", "decision"];
export const CODER_SLUG_RE = /cod(e|er|ing|estral)|devstral|starcoder/;
const MULTILINGUAL_RE = /multilingual/i;
const MS_PER_DAY = 86_400_000;

export function qualityTierFor(score: number): QualityTier {
  if (score >= TIER_FRONTIER_MIN) return "frontier";
  if (score >= TIER_STRONG_MIN) return "strong";
  if (score >= TIER_GOOD_MIN) return "good";
  return "lightweight";
}

function monthIndex(date: string): number {
  return Number(date.slice(0, 4)) * 12 + Number(date.slice(5, 7)) - 1;
}

function dayNumber(date: string): number {
  const day = date.length >= 10 ? Number(date.slice(8, 10)) : 1;
  return Date.UTC(Number(date.slice(0, 4)), Number(date.slice(5, 7)) - 1, day) / MS_PER_DAY;
}

function tagSlug(tag: string): string {
  return tag.split(":")[0];
}

function curatedSlug(row: ModelEntry): string {
  return tagSlug(row.ollamaTag ?? row.id);
}

function runningMinimum(cards: LibraryCatalogModel[]): Array<string | undefined> {
  let minimum: string | undefined;
  return cards.map((card) => {
    if (!card.updated) return undefined;
    if (minimum === undefined || card.updated < minimum) minimum = card.updated;
    return minimum;
  });
}

function estimatesAgreeWithCurated(
  curated: ModelEntry[],
  cards: LibraryCatalogModel[],
  estimates: Array<string | undefined>,
): boolean {
  const estimateBySlug = new Map<string, string>();
  cards.forEach((card, i) => {
    const estimate = estimates[i];
    if (estimate && !estimateBySlug.has(card.slug)) estimateBySlug.set(card.slug, estimate);
  });
  // One anchor per slug, dated by its earliest curated size, so big families do not outvote the rest.
  const earliestBySlug = new Map<string, string>();
  for (const row of curated) {
    if (!row.ollamaTag) continue;
    const slug = tagSlug(row.ollamaTag);
    const earliest = earliestBySlug.get(slug);
    if (earliest === undefined || row.releaseDate < earliest) earliestBySlug.set(slug, row.releaseDate);
  }
  let anchors = 0;
  let agreeing = 0;
  for (const [slug, releaseDate] of earliestBySlug) {
    const estimate = estimateBySlug.get(slug);
    if (!estimate) continue;
    anchors++;
    if (Math.abs(monthIndex(estimate) - monthIndex(releaseDate)) <= GUARD_TOLERANCE_MONTHS) agreeing++;
  }
  return anchors < GUARD_MIN_ANCHORS || agreeing / anchors >= GUARD_MIN_AGREEMENT;
}

interface SizeVariant {
  size: string | null; // badge text as shown on the card; null for size-less cards
  billions: number;
}

function sizeVariants(card: LibraryCatalogModel): SizeVariant[] {
  const badges: SizeVariant[] = [];
  for (const size of card.parameterSizes) {
    const match = SIZE_BADGE_RE.exec(size);
    if (!match) continue;
    const billions = match[2] === "m" ? Number(match[1]) / 1000 : Number(match[1]);
    if (billions >= MIN_PARAMS_B) badges.push({ size, billions });
  }
  if (card.parameterSizes.length > 0) return badges;
  const total = DESCRIPTION_TOTAL_RE.exec(card.description);
  const billions = total ? Number(total[1]) : 0;
  return billions >= MIN_PARAMS_B ? [{ size: null, billions }] : [];
}

// Active/total ratio stated by the card or implied by an MoE mention; undefined for dense models.
function activeRatio(description: string, variants: SizeVariant[]): number | undefined {
  const pair = ACTIVE_PAIR_RE.exec(description);
  if (pair) return validRatio(Number(pair[2]) / Number(pair[1]));
  const active = ACTIVE_COUNT_RE.exec(description) ?? ACTIVATING_COUNT_RE.exec(description);
  if (active) {
    // A bare active count describes the flagship (largest) size.
    const largest = Math.max(...variants.map((v) => v.billions));
    return validRatio(Number(active[1]) / largest);
  }
  return MOE_MENTION_RE.test(description) ? MOE_DEFAULT_ACTIVE_FRACTION : undefined;
}

function validRatio(ratio: number): number | undefined {
  return ratio > 0 && ratio < 1 ? ratio : undefined;
}

function effectiveBillions(total: number, active: number | undefined, ternary: boolean): number {
  const base = active === undefined ? total : total ** MOE_TOTAL_EXPONENT * active ** (1 - MOE_TOTAL_EXPONENT);
  return ternary ? base * TERNARY_QUALITY_SCALE : base;
}

function libraryCategories(card: LibraryCatalogModel): ModelCategory[] {
  const thinks = card.capabilities.includes("thinking");
  if (CODER_SLUG_RE.test(card.slug)) return thinks ? ["coding", "reasoning"] : ["coding"];
  const categories: ModelCategory[] = ["general"];
  if (card.capabilities.includes("tools")) categories.push("coding");
  if (thinks) categories.push("reasoning");
  if (MULTILINGUAL_RE.test(card.description)) categories.push("multilingual");
  return categories;
}

// Everything a row needs to be scored once the popularity median is known.
interface Draft {
  entry: ModelEntry;
  effectiveB: number;
  releaseDay: string; // YYYY-MM or YYYY-MM-DD
  pulls?: number;
  coder: boolean;
}

function isSkipped(card: LibraryCatalogModel): boolean {
  return (
    (card.cloud === true && card.parameterSizes.length === 0) ||
    card.capabilities.some((c) => SKIPPED_CAPABILITIES.includes(c)) ||
    SKIP_SLUG_RE.test(card.slug) ||
    SKIP_DESCRIPTION_RE.test(card.description)
  );
}

function draftsForCard(card: LibraryCatalogModel, release: string): Draft[] {
  const variants = sizeVariants(card);
  const ratio = variants.length > 0 ? activeRatio(card.description, variants) : undefined;
  const ternary = TERNARY_RE.test(card.description);
  const bitsPerWeight = ternary ? TERNARY_BITS_PER_WEIGHT : DEFAULT_BITS_PER_WEIGHT;
  const releaseMonth = release.slice(0, 7);
  const categories = libraryCategories(card);
  const seen = new Set<string>();

  return variants.flatMap(({ size, billions }) => {
    const tag = size ? `${card.slug}:${size}` : card.slug;
    if (seen.has(tag)) return [];
    seen.add(tag);
    const active = ratio === undefined ? undefined : Math.round(billions * ratio * ACTIVE_ROUNDING_SCALE) / ACTIVE_ROUNDING_SCALE;
    const entry: ModelEntry = {
      id: tag,
      name: size ? `${card.slug} ${size.toUpperCase()}` : card.slug,
      provider: "Ollama library",
      parametersBillion: billions,
      contextWindow: releaseMonth >= LONG_CONTEXT_FROM ? LONG_CONTEXT_TOKENS : SHORT_CONTEXT_TOKENS,
      categories,
      qualityTier: "lightweight",
      qualityScore: 0,
      quantizations: [
        {
          name: ternary ? TERNARY_QUANT_NAME : DEFAULT_QUANT_NAME,
          bitsPerWeight,
          // Weights only: KV cache and compute buffer are added by the memory model.
          vramMb: Math.round((billions * 1e9 * bitsPerWeight) / 8 / 2 ** 20),
          // Default builds are Q4_K_M-class files (calc spec F8); ternary rows share it as a rough stand-in.
          qualityRetention: quantRetention("Q4_K_M").retention,
        },
      ],
      ollamaTag: tag,
      releaseDate: releaseMonth,
      source: "library",
      contextWindowEstimated: true,
      ...(active === undefined ? {} : { activeParametersBillion: active }),
    };
    return [
      {
        entry,
        effectiveB: effectiveBillions(billions, active, ternary),
        releaseDay: release,
        pulls: card.pulls,
        coder: CODER_SLUG_RE.test(card.slug),
      },
    ];
  });
}

function median(values: number[]): number | undefined {
  if (values.length === 0) return undefined;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function pullVelocity(pulls: number | undefined, releaseDay: string, fetchedDay: number): number | undefined {
  if (pulls === undefined) return undefined;
  return pulls / Math.max(MIN_VELOCITY_DAYS, fetchedDay - dayNumber(releaseDay));
}

function scoreDraft(draft: Draft, refMonth: number, fetchedDay: number, medianVelocity: number | undefined): ModelEntry {
  const velocity = pullVelocity(draft.pulls, draft.releaseDay, fetchedDay);
  const popularity =
    velocity === undefined || !medianVelocity
      ? 0
      : Math.min(
          POPULARITY_CLAMP,
          Math.max(-POPULARITY_CLAMP, POPULARITY_POINTS_PER_DECADE * Math.log10(velocity / medianVelocity)),
        );
  const age = refMonth - monthIndex(draft.releaseDay);
  const L = Math.log2(Math.min(draft.effectiveB, QUALITY_PARAMS_CAP_B));
  const raw =
    QUALITY_INTERCEPT +
    QUALITY_LINEAR * L +
    QUALITY_QUADRATIC * L * L -
    AGE_PENALTY_PER_MONTH * Math.max(0, age - AGE_GRACE_MONTHS) +
    popularity +
    (draft.coder ? CODER_BONUS : 0);
  const qualityScore = Math.min(100, Math.max(0, Math.round(raw)));
  return { ...draft.entry, qualityScore, qualityTier: qualityTierFor(qualityScore) };
}

function curatedDraft(row: ModelEntry, pulls: number | undefined): Draft {
  return {
    entry: { ...row, source: "curated" },
    effectiveB: effectiveBillions(row.parametersBillion, row.activeParametersBillion, false),
    releaseDay: row.releaseDate,
    pulls,
    coder: CODER_SLUG_RE.test(curatedSlug(row)),
  };
}

function isDuplicate(row: ModelEntry, curated: ModelEntry[]): boolean {
  const slug = tagSlug(row.id);
  return curated.some(
    (c) =>
      c.id === row.id ||
      c.ollamaTag === row.id ||
      (c.ollamaTag !== null &&
        tagSlug(c.ollamaTag) === slug &&
        Math.abs(c.parametersBillion - row.parametersBillion) < DEDUPE_PARAMS_TOLERANCE_B),
  );
}

/**
 * Appends rows synthesized from the live ollama.com library to the curated list and
 * recomputes every quality score on one scale. Pure: fetchedAt is the only clock.
 */
export function mergeLiveCatalog(
  curated: ModelEntry[],
  cards: LibraryCatalogModel[],
  fetchedAt: Date,
): ModelEntry[] {
  const fetchedDay = fetchedAt.getTime() / MS_PER_DAY;
  const fetchedMonth = fetchedAt.getUTCFullYear() * 12 + fetchedAt.getUTCMonth();

  const runningMin = runningMinimum(cards);
  const estimates = estimatesAgreeWithCurated(curated, cards, runningMin)
    ? runningMin
    : cards.map((card) => card.updated);
  const known = estimates.filter((e): e is string => e !== undefined);
  // A loop, not Math.max(...spread): a corrupt cache file may hold more cards than an argument list allows.
  let newestMonth = -Infinity;
  for (const date of known) newestMonth = Math.max(newestMonth, monthIndex(date));
  const refMonth = known.length > 0 ? Math.min(fetchedMonth, newestMonth) : fetchedMonth;

  const curatedSlugs = new Set(curated.flatMap((row) => (row.ollamaTag ? [tagSlug(row.ollamaTag)] : [])));
  const pullsBySlug = new Map<string, number>();
  for (const card of cards) {
    if (card.pulls !== undefined && !pullsBySlug.has(card.slug)) pullsBySlug.set(card.slug, card.pulls);
  }

  const drafts = cards.flatMap((card, i) => {
    const release = estimates[i];
    if (!release || isSkipped(card)) return [];
    if (!curatedSlugs.has(card.slug) && refMonth - monthIndex(release) > WINDOW_MONTHS) return [];
    return draftsForCard(card, release);
  });
  const medianVelocity = median(
    drafts.flatMap((d) => pullVelocity(d.pulls, d.releaseDay, fetchedDay) ?? []),
  );

  const score = (draft: Draft) => scoreDraft(draft, refMonth, fetchedDay, medianVelocity);
  const curatedRows = curated.map((row) =>
    score(curatedDraft(row, row.ollamaTag ? pullsBySlug.get(tagSlug(row.ollamaTag)) : undefined)),
  );
  const libraryRows = drafts.filter((d) => !isDuplicate(d.entry, curated)).map(score);
  return [...curatedRows, ...libraryRows];
}
