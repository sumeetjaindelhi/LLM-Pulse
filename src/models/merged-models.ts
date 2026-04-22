import { getAllModels, getModelByTag } from "./database.js";
import { fetchOllamaModels } from "./ollama-models.js";
import { getLibraryCatalog } from "./library-cache.js";
import type {
  LibraryCatalogModel,
  MergedModel,
  MergedModelSource,
  OllamaModel,
} from "../core/types.js";

export interface MergeOptions {
  ollamaHost?: string;
  includeLibrary?: boolean;
  refresh?: boolean;
}

export async function getMergedModels(
  hostOrOpts?: string | MergeOptions,
): Promise<MergedModel[]> {
  const opts: MergeOptions =
    typeof hostOrOpts === "string" ? { ollamaHost: hostOrOpts } : hostOrOpts ?? {};

  const dbModels = getAllModels();
  const ollamaPromise = fetchOllamaModels(opts.ollamaHost);
  const libraryPromise = opts.includeLibrary
    ? getLibraryCatalog({ refresh: opts.refresh ?? false })
    : Promise.resolve<LibraryCatalogModel[]>([]);

  const [ollamaModels, libraryModels] = await Promise.all([ollamaPromise, libraryPromise]);

  const ollamaByName = new Map<string, OllamaModel>();
  for (const om of ollamaModels) ollamaByName.set(om.name, om);

  const libraryBySlug = new Map<string, LibraryCatalogModel>();
  for (const lib of libraryModels) libraryBySlug.set(lib.slug, lib);

  const merged: MergedModel[] = [];
  const matchedOllamaTags = new Set<string>();
  const matchedLibrarySlugs = new Set<string>();

  // Pass 1: curated DB rows — attach matching installed / library records
  for (const entry of dbModels) {
    const tag = entry.ollamaTag;
    const ollamaModel = tag ? ollamaByName.get(tag) ?? null : null;
    const librarySlug = tag ? tag.split(":")[0] : null;
    const libraryModel = librarySlug ? libraryBySlug.get(librarySlug) ?? null : null;

    if (tag && ollamaModel) matchedOllamaTags.add(tag);
    if (librarySlug && libraryModel) matchedLibrarySlugs.add(librarySlug);

    const sources: MergedModelSource[] = ["curated"];
    if (libraryModel) sources.push("library");
    if (ollamaModel) sources.push("installed");

    merged.push({
      entry,
      ollamaModel,
      libraryModel,
      installed: ollamaModel !== null,
      ollamaTag: tag,
      sources,
    });
  }

  // Pass 2: library models NOT in curated DB — discoverable catalog entries
  for (const lib of libraryModels) {
    if (matchedLibrarySlugs.has(lib.slug)) continue;

    // Library slug may also match an installed Ollama tag (e.g. user installed
    // "gpt-oss:20b", library slug is "gpt-oss") — merge those too.
    const installedForSlug = ollamaModels.find(
      (om) => om.name === lib.slug || om.name.startsWith(`${lib.slug}:`),
    );
    if (installedForSlug) matchedOllamaTags.add(installedForSlug.name);

    const sources: MergedModelSource[] = ["library"];
    if (installedForSlug) sources.push("installed");

    merged.push({
      entry: null,
      ollamaModel: installedForSlug ?? null,
      libraryModel: lib,
      installed: installedForSlug !== undefined,
      ollamaTag: installedForSlug?.name ?? lib.slug,
      sources,
    });
  }

  // Pass 3: locally installed Ollama models we haven't matched yet
  // (e.g. user-pulled tags not in the library page or curated DB — custom
  // fine-tunes, sideloaded GGUFs, etc).
  for (const om of ollamaModels) {
    if (matchedOllamaTags.has(om.name)) continue;

    // Try one more time: the Ollama tag might match a curated entry by tag
    // (e.g. "llama3.1:latest" after redirect). getModelByTag handles this.
    const curatedByTag = getModelByTag(om.name);
    if (curatedByTag) continue;

    merged.push({
      entry: null,
      ollamaModel: om,
      libraryModel: null,
      installed: true,
      ollamaTag: om.name,
      sources: ["installed"],
    });
  }

  // Sort: installed first, then curated, then library-only, then by name
  const sourceRank = (m: MergedModel): number => {
    if (m.installed) return 0;
    if (m.entry) return 1;
    return 2;
  };

  merged.sort((a, b) => {
    const ra = sourceRank(a);
    const rb = sourceRank(b);
    if (ra !== rb) return ra - rb;
    const nameA =
      a.entry?.name ?? a.libraryModel?.slug ?? a.ollamaModel?.name ?? "";
    const nameB =
      b.entry?.name ?? b.libraryModel?.slug ?? b.ollamaModel?.name ?? "";
    return nameA.localeCompare(nameB);
  });

  return merged;
}
