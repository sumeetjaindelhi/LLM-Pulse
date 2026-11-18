import { execa } from "execa";
import { OLLAMA_API_URL } from "../core/constants.js";
import { archFactsFromMetadata } from "../models/arch-facts.js";
import { fetchRegistryFacts, type RegistryFacts } from "../models/ollama-registry.js";
import { quantFileBpw, quantRetention } from "../models/quant-facts.js";
import {
  FileType,
  OllamaVersionSchema,
  OllamaTagsSchema,
  OllamaPsSchema,
  OllamaShowSchema,
} from "../core/api-schemas.js";
import type { ArchFacts, LoadedModel, ModelEntry, QuantizationVariant, RuntimeInfo } from "../core/types.js";

export async function detectOllama(host?: string): Promise<RuntimeInfo> {
  const baseUrl = host || OLLAMA_API_URL;
  const info: RuntimeInfo = {
    name: "Ollama",
    status: "not_found",
    version: null,
    path: null,
    models: [],
  };

  // Check if binary exists. A missing binary is not conclusive: the API may
  // still be served by a container, a remote host, or an app bundle not on PATH.
  try {
    const whichCmd = process.platform === "win32" ? "where" : "which";
    const { stdout } = await execa(whichCmd, ["ollama"], { timeout: 5000 });
    info.path = stdout.trim().split("\n")[0];
    info.status = "installed";
  } catch {
    // not on PATH
  }

  // Check if API is running and get version
  try {
    const response = await fetch(`${baseUrl}/api/version`, {
      signal: AbortSignal.timeout(3000),
      redirect: "error",
    });
    if (response.ok) {
      const data = OllamaVersionSchema.parse(await response.json());
      info.version = data.version;
      info.status = "running";
    }
  } catch {
    // API not running, try getting version from the CLI if it is installed
    if (info.path) {
      try {
        const { stdout } = await execa("ollama", ["--version"], { timeout: 5000 });
        const match = stdout.match(/(\d+\.\d+\.\d+)/);
        if (match) info.version = match[1];
      } catch {
        // ignore
      }
    }
    return info;
  }

  // Get installed models
  try {
    const response = await fetch(`${baseUrl}/api/tags`, {
      signal: AbortSignal.timeout(3000),
      redirect: "error",
    });
    if (response.ok) {
      const data = OllamaTagsSchema.parse(await response.json());
      info.models = data.models.map((m) => m.name);
    }
  } catch {
    // ignore
  }

  return info;
}

const API_TIMEOUT_MS = 3000;
const MAX_CONTEXT_LENGTH = 2 ** 24;
const MIB = 2 ** 20;
const MAX_FILE_BYTES = 2 ** 50; // definition: ~1 PiB, same bound as the registry manifest sizes (core/api-schemas.ts)

// Returns the value when it is an integer in [min, max], else null.
function boundedInt(value: unknown, min: number, max: number): number | null {
  return typeof value === "number" && Number.isInteger(value) && value >= min && value <= max
    ? value
    : null;
}

/** Architecture facts of an installed model, read from /api/show; `contextLength` is always set.
 *  null when the model isn't installed, Ollama is unreachable, or a required field is missing or out of range. */
export async function fetchInstalledArch(host: string, tag: string): Promise<ArchFacts | null> {
  try {
    const response = await fetch(`${host}/api/show`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: tag }),
      signal: AbortSignal.timeout(API_TIMEOUT_MS),
      redirect: "error",
    });
    if (!response.ok) return null;
    const show = OllamaShowSchema.parse(await response.json());
    // A safetensors model's model_info is built from config.json (O server/show_safetensors.go), not read from GGUF metadata.
    if (show.details?.format !== "gguf") return null;
    const arch = archFactsFromMetadata(show.model_info);
    return arch?.contextLength ? arch : null;
  } catch {
    return null;
  }
}

interface InstalledFile {
  bytes: number;
  quantization: string;
}

/** File size and quantization of `tag` as listed by /api/tags; null when it is not installed or Ollama is unreachable. */
async function fetchInstalledFile(host: string, tag: string): Promise<InstalledFile | null> {
  try {
    const response = await fetch(`${host}/api/tags`, {
      signal: AbortSignal.timeout(API_TIMEOUT_MS),
      redirect: "error",
    });
    if (!response.ok) return null;
    const fullTag = tag.includes(":") ? tag : `${tag}:latest`;
    const listed = OllamaTagsSchema.parse(await response.json()).models.find((m) => m.name === fullTag);
    // The type can become a quant name on screen (a library row's sole quant), so it must be a plain file type name.
    const quantization = FileType.safeParse(listed?.details.quantization_level);
    if (!listed || !quantization.success || !Number.isSafeInteger(listed.size) || listed.size < MIB || listed.size > MAX_FILE_BYTES) return null;
    return { bytes: listed.size, quantization: quantization.data.toUpperCase() };
  } catch {
    return null;
  }
}

/** Models currently loaded in Ollama (/api/ps), with the context length each was
 *  allocated. [] when Ollama is unreachable. */
export async function fetchLoadedModels(host: string): Promise<LoadedModel[]> {
  try {
    const response = await fetch(`${host}/api/ps`, {
      signal: AbortSignal.timeout(API_TIMEOUT_MS),
      redirect: "error",
    });
    if (!response.ok) return [];
    const data = OllamaPsSchema.parse(await response.json());
    return data.models.map((m) => ({
      name: m.name,
      contextLength: boundedInt(m.context_length, 1, MAX_CONTEXT_LENGTH),
      quantizationLevel: m.details?.quantization_level,
    }));
  } catch {
    return [];
  }
}

/** Loaded models, each with its architecture facts (exact KV sizing) when /api/show can read them. */
export async function fetchLoadedModelsWithKv(host: string): Promise<LoadedModel[]> {
  const loaded = await fetchLoadedModels(host);
  return Promise.all(
    loaded.map(async (m) => {
      const arch = await fetchInstalledArch(host, m.name);
      return arch ? { ...m, arch } : m;
    }),
  );
}

/** The model with these architecture facts. Ollama caps num_ctx at the file's training context
 *  (O llm/server.go:111-114), so a known one replaces the listed window. /api/show leaves out the token array, so facts
 *  read there often lack the vocabulary size; the one the row already has for the same tag is kept. */
function withArch(model: ModelEntry, facts: ArchFacts): ModelEntry {
  const vocabSize = facts.vocabSize ?? model.arch?.vocabSize;
  const arch = vocabSize === undefined ? facts : { ...facts, vocabSize };
  if (arch.contextLength === undefined) return { ...model, arch };
  const { contextWindowEstimated, ...exact } = model;
  return { ...exact, arch, contextWindow: arch.contextLength };
}

/** The model's quants with the file its tag installs (type `fileType`, `vramMb` MiB) as the exact one: the quant of that
 *  name, or a library row's sole quant, a placeholder for that file, which also takes the file's type as its name and,
 *  where quant-facts.ts has them, that type's bits per weight and retention. */
function withTagFile(model: ModelEntry, fileType: string, vramMb: number): QuantizationVariant[] {
  if (model.source === "library" && model.quantizations.length === 1) {
    const [placeholder] = model.quantizations;
    const retention = quantRetention(fileType);
    return [{
      ...placeholder,
      name: fileType,
      bitsPerWeight: quantFileBpw(fileType)?.bpw ?? placeholder.bitsPerWeight,
      vramMb,
      qualityRetention: retention.published ? retention.retention : placeholder.qualityRetention,
      exact: true,
    }];
  }
  return model.quantizations.map((q) => (q.name.toUpperCase() === fileType.toUpperCase() ? { ...q, vramMb, exact: true } : q));
}

/** The model with its estimates replaced by exact figures when it is installed on this host: architecture facts and
 *  training context (/api/show) and the installed file (/api/tags, withTagFile). Unchanged otherwise (never throws,
 *  never blocks long). */
export async function withInstalledKv(host: string, model: ModelEntry): Promise<ModelEntry> {
  if (!model.ollamaTag) return model;
  const [arch, installed] = await Promise.all([
    fetchInstalledArch(host, model.ollamaTag),
    fetchInstalledFile(host, model.ollamaTag),
  ]);
  const result = arch ? withArch(model, arch) : model;
  if (!installed) return result;
  // At most one quant is exact, the file `ollama pull <tag>` installs: another local file under the tag (a re-pushed tag,
  // a custom model) does not take its place.
  const pulled = model.quantizations.find((q) => q.exact);
  if (pulled && pulled.name.toUpperCase() !== installed.quantization) return result;
  return { ...result, quantizations: withTagFile(result, installed.quantization, Math.round(installed.bytes / MIB)) };
}

/** The model with the registry's facts about the file `ollama pull <tag>` installs: its architecture facts when the model
 *  has none, and the file itself (withTagFile), whose size counts the projector and draft layers /api/tags also counts. */
function withRegistryFacts(model: ModelEntry, facts: RegistryFacts): ModelEntry {
  const vramMb = Math.round((facts.weightsBytes + (facts.projectorBytes ?? 0) + (facts.draftBytes ?? 0)) / MIB);
  const arch = model.arch ?? facts.arch;
  const fromRegistry = !model.arch && facts.arch !== undefined;
  // Ollama loads the draft model by default (DraftNumPredict 4, O api/types.go:1364, O server/routes.go:168-170); its KV
  // and compute are not modelled.
  const draftNote = { inexactReason: "draft model: its KV and compute buffer are not modelled" };
  const withFacts = !arch || !(fromRegistry || facts.draftBytes) ? model : withArch(model, facts.draftBytes ? { ...arch, ...draftNote } : arch);
  return { ...withFacts, quantizations: withTagFile(model, facts.fileType, vramMb) };
}

/** The model with the most exact facts available: from Ollama on `host` when installed (withInstalledKv), else from
 *  the ollama.com registry entry of its tag (none when offline), else unchanged. Never throws. */
export async function withExactFacts(host: string, model: ModelEntry): Promise<ModelEntry> {
  const installed = await withInstalledKv(host, model);
  // An exact quant means the tag's file is known: curated rows were built from registry facts, installed ones read from
  // Ollama. The registry reads that same file, so it has nothing to add, not even an arch still missing (mllama).
  if (!installed.ollamaTag || installed.quantizations.some((q) => q.exact)) return installed;
  const facts = await fetchRegistryFacts(installed.ollamaTag);
  return facts ? withRegistryFacts(installed, facts) : installed;
}
