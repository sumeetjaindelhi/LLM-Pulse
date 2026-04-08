import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { detectHardware } from "../hardware/index.js";
import { clearHardwareCache } from "../hardware/index.js";
import { HardwareMonitor } from "../hardware/monitor.js";
import { detectAllRuntimes } from "../runtimes/index.js";
import { getAllModels, searchModels, filterByCategory, resolveModel } from "../models/database.js";
import { fetchOllamaModels, clearOllamaCache } from "../models/ollama-models.js";
import { getRecommendations } from "../analysis/recommender.js";
import { scoreModel, deriveVerdict, getAvailableVram } from "../analysis/scorer.js";
import { runDiagnostics } from "../analysis/doctor.js";
import { resolveOllamaHost } from "../core/config.js";
import { VERSION } from "../core/constants.js";
import type { ModelCategory } from "../core/types.js";

const CategoryEnum = z.enum(["general", "coding", "reasoning", "creative", "multilingual", "all"]);

const server = new McpServer({
  name: "llm-pulse",
  version: VERSION,
});

// ── scan ─────────────────────────────────────────────
server.tool(
  "scan",
  "Full hardware scan — detects CPU, GPU, RAM, disk, LLM runtimes (Ollama, llama.cpp, LM Studio), and recommends compatible models",
  {
    category: CategoryEnum.optional().default("all").describe("Filter recommendations by model category"),
    top: z.number().int().min(1).max(50).optional().default(5).describe("Number of recommendations to return"),
    host: z.string().optional().describe("Ollama API host URL (default: http://127.0.0.1:11434)"),
  },
  async ({ category, top, host }) => {
    try {
      clearHardwareCache();
      clearOllamaCache();
      const ollamaHost = resolveOllamaHost(host);

      const [hardware, runtimes] = await Promise.all([
        detectHardware(),
        detectAllRuntimes(ollamaHost),
      ]);

      const recommendations = getRecommendations(hardware, {
        category,
        top,
        onlyFitting: true,
      });

      const result = {
        hardware,
        runtimes,
        recommendations: recommendations.map((r) => ({
          rank: r.rank,
          model: r.score.model.name,
          quantization: r.score.quantization.name,
          fitLevel: r.score.fitLevel,
          vramMb: r.score.quantization.vramMb,
          compositeScore: r.score.compositeScore,
          pullCommand: r.pullCommand,
        })),
      };

      return { content: [{ type: "text" as const, text: JSON.stringify(result, null, 2) }] };
    } catch (err) {
      return {
        isError: true,
        content: [{ type: "text" as const, text: JSON.stringify({ error: err instanceof Error ? err.message : String(err) }) }],
      };
    }
  },
);

// ── check ────────────────────────────────────────────
server.tool(
  "check",
  "Check if your hardware can run a specific model — shows verdict (yes/maybe/no), best quantization, VRAM fit, and speed estimate",
  {
    model: z.string().describe("Model name, ID, or Ollama tag (e.g. 'llama3.1:8b', 'deepseek-coder-v2')"),
    quant: z.string().optional().describe("Specific quantization to check (e.g. 'Q4_K_M', 'Q8_0')"),
    host: z.string().optional().describe("Ollama API host URL"),
  },
  async ({ model: modelArg, quant, host }) => {
    try {
      clearHardwareCache();

      const hardware = await detectHardware();
      const model = resolveModel(modelArg);

      if (!model) {
        const suggestions = searchModels(modelArg).slice(0, 5);
        return {
          isError: true,
          content: [{
            type: "text" as const,
            text: JSON.stringify({
              error: "Model not found",
              query: modelArg,
              suggestions: suggestions.map((s) => ({ id: s.id, name: s.name, ollamaTag: s.ollamaTag })),
            }, null, 2),
          }],
        };
      }

      // Filter quantizations if specific quant requested
      let quants = model.quantizations;
      if (quant) {
        const match = quants.find((q) => q.name.toLowerCase() === quant.toLowerCase());
        if (match) quants = [match];
      }

      // Score all quantizations
      const scores = quants
        .map((q) => scoreModel(model, q, hardware))
        .sort((a, b) => b.compositeScore - a.compositeScore);

      const bestScore = scores[0];
      const verdict = deriveVerdict(bestScore.fitLevel);
      const availableVramMb = getAvailableVram(hardware);
      const pullCommand = model.ollamaTag ? `ollama pull ${model.ollamaTag}` : null;

      const result = {
        model: {
          id: model.id,
          name: model.name,
          provider: model.provider,
          parametersBillion: model.parametersBillion,
          contextWindow: model.contextWindow,
          qualityTier: model.qualityTier,
          ollamaTag: model.ollamaTag,
        },
        hardware: {
          gpu: hardware.primaryGpu?.model ?? null,
          vramMb: availableVramMb,
          ramMb: hardware.memory.totalMb,
        },
        verdict,
        bestQuantization: {
          name: bestScore.quantization.name,
          bitsPerWeight: bestScore.quantization.bitsPerWeight,
          vramMb: bestScore.quantization.vramMb,
          fitLevel: bestScore.fitLevel,
          fitRatio: Math.round(bestScore.fitRatio * 100) / 100,
          compositeScore: bestScore.compositeScore,
          speedEstimate: bestScore.speedEstimate,
        },
        allQuantizations: scores.map((s) => ({
          name: s.quantization.name,
          vramNeeded: s.quantization.vramMb,
          vramAvailable: availableVramMb,
          fitLevel: s.fitLevel,
          compositeScore: s.compositeScore,
          speedEstimate: s.speedEstimate,
        })),
        pullCommand,
      };

      return { content: [{ type: "text" as const, text: JSON.stringify(result, null, 2) }] };
    } catch (err) {
      return {
        isError: true,
        content: [{ type: "text" as const, text: JSON.stringify({ error: err instanceof Error ? err.message : String(err) }) }],
      };
    }
  },
);

// ── recommend ────────────────────────────────────────
server.tool(
  "recommend",
  "Get model recommendations for your current hardware — returns ranked list sorted by composite score (quality + quantization + tier + fit)",
  {
    category: CategoryEnum.optional().default("all").describe("Filter by model category"),
    top: z.number().int().min(1).max(50).optional().default(5).describe("Number of recommendations"),
    onlyFitting: z.boolean().optional().default(true).describe("Exclude models that cannot run on this hardware"),
    host: z.string().optional().describe("Ollama API host URL"),
  },
  async ({ category, top, onlyFitting }) => {
    try {
      clearHardwareCache();

      const hardware = await detectHardware();
      const recommendations = getRecommendations(hardware, { category, top, onlyFitting });

      const result = {
        hardware: {
          gpu: hardware.primaryGpu?.model ?? null,
          vramMb: getAvailableVram(hardware),
          ramMb: hardware.memory.totalMb,
        },
        recommendations: recommendations.map((r) => ({
          rank: r.rank,
          model: {
            id: r.score.model.id,
            name: r.score.model.name,
            provider: r.score.model.provider,
            parametersBillion: r.score.model.parametersBillion,
            contextWindow: r.score.model.contextWindow,
            qualityTier: r.score.model.qualityTier,
            categories: r.score.model.categories,
            ollamaTag: r.score.model.ollamaTag,
          },
          quantization: r.score.quantization.name,
          vramMb: r.score.quantization.vramMb,
          fitLevel: r.score.fitLevel,
          compositeScore: r.score.compositeScore,
          speedEstimate: r.score.speedEstimate,
          pullCommand: r.pullCommand,
        })),
      };

      return { content: [{ type: "text" as const, text: JSON.stringify(result, null, 2) }] };
    } catch (err) {
      return {
        isError: true,
        content: [{ type: "text" as const, text: JSON.stringify({ error: err instanceof Error ? err.message : String(err) }) }],
      };
    }
  },
);

// ── doctor ───────────────────────────────────────────
server.tool(
  "doctor",
  "System health check — scores your hardware for local LLM readiness (0-100), checks CPU/GPU/RAM/disk/runtimes, provides actionable suggestions",
  {
    host: z.string().optional().describe("Ollama API host URL"),
  },
  async ({ host }) => {
    try {
      clearHardwareCache();
      clearOllamaCache();
      const ollamaHost = resolveOllamaHost(host);

      const [hardware, runtimes] = await Promise.all([
        detectHardware(),
        detectAllRuntimes(ollamaHost),
      ]);

      const report = runDiagnostics(hardware, runtimes);

      return { content: [{ type: "text" as const, text: JSON.stringify(report, null, 2) }] };
    } catch (err) {
      return {
        isError: true,
        content: [{ type: "text" as const, text: JSON.stringify({ error: err instanceof Error ? err.message : String(err) }) }],
      };
    }
  },
);

// ── models ───────────────────────────────────────────
server.tool(
  "models",
  "Browse and search the model database — filter by name, category, or hardware compatibility",
  {
    search: z.string().optional().describe("Search query — matches model name, ID, or provider"),
    category: CategoryEnum.optional().default("all").describe("Filter by model category"),
    fits: z.boolean().optional().default(false).describe("Only show models that fit your hardware (triggers hardware scan)"),
    host: z.string().optional().describe("Ollama API host URL"),
  },
  async ({ search, category, fits }) => {
    try {
      // Resolve models from database
      let models;
      if (search) {
        models = searchModels(search);
      } else if (category !== "all") {
        models = filterByCategory(category as ModelCategory);
      } else {
        models = getAllModels();
      }

      // If --fits, scan hardware and score/filter
      if (fits) {
        clearHardwareCache();
        const hardware = await detectHardware();

        const scored = models
          .map((model) => {
            const scores = model.quantizations
              .map((q) => scoreModel(model, q, hardware))
              .filter((s) => s.fitLevel !== "cannot_run")
              .sort((a, b) => b.compositeScore - a.compositeScore);

            if (scores.length === 0) return null;
            const best = scores[0];

            return {
              id: model.id,
              name: model.name,
              provider: model.provider,
              parametersBillion: model.parametersBillion,
              categories: model.categories,
              quantization: best.quantization.name,
              vramMb: best.quantization.vramMb,
              ollamaTag: model.ollamaTag,
              fitLevel: best.fitLevel,
              compositeScore: best.compositeScore,
            };
          })
          .filter(Boolean);

        return {
          content: [{
            type: "text" as const,
            text: JSON.stringify({ count: scored.length, models: scored }, null, 2),
          }],
        };
      }

      // No hardware filter — show all with smallest quant info
      const result = models.map((m) => {
        const smallest = m.quantizations[0];
        return {
          id: m.id,
          name: m.name,
          provider: m.provider,
          parametersBillion: m.parametersBillion,
          categories: m.categories,
          quantization: smallest.name,
          vramMb: smallest.vramMb,
          ollamaTag: m.ollamaTag,
        };
      });

      return {
        content: [{
          type: "text" as const,
          text: JSON.stringify({ count: result.length, models: result }, null, 2),
        }],
      };
    } catch (err) {
      return {
        isError: true,
        content: [{ type: "text" as const, text: JSON.stringify({ error: err instanceof Error ? err.message : String(err) }) }],
      };
    }
  },
);

// ── monitor ──────────────────────────────────────────
server.tool(
  "monitor",
  "Take a one-shot snapshot of live hardware state — CPU/GPU utilization, VRAM usage, temperature, power, and active Ollama model with tokens/sec",
  {
    host: z.string().optional().describe("Ollama API host URL (default: http://127.0.0.1:11434)"),
  },
  async ({ host }) => {
    try {
      const ollamaHost = resolveOllamaHost(host);
      const monitor = new HardwareMonitor(ollamaHost);
      const snapshot = await monitor.takeSnapshot();

      return { content: [{ type: "text" as const, text: JSON.stringify(snapshot, null, 2) }] };
    } catch (err) {
      return {
        isError: true,
        content: [{ type: "text" as const, text: JSON.stringify({ error: err instanceof Error ? err.message : String(err) }) }],
      };
    }
  },
);

// ── Start ────────────────────────────────────────────
export async function startServer(): Promise<void> {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}
