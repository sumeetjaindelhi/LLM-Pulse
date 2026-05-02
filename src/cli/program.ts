import { Command } from "commander";
import { VERSION } from "../core/constants.js";
import { loadConfig } from "../core/config.js";
import { scanCommand } from "./commands/scan.js";
import { doctorCommand } from "./commands/doctor.js";
import { modelsCommand } from "./commands/models.js";
import { benchmarkCommand } from "./commands/benchmark.js";
import { compareCommand } from "./commands/compare.js";
import { checkCommand } from "./commands/check.js";
import { quantAdviceCommand } from "./commands/quant-advice.js";
import { profileCommand } from "./commands/profile.js";
import { parseIntSafe } from "./utils/parse-int-safe.js";
import type { ScanOptions, ModelCategory, OutputFormat, CheckOptions, QuantAdviceOptions } from "../core/types.js";

export function createProgram(): Command {
  const config = loadConfig();
  const defaultFormat = config.defaultFormat ?? "table";
  const defaultCategory = config.defaultCategory ?? "all";
  const defaultTopNum = config.defaultTop ?? 5;
  const defaultTop = String(defaultTopNum);
  const DEFAULT_ROUNDS = 3;
  const DEFAULT_CONTEXT_SIZE = 2048;
  const DEFAULT_COMPARE_TOP = 3;

  const program = new Command();

  program
    .name("llm-pulse")
    .description("Zero-config CLI for monitoring your local LLM hardware, runtimes, and model compatibility")
    .version(VERSION)
    .enablePositionalOptions();

  // Default action (no subcommand) runs scan
  program
    .option("-f, --format <format>", "Output format (table, json, csv)", defaultFormat)
    .option("-c, --category <category>", "Filter by category (general, coding, reasoning, creative, multilingual)", defaultCategory)
    .option("-t, --top <n>", "Number of recommendations", defaultTop)
    .option("-H, --host <url>", "Ollama API host URL")
    .action(async (opts) => {
      const options: ScanOptions = {
        format: opts.format as OutputFormat,
        category: opts.category as ModelCategory | "all",
        top: parseIntSafe(opts.top, defaultTopNum, "top"),
        host: opts.host,
      };
      await scanCommand(options);
    });

  // Explicit scan subcommand (same as default)
  program
    .command("scan")
    .description("Full hardware scan + model recommendations")
    .option("-f, --format <format>", "Output format (table, json, csv)", defaultFormat)
    .option("-c, --category <category>", "Filter by category", defaultCategory)
    .option("-t, --top <n>", "Number of recommendations", defaultTop)
    .option("-H, --host <url>", "Ollama API host URL")
    .action(async (opts) => {
      const options: ScanOptions = {
        format: opts.format as OutputFormat,
        category: opts.category as ModelCategory | "all",
        top: parseIntSafe(opts.top, defaultTopNum, "top"),
        host: opts.host,
      };
      await scanCommand(options);
    });

  // Doctor command
  program
    .command("doctor")
    .description("System health check with actionable advice")
    .option("-f, --format <format>", "Output format (table, json, csv)", defaultFormat)
    .option("--fix", "Automatically apply available fixes", false)
    .option("-H, --host <url>", "Ollama API host URL")
    .action(async (opts) => {
      await doctorCommand({ format: opts.format, fix: opts.fix, host: opts.host });
    });

  // Models command
  program
    .command("models")
    .description("Browse model database filtered for your hardware")
    .option("-s, --search <query>", "Search models by name")
    .option("-c, --category <category>", "Filter by category", defaultCategory)
    .option("--fits", "Only show models that fit your hardware", false)
    .option("--live", "Include live models from Ollama", false)
    .option("--installed", "Show only installed Ollama models", false)
    .option("--library", "Include full Ollama library catalog (ollama.com/library)", false)
    .option("--refresh", "Bypass cache and re-scrape ollama.com/library", false)
    .option("-f, --format <format>", "Output format (table, json, csv)", defaultFormat)
    .option("-H, --host <url>", "Ollama API host URL")
    .action(async (opts) => {
      await modelsCommand({
        search: opts.search,
        category: opts.category as ModelCategory | "all",
        fits: opts.fits,
        live: opts.live || opts.installed || opts.library || opts.refresh,
        installed: opts.installed,
        library: opts.library || opts.refresh,
        refresh: opts.refresh,
        format: opts.format,
        host: opts.host,
      });
    });

  // Compare command
  program
    .command("compare [models...]")
    .description("Compare models side-by-side against your hardware")
    .option("-f, --format <format>", "Output format (table, json, csv)", defaultFormat)
    .option("-c, --category <category>", "Auto-pick top models from category", defaultCategory)
    .option("-t, --top <n>", "Number of models to compare (with --category)", "3")
    .option("-q, --quant <quant>", "Force specific quantization (e.g. Q4_K_M)")
    .option("-H, --host <url>", "Ollama API host URL")
    .action(async (models: string[], opts) => {
      await compareCommand(models, {
        format: opts.format as OutputFormat,
        category: opts.category as ModelCategory | "all",
        top: parseIntSafe(opts.top, DEFAULT_COMPARE_TOP, "top"),
        quant: opts.quant,
        host: opts.host,
      });
    });

  // Check command
  program
    .command("check <model>")
    .description("Check if your hardware can run a specific model")
    .option("-q, --quant <name>", "Check specific quantization (e.g. Q4_K_M)")
    .option("-f, --format <format>", "Output format (table, json, csv)", defaultFormat)
    .option("-v, --verbose", "Show extra details", false)
    .option("-H, --host <url>", "Ollama API host URL")
    .action(async (model: string, opts) => {
      const options: CheckOptions = {
        quant: opts.quant,
        format: opts.format as OutputFormat,
        verbose: opts.verbose,
        host: opts.host,
      };
      await checkCommand(model, options);
    });

  // Quant-advice command — quality-vs-VRAM tradeoff picker for a model.
  program
    .command("quant-advice <model>")
    .description("Which quantization should I use? Quality-vs-VRAM tradeoff table with a sweet-spot pick for your hardware")
    .option("-f, --format <format>", "Output format (table, json, csv)", defaultFormat)
    .option("-v, --verbose", "Show extra details", false)
    .action(async (model: string, opts) => {
      const options: QuantAdviceOptions = {
        format: opts.format as OutputFormat,
        verbose: opts.verbose,
      };
      await quantAdviceCommand(model, options);
    });

  // Monitor command
  program
    .command("monitor")
    .description("Live-updating system monitor (like htop for LLMs)")
    .option("-H, --host <url>", "Ollama API host URL")
    .action(async (opts) => {
      const { monitorCommand } = await import("./commands/monitor.js");
      await monitorCommand({ host: opts.host });
    });

  // Benchmark command
  program
    .command("benchmark")
    .description("Run a quick inference benchmark via Ollama")
    .option("-m, --model <model>", "Model to benchmark", "")
    .option("-r, --rounds <n>", "Number of test rounds", "3")
    .option("-f, --format <format>", "Output format (table, json, csv)", defaultFormat)
    .option("-H, --host <url>", "Ollama API host URL")
    .action(async (opts) => {
      await benchmarkCommand({
        model: opts.model,
        rounds: parseIntSafe(opts.rounds, DEFAULT_ROUNDS, "rounds"),
        format: opts.format as OutputFormat,
        host: opts.host,
      });
    });

  // Profile command
  program
    .command("profile")
    .description("Run inference with hardware profiling (latency, VRAM, GPU timeline)")
    .option("-m, --model <model>", "Model to profile", "")
    .option("-p, --prompt <prompt>", "Custom prompt (default: short/medium/long set)")
    .option("-c, --context-size <n>", "Context size", "2048")
    .option("-f, --format <format>", "Output format (table, json, csv)", defaultFormat)
    .option("-H, --host <url>", "Ollama API host URL")
    .action(async (opts) => {
      await profileCommand({
        model: opts.model,
        prompt: opts.prompt,
        contextSize: parseIntSafe(opts.contextSize, DEFAULT_CONTEXT_SIZE, "context-size"),
        format: opts.format as OutputFormat,
        host: opts.host,
      });
    });

  return program;
}
