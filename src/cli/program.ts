import { Command, InvalidArgumentError, Option } from "commander";
import { VERSION } from "../core/constants.js";
import { CATEGORY_FILTERS, OUTPUT_FORMATS, loadConfig } from "../core/config.js";
import { scanCommand } from "./commands/scan.js";
import { doctorCommand } from "./commands/doctor.js";
import { modelsCommand } from "./commands/models.js";
import { benchmarkCommand } from "./commands/benchmark.js";
import { compareCommand } from "./commands/compare.js";
import { checkCommand } from "./commands/check.js";
import { quantAdviceCommand } from "./commands/quant-advice.js";
import { optimizeCommand } from "./commands/optimize.js";
import { contextFitCommand } from "./commands/context-fit.js";
import { reportCommand } from "./commands/report.js";
import { profileCommand } from "./commands/profile.js";
import type { ScanOptions, ModelCategory, OutputFormat, CheckOptions, QuantAdviceOptions, OptimizeOptions, ContextFitOptions } from "../core/types.js";

// Commander's argParser for numeric flags. Only plain decimal digits are accepted:
// parseInt would silently read "50k" as 50 and "1e6" as 1.
function wholeNumberAtLeast(min: number): (value: string) => number {
  return (value) => {
    const n = /^\d+$/.test(value) ? Number(value) : NaN;
    if (!Number.isSafeInteger(n) || n < min) {
      throw new InvalidArgumentError(`Expected a whole number of at least ${min}.`);
    }
    return n;
  };
}

export function createProgram(): Command {
  const config = loadConfig();
  const defaultFormat = config.defaultFormat ?? "table";
  const defaultCategory = config.defaultCategory ?? "all";
  const defaultTop = config.defaultTop ?? 5;
  const DEFAULT_ROUNDS = 3;
  const DEFAULT_CONTEXT_SIZE = 2048;
  const DEFAULT_COMPARE_TOP = 3;
  const DEFAULT_RESPONSE_TOKENS = 512;

  const formatOption = () =>
    new Option("-f, --format <format>", "Output format").choices(OUTPUT_FORMATS).default(defaultFormat);
  const categoryOption = (description: string) =>
    new Option("-c, --category <category>", description).choices(CATEGORY_FILTERS).default(defaultCategory);

  const program = new Command();

  program
    .name("llm-pulse")
    .description("Zero-config CLI for monitoring your local LLM hardware, runtimes, and model compatibility")
    .version(VERSION)
    .enablePositionalOptions();

  // Default action (no subcommand) runs scan
  program
    .addOption(formatOption())
    .addOption(categoryOption("Filter by category"))
    .option("-t, --top <n>", "Number of recommendations", wholeNumberAtLeast(1), defaultTop)
    .option("-H, --host <url>", "Ollama API host URL")
    .action(async (opts) => {
      const options: ScanOptions = {
        format: opts.format as OutputFormat,
        category: opts.category as ModelCategory | "all",
        top: opts.top,
        host: opts.host,
      };
      await scanCommand(options);
    });

  // Explicit scan subcommand (same as default)
  program
    .command("scan")
    .description("Full hardware scan + model recommendations")
    .addOption(formatOption())
    .addOption(categoryOption("Filter by category"))
    .option("-t, --top <n>", "Number of recommendations", wholeNumberAtLeast(1), defaultTop)
    .option("-H, --host <url>", "Ollama API host URL")
    .action(async (opts) => {
      const options: ScanOptions = {
        format: opts.format as OutputFormat,
        category: opts.category as ModelCategory | "all",
        top: opts.top,
        host: opts.host,
      };
      await scanCommand(options);
    });

  // Doctor command
  program
    .command("doctor")
    .description("System health check with actionable advice")
    .addOption(formatOption())
    .option("--fix", "Automatically apply available fixes", false)
    .option("--dry-run", "Preview the exact commands --fix would run, without executing them", false)
    .option("-H, --host <url>", "Ollama API host URL")
    .action(async (opts) => {
      await doctorCommand({ format: opts.format, fix: opts.fix, dryRun: opts.dryRun, host: opts.host });
    });

  // Models command
  program
    .command("models")
    .description("Browse model database filtered for your hardware")
    .option("-s, --search <query>", "Search models by name")
    .addOption(categoryOption("Filter by category"))
    .option("--fits", "Only show models that fit your hardware", false)
    .option("--live", "Include live models from Ollama", false)
    .option("--installed", "Show only installed Ollama models", false)
    .option("--library", "Include full Ollama library catalog (ollama.com/library)", false)
    .option("--refresh", "Bypass cache and re-scrape ollama.com/library", false)
    .addOption(formatOption())
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
    .addOption(formatOption())
    .addOption(categoryOption("Auto-pick top models from category"))
    .option("-t, --top <n>", "Number of models to compare (with --category)", wholeNumberAtLeast(2), DEFAULT_COMPARE_TOP)
    .option("-q, --quant <quant>", "Force specific quantization (e.g. Q4_K_M)")
    .option("-H, --host <url>", "Ollama API host URL")
    .action(async (models: string[], opts) => {
      await compareCommand(models, {
        format: opts.format as OutputFormat,
        category: opts.category as ModelCategory | "all",
        top: opts.top,
        quant: opts.quant,
        host: opts.host,
      });
    });

  // Check command
  program
    .command("check <model>")
    .description("Check if your hardware can run a specific model")
    .option("-q, --quant <name>", "Check specific quantization (e.g. Q4_K_M)")
    .addOption(formatOption())
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
    .addOption(formatOption())
    .option("-v, --verbose", "Show extra details", false)
    .action(async (model: string, opts) => {
      const options: QuantAdviceOptions = {
        format: opts.format as OutputFormat,
        verbose: opts.verbose,
      };
      await quantAdviceCommand(model, options);
    });

  // Optimize command — recommend tuned Ollama runtime parameters for a model.
  program
    .command("optimize <model>")
    .description("Recommend tuned Ollama runtime parameters (num_ctx, num_gpu, num_thread, num_batch) for your hardware")
    .option("-q, --quant <name>", "Pin a specific quantization instead of the sweet-spot pick")
    .addOption(formatOption())
    .option("-v, --verbose", "Show extra details", false)
    .action(async (model: string, opts) => {
      const options: OptimizeOptions = {
        quant: opts.quant,
        format: opts.format as OutputFormat,
        verbose: opts.verbose,
      };
      await optimizeCommand(model, options);
    });

  // Context-fit command — will a prompt of N tokens fit on this hardware?
  program
    .command("context-fit <model>")
    .description("Check whether a prompt of N tokens fits a model's context on your hardware (native window + hardware-afforded KV cache)")
    .requiredOption("--prompt-tokens <n>", "Number of prompt/input tokens to check", wholeNumberAtLeast(0))
    .option("--response-tokens <n>", "Tokens to reserve for the model's response", wholeNumberAtLeast(0), DEFAULT_RESPONSE_TOKENS)
    .option("-q, --quant <name>", "Pin a specific quantization instead of the sweet-spot pick")
    .addOption(formatOption())
    .action(async (model: string, opts) => {
      const options: ContextFitOptions = {
        promptTokens: opts.promptTokens,
        responseTokens: opts.responseTokens,
        quant: opts.quant,
        format: opts.format as OutputFormat,
      };
      await contextFitCommand(model, options);
    });

  // Report command — paste-ready markdown system report for sharing.
  program
    .command("report")
    .description("Paste-ready markdown system report (hardware, runtimes, health, recommended models) for GitHub / Reddit / Discord")
    .addOption(categoryOption("Filter by category"))
    .option("-t, --top <n>", "Number of recommendations", wholeNumberAtLeast(1), defaultTop)
    .option("-H, --host <url>", "Ollama API host URL")
    .action(async (opts) => {
      await reportCommand({
        category: opts.category as ModelCategory | "all",
        top: opts.top,
        host: opts.host,
      });
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
    .option("-r, --rounds <n>", "Number of test rounds", wholeNumberAtLeast(1), DEFAULT_ROUNDS)
    .addOption(formatOption())
    .option("-H, --host <url>", "Ollama API host URL")
    .action(async (opts) => {
      await benchmarkCommand({
        model: opts.model,
        rounds: opts.rounds,
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
    .option("-c, --context-size <n>", "Context size", wholeNumberAtLeast(1), DEFAULT_CONTEXT_SIZE)
    .addOption(formatOption())
    .option("-H, --host <url>", "Ollama API host URL")
    .action(async (opts) => {
      await profileCommand({
        model: opts.model,
        prompt: opts.prompt,
        contextSize: opts.contextSize,
        format: opts.format as OutputFormat,
        host: opts.host,
      });
    });

  return program;
}
