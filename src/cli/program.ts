import { Command } from "commander";
import { VERSION } from "../core/constants.js";
import { scanCommand } from "./commands/scan.js";
import { doctorCommand } from "./commands/doctor.js";
import { modelsCommand } from "./commands/models.js";
import { monitorCommand } from "./commands/monitor.js";
import { benchmarkCommand } from "./commands/benchmark.js";
import type { ScanOptions, ModelCategory, OutputFormat } from "../core/types.js";

export function createProgram(): Command {
  const program = new Command();

  program
    .name("llm-pulse")
    .description("Zero-config CLI for monitoring your local LLM hardware, runtimes, and model compatibility")
    .version(VERSION);

  // Default action (no subcommand) runs scan
  program
    .option("-f, --format <format>", "Output format (table or json)", "table")
    .option("-c, --category <category>", "Filter by category (general, coding, reasoning, creative, multilingual)", "all")
    .option("-t, --top <n>", "Number of recommendations", "5")
    .option("-v, --verbose", "Show detailed output", false)
    .action(async (opts) => {
      const options: ScanOptions = {
        format: opts.format as OutputFormat,
        category: opts.category as ModelCategory | "all",
        top: parseInt(opts.top, 10),
        verbose: opts.verbose,
      };
      await scanCommand(options);
    });

  // Explicit scan subcommand (same as default)
  program
    .command("scan")
    .description("Full hardware scan + model recommendations")
    .option("-f, --format <format>", "Output format (table or json)", "table")
    .option("-c, --category <category>", "Filter by category", "all")
    .option("-t, --top <n>", "Number of recommendations", "5")
    .option("-v, --verbose", "Show detailed output", false)
    .action(async (opts) => {
      const options: ScanOptions = {
        format: opts.format as OutputFormat,
        category: opts.category as ModelCategory | "all",
        top: parseInt(opts.top, 10),
        verbose: opts.verbose,
      };
      await scanCommand(options);
    });

  // Doctor command
  program
    .command("doctor")
    .description("System health check with actionable advice")
    .option("-f, --format <format>", "Output format (table or json)", "table")
    .action(async (opts) => {
      await doctorCommand({ format: opts.format });
    });

  // Models command
  program
    .command("models")
    .description("Browse model database filtered for your hardware")
    .option("-s, --search <query>", "Search models by name")
    .option("-c, --category <category>", "Filter by category", "all")
    .option("--fits", "Only show models that fit your hardware", false)
    .option("-f, --format <format>", "Output format (table or json)", "table")
    .action(async (opts) => {
      await modelsCommand({
        search: opts.search,
        category: opts.category as ModelCategory | "all",
        fits: opts.fits,
        format: opts.format,
      });
    });

  // Monitor command
  program
    .command("monitor")
    .description("Live-updating system monitor (like htop for LLMs)")
    .action(async () => {
      await monitorCommand();
    });

  // Benchmark command
  program
    .command("benchmark")
    .description("Run a quick inference benchmark via Ollama")
    .option("-m, --model <model>", "Model to benchmark", "")
    .option("-r, --rounds <n>", "Number of test rounds", "3")
    .action(async (opts) => {
      await benchmarkCommand({
        model: opts.model,
        rounds: parseInt(opts.rounds, 10),
      });
    });

  return program;
}
