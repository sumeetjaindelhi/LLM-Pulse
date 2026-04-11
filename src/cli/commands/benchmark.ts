import ora from "ora";
import { theme } from "../ui/colors.js";
import { sectionHeader } from "../ui/boxes.js";
import { toCsv } from "../ui/csv.js";
import { resolveOllamaHost } from "../../core/config.js";
import { pickOllamaModel } from "../utils/ollama-helpers.js";
import { BenchmarkLineSchema } from "../../core/api-schemas.js";
import type { OutputFormat } from "../../core/types.js";

interface BenchmarkOptions {
  model: string;
  rounds: number;
  format: OutputFormat;
  host?: string;
}

const TEST_PROMPTS = [
  "Explain what a hash table is in one paragraph.",
  "Write a Python function that checks if a string is a palindrome.",
  "What are the three laws of thermodynamics? Be brief.",
  "Translate this to French: The weather is beautiful today.",
  "What is the time complexity of merge sort and why?",
];

interface RoundResult {
  prompt: string;
  tokensGenerated: number;
  totalMs: number;
  tokensPerSec: number;
  ttftMs: number;
}

export async function benchmarkCommand(options: BenchmarkOptions): Promise<void> {
  const baseUrl = resolveOllamaHost(options.host);
  const isJson = options.format === "json";
  const isCsv = options.format === "csv";
  const silent = isJson || isCsv;

  // Check if Ollama is running
  const spinner = silent ? null : ora({ text: "Checking Ollama...", color: "cyan" }).start();

  let isRunning = false;
  try {
    const res = await fetch(`${baseUrl}/api/version`, {
      signal: AbortSignal.timeout(3000),
    });
    isRunning = res.ok;
  } catch {
    // not running
  }

  if (!isRunning) {
    spinner?.fail("Ollama is not running");
    if (!silent) {
      console.log(`\n  ${theme.fail("Ollama must be running for benchmarks.")}`);
      console.log(`  Start it with: ${theme.command("ollama serve")}`);
      console.log(`  Install from:  ${theme.command("https://ollama.com")}\n`);
    } else {
      console.log(isJson ? JSON.stringify({ error: "Ollama is not running" }) : "");
    }
    return;
  }

  // Determine which model to use
  let model = options.model;
  if (!model) {
    spinner && (spinner.text = "Finding a model to benchmark...");
    const picked = await pickOllamaModel(baseUrl);
    model = picked ?? "";
    if (!model) {
      spinner?.fail("No models available");
      if (!silent) {
        console.log(`\n  ${theme.warning("No models installed in Ollama.")}`);
        console.log(`  Pull one first: ${theme.command("ollama pull tinyllama")}\n`);
      } else {
        console.log(isJson ? JSON.stringify({ error: "No models installed" }) : "");
      }
      return;
    }
  }

  spinner?.succeed(`Benchmarking ${theme.highlight(model)}`);
  // The prompt library has a finite size. If the user asks for more rounds
  // than we have prompts, honestly tell them instead of silently running fewer.
  const effectiveRounds = Math.min(options.rounds, TEST_PROMPTS.length);
  if (!silent) {
    console.log(sectionHeader(`Inference Benchmark — ${model}`));
    if (options.rounds > TEST_PROMPTS.length) {
      console.log(
        `\n  ${theme.muted(`Note: --rounds capped at ${TEST_PROMPTS.length} (benchmark prompt library size).`)}`,
      );
    }
    console.log(`\n  Running ${effectiveRounds} rounds...\n`);
  }

  const results: RoundResult[] = [];
  const prompts = TEST_PROMPTS.slice(0, effectiveRounds);

  for (let i = 0; i < prompts.length; i++) {
    const roundSpinner = silent ? null : ora({
      text: `  Round ${i + 1}/${prompts.length}...`,
      color: "cyan",
    }).start();

    const result = await runInference(baseUrl, model, prompts[i]);

    if (result) {
      results.push(result);
      roundSpinner?.succeed(
        `  Round ${i + 1}: ${theme.number(`${result.tokensPerSec.toFixed(1)} tok/s`)}  TTFT: ${theme.number(`${result.ttftMs}ms`)}  Tokens: ${result.tokensGenerated}`,
      );
    } else {
      roundSpinner?.fail(`  Round ${i + 1}: failed`);
    }
  }

  if (results.length === 0) {
    if (!silent) console.log(`\n  ${theme.fail("All rounds failed. Check Ollama logs.")}\n`);
    else console.log(isJson ? JSON.stringify({ error: "All rounds failed" }) : "");
    return;
  }

  // Summary
  const avgTps = results.reduce((s, r) => s + r.tokensPerSec, 0) / results.length;
  const avgTtft = results.reduce((s, r) => s + r.ttftMs, 0) / results.length;
  const totalTokens = results.reduce((s, r) => s + r.tokensGenerated, 0);

  if (isJson) {
    console.log(JSON.stringify({
      model,
      rounds: results.length,
      avgTokensPerSec: Math.round(avgTps * 10) / 10,
      avgTtftMs: Math.round(avgTtft),
      totalTokens,
      results: results.map((r) => ({
        prompt: r.prompt.slice(0, 80),
        tokensGenerated: r.tokensGenerated,
        totalMs: r.totalMs,
        tokensPerSec: Math.round(r.tokensPerSec * 10) / 10,
        ttftMs: r.ttftMs,
      })),
    }, null, 2));
    return;
  }

  if (isCsv) {
    const headers = ["round", "prompt", "tokensGenerated", "totalMs", "tokensPerSec", "ttftMs"];
    const rows = results.map((r, i) => [
      i + 1, r.prompt.slice(0, 80), r.tokensGenerated, r.totalMs,
      Math.round(r.tokensPerSec * 10) / 10, r.ttftMs,
    ]);
    console.log(toCsv(headers, rows));
    return;
  }

  console.log(`\n  ${theme.subheader("Results")}:`);
  console.log(`  Avg tokens/sec:  ${theme.pass(avgTps.toFixed(1))}`);
  console.log(`  Avg TTFT:        ${theme.number(`${Math.round(avgTtft)}ms`)}`);
  console.log(`  Total tokens:    ${theme.number(String(totalTokens))}`);
  console.log(`  Rounds:          ${results.length}/${prompts.length}`);

  // Performance rating
  let rating: string;
  if (avgTps >= 40) rating = theme.pass("Excellent — fast interactive use");
  else if (avgTps >= 20) rating = theme.pass("Good — smooth for most tasks");
  else if (avgTps >= 10) rating = theme.warning("Moderate — usable but noticeable lag");
  else rating = theme.fail("Slow — consider a smaller model or quantization");

  console.log(`  Rating:          ${rating}\n`);
}

async function runInference(baseUrl: string, model: string, prompt: string): Promise<RoundResult | null> {
  try {
    const startTime = performance.now();
    let firstTokenTime: number | null = null;
    let tokensGenerated = 0;

    const res = await fetch(`${baseUrl}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        prompt,
        stream: true,
        options: { num_predict: 100 },
      }),
      signal: AbortSignal.timeout(60000),
    });

    if (!res.ok || !res.body) return null;

    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const text = decoder.decode(value, { stream: true });
      const lines = text.split("\n").filter(Boolean);

      for (const line of lines) {
        try {
          const json = BenchmarkLineSchema.parse(JSON.parse(line));
          if (json.response) {
            if (firstTokenTime === null) {
              firstTokenTime = performance.now();
            }
            tokensGenerated++;
          }
        } catch {
          // skip malformed lines
        }
      }
    }

    const endTime = performance.now();
    const totalMs = endTime - startTime;
    const ttftMs = firstTokenTime !== null ? firstTokenTime - startTime : totalMs;

    return {
      prompt,
      tokensGenerated,
      totalMs: Math.round(totalMs),
      tokensPerSec: tokensGenerated / (totalMs / 1000),
      ttftMs: Math.round(ttftMs),
    };
  } catch {
    return null;
  }
}
