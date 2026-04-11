import ora from "ora";
import { execa } from "execa";
import si from "systeminformation";
import { theme } from "../ui/colors.js";
import { sectionHeader } from "../ui/boxes.js";
import { toCsv } from "../ui/csv.js";
import { resolveOllamaHost } from "../../core/config.js";
import { pickOllamaModel } from "../utils/ollama-helpers.js";
import { BenchmarkLineSchema } from "../../core/api-schemas.js";
import type { ProfileOptions, ProfileResult, HardwareSnapshot, OutputFormat } from "../../core/types.js";

const DEFAULT_PROMPTS = [
  "What is 2+2?",
  "Explain what a hash table is and describe its time complexity for common operations including insertion, deletion, and lookup.",
  "Write a detailed essay about the history of computing, starting from Charles Babbage's Analytical Engine through to modern quantum computers. Cover key milestones including ENIAC, the transistor, integrated circuits, personal computers, the internet, and artificial intelligence. Discuss how each innovation built upon its predecessors.",
];

const PROMPT_LABELS = ["short", "medium", "long"];

export async function profileCommand(options: ProfileOptions): Promise<void> {
  const baseUrl = resolveOllamaHost(options.host);
  const isJson = options.format === "json";
  const isCsv = options.format === "csv";
  const silent = isJson || isCsv;

  const spinner = silent ? null : ora({ text: "Checking Ollama...", color: "cyan" }).start();

  // Check Ollama is running
  let isRunning = false;
  try {
    const res = await fetch(`${baseUrl}/api/version`, { signal: AbortSignal.timeout(3000) });
    isRunning = res.ok;
  } catch { /* not running */ }

  if (!isRunning) {
    spinner?.fail("Ollama is not running");
    if (!silent) {
      console.log(`\n  ${theme.fail("Ollama must be running for profiling.")}`);
      console.log(`  Start it with: ${theme.command("ollama serve")}\n`);
    } else {
      console.log(JSON.stringify({ error: "Ollama is not running" }));
    }
    return;
  }

  // Determine model
  let model = options.model;
  if (!model) {
    spinner && (spinner.text = "Finding a model to profile...");
    const picked = await pickOllamaModel(baseUrl);
    model = picked ?? "";
    if (!model) {
      spinner?.fail("No models available");
      if (!silent) {
        console.log(`\n  ${theme.warning("No models installed in Ollama.")}`);
        console.log(`  Pull one first: ${theme.command("ollama pull tinyllama")}\n`);
      } else {
        console.log(JSON.stringify({ error: "No models installed" }));
      }
      return;
    }
  }

  spinner?.succeed(`Profiling ${theme.highlight(model)}`);

  // Determine prompts
  const prompts: { label: string; text: string }[] = [];
  if (options.prompt) {
    prompts.push({ label: "custom", text: options.prompt });
  } else {
    for (let i = 0; i < DEFAULT_PROMPTS.length; i++) {
      prompts.push({ label: PROMPT_LABELS[i], text: DEFAULT_PROMPTS[i] });
    }
  }

  if (!silent) {
    console.log(sectionHeader(`Inference Profile — ${model}`));
    console.log(`\n  Running ${prompts.length} prompt(s) with hardware monitoring...\n`);
  }

  const results: ProfileResult[] = [];

  for (let i = 0; i < prompts.length; i++) {
    const { label, text } = prompts[i];
    const roundSpinner = silent ? null : ora({ text: `  Prompt ${i + 1}/${prompts.length} (${label})...`, color: "cyan" }).start();

    const result = await runProfiledInference(baseUrl, model, text, options.contextSize);

    if (result) {
      results.push(result);
      roundSpinner?.succeed(
        `  ${label}: ${theme.number(`${result.tokensPerSec.toFixed(1)} tok/s`)}  TTFT: ${theme.number(`${result.ttftMs}ms`)}  Peak VRAM: ${result.peakVramMb !== null ? theme.number(`${result.peakVramMb} MB`) : theme.muted("n/a")}`,
      );
    } else {
      roundSpinner?.fail(`  ${label}: failed`);
    }
  }

  if (results.length === 0) {
    if (!silent) console.log(`\n  ${theme.fail("All prompts failed.")}\n`);
    else console.log(JSON.stringify({ error: "All prompts failed" }));
    return;
  }

  outputResults(results, options.format);
}

async function runProfiledInference(
  baseUrl: string,
  model: string,
  prompt: string,
  contextSize: number,
): Promise<ProfileResult | null> {
  try {
    const snapshots: HardwareSnapshot[] = [];
    let currentPhase: HardwareSnapshot["phase"] = "idle";
    let stopPolling = false;

    // Take idle snapshot
    const idleSnap = await takeSnapshot("idle");
    snapshots.push(idleSnap);

    // Start hardware polling
    const pollInterval = setInterval(async () => {
      if (stopPolling) return;
      const snap = await takeSnapshot(currentPhase);
      snapshots.push(snap);
    }, 500);

    const startTime = performance.now();
    let firstTokenTime: number | null = null;
    let tokensGenerated = 0;

    currentPhase = "prompt";

    const res = await fetch(`${baseUrl}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        prompt,
        stream: true,
        options: { num_ctx: contextSize, num_predict: 150 },
      }),
      signal: AbortSignal.timeout(120000),
    });

    if (!res.ok || !res.body) {
      stopPolling = true;
      clearInterval(pollInterval);
      return null;
    }

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
              currentPhase = "generation";
            }
            tokensGenerated++;
          }
        } catch {
          // skip malformed lines
        }
      }
    }

    const endTime = performance.now();
    currentPhase = "complete";

    // Take final snapshot
    const finalSnap = await takeSnapshot("complete");
    snapshots.push(finalSnap);

    stopPolling = true;
    clearInterval(pollInterval);

    const totalMs = endTime - startTime;
    const ttftMs = firstTokenTime !== null ? firstTokenTime - startTime : totalMs;
    const promptProcessMs = firstTokenTime !== null ? firstTokenTime - startTime : totalMs;
    const generationMs = firstTokenTime !== null ? endTime - firstTokenTime : 0;

    // Compute peak VRAM
    let peakVramMb: number | null = null;
    for (const s of snapshots) {
      if (s.gpuVramUsedMb !== null) {
        if (peakVramMb === null || s.gpuVramUsedMb > peakVramMb) {
          peakVramMb = s.gpuVramUsedMb;
        }
      }
    }

    // Compute average GPU utilization per phase
    const avgGpuByPhase: Record<string, number | null> = {};
    for (const phase of ["idle", "prompt", "generation", "complete"] as const) {
      const phaseSnaps = snapshots.filter((s) => s.phase === phase && s.gpuUtilPercent !== null);
      if (phaseSnaps.length > 0) {
        avgGpuByPhase[phase] = Math.round(phaseSnaps.reduce((sum, s) => sum + s.gpuUtilPercent!, 0) / phaseSnaps.length);
      } else {
        avgGpuByPhase[phase] = null;
      }
    }

    return {
      model,
      prompt,
      contextSize,
      ttftMs: Math.round(ttftMs),
      promptProcessMs: Math.round(promptProcessMs),
      generationMs: Math.round(generationMs),
      totalMs: Math.round(totalMs),
      tokensGenerated,
      // Guard against div-by-zero when generation finishes in a single chunk
      // (generationMs === 0). Infinity would serialize unpredictably.
      tokensPerSec:
        tokensGenerated > 0 && generationMs > 0
          ? tokensGenerated / (generationMs / 1000)
          : 0,
      peakVramMb,
      avgGpuByPhase,
      snapshots,
    };
  } catch {
    return null;
  }
}

async function takeSnapshot(phase: HardwareSnapshot["phase"]): Promise<HardwareSnapshot> {
  const snap: HardwareSnapshot = {
    timestampMs: Date.now(),
    phase,
    gpuUtilPercent: null,
    gpuVramUsedMb: null,
    gpuVramTotalMb: null,
    gpuTempCelsius: null,
    gpuPowerWatt: null,
    cpuPercent: null,
    ramUsedMb: null,
  };

  try {
    const [cpu, mem, gpu] = await Promise.all([
      si.currentLoad().catch(() => null),
      si.mem().catch(() => null),
      pollGpuSnapshot(),
    ]);

    if (cpu) snap.cpuPercent = Math.round(cpu.currentLoad);
    if (mem) {
      // Match detectMemory()/pollMemory() convention: exclude reclaimable
      // buffcache from "used" so macOS doesn't always look like 98%.
      const totalMb = Math.round(mem.total / (1024 * 1024));
      const availableMb = Math.round(mem.available / (1024 * 1024));
      snap.ramUsedMb = Math.max(0, totalMb - availableMb);
    }
    if (gpu) {
      snap.gpuUtilPercent = gpu.percent;
      snap.gpuVramUsedMb = gpu.vramUsedMb;
      snap.gpuVramTotalMb = gpu.vramTotalMb;
      snap.gpuTempCelsius = gpu.temp;
      snap.gpuPowerWatt = gpu.powerWatt;
    }
  } catch {
    // return partial snapshot
  }

  return snap;
}

async function pollGpuSnapshot(): Promise<{
  percent: number | null;
  vramUsedMb: number | null;
  vramTotalMb: number | null;
  temp: number | null;
  powerWatt: number | null;
} | null> {
  try {
    const { stdout } = await execa("nvidia-smi", [
      "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total,power.draw",
      "--format=csv,noheader,nounits",
    ], { timeout: 3000 });
    const parts = stdout.trim().split(",").map((s) => s.trim());
    // Explicit NaN check instead of `|| null` so legitimate 0 values (idle GPU,
    // 0 W power on headless setups) are preserved rather than dropped.
    const toNum = (s: string | undefined): number | null => {
      if (s === undefined) return null;
      const n = parseInt(s, 10);
      return Number.isNaN(n) ? null : n;
    };
    const toFloat = (s: string | undefined): number | null => {
      if (s === undefined) return null;
      const n = parseFloat(s);
      return Number.isNaN(n) ? null : n;
    };
    return {
      percent: toNum(parts[0]),
      temp: toNum(parts[1]),
      vramUsedMb: toNum(parts[2]),
      vramTotalMb: toNum(parts[3]),
      powerWatt: toFloat(parts[4]),
    };
  } catch {
    return null;
  }
}

function outputResults(results: ProfileResult[], format: OutputFormat): void {
  if (format === "json") {
    const output = results.map((r) => ({
      model: r.model,
      prompt: r.prompt.slice(0, 80) + (r.prompt.length > 80 ? "..." : ""),
      contextSize: r.contextSize,
      ttftMs: r.ttftMs,
      promptProcessMs: r.promptProcessMs,
      generationMs: r.generationMs,
      totalMs: r.totalMs,
      tokensGenerated: r.tokensGenerated,
      tokensPerSec: Math.round(r.tokensPerSec * 10) / 10,
      peakVramMb: r.peakVramMb,
      avgGpuByPhase: r.avgGpuByPhase,
      snapshotCount: r.snapshots.length,
    }));
    console.log(JSON.stringify(output, null, 2));
    return;
  }

  if (format === "csv") {
    const headers = [
      "model", "prompt", "contextSize", "ttftMs", "promptProcessMs",
      "generationMs", "totalMs", "tokensGenerated", "tokensPerSec",
      "peakVramMb", "avgGpuIdle", "avgGpuPrompt", "avgGpuGeneration",
    ];
    const rows = results.map((r) => [
      r.model,
      r.prompt.slice(0, 80),
      r.contextSize,
      r.ttftMs,
      r.promptProcessMs,
      r.generationMs,
      r.totalMs,
      r.tokensGenerated,
      Math.round(r.tokensPerSec * 10) / 10,
      r.peakVramMb,
      r.avgGpuByPhase["idle"],
      r.avgGpuByPhase["prompt"],
      r.avgGpuByPhase["generation"],
    ]);
    console.log(toCsv(headers, rows));
    return;
  }

  // Table output
  console.log();
  for (const r of results) {
    const promptPreview = r.prompt.length > 60 ? r.prompt.slice(0, 57) + "..." : r.prompt;
    console.log(`  ${theme.subheader("Prompt:")} ${theme.muted(promptPreview)}`);
    console.log(`  ${theme.muted("Timing:")}`);
    console.log(`    TTFT:              ${theme.number(`${r.ttftMs}ms`)}`);
    console.log(`    Prompt processing: ${theme.number(`${r.promptProcessMs}ms`)}`);
    console.log(`    Generation:        ${theme.number(`${r.generationMs}ms`)}`);
    console.log(`    Total:             ${theme.number(`${r.totalMs}ms`)}`);
    console.log(`  ${theme.muted("Throughput:")}`);
    console.log(`    Tokens generated:  ${theme.number(String(r.tokensGenerated))}`);
    console.log(`    Tokens/sec:        ${theme.pass(`${r.tokensPerSec.toFixed(1)}`)}`);
    console.log(`  ${theme.muted("Memory:")}`);
    console.log(`    Peak VRAM:         ${r.peakVramMb !== null ? theme.number(`${r.peakVramMb} MB`) : theme.muted("n/a")}`);
    console.log(`  ${theme.muted("Avg GPU % by phase:")}`);
    for (const phase of ["idle", "prompt", "generation"] as const) {
      const val = r.avgGpuByPhase[phase];
      console.log(`    ${phase.padEnd(15)} ${val !== null ? theme.number(`${val}%`) : theme.muted("n/a")}`);
    }
    console.log();
  }
}
