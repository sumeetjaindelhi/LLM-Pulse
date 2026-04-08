import ora from "ora";
import Table from "cli-table3";
import { detectHardware } from "../../hardware/index.js";
import { scoreModel, deriveVerdict, getAvailableVram } from "../../analysis/scorer.js";
import { resolveModel, searchModels } from "../../models/database.js";
import { titleBox, sectionHeader, keyValue, subLine } from "../ui/boxes.js";
import { theme } from "../ui/colors.js";
import { fitBadge } from "../ui/badges.js";
import { formatMb } from "../ui/progress.js";
import { toCsv } from "../ui/csv.js";
import type {
  ModelEntry,
  ModelScore,
  HardwareProfile,
  Verdict,
  CheckOptions,
} from "../../core/types.js";

function formatContext(ctx: number): string {
  if (ctx >= 1_000_000) return `${(ctx / 1_000_000).toFixed(1)}M tokens`;
  if (ctx >= 1_000) return `${Math.round(ctx / 1_000)}K tokens`;
  return `${ctx} tokens`;
}

function verdictLine(verdict: Verdict, modelName: string): string {
  switch (verdict) {
    case "yes":
      return `${theme.pass("✓ YES")} ${theme.pass(`— You can run ${modelName}`)}`;
    case "maybe":
      return `${theme.warning("⚠ MAYBE")} ${theme.warning(`— ${modelName} might run with limitations`)}`;
    case "no":
      return `${theme.fail("✗ NO")} ${theme.fail(`— ${modelName} won't fit on your hardware`)}`;
  }
}

function quantTable(scores: ModelScore[], bestIdx: number, availableVramMb: number): string {
  const table = new Table({
    chars: {
      top: "", "top-mid": "", "top-left": "", "top-right": "",
      bottom: "", "bottom-mid": "", "bottom-left": "", "bottom-right": "",
      left: "  ", "left-mid": "", mid: "", "mid-mid": "",
      right: "", "right-mid": "", middle: "  ",
    },
    style: { "padding-left": 0, "padding-right": 0, head: ["gray"] },
  });

  table.push(["Quant", "VRAM", "Available", "Fit", "Speed", "Score"]);

  for (let i = 0; i < scores.length; i++) {
    const s = scores[i];
    const marker = i === bestIdx ? theme.pass("★ ") : "  ";
    const quantName = `${marker}${s.quantization.name}`;
    const vramNeeded = formatMb(s.quantization.vramMb);
    const vramAvail = formatMb(availableVramMb);
    const fit = fitBadge(s.fitLevel);
    const speed = s.speedEstimate === "fast" ? theme.pass(s.speedEstimate)
      : s.speedEstimate === "moderate" ? theme.warning(s.speedEstimate)
      : theme.fail(s.speedEstimate);
    const score = s.fitLevel === "cannot_run"
      ? theme.fail("—")
      : theme.number(String(s.compositeScore));

    table.push([quantName, vramNeeded, vramAvail, fit, speed, score]);
  }

  return table.toString();
}

export async function checkCommand(
  modelArg: string,
  options: CheckOptions,
): Promise<void> {
  const isJson = options.format === "json";
  const isCsv = options.format === "csv";
  const silent = isJson || isCsv;

  // 1. Detect hardware
  const spinner = silent ? null : ora({ text: "Detecting hardware...", color: "cyan" }).start();
  const hardware = await detectHardware();
  spinner?.succeed("Hardware detected");

  // 2. Resolve model
  const model = resolveModel(modelArg);

  if (!model) {
    const suggestions = searchModels(modelArg).slice(0, 5);
    if (silent) {
      console.log(JSON.stringify({
        error: "Model not found",
        query: modelArg,
        suggestions: suggestions.map((s) => ({ id: s.id, name: s.name, ollamaTag: s.ollamaTag })),
      }, null, 2));
      return;
    }

    console.log(`\n  ${theme.fail("✗")} Model not found: ${theme.value(modelArg)}`);
    if (suggestions.length > 0) {
      console.log(`  ${theme.muted("Did you mean:")}`);
      for (const s of suggestions) {
        const tag = s.ollamaTag ? theme.muted(` (${s.ollamaTag})`) : "";
        console.log(`    ${theme.muted("•")} ${s.name}${tag}`);
      }
    }
    console.log(`  ${theme.muted("Browse all models:")} ${theme.command("llm-pulse models")}`);
    return;
  }

  // 3. Filter quantizations
  let quants = model.quantizations;
  let quantWarning: string | null = null;
  if (options.quant) {
    const match = quants.find((q) => q.name.toLowerCase() === options.quant!.toLowerCase());
    if (match) {
      quants = [match];
    } else {
      quantWarning = `Quantization ${options.quant} not available for ${model.name}, showing all`;
    }
  }

  // 4. Score all quantizations
  const scores = quants
    .map((q) => scoreModel(model, q, hardware))
    .sort((a, b) => b.compositeScore - a.compositeScore);

  // 5. Best quantization & verdict
  const bestScore = scores[0]; // Already sorted by compositeScore descending
  const verdict = deriveVerdict(bestScore.fitLevel);
  const bestIdx = verdict !== "no" ? 0 : -1; // No star marker when nothing fits
  const availableVramMb = getAvailableVram(hardware);
  const pullCommand = model.ollamaTag ? `ollama pull ${model.ollamaTag}` : null;

  // 6. Output
  if (isJson) {
    outputJson(model, hardware, scores, bestScore, verdict, availableVramMb, pullCommand);
  } else if (isCsv) {
    outputCsv(scores, availableVramMb, bestIdx);
  } else {
    outputTable(model, hardware, scores, bestScore, bestIdx, verdict, availableVramMb, pullCommand, quantWarning, options.verbose);
  }
}

function outputTable(
  model: ModelEntry,
  hardware: HardwareProfile,
  scores: ModelScore[],
  bestScore: ModelScore,
  bestIdx: number,
  verdict: Verdict,
  availableVramMb: number,
  pullCommand: string | null,
  quantWarning: string | null,
  verbose: boolean,
): void {
  const lines: string[] = [];

  // Verdict headline
  lines.push("");
  lines.push(`  ${verdictLine(verdict, model.name)}`);

  // Model info
  lines.push(sectionHeader("Model"));
  lines.push(keyValue("Name", model.name));
  lines.push(keyValue("By", model.provider));
  lines.push(keyValue("Param", `${model.parametersBillion}B`));
  lines.push(keyValue("Ctx", formatContext(model.contextWindow)));
  if (verbose) {
    lines.push(keyValue("Tier", model.qualityTier));
    lines.push(keyValue("Score", String(model.qualityScore)));
  }

  // Hardware summary
  lines.push(sectionHeader("Your Hardware"));
  if (hardware.primaryGpu) {
    lines.push(keyValue("GPU", hardware.primaryGpu.model));
    lines.push(keyValue("VRAM", formatMb(availableVramMb)));
    if (hardware.primaryGpu.vendor === "Apple") {
      lines.push(subLine("(unified memory — 75% usable for inference)"));
    }
  } else {
    lines.push(keyValue("GPU", theme.muted("None — CPU inference (using RAM)")));
  }
  lines.push(keyValue("RAM", `${formatMb(hardware.memory.totalMb)} ${hardware.memory.type}`));
  if (verbose) {
    lines.push(keyValue("CPU", hardware.cpu.brand));
    lines.push(subLine(`${hardware.cpu.threads} threads · AVX2 ${hardware.cpu.hasAvx2 ? "✓" : "✗"}`));
  }

  // Quant warning
  if (quantWarning) {
    lines.push("");
    lines.push(`  ${theme.warning("⚠")} ${theme.warning(quantWarning)}`);
  }

  // Quantization breakdown
  lines.push(sectionHeader("Quantizations"));
  lines.push(quantTable(scores, bestIdx, availableVramMb));

  // Recommendation
  lines.push("");
  if (verdict !== "no") {
    lines.push(`  ${theme.pass("★")} Recommended: ${theme.pass(bestScore.quantization.name)} — ${bestScore.speedEstimate} speed, ${formatMb(bestScore.quantization.vramMb)} VRAM`);
    if (pullCommand) {
      lines.push(`  ${theme.muted("→")} ${theme.command(pullCommand)}`);
    }
  } else {
    lines.push(`  ${theme.fail("✗")} No quantization fits your hardware.`);
    lines.push(`  ${theme.muted("Try a smaller model:")} ${theme.command("llm-pulse models --fits")}`);
  }

  console.log(titleBox(lines.join("\n")));
}

function outputJson(
  model: ModelEntry,
  hardware: HardwareProfile,
  scores: ModelScore[],
  bestScore: ModelScore,
  verdict: Verdict,
  availableVramMb: number,
  pullCommand: string | null,
): void {
  const output = {
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

  console.log(JSON.stringify(output, null, 2));
}

function outputCsv(scores: ModelScore[], availableVramMb: number, bestIdx: number): void {
  const headers = [
    "quantization", "bitsPerWeight", "vramNeeded", "vramAvailable",
    "fitLevel", "compositeScore", "speedEstimate", "isRecommended",
  ];
  const rows = scores.map((s, i) => [
    s.quantization.name, s.quantization.bitsPerWeight, s.quantization.vramMb,
    availableVramMb, s.fitLevel, s.compositeScore, s.speedEstimate, i === bestIdx,
  ]);
  console.log(toCsv(headers, rows));
}
