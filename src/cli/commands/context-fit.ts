import ora from "ora";
import { detectHardware } from "../../hardware/index.js";
import { checkContextFit } from "../../analysis/context-fit.js";
import { getAvailableVram } from "../../analysis/scorer.js";
import { resolveModel } from "../../models/database.js";
import { titleBox, sectionHeader, keyValue } from "../ui/boxes.js";
import { theme } from "../ui/colors.js";
import { formatMb } from "../ui/progress.js";
import { renderModelNotFound } from "../ui/errors.js";
import { toCsv } from "../ui/csv.js";
import type {
  ContextFitOptions,
  ContextFitResult,
  HardwareProfile,
  QuantizationVariant,
} from "../../core/types.js";

// Pure CSV serialization — one flat row. Exported so it can be unit-tested
// without invoking the command's hardware detection / console output.
export function toContextFitCsv(result: ContextFitResult): string {
  const headers = [
    "model", "quant", "quantForced", "promptTokens", "responseTokens", "neededTokens",
    "nativeWindow", "affordedByVramTokens", "effectiveMaxTokens", "limitedBy", "verdict",
    "headroomTokens", "remedy",
  ];
  const row: (string | number | boolean)[] = [
    result.model.id,
    result.quant.name,
    result.quantForced,
    result.promptTokens,
    result.responseTokens,
    result.neededTokens,
    result.nativeWindow,
    result.affordedByVramTokens,
    result.effectiveMaxTokens,
    result.limitedBy,
    result.verdict,
    result.headroomTokens,
    result.remedy ?? "",
  ];
  return toCsv(headers, [row]);
}

function verdictLabel(result: ContextFitResult): string {
  const headroom = result.headroomTokens.toLocaleString();
  if (result.verdict === "yes") return theme.pass(`YES — fits (${headroom} tok headroom)`);
  if (result.verdict === "tight") return theme.warning(`TIGHT — fits, ${headroom} tok headroom`);
  return theme.fail(`NO — short by ${Math.abs(result.headroomTokens).toLocaleString()} tok`);
}

function outputTable(result: ContextFitResult, hardware: HardwareProfile, availableVramMb: number): void {
  const lines: string[] = [];
  lines.push("");
  lines.push(`  ${theme.header("Context Fit")} — ${theme.value(result.model.name)}`);

  const quantTag = result.quantForced ? "(forced)" : "(sweet spot — override with --quant)";
  lines.push(sectionHeader("Quant"));
  lines.push(`  ${theme.pass(`★ ${result.quant.name}`)} ${theme.muted(quantTag)}`);

  lines.push(sectionHeader("Workload"));
  lines.push(`  ${theme.number(result.promptTokens.toLocaleString())} prompt ${theme.muted("+")} ${theme.number(result.responseTokens.toLocaleString())} response ${theme.muted("=")} ${theme.number(result.neededTokens.toLocaleString())} tokens`);

  lines.push(sectionHeader("Hardware"));
  if (hardware.primaryGpu) {
    lines.push(keyValue("GPU", `${hardware.primaryGpu.model} · ${formatMb(availableVramMb)}`));
  } else {
    lines.push(keyValue("Memory", `${formatMb(availableVramMb)} usable (CPU inference)`));
  }

  lines.push(sectionHeader("Ceilings"));
  lines.push(keyValue("Native window", `${result.nativeWindow.toLocaleString()} tok`));
  lines.push(keyValue("Hardware-afforded", `${result.affordedByVramTokens.toLocaleString()} tok`));
  lines.push(keyValue("Effective max", `${result.effectiveMaxTokens.toLocaleString()} tok  (limited by ${result.limitedBy})`));

  lines.push("");
  lines.push(`  ${verdictLabel(result)}`);
  if (result.remedy) {
    lines.push(`  ${theme.label("Remedy")} ${theme.muted(result.remedy)}`);
  }

  console.log(titleBox(lines.join("\n")));
}

export async function contextFitCommand(
  modelArg: string,
  options: ContextFitOptions,
): Promise<void> {
  const isJson = options.format === "json";
  const isCsv = options.format === "csv";
  const silent = isJson || isCsv;

  const spinner = silent ? null : ora({ text: "Detecting hardware...", color: "cyan" }).start();
  const hardware = await detectHardware();
  spinner?.succeed("Hardware detected");

  const model = resolveModel(modelArg);
  if (!model) {
    renderModelNotFound(modelArg, { silent });
    return;
  }

  if (model.quantizations.length === 0) {
    const payload = { error: `Model ${model.name} has no quantizations defined in the database` };
    if (silent) console.log(JSON.stringify(payload, null, 2));
    else console.log(`\n  ${theme.fail("✗")} ${payload.error}`);
    return;
  }

  // Resolve --quant if given; a typo is an error (with the available list) rather
  // than a silent fall-through to the sweet spot.
  let forcedQuant: QuantizationVariant | undefined;
  if (options.quant) {
    const match = model.quantizations.find((q) => q.name.toLowerCase() === options.quant!.toLowerCase());
    if (!match) {
      const payload = {
        error: `Quantization "${options.quant}" not found for ${model.name}`,
        availableQuantizations: model.quantizations.map((q) => q.name),
      };
      if (silent) {
        console.log(JSON.stringify(payload, null, 2));
      } else {
        console.log(`\n  ${theme.fail("✗")} ${payload.error}`);
        console.log(`  ${theme.muted("Available:")} ${payload.availableQuantizations.join(", ")}`);
      }
      return;
    }
    forcedQuant = match;
  }

  const result = checkContextFit(model, hardware, {
    promptTokens: options.promptTokens,
    responseTokens: options.responseTokens,
    quant: forcedQuant,
  });

  if (isJson) {
    console.log(JSON.stringify(result, null, 2));
  } else if (isCsv) {
    console.log(toContextFitCsv(result));
  } else {
    outputTable(result, hardware, getAvailableVram(hardware));
  }
}
