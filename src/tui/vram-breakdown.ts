import type { VramBreakdown } from "../core/types.js";
import type { MonitorSnapshot } from "../hardware/monitor.js";
import { getCuratedModels } from "../models/database.js";
import { memoryNeedMb } from "../analysis/context-capacity.js";
import { budgetMb } from "../analysis/memory-model.js";

export const FIT_CONTEXT_TOKENS = 4096;
const MAX_FITS = 3;

/**
 * The monitor reads one GPU while Ollama's size_vram sums the model's buffers over every GPU it loaded on, so the
 * Loaded/Other split is only meaningful when that one GPU is the only one (`split`); otherwise it is used/free only.
 */
export function vramBreakdown(
  snapshot: Pick<MonitorSnapshot, "gpuVramUsedMb" | "gpuVramTotalMb" | "modelVramMb" | "gpuCount">,
): VramBreakdown | null {
  const { gpuVramUsedMb: usedMb, gpuVramTotalMb: totalMb } = snapshot;
  if (usedMb === null || totalMb === null) return null;
  const split = snapshot.gpuCount === 1;
  const modelVramMb = split ? Math.min(snapshot.modelVramMb ?? 0, usedMb) : 0;
  return {
    totalMb,
    usedMb,
    freeMb: Math.max(0, totalMb - usedMb),
    modelVramMb,
    otherMb: usedMb - modelVramMb,
    split,
  };
}

/** Memory budget (MiB) of a GPU with `freeMb` free: Metal and discrete as the scan budgets them; an AMD APU's free GTT
 *  is also capped by `apuRamAvailableMb`, the RAM it draws from (null for a card with its own VRAM). */
function freeBudgetMb(freeMb: number, gpuVendor: string | null, apuRamAvailableMb: number | null): number {
  if (gpuVendor === "Apple") return budgetMb({ kind: "apple-metal", vramMb: freeMb });
  if (apuRamAvailableMb !== null) return budgetMb({ kind: "amd-apu", gttMb: freeMb, availableMb: apuRamAvailableMb });
  return budgetMb({ kind: "discrete-gpu", vramMb: freeMb });
}

/** Curated models whose `ollama pull` file plus KV cache and compute buffer at FIT_CONTEXT_TOKENS fit in `freeMb` once the
 *  runtime's reserves are taken (budgetMb), largest first. */
export function fitsInFreeVram(
  freeMb: number,
  gpuVendor: string | null,
  apuRamAvailableMb: number | null = null,
): { name: string; needMb: number }[] {
  const usableMb = freeBudgetMb(freeMb, gpuVendor, apuRamAvailableMb);
  const fits: { name: string; needMb: number }[] = [];
  for (const model of getCuratedModels()) {
    const quant = model.ollamaTag ? model.quantizations.find((q) => q.exact) : undefined;
    if (!quant) continue;
    const needMb = memoryNeedMb(model, quant, FIT_CONTEXT_TOKENS);
    if (needMb <= usableMb) fits.push({ name: model.name, needMb });
  }
  return fits.sort((a, b) => b.needMb - a.needMb).slice(0, MAX_FITS);
}
