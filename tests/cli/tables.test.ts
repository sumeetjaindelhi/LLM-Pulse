import { describe, it, expect, beforeEach, afterEach } from "vitest";
import chalk from "chalk";
import stringWidth from "string-width";
import highEnd from "../fixtures/hardware-profiles/high-end-nvidia.json";
import type { HardwareProfile } from "../../src/core/types.js";
import { getRecommendations } from "../../src/analysis/recommender.js";
import { recommendationTable } from "../../src/cli/ui/tables.js";

const hardware = highEnd as HardwareProfile;
const originalChalkLevel = chalk.level;
beforeEach(() => { chalk.level = 0; });
afterEach(() => { chalk.level = originalChalkLevel; });

describe("recommendationTable", () => {
  it.each(["nemotron-cascade-2-instruct-preview", "deepseek-r1-distill-qwen-32b-preview"])("shows the single-word name %s in full", (slug) => {
    const [first, ...rest] = getRecommendations(hardware, { category: "all", top: 3, onlyFitting: true });
    const rename = (rec: typeof first, name: string) => ({ ...rec, score: { ...rec.score, model: { ...rec.score.model, name } } });
    // A long multi-word name keeps the Model column at its cap, so the single word has to be broken.
    const out = recommendationTable([rename(first, slug), rename(rest[0], "Qwen 3 14B Instruct Preview Edition Extended"), ...rest.slice(1)], hardware, new Set([first.score.model.ollamaTag ?? ""]));
    const lines = out.split("\n");
    const [from, to] = [lines[0].indexOf("Model"), lines[0].indexOf("Quant")];
    const joined = lines.slice(1).map((line) => line.slice(from, to).replace("●", "").trim()).join("");
    expect(joined).toContain(slug);
    expect(out).not.toContain("…");
    expect(out.split("\n").filter((l) => stringWidth(l) > 72)).toEqual([]);
  });
});
