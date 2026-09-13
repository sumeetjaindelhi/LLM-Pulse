import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import chalk from "chalk";

vi.mock("../../src/hardware/index.js", async () => ({
  detectHardware: async () => (await import("../fixtures/hardware-profiles/apple-m2.json")).default,
}));

import { checkCommand } from "../../src/cli/commands/check.js";
import { quantAdviceCommand } from "../../src/cli/commands/quant-advice.js";

let stdout: string[];
const originalChalkLevel = chalk.level;

beforeEach(() => {
  stdout = [];
  // Theme colours off, as in a pipe, so any escape code left comes from cli-table3.
  chalk.level = 0;
  vi.spyOn(console, "log").mockImplementation((...args) => { stdout.push(args.join(" ")); });
});

afterEach(() => {
  vi.restoreAllMocks();
  chalk.level = originalChalkLevel;
});

describe("check table on Apple Silicon", () => {
  it("describes unified memory without a stale percentage", async () => {
    await checkCommand("llama3.1:8b", { format: "table", verbose: false });
    const out = stdout.join("\n");
    expect(out).toContain("(unified memory — shared with system RAM)");
    expect(out).not.toContain("75% usable");
  });

  it("emits no ANSI escape codes when colours are off", async () => {
    await checkCommand("llama3.1:8b", { format: "table", verbose: false });
    expect(stdout.join("\n")).not.toContain("[");
  });
});

describe("quant-advice table", () => {
  it("emits no ANSI escape codes when colours are off", async () => {
    await quantAdviceCommand("llama3.1:8b", { format: "table", verbose: false });
    expect(stdout.join("\n")).not.toContain("[");
  });
});
