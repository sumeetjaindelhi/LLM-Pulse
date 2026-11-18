import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import chalk from "chalk";
import highEnd from "../fixtures/hardware-profiles/high-end-nvidia.json";

vi.mock("../../src/hardware/index.js", async () => ({ detectHardware: async () => highEnd }));
vi.mock("../../src/runtimes/ollama.js", () => ({ withExactFacts: async (_host: string, model: unknown) => model }));
vi.mock("ora", () => ({ default: () => ({ start() { return this; }, succeed() {}, stop() {}, text: "" }) }));
vi.mock("../../src/cli/ui/boxes.js", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../src/cli/ui/boxes.js")>()),
  titleBox: (content: string) => content,
}));

import { checkCommand } from "../../src/cli/commands/check.js";
import { compareCommand } from "../../src/cli/commands/compare.js";

let stdout: string[];
const originalChalkLevel = chalk.level;

beforeEach(() => {
  stdout = [];
  vi.spyOn(console, "log").mockImplementation((...args) => { stdout.push(args.join(" ")); });
});

afterEach(() => {
  vi.restoreAllMocks();
  chalk.level = originalChalkLevel;
});

describe("check offload hint", () => {
  it("keeps /set parameter num_gpu green on its own indented line", async () => {
    chalk.level = 1;
    await checkCommand("qwen2.5:72b", { format: "table", verbose: false });
    const line = stdout.join("\n").split("\n").find((l) => l.includes("/set parameter num_gpu"));
    expect(line).toBeDefined();
    expect(line).toMatch(/^ {8}\x1b\[32m\/set parameter num_gpu \d+\x1b\[39m$/);
  });
});

describe("compare with four models", () => {
  it("separator cells are as wide as the columns: 13 label, then (72 - 2 - 8 - 13) / 4 = 12 each", async () => {
    chalk.level = 0;
    await compareCommand(["llama3.1:8b", "qwen2.5:7b", "qwen2.5-coder:14b", "mistral-small:22b"], { format: "table", category: "all", top: 3 });
    const sep = stdout.join("\n").split("\n").find((l) => l.includes("──"));
    expect(sep?.match(/─+/g)?.map((run) => run.length)).toEqual([13, 12, 12, 12, 12]);
  });
});

describe("keyValue", () => {
  // The label column is 6 wide; a 6-letter label such as "Memory" ran into its value ("Memory22.4 GiB").
  it("keeps at least one space between a label and its value", async () => {
    const { keyValue } = await import("../../src/cli/ui/boxes.js");
    chalk.level = 0;
    expect(keyValue("Memory", "22.4 GiB")).toBe("  Memory 22.4 GiB");
    expect(keyValue("GPU", "x")).toBe("  GPU   x");
  });
});
