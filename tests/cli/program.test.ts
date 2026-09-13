import { describe, it, expect, vi, beforeEach } from "vitest";
import type { Command, CommanderError } from "commander";

// If commander ever accepted a bad value, the action would run. Make that fail
// fast instead of probing real hardware or a local Ollama.
vi.mock("../../src/hardware/index.js", () => ({
  detectHardware: vi.fn().mockRejectedValue(new Error("action should not run")),
}));

import { createProgram } from "../../src/cli/program.js";

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("action should not run")));
});

function quietProgram(): Command {
  const program = createProgram();
  for (const cmd of [program, ...program.commands]) {
    cmd.exitOverride();
    cmd.configureOutput({ writeOut: () => {}, writeErr: () => {} });
  }
  return program;
}

async function rejectionCode(args: string[]): Promise<string> {
  try {
    await quietProgram().parseAsync(args, { from: "user" });
  } catch (err) {
    return (err as CommanderError).code ?? (err as Error).message;
  }
  return "accepted";
}

function findOption(commandName: string, long: string) {
  const cmd = createProgram().commands.find((c) => c.name() === commandName);
  const option = cmd?.options.find((o) => o.long === long);
  if (!option) throw new Error(`no ${long} on ${commandName}`);
  return option;
}

describe("numeric flags", () => {
  it.each([
    ["context-fit", "llama3.1:8b", "--prompt-tokens", "50k"],
    ["context-fit", "llama3.1:8b", "--prompt-tokens", "50,000"],
    ["context-fit", "llama3.1:8b", "--prompt-tokens", "1e6"],
    ["context-fit", "llama3.1:8b", "--prompt-tokens", "1.5"],
    ["context-fit", "llama3.1:8b", "--prompt-tokens", "-1"],
    ["context-fit", "llama3.1:8b", "--prompt-tokens", "abc"],
    ["context-fit", "llama3.1:8b", "--prompt-tokens", "99999999999999999999"],
  ])("rejects %s %s %s %s", async (...args) => {
    expect(await rejectionCode(args)).toBe("commander.invalidArgument");
  });

  it.each([
    [["scan", "--top", "0"]],
    [["--top", "2x"]],
    [["report", "--top", "0"]],
    [["compare", "--top", "1"]],
    [["benchmark", "--rounds", "0"]],
    [["profile", "--context-size", "0"]],
    [["context-fit", "llama3.1:8b", "--prompt-tokens", "10", "--response-tokens", "1.5"]],
  ])("rejects values below the command's lower bound: %j", async (args) => {
    expect(await rejectionCode(args)).toBe("commander.invalidArgument");
  });

  it("parses whole decimal integers to numbers, including 0 where the command accepts it", () => {
    expect(findOption("context-fit", "--prompt-tokens").parseArg!("50000", undefined)).toBe(50000);
    expect(findOption("context-fit", "--prompt-tokens").parseArg!("0", undefined)).toBe(0);
    expect(findOption("context-fit", "--response-tokens").parseArg!("0", undefined)).toBe(0);
    expect(findOption("compare", "--top", ).parseArg!("2", undefined)).toBe(2);
  });

  it("keeps the existing defaults", () => {
    expect(findOption("compare", "--top").defaultValue).toBe(3);
    expect(findOption("context-fit", "--response-tokens").defaultValue).toBe(512);
    expect(findOption("benchmark", "--rounds").defaultValue).toBe(3);
    expect(findOption("profile", "--context-size").defaultValue).toBe(2048);
  });
});

describe("--format and --category", () => {
  it.each([
    [["check", "llama3.1:8b", "--format", "JSON"]],
    [["models", "--format", "xml"]],
    [["--format", "yaml"]],
    [["scan", "--category", "bogus"]],
    [["compare", "--category", "bogus"]],
    [["report", "--category", "bogus"]],
  ])("rejects an unknown value: %j", async (args) => {
    expect(await rejectionCode(args)).toBe("commander.invalidArgument");
  });

  it("offers the same choices on every command that has the flag", () => {
    const program = createProgram();
    for (const cmd of [program, ...program.commands]) {
      const format = cmd.options.find((o) => o.long === "--format");
      if (format) expect(format.argChoices).toEqual(["table", "json", "csv"]);
      const category = cmd.options.find((o) => o.long === "--category");
      if (category) {
        expect(category.argChoices).toEqual(["general", "coding", "reasoning", "creative", "multilingual", "all"]);
      }
    }
  });
});
