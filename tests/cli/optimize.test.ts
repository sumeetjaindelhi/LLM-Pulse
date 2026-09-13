import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

vi.mock("../../src/hardware/index.js", async () => ({
  detectHardware: async () => (await import("../fixtures/hardware-profiles/cpu-only.json")).default,
}));

import { optimizeCommand } from "../../src/cli/commands/optimize.js";

let stdout: string[];

beforeEach(() => {
  stdout = [];
  vi.spyOn(console, "log").mockImplementation((...args) => { stdout.push(args.join(" ")); });
  process.exitCode = undefined;
});

afterEach(() => {
  vi.restoreAllMocks();
  process.exitCode = undefined;
});

describe("optimize", () => {
  it("json: no fitting quantization prints the error payload and exits 1", async () => {
    await optimizeCommand("llama-4-scout", { format: "json", verbose: false });
    expect(JSON.parse(stdout.join("\n")).error).toBe("No quantization fits");
    expect(process.exitCode).toBe(1);
  });

  it("leaves the exit code alone on success", async () => {
    await optimizeCommand("llama3.1:8b", { format: "json", verbose: false });
    expect(JSON.parse(stdout.join("\n")).parameters).toBeDefined();
    expect(process.exitCode).toBeUndefined();
  });
});
