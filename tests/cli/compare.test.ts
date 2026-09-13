import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

vi.mock("../../src/hardware/index.js", async () => ({
  detectHardware: async () => (await import("../fixtures/hardware-profiles/high-end-nvidia.json")).default,
}));

import { compareCommand } from "../../src/cli/commands/compare.js";
import type { CompareOptions } from "../../src/core/types.js";

let stdout: string[];
let stderr: string[];

beforeEach(() => {
  stdout = [];
  stderr = [];
  vi.spyOn(console, "log").mockImplementation((...args) => { stdout.push(args.join(" ")); });
  vi.spyOn(console, "error").mockImplementation((...args) => { stderr.push(args.join(" ")); });
  process.exitCode = undefined;
});

afterEach(() => {
  vi.restoreAllMocks();
  process.exitCode = undefined;
});

function options(over: Partial<CompareOptions> = {}): CompareOptions {
  return { format: "json", category: "all", top: 3, ...over };
}

describe("compare", () => {
  it("matches --quant case-insensitively", async () => {
    await compareCommand(["llama3.1:8b", "qwen2.5:7b"], options({ quant: "q4_k_m" }));
    const output = JSON.parse(stdout.join("\n"));
    expect(output.models.map((m: { quantization: string }) => m.quantization)).toEqual(["Q4_K_M", "Q4_K_M"]);
    expect(stderr).toEqual([]);
  });

  it("json: an unavailable --quant warns on stderr and stdout stays parseable", async () => {
    await compareCommand(["llama3.1:8b", "qwen2.5:7b"], options({ quant: "Q9_Z" }));
    expect(() => JSON.parse(stdout.join("\n"))).not.toThrow();
    expect(stderr.join("\n")).toContain("quant Q9_Z not available, using best fit");
  });

  it("json: an unknown model is reported on stderr, stdout stays parseable, exit code 1", async () => {
    await compareCommand(["llama3.1:8b", "nosuchmodelxyz", "qwen2.5:7b"], options());
    expect(JSON.parse(stdout.join("\n")).models).toHaveLength(2);
    expect(stderr.join("\n")).toContain("Model not found: nosuchmodelxyz");
    expect(process.exitCode).toBe(1);
  });

  it("csv: diagnostics stay off stdout", async () => {
    await compareCommand(["llama3.1:8b", "nosuchmodelxyz", "qwen2.5:7b"], options({ format: "csv", quant: "Q9_Z" }));
    expect(stdout.join("\n").startsWith("id,name,")).toBe(true);
  });

  it("json: fewer than 2 models prints the error payload and exits 1", async () => {
    await compareCommand(["llama3.1:8b"], options());
    expect(JSON.parse(stdout.join("\n"))).toEqual({ error: "Need at least 2 models to compare" });
    expect(process.exitCode).toBe(1);
  });

  it("csv: fewer than 2 models writes the error to stderr only", async () => {
    await compareCommand(["llama3.1:8b"], options({ format: "csv" }));
    expect(stdout).toEqual([]);
    expect(stderr.join("\n")).toContain("Need at least 2 models to compare");
    expect(process.exitCode).toBe(1);
  });

  it("leaves the exit code alone on success", async () => {
    await compareCommand(["llama3.1:8b", "qwen2.5:7b"], options());
    expect(process.exitCode).toBeUndefined();
  });
});
