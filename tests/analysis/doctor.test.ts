import { describe, it, expect } from "vitest";
import { runDiagnostics } from "../../src/analysis/doctor.js";
import type { HardwareProfile, RuntimeInfo } from "../../src/core/types.js";
import highEnd from "../fixtures/hardware-profiles/high-end-nvidia.json";
import cpuOnly from "../fixtures/hardware-profiles/cpu-only.json";

const noRuntimes: RuntimeInfo[] = [
  { name: "Ollama", status: "not_found", version: null, path: null, models: [] },
  { name: "llama.cpp", status: "not_found", version: null, path: null, models: [] },
  { name: "LM Studio", status: "not_found", version: null, path: null, models: [] },
];

const withOllama: RuntimeInfo[] = [
  { name: "Ollama", status: "running", version: "0.5.1", path: "/usr/local/bin/ollama", models: ["llama3.1:8b"] },
  { name: "llama.cpp", status: "not_found", version: null, path: null, models: [] },
  { name: "LM Studio", status: "not_found", version: null, path: null, models: [] },
];

describe("runDiagnostics", () => {
  it("gives high score to well-equipped system", () => {
    const report = runDiagnostics(highEnd as HardwareProfile, withOllama);
    expect(report.score).toBeGreaterThanOrEqual(80);
    expect(report.summary).toContain("Great");
  });

  it("gives lower score to CPU-only system without runtimes", () => {
    const report = runDiagnostics(cpuOnly as HardwareProfile, noRuntimes);
    expect(report.score).toBeLessThan(80);
  });

  it("flags missing AVX2 as warning", () => {
    const noAvx: HardwareProfile = {
      ...(highEnd as HardwareProfile),
      cpu: { ...(highEnd as HardwareProfile).cpu, hasAvx2: false },
    };
    const report = runDiagnostics(noAvx, withOllama);
    const avxCheck = report.checks.find((c) => c.label === "AVX2");
    expect(avxCheck?.severity).toBe("warning");
  });

  it("flags no GPU as warning", () => {
    const report = runDiagnostics(cpuOnly as HardwareProfile, withOllama);
    const gpuCheck = report.checks.find((c) => c.label === "GPU");
    expect(gpuCheck?.severity).toBe("warning");
  });

  it("flags no runtime as fail", () => {
    const report = runDiagnostics(highEnd as HardwareProfile, noRuntimes);
    const rtCheck = report.checks.find((c) => c.label === "Runtime");
    expect(rtCheck?.severity).toBe("fail");
  });

  it("provides actionable suggestion", () => {
    const report = runDiagnostics(cpuOnly as HardwareProfile, noRuntimes);
    expect(report.topSuggestion).toBeTruthy();
  });

  it("score is always 0-100", () => {
    const report1 = runDiagnostics(highEnd as HardwareProfile, withOllama);
    const report2 = runDiagnostics(cpuOnly as HardwareProfile, noRuntimes);
    expect(report1.score).toBeGreaterThanOrEqual(0);
    expect(report1.score).toBeLessThanOrEqual(100);
    expect(report2.score).toBeGreaterThanOrEqual(0);
    expect(report2.score).toBeLessThanOrEqual(100);
  });
});
