import { describe, it, expect } from "vitest";
import { runDiagnostics } from "../../src/analysis/doctor.js";
import type { HardwareProfile, RuntimeInfo } from "../../src/core/types.js";
import highEnd from "../fixtures/hardware-profiles/high-end-nvidia.json";
import cpuOnly from "../fixtures/hardware-profiles/cpu-only.json";
import appleM2 from "../fixtures/hardware-profiles/apple-m2.json";
import amdRdna3 from "../fixtures/hardware-profiles/amd-rdna3.json";
import windowsNvidia from "../fixtures/hardware-profiles/windows-nvidia.json";

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

  it("shows unified memory info for Apple Silicon", () => {
    const report = runDiagnostics(appleM2 as HardwareProfile, withOllama);
    const memCheck = report.checks.find((c) => c.label === "Unified Memory");
    expect(memCheck).toBeDefined();
    expect(memCheck!.severity).toBe("info");
    expect(memCheck!.message).toContain("75%");
  });

  it("warns about missing ROCm on AMD GPU without accelerator", () => {
    const amdNoRocm: HardwareProfile = {
      ...(amdRdna3 as HardwareProfile),
      primaryGpu: {
        ...(amdRdna3 as HardwareProfile).primaryGpu!,
        acceleratorVersion: null,
      },
    };
    const report = runDiagnostics(amdNoRocm, withOllama);
    const rocmCheck = report.checks.find((c) => c.label === "ROCm");
    expect(rocmCheck).toBeDefined();
    expect(rocmCheck!.severity).toBe("warning");
    expect(rocmCheck!.message).toContain("rocm-smi not found");
  });

  it("does not warn about ROCm when AMD GPU has accelerator version", () => {
    const report = runDiagnostics(amdRdna3 as HardwareProfile, withOllama);
    const rocmCheck = report.checks.find((c) => c.label === "ROCm");
    expect(rocmCheck).toBeUndefined();
  });

  it("recognizes ARM CPU NEON as pass for SIMD", () => {
    const report = runDiagnostics(appleM2 as HardwareProfile, withOllama);
    const simdCheck = report.checks.find((c) => c.label === "SIMD");
    expect(simdCheck).toBeDefined();
    expect(simdCheck!.severity).toBe("pass");
  });

  it("windows-nvidia with driver 560 passes the driver-version check", () => {
    const report = runDiagnostics(windowsNvidia as HardwareProfile, withOllama);
    const driverCheck = report.checks.find((c) => c.label === "GPU Driver");
    expect(driverCheck).toBeDefined();
    expect(driverCheck!.severity).toBe("pass");
    expect(driverCheck!.message).toContain("up to date");
  });

  it("windows-nvidia without any runtime still scores well due to GPU+RAM", () => {
    const report = runDiagnostics(windowsNvidia as HardwareProfile, noRuntimes);
    // 24GB VRAM + 64GB DDR5 + NVMe + AVX2 — hardware is great,
    // runtime missing is the only gap. Should still clear 65/100.
    expect(report.score).toBeGreaterThanOrEqual(65);
    const rtCheck = report.checks.find((c) => c.label === "Runtime");
    expect(rtCheck?.severity).toBe("fail");
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
