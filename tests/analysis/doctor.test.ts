import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { runDiagnostics, planFixes } from "../../src/analysis/doctor.js";
import type { CatalogHealth, FixAction, HardwareProfile, LoadedModel, RuntimeInfo } from "../../src/core/types.js";
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
    // Fixture: 10977 MiB wired on a 16384 MiB M2 → 67% of RAM. Message shape:
    // "Apple Silicon unified memory — ~11 GiB wired to GPU (67% of 16 GiB total)"
    expect(memCheck!.message).toMatch(/\d+%\s+of\s+\d+ GiB/);
    expect(memCheck!.message).toContain("wired to GPU");
  });

  // Every size here is MiB / 1024 (disk: bytes / 2^30), the same figures scan and report print as GiB.
  it("labels 1024-based sizes GiB", () => {
    const report = runDiagnostics(highEnd as HardwareProfile, withOllama);
    for (const label of ["GPU VRAM", "RAM", "Disk Space"]) {
      expect(report.checks.find((c) => c.label === label)!.message, label).toMatch(/^\d+ GiB /);
    }
  });

  // The memory model, not a size table, decides what fits: an 18 GiB card rates 32B models "barely".
  it("states the VRAM without a fixed model-size range, and points at scan", () => {
    const message = runDiagnostics(highEnd as HardwareProfile, withOllama).checks.find((c) => c.label === "GPU VRAM")!.message;
    expect(message).toBe("24 GiB VRAM — see `llm-pulse scan` for the models that fit");
    expect(message).not.toMatch(/\dB/);
  });

  it("uses GiB in every suggestion", () => {
    const report = runDiagnostics(cpuOnly as HardwareProfile, noRuntimes);
    expect(JSON.stringify(report)).not.toMatch(/\bGB\b/);
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

describe("planFixes", () => {
  it("marks allowlisted fixes as would-run and carries display fields through", () => {
    const plan = planFixes([
      {
        label: "Pull a starter model",
        command: "ollama pull llama3.2:3b",
        argv: ["ollama", "pull", "llama3.2:3b"],
        description: "Downloads Llama 3.2 3B (~2 GiB) — a great starter model",
      },
    ]);
    expect(plan).toHaveLength(1);
    expect(plan[0].status).toBe("would-run");
    expect(plan[0].label).toBe("Pull a starter model");
    expect(plan[0].command).toBe("ollama pull llama3.2:3b");
    expect(plan[0].description).toContain("Llama 3.2 3B");
  });

  it("marks fixes whose binary is not allowlisted as blocked", () => {
    const plan = planFixes([
      {
        label: "Sketchy",
        command: "evil --pwn",
        argv: ["evil", "--pwn"],
        description: "Should never run",
      },
    ]);
    expect(plan[0].status).toBe("blocked");
  });

  it("marks fixes with a missing or empty argv as malformed", () => {
    const noArgv = { label: "No argv", command: "x", description: "d" };
    const emptyArgv = { label: "Empty argv", command: "x", argv: [], description: "d" };
    const plan = planFixes([noArgv as FixAction, emptyArgv as FixAction]);
    expect(plan[0].status).toBe("malformed");
    expect(plan[1].status).toBe("malformed");
  });

  it("accepts every fix that runDiagnostics itself emits", () => {
    // The fixes shipped in diagnostics must always clear the runner allowlist —
    // a blocked/malformed entry here means the two halves drifted apart.
    const report = runDiagnostics(cpuOnly as HardwareProfile, noRuntimes);
    const fixes = report.checks.flatMap((c) => (c.fix ? [c.fix] : []));
    expect(fixes.length).toBeGreaterThan(0);
    const plan = planFixes(fixes);
    expect(plan.every((p) => p.status === "would-run")).toBe(true);
  });
});

describe("runDiagnostics Context check", () => {
  const hw = highEnd as HardwareProfile;
  const contextRow = (loaded: LoadedModel[], runtimes = withOllama, hardware = hw) =>
    runDiagnostics(hardware, runtimes, loaded).checks.filter((c) => c.label === "Context");

  it("warns for a loaded model below the 64K guidance and what the GPU holds, sized by the loaded quant", () => {
    // Curated llama3.1:8b facts on the 24 GB GPU (budget 23095): need(131072) = 4693 + 16384 KV + 1102.2 compute
    // = 22179.2 fits, so the whole 131072 training context does.
    const rows = contextRow([{ name: "llama3.1:8b", contextLength: 4096, quantizationLevel: "Q4_K_M" }]);
    expect(rows).toHaveLength(1);
    expect(rows[0].severity).toBe("warning");
    expect(rows[0].message).toContain("4,096");
    expect(rows[0].message).toContain("131,072");
    expect(rows[0].suggestion).toContain("OLLAMA_CONTEXT_LENGTH=131072 ollama serve");
    expect(rows[0].suggestion).toContain("64,000");
    expect(rows[0].fix).toBeUndefined();
  });

  it("does not change the score", () => {
    const base = runDiagnostics(hw, withOllama);
    const withLoaded = runDiagnostics(hw, withOllama, [{ name: "llama3.1:8b", contextLength: 4096, quantizationLevel: "Q4_K_M" }]);
    expect(withLoaded.score).toBe(base.score);
  });

  it("sizes KV from the loaded model's own architecture facts", () => {
    // A loaded file with 32 KV heads: 32 x 32 x 256 x 2 B = 0.5 MiB/token. Above 32768 tokens (compute 1102.2):
    // c <= (23095 - 4693 - 1102.2) / 0.5 = 34599 -> 34560 -> tier 32768 (the curated facts give 131072, first test).
    const arch = { architecture: "llama", layers: 32, headCountKv: 32, keyLength: 128, valueLength: 128, vocabSize: 128256, contextLength: 131072 };
    const rows = contextRow([{ name: "llama3.1:8b", contextLength: 4096, quantizationLevel: "Q4_K_M", arch }]);
    expect(rows[0].suggestion).toContain("OLLAMA_CONTEXT_LENGTH=32768 ");
  });

  it("falls back to the sweet-spot quant when the loaded level is unknown", () => {
    expect(contextRow([{ name: "llama3.1:8b", contextLength: 4096 }])).toHaveLength(1);
  });

  it("is quiet at 65536", () => {
    expect(contextRow([{ name: "llama3.1:8b", contextLength: 65536 }])).toHaveLength(0);
  });

  it("is quiet when no context tier fits the machine", () => {
    // Metal budget 5500 - 1536 = 3964 MiB is below the 4693 MiB Q4_K_M file, so max context is 0.
    const base = appleM2 as HardwareProfile;
    const smallGpu = { ...base.primaryGpu!, vramMb: 5500 };
    const small = { ...base, gpus: [smallGpu], primaryGpu: smallGpu };
    expect(contextRow([{ name: "llama3.1:8b", contextLength: 1024, quantizationLevel: "Q4_K_M" }], withOllama, small)).toHaveLength(0);
  });

  it.each([
    ["unknown model", [{ name: "unknown:1b", contextLength: 4096 }]],
    ["unknown context", [{ name: "llama3.1:8b", contextLength: null }]],
    ["nothing loaded", []],
  ] as [string, LoadedModel[]][])("no row for %s", (_label, loaded) => {
    expect(contextRow(loaded)).toHaveLength(0);
  });

  it("no row when Ollama is not running", () => {
    const installed = withOllama.map((r) => (r.name === "Ollama" ? { ...r, status: "installed" as const } : r));
    expect(contextRow([{ name: "llama3.1:8b", contextLength: 4096 }], installed)).toHaveLength(0);
  });
});

describe("runDiagnostics model catalog row", () => {
  const NOW = Date.UTC(2026, 10, 18, 12);
  const DAY_MS = 24 * 60 * 60 * 1000;
  const live: CatalogHealth = { source: "live", fetchedAt: NOW - 3 * DAY_MS, liveModels: 142, lastError: null, lastSuccessAt: NOW - 3 * DAY_MS, offline: false };
  const hardware = highEnd as HardwareProfile;
  const catalogCheck = (catalog?: CatalogHealth) =>
    runDiagnostics(hardware, withOllama, [], catalog).checks.find((c) => c.label === "Model catalog");

  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(NOW);
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("adds no row when the caller passes no catalog", () => {
    expect(catalogCheck()).toBeUndefined();
  });

  it("reports the live count and age as info", () => {
    expect(catalogCheck(live)).toMatchObject({ severity: "info", message: "142 live models from ollama.com, updated 3 days ago" });
  });

  it("reports hours for a catalog under a day old", () => {
    expect(catalogCheck({ ...live, fetchedAt: NOW - 5 * 60 * 60 * 1000 })?.message).toBe("142 live models from ollama.com, updated 5 hours ago");
  });

  it("uses the singular for one hour", () => {
    expect(catalogCheck({ ...live, fetchedAt: NOW - 90 * 60 * 1000 })?.message).toContain("updated 1 hour ago");
  });

  it("says less than an hour for a fresh catalog", () => {
    expect(catalogCheck({ ...live, fetchedAt: NOW - 60 * 1000 })?.message).toContain("updated less than an hour ago");
  });

  it("points a failing refresh at the offline switch", () => {
    const check = catalogCheck({ source: "bundled", fetchedAt: null, liveModels: 0, lastError: "HTTP 503", lastSuccessAt: null, offline: false });
    expect(check?.suggestion).toContain("LLMPULSE_OFFLINE=1");
  });

  it("says offline when the offline switch is on", () => {
    expect(catalogCheck({ source: "bundled", fetchedAt: null, liveModels: 0, lastError: null, lastSuccessAt: null, offline: true }))
      .toMatchObject({ severity: "info", message: "offline: built-in list (LLMPULSE_OFFLINE / offline config)" });
  });

  it("says no catalog is cached yet when online with nothing fetched", () => {
    expect(catalogCheck({ source: "bundled", fetchedAt: null, liveModels: 0, lastError: null, lastSuccessAt: null, offline: false }))
      .toMatchObject({ severity: "info", message: "online, but no catalog cached yet: built-in list", suggestion: "Run `llm-pulse models --refresh` to fetch it" });
  });

  it("warns with the error when a fetch failed and nothing ever succeeded", () => {
    const check = catalogCheck({ source: "bundled", fetchedAt: null, liveModels: 0, lastError: "fetch failed: ECONNREFUSED", lastSuccessAt: null, offline: false });
    expect(check?.severity).toBe("warning");
    expect(check?.message).toContain("fetch failed: ECONNREFUSED");
  });

  it("warns when the last success is more than 7 days old", () => {
    const stale = NOW - 8 * DAY_MS;
    const check = catalogCheck({ ...live, fetchedAt: stale, lastSuccessAt: stale, lastError: "HTTP 503" });
    expect(check?.severity).toBe("warning");
    expect(check?.message).toContain("HTTP 503");
  });

  it("stays info when the last error is recent news and the last success is under 7 days old", () => {
    expect(catalogCheck({ ...live, lastError: "HTTP 503" })?.severity).toBe("info");
  });

  it("never changes the score", () => {
    const bare = runDiagnostics(hardware, withOllama).score;
    const failing = runDiagnostics(hardware, withOllama, [], { source: "bundled", fetchedAt: null, liveModels: 0, lastError: "x", lastSuccessAt: null, offline: false }).score;
    expect(failing).toBe(bare);
  });
});
