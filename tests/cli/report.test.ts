import { describe, it, expect, beforeEach, afterEach } from "vitest";
import chalk from "chalk";
import { renderMarkdownReport } from "../../src/cli/commands/report.js";
import type { ReportInput } from "../../src/cli/commands/report.js";
import { runDiagnostics } from "../../src/analysis/doctor.js";
import { getRecommendations } from "../../src/analysis/recommender.js";
import { getAvailableVram } from "../../src/analysis/scorer.js";
import { theme } from "../../src/cli/ui/colors.js";
import { formatMb } from "../../src/cli/ui/progress.js";
import type { HardwareProfile, RuntimeInfo } from "../../src/core/types.js";
import highEnd from "../fixtures/hardware-profiles/high-end-nvidia.json";
import windowsNvidia from "../fixtures/hardware-profiles/windows-nvidia.json";
import cpuOnly from "../fixtures/hardware-profiles/cpu-only.json";
import appleM2 from "../fixtures/hardware-profiles/apple-m2.json";

const HIGH_END = highEnd as HardwareProfile;
const WINDOWS_NVIDIA = windowsNvidia as HardwareProfile;
const CPU_ONLY = cpuOnly as HardwareProfile;
const APPLE_M2 = appleM2 as HardwareProfile;

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

// Builds a ReportInput whose derived fields (health, recommendations, usable
// memory) are computed from the given hardware and runtimes, like reportCommand.
function input(
  hardware: HardwareProfile,
  runtimes: RuntimeInfo[],
  overrides: Partial<ReportInput> = {},
): ReportInput {
  return {
    version: "1.1.0",
    generatedAt: new Date("2026-09-13T12:00:00Z"),
    platform: "linux/x64",
    hardware,
    runtimes,
    installedTags: new Set<string>(),
    recommendations: getRecommendations(hardware, { top: 5, onlyFitting: true }),
    health: runDiagnostics(hardware, runtimes),
    availableVramMb: getAvailableVram(hardware),
    category: "all",
    top: 5,
    ...overrides,
  };
}

function lines(markdown: string): string[] {
  return markdown.split("\n");
}

describe("renderMarkdownReport — hardware", () => {
  it("renders the NVIDIA hardware table", () => {
    const md = renderMarkdownReport(input(HIGH_END, withOllama));
    expect(md).toContain("| CPU | AMD Ryzen 9 7950X · 16 cores / 32 threads · AVX2 |");
    expect(md).toContain("| GPU | NVIDIA GeForce RTX 4090 · 24.0 GB VRAM · CUDA 12.5 |");
    expect(md).toContain("| RAM | 64.0 GB DDR5 @ 5600 MHz |");
    expect(md).toContain("| Disk | NVMe · 1200 GB free |");
    expect(md).toContain("| Usable for inference | 24.0 GB |");
    expect(md).not.toContain("Unified memory");
  });

  it("does not repeat the vendor when the model name already starts with it", () => {
    const md = renderMarkdownReport(input(WINDOWS_NVIDIA, withOllama));
    expect(md).toContain("| GPU | NVIDIA GeForce RTX 4090 ·");
    expect(md).not.toContain("NVIDIA NVIDIA");
  });

  it("renders a CPU-only machine with system RAM as the usable pool", () => {
    const md = renderMarkdownReport(input(CPU_ONLY, noRuntimes));
    expect(md).toContain("| GPU | None — CPU inference |");
    expect(md).toContain("| Usable for inference | 9.8 GB |");
  });

  it("renders Apple Silicon with NEON, the headroom-adjusted pool and the unified memory note", () => {
    const md = renderMarkdownReport(input(APPLE_M2, withOllama));
    const cpuRow = lines(md).find((line) => line.startsWith("| CPU |"));
    expect(cpuRow?.endsWith("· NEON |")).toBe(true);
    expect(md).toContain("| GPU | Apple M2 Pro · 10.7 GB VRAM |");
    expect(md).toContain("| Usable for inference | 4.7 GB |");
    expect(md).toContain("Unified memory — estimate reserves ~6 GB");
    expect(md).not.toContain("75%");
  });
});

describe("renderMarkdownReport — runtimes and privacy", () => {
  it("renders runtime rows without install paths", () => {
    const md = renderMarkdownReport(input(HIGH_END, withOllama));
    expect(md).toContain("| Ollama | running | 0.5.1 | 1 |");
    expect(md).toContain("| llama.cpp | not found | — | — |");
    expect(md).not.toContain("/usr/local/bin/ollama");
  });

  it("never prints installed model names", () => {
    const runtimes: RuntimeInfo[] = [
      { ...withOllama[0], models: ["secret-finetune:latest"] },
      ...withOllama.slice(1),
    ];
    const md = renderMarkdownReport(
      input(HIGH_END, runtimes, { installedTags: new Set(["secret-finetune:latest"]) }),
    );
    expect(md).not.toContain("secret-finetune");
  });

  it("renders an installed-but-offline Ollama with no version or model count", () => {
    const runtimes: RuntimeInfo[] = [
      { name: "Ollama", status: "installed", version: null, path: "/usr/local/bin/ollama", models: [] },
      ...noRuntimes.slice(1),
    ];
    expect(renderMarkdownReport(input(HIGH_END, runtimes))).toContain("| Ollama | installed | — | — |");
  });
});

describe("renderMarkdownReport — health", () => {
  it("renders the score heading, one list item per check, and the suggestion only when present", () => {
    const healthy = input(HIGH_END, withOllama);
    const healthyMd = renderMarkdownReport(healthy);
    expect(lines(healthyMd)).toContain(`### Health — ${healthy.health.score}/100 · ${healthy.health.summary}`);
    expect(lines(healthyMd).filter((line) => line.startsWith("- "))).toHaveLength(healthy.health.checks.length);
    expect(healthy.health.topSuggestion).toBeNull();
    expect(healthyMd).not.toContain("> Suggestion: ");

    const struggling = input(CPU_ONLY, noRuntimes);
    expect(struggling.health.topSuggestion).not.toBeNull();
    expect(renderMarkdownReport(struggling)).toContain("> Suggestion: ");
  });
});

describe("renderMarkdownReport — recommendations", () => {
  it("marks only recommended models whose Ollama tag is installed", () => {
    const recs = getRecommendations(HIGH_END, { top: 5, onlyFitting: true });
    const marked = lines(renderMarkdownReport(
      input(HIGH_END, withOllama, { recommendations: recs, installedTags: new Set([recs[0].score.model.ollamaTag!]) }),
    ));
    expect(marked.find((line) => line.startsWith("| 1 |"))?.endsWith("| yes |")).toBe(true);
    expect(marked.find((line) => line.startsWith("| 2 |"))?.endsWith("| — |")).toBe(true);

    const unmarked = lines(renderMarkdownReport(input(HIGH_END, withOllama, { recommendations: recs })));
    expect(unmarked.some((line) => line.endsWith("| yes |"))).toBe(false);
  });

  it("fills the recommendation row from the model score", () => {
    const report = input(HIGH_END, withOllama);
    const { score } = report.recommendations[0];
    const row = lines(renderMarkdownReport(report)).find((line) => line.startsWith("| 1 |"));
    expect(row).toContain(`| ${score.model.name} |`);
    expect(row).toContain(`| ${score.quantization.name} |`);
    expect(row).toContain(`| ${score.fitLevel} |`);
    expect(row).toContain(`| ${formatMb(score.quantization.vramMb)} |`);
    expect(row).toContain(`| ${score.speedEstimate} |`);
  });

  it("replaces the table with an empty-state line when nothing is recommended", () => {
    const md = renderMarkdownReport(
      input(HIGH_END, withOllama, { recommendations: [], category: "coding", top: 3 }),
    );
    expect(md).toContain(
      "_No curated models fit this hardware with the current filters (category: coding, top: 3)._",
    );
    expect(md).not.toContain("| # | Model |");
  });
});

describe("renderMarkdownReport — hostile input", () => {
  it("keeps a CPU brand with a pipe and newline inside its own table row", () => {
    const evil: HardwareProfile = { ...HIGH_END, cpu: { ...HIGH_END.cpu, brand: "Evil | CPU\n| Fake | row" } };
    const md = renderMarkdownReport(input(evil, withOllama));
    expect(md).toContain("| CPU | Evil \\| CPU \\| Fake \\| row ·");
    const hardwareSection = md.slice(md.indexOf("### Hardware"), md.indexOf("### Runtimes"));
    expect(lines(hardwareSection).filter((line) => line.startsWith("|"))).toHaveLength(7);
  });

  it("neutralises terminal escapes and HTML in a runtime version", () => {
    const runtimes: RuntimeInfo[] = [
      { ...withOllama[0], version: "0.5.1\u001b[2J<img src=x>" },
      ...withOllama.slice(1),
    ];
    const md = renderMarkdownReport(input(HIGH_END, runtimes));
    expect(md).not.toContain("\u001b");
    expect(md).toContain("\\<img src=x\\>");
  });
});

describe("renderMarkdownReport — with terminal colour enabled", () => {
  let savedLevel = chalk.level;

  beforeEach(() => {
    savedLevel = chalk.level;
    chalk.level = 1;
  });

  afterEach(() => {
    chalk.level = savedLevel;
  });

  it("emits no ANSI escape sequences", () => {
    // Guard against a vacuous pass: colour must really be on for this test.
    expect(theme.pass("x")).not.toBe("x");
    expect(renderMarkdownReport(input(APPLE_M2, withOllama))).not.toMatch(/\u001b\[/);
  });
});

describe("renderMarkdownReport — document structure", () => {
  it("emits the sections in order", () => {
    const md = renderMarkdownReport(input(HIGH_END, withOllama));
    const order = [
      "## llm-pulse report",
      "### Hardware",
      "### Runtimes",
      "### Health",
      "### Recommended models",
      "_Generated by [llm-pulse]",
    ].map((marker) => md.indexOf(marker));
    expect(order.every((position) => position >= 0)).toBe(true);
    expect([...order].sort((a, b) => a - b)).toEqual(order);
  });

  it("renders the meta line from version, platform and the UTC date", () => {
    expect(renderMarkdownReport(input(HIGH_END, withOllama))).toContain(
      "_llm-pulse v1.1.0 · linux/x64 · 2026-09-13_",
    );
  });
});
