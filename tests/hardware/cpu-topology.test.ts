import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

describe("detectCpuTopology", () => {
  const originalPlatform = process.platform;
  const originalArch = process.arch;

  beforeEach(() => {
    vi.resetModules();
  });

  afterEach(() => {
    Object.defineProperty(process, "platform", { value: originalPlatform });
    Object.defineProperty(process, "arch", { value: originalArch });
    vi.restoreAllMocks();
  });

  it("returns nulls on Windows (unsupported platform for this pass)", async () => {
    Object.defineProperty(process, "platform", { value: "win32" });
    const { detectCpuTopology } = await import("../../src/hardware/cpu-topology.js");
    const t = await detectCpuTopology();
    expect(t.performanceCores).toBeNull();
    expect(t.efficiencyCores).toBeNull();
  });

  it("reads sysctl on Apple Silicon", async () => {
    Object.defineProperty(process, "platform", { value: "darwin" });
    Object.defineProperty(process, "arch", { value: "arm64" });
    vi.doMock("execa", () => ({
      execa: vi.fn().mockImplementation(async (_cmd: string, args: string[]) => {
        if (args.includes("hw.perflevel0.physicalcpu")) return { stdout: "8\n" };
        if (args.includes("hw.perflevel1.physicalcpu")) return { stdout: "4\n" };
        throw new Error("unexpected sysctl");
      }),
    }));
    const { detectCpuTopology } = await import("../../src/hardware/cpu-topology.js");
    const t = await detectCpuTopology();
    expect(t.performanceCores).toBe(8);
    expect(t.efficiencyCores).toBe(4);
  });

  it("returns nulls on Apple Silicon when sysctl is unavailable", async () => {
    Object.defineProperty(process, "platform", { value: "darwin" });
    Object.defineProperty(process, "arch", { value: "arm64" });
    vi.doMock("execa", () => ({
      execa: vi.fn().mockRejectedValue(Object.assign(new Error("ENOENT"), { code: "ENOENT" })),
    }));
    const { detectCpuTopology } = await import("../../src/hardware/cpu-topology.js");
    const t = await detectCpuTopology();
    expect(t.performanceCores).toBeNull();
    expect(t.efficiencyCores).toBeNull();
  });

  it("reads hybrid topology via sysfs on Linux Alder Lake+ (SMT P-cores)", async () => {
    Object.defineProperty(process, "platform", { value: "linux" });
    vi.doMock("node:fs/promises", () => ({
      readFile: vi.fn().mockImplementation(async (path: string) => {
        if (path.endsWith("cpu_core/cpus")) return "0-15"; // 16 logical (8P × SMT)
        if (path.endsWith("cpu_atom/cpus")) return "16-19"; // 4 E-cores
        const core = path.match(/cpu(\d+)\/topology\/core_cpus_list$/);
        if (core) {
          const first = Math.floor(Number(core[1]) / 2) * 2;
          return `${first}-${first + 1}\n`;
        }
        throw Object.assign(new Error("ENOENT"), { code: "ENOENT" });
      }),
    }));
    const { detectCpuTopology } = await import("../../src/hardware/cpu-topology.js");
    const t = await detectCpuTopology();
    expect(t.performanceCores).toBe(8);
    expect(t.efficiencyCores).toBe(4);
  });

  it("does not halve P-cores on hybrid CPUs without SMT (Core Ultra 200S)", async () => {
    Object.defineProperty(process, "platform", { value: "linux" });
    vi.doMock("node:fs/promises", () => ({
      readFile: vi.fn().mockImplementation(async (path: string) => {
        if (path.endsWith("cpu_core/cpus")) return "0-7";
        if (path.endsWith("cpu_atom/cpus")) return "8-23";
        const core = path.match(/cpu(\d+)\/topology\/core_cpus_list$/);
        if (core) return `${core[1]}\n`;
        throw Object.assign(new Error("ENOENT"), { code: "ENOENT" });
      }),
    }));
    const { detectCpuTopology } = await import("../../src/hardware/cpu-topology.js");
    const t = await detectCpuTopology();
    expect(t.performanceCores).toBe(8);
    expect(t.efficiencyCores).toBe(16);
  });

  it("falls back to thread_siblings_list on kernels without core_cpus_list", async () => {
    Object.defineProperty(process, "platform", { value: "linux" });
    vi.doMock("node:fs/promises", () => ({
      readFile: vi.fn().mockImplementation(async (path: string) => {
        if (path.endsWith("cpu_core/cpus")) return "0-3";
        const core = path.match(/cpu(\d+)\/topology\/thread_siblings_list$/);
        if (core) return Number(core[1]) < 2 ? "0-1\n" : "2-3\n";
        throw Object.assign(new Error("ENOENT"), { code: "ENOENT" });
      }),
    }));
    const { detectCpuTopology } = await import("../../src/hardware/cpu-topology.js");
    const t = await detectCpuTopology();
    expect(t.performanceCores).toBe(2);
  });

  it("returns nulls on Linux symmetric CPUs (no hybrid topology files)", async () => {
    Object.defineProperty(process, "platform", { value: "linux" });
    vi.doMock("node:fs/promises", () => ({
      readFile: vi.fn().mockRejectedValue(Object.assign(new Error("ENOENT"), { code: "ENOENT" })),
    }));
    const { detectCpuTopology } = await import("../../src/hardware/cpu-topology.js");
    const t = await detectCpuTopology();
    expect(t.performanceCores).toBeNull();
    expect(t.efficiencyCores).toBeNull();
  });
});
