import { describe, it, expect, afterEach, vi } from "vitest";
import { budgetGpus, defaultContextFor, getAvailableVram, ignoredGpu, ignoredGpuLabel, isCpuOnly, suggestGpuOffload } from "../../src/analysis/scorer.js";
import { getModelById } from "../../src/models/database.js";
import type { GpuInfo, HardwareProfile } from "../../src/core/types.js";
import cpuOnly from "../fixtures/hardware-profiles/cpu-only.json";

const gpuBase: GpuInfo = {
  vendor: "NVIDIA",
  model: "GPU",
  vramMb: 0,
  driverVersion: "",
  acceleratorVersion: null,
  acceleratorType: "cuda",
  utilizationPercent: null,
  temperatureCelsius: null,
  vramUsedMb: null,
};

function withGpus(gpus: GpuInfo[], availableMb = 50_000): HardwareProfile {
  const base = cpuOnly as HardwareProfile;
  const primary = gpus.filter((g) => !g.unifiedMemory).sort((a, b) => b.vramMb - a.vramMb)[0] ?? gpus[0] ?? null;
  return { ...base, memory: { ...base.memory, availableMb }, gpus, primaryGpu: primary };
}

const apu = (gfxTarget: string): GpuInfo => ({
  ...gpuBase,
  vendor: "AMD",
  model: "Radeon APU",
  vramMb: 32768,
  acceleratorType: "rocm",
  unifiedMemory: true,
  gfxTarget,
});
const rtx4090: GpuInfo = { ...gpuBase, vramMb: 24564 };
const rx7900: GpuInfo = { ...gpuBase, vendor: "AMD", vramMb: 24576, acceleratorType: "rocm" };

afterEach(() => vi.unstubAllEnvs());

describe("integrated GPU admission (O discover/runner.go:31-34, 382-435)", () => {
  const model = getModelById("llama-3.1-8b")!;

  it("drops an AMD APU that is not gfx1151: CPU-only budget and the 4096 default context", () => {
    const hw = withGpus([apu("gfx1103")], 50_000); // Radeon 780M
    expect(isCpuOnly(hw)).toBe(true);
    expect(getAvailableVram(hw)).toBe(50_000);
    expect(defaultContextFor(model, hw)).toBe(4096);
    expect(budgetGpus(hw)).toEqual([]);
  });

  it("keeps gfx1151 (Strix Halo): min(GTT 32768, available 50000) - 457 - 1024 = 31287", () => {
    const hw = withGpus([apu("gfx1151")], 50_000);
    expect(isCpuOnly(hw)).toBe(false);
    expect(getAvailableVram(hw)).toBe(31287);
  });

  it("keeps any integrated AMD GPU when OLLAMA_IGPU_ENABLE is truthy, drops it again when false", () => {
    vi.stubEnv("OLLAMA_IGPU_ENABLE", "1");
    expect(isCpuOnly(withGpus([apu("gfx1103")]))).toBe(false);
    vi.stubEnv("OLLAMA_IGPU_ENABLE", "false");
    expect(isCpuOnly(withGpus([apu("gfx1103")]))).toBe(true);
  });

  it("explicit OLLAMA_IGPU_ENABLE=0 drops every non-Metal integrated GPU, GB10 and gfx1151 too (O integratedGPUAdmission)", () => {
    const gb10: GpuInfo = { ...gpuBase, vramMb: 100_000, unifiedMemory: true };
    const metal: GpuInfo = { ...gpuBase, vendor: "Apple", vramMb: 18186, acceleratorType: "metal", unifiedMemory: true };
    vi.stubEnv("OLLAMA_IGPU_ENABLE", "0");
    expect(isCpuOnly(withGpus([gb10]))).toBe(true);
    expect(isCpuOnly(withGpus([apu("gfx1151")]))).toBe(true);
    expect(isCpuOnly(withGpus([metal]))).toBe(false);
    vi.stubEnv("OLLAMA_IGPU_ENABLE", "");
    expect(isCpuOnly(withGpus([gb10]))).toBe(false);
    expect(isCpuOnly(withGpus([apu("gfx1103")]))).toBe(true);
  });

  it("explicit OLLAMA_IGPU_ENABLE=1 admits every integrated GPU", () => {
    vi.stubEnv("OLLAMA_IGPU_ENABLE", "1");
    expect(budgetGpus(withGpus([apu("gfx1103")]))).toHaveLength(1);
  });

  it("treats a lone non-unified primary GPU as discrete when the GPU list is empty", () => {
    const hw = { ...withGpus([]), primaryGpu: rx7900 };
    expect(isCpuOnly(hw)).toBe(false);
    expect(getAvailableVram(hw)).toBe(24576 - 457 - 1024);
  });

  it("gives no offload split when the budget group is the unified APU, though the primary GPU is discrete", () => {
    const hw = withGpus([rtx4090, apu("gfx1151")]); // ROCm group 31287 beats CUDA 23083; the primary is the 4090
    const big = getModelById("llama-3.3-70b")!;
    expect(hw.primaryGpu).toBe(rtx4090);
    expect(suggestGpuOffload(big, big.quantizations.find((q) => q.name === "Q4_K_M")!, hw, 4096)).toBeNull();
  });

  it("keeps CUDA unified memory (GB10): available RAM - 457 - 1024", () => {
    const gb10: GpuInfo = { ...gpuBase, vramMb: 100_000, unifiedMemory: true };
    expect(getAvailableVram(withGpus([gb10], 90_000))).toBe(90_000 - 457 - 1024);
  });

  it("keeps Strix Halo beside a discrete GPU, in its own ROCm group", () => {
    const hw = withGpus([rtx4090, apu("gfx1151")]);
    // groups: CUDA 24564 - 1481 = 23083; ROCm min(32768, 50000) - 1481 = 31287 -> ROCm wins
    expect(getAvailableVram(hw)).toBe(31287);
    // default context still counts every kept GPU: 24564 + 32768 = 57332 MiB >= 47 GiB -> 262144, clamped to the 131072 training context
    expect(defaultContextFor(model, hw)).toBe(131072);
  });
});

describe("one runtime library per model (O ml/device.go ByLibrary, O server/sched.go:1008-1128)", () => {
  const model = getModelById("llama-3.1-8b")!;
  const mixed = withGpus([rtx4090, rx7900]);

  it("budgets the group with the largest summed budget, not the sum across vendors", () => {
    // CUDA 24564 - 1481 = 23083; ROCm 24576 - 1481 = 23095
    expect(getAvailableVram(mixed)).toBe(23095);
    expect(budgetGpus(mixed)).toEqual([rx7900]);
  });

  it("still sums every GPU for the default context: 49140 MiB >= 47 GiB", () => {
    expect(defaultContextFor(model, mixed)).toBe(131072); // 262144 clamped to the training context
  });

  it("sums GPUs of one library: 2 x 24564 -> 2 x 23083", () => {
    const pair = withGpus([rtx4090, rtx4090]);
    expect(getAvailableVram(pair)).toBe(46166);
    expect(budgetGpus(pair)).toHaveLength(2);
  });

  it("offload suggestions use the chosen group's budget", () => {
    const big = getModelById("llama-3.3-70b")!;
    const q4 = big.quantizations.find((q) => q.name === "Q4_K_M")!;
    const offload = suggestGpuOffload(big, q4, mixed, 4096)!;
    expect(offload.reason).toBe("partial_offload");
    const single = suggestGpuOffload(big, q4, withGpus([rx7900]), 4096)!;
    expect(offload.gpuLayers).toBe(single.gpuLayers);
  });
});

describe("ignoredGpu", () => {
  it("is the detected GPU Ollama drops, else null", () => {
    const dropped = apu("gfx1103");
    expect(ignoredGpu(withGpus([dropped]))).toBe(dropped);
    expect(ignoredGpu(withGpus([apu("gfx1151")]))).toBeNull();
    expect(ignoredGpu(withGpus([rtx4090]))).toBeNull();
    expect(ignoredGpu(withGpus([]))).toBeNull();
  });

  it("labels it with its model and how to enable it", () => {
    expect(ignoredGpuLabel(apu("gfx1103"))).toBe("Radeon APU not used by Ollama (integrated; OLLAMA_IGPU_ENABLE=1 enables it)");
  });
});
