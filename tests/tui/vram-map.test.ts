import { describe, it, expect } from "vitest";
import { Writable } from "node:stream";
import React from "react";
import { render } from "ink";
import { VramMap } from "../../src/tui/VramMap.js";
import type { MonitorSnapshot } from "../../src/hardware/monitor.js";
import { fitsInFreeVram } from "../../src/tui/vram-breakdown.js";

const snapshot: MonitorSnapshot = {
  cpuPercent: 0, cpuTemp: null, gpuPercent: null, gpuTemp: null, gpuVramUsedMb: null, gpuVramTotalMb: null,
  gpuPowerWatt: null, gpuClockMhz: null, gpuCount: null, gpuUnifiedMemory: false, ramUsedMb: 0, ramTotalMb: 1, ramPercent: 0,
  gpuVendor: null, gpuModel: null, activeModel: null, tokensPerSec: null, modelSize: null, modelVramMb: null,
  modelQuantization: null, modelContextLength: null, modelMaxContext: null,
};

async function renderText(over: Partial<MonitorSnapshot>): Promise<string> {
  let output = "";
  const stdout = Object.assign(
    new Writable({ write(chunk, _enc, done) { output += chunk.toString(); done(); } }),
    { columns: 100, rows: 40, isTTY: false },
  );
  const app = render(React.createElement(VramMap, { snapshot: { ...snapshot, ...over } }), {
    stdout: stdout as unknown as NodeJS.WriteStream,
    debug: true,
    patchConsole: false,
  });
  await new Promise((resolve) => setTimeout(resolve, 20));
  app.unmount();
  return output;
}

describe("VramMap", () => {
  it("says a GPU with utilization but no memory figures is unified memory, not missing (GB10)", async () => {
    const text = await renderText({ gpuPercent: 12, gpuVendor: "NVIDIA", gpuModel: "NVIDIA GB10" });
    expect(text).toContain("GPU reports no memory figures (unified memory");
    expect(text).not.toContain("no supported GPU detected");
  });

  it("keeps the no-GPU message when nothing is read", async () => {
    expect(await renderText({})).toContain("no supported GPU detected");
  });

  it("shows Used instead of the Loaded/Other split on a multi-GPU machine", async () => {
    const text = await renderText({ gpuVramUsedMb: 21504, gpuVramTotalMb: 24576, modelVramMb: 41984, gpuCount: 2, activeModel: "llama3.3:70b" });
    expect(text).not.toContain("Loaded (Ollama)");
    expect(text).toContain("Used");
    expect(text).toContain("88% used");
  });

  it("shows the Loaded/Other split when the monitor reads the only GPU", async () => {
    const text = await renderText({ gpuVramUsedMb: 6144, gpuVramTotalMb: 16384, modelVramMb: 4096, gpuCount: 1, activeModel: "llama3.1:8b" });
    expect(text).toContain("Loaded (Ollama)");
    expect(text).toContain("Other GPU use");
  });

  // The APU pool is GTT, a slice of system RAM: free GTT 32768 - 100 = 32668 MiB, RAM available 9000 - 1000 = 8000 MiB,
  // so the budget is 8000 - 457 - 1024 = 6519, not 32668 - 1481 = 31187.
  it("caps the Can-still-fit list by the free RAM on an AMD APU", async () => {
    const capped = fitsInFreeVram(32668, "AMD", 8000);
    const uncapped = fitsInFreeVram(32668, "AMD", null);
    expect(capped[0].name).not.toBe(uncapped[0].name);
    const text = await renderText({
      gpuVendor: "AMD", gpuUnifiedMemory: true, gpuVramUsedMb: 100, gpuVramTotalMb: 32768, gpuCount: 1,
      ramUsedMb: 1000, ramTotalMb: 9000,
    });
    expect(text).toContain(capped[0].name);
    expect(text).not.toContain(uncapped[0].name);
  });
});
