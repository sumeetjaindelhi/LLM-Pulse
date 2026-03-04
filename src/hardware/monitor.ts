import { EventEmitter } from "node:events";
import si from "systeminformation";
import { execa } from "execa";
import { OLLAMA_API_URL } from "../core/constants.js";

export interface MonitorSnapshot {
  cpuPercent: number;
  cpuTemp: number | null;
  gpuPercent: number | null;
  gpuTemp: number | null;
  gpuVramUsedMb: number | null;
  gpuVramTotalMb: number | null;
  ramUsedMb: number;
  ramTotalMb: number;
  ramPercent: number;
  activeModel: string | null;
  tokensPerSec: number | null;
}

export class HardwareMonitor extends EventEmitter {
  private interval: ReturnType<typeof setInterval> | null = null;
  private running = false;

  start(intervalMs = 1000): void {
    if (this.running) return;
    this.running = true;

    // Fire immediately, then on interval
    this.poll();
    this.interval = setInterval(() => this.poll(), intervalMs);
  }

  stop(): void {
    this.running = false;
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
    }
  }

  private async poll(): Promise<void> {
    if (!this.running) return;

    try {
      const [cpu, mem, gpu, ollama] = await Promise.all([
        this.pollCpu(),
        this.pollMemory(),
        this.pollGpu(),
        this.pollOllama(),
      ]);

      const snapshot: MonitorSnapshot = {
        cpuPercent: cpu.percent,
        cpuTemp: cpu.temp,
        gpuPercent: gpu.percent,
        gpuTemp: gpu.temp,
        gpuVramUsedMb: gpu.vramUsedMb,
        gpuVramTotalMb: gpu.vramTotalMb,
        ramUsedMb: mem.usedMb,
        ramTotalMb: mem.totalMb,
        ramPercent: mem.percent,
        activeModel: ollama.model,
        tokensPerSec: ollama.tokensPerSec,
      };

      this.emit("snapshot", snapshot);
    } catch {
      // Swallow polling errors to keep running
    }
  }

  private async pollCpu() {
    const [load, temp] = await Promise.all([
      si.currentLoad(),
      si.cpuTemperature().catch(() => null),
    ]);
    return {
      percent: Math.round(load.currentLoad),
      temp: temp?.main ?? null,
    };
  }

  private async pollMemory() {
    const mem = await si.mem();
    const totalMb = Math.round(mem.total / (1024 * 1024));
    const usedMb = Math.round(mem.used / (1024 * 1024));
    return {
      totalMb,
      usedMb,
      percent: Math.round((usedMb / totalMb) * 100),
    };
  }

  private async pollGpu() {
    try {
      const { stdout } = await execa("nvidia-smi", [
        "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
      ]);
      const [util, temp, used, total] = stdout.trim().split(",").map((s) => s.trim());
      return {
        percent: parseInt(util, 10),
        temp: parseInt(temp, 10),
        vramUsedMb: parseInt(used, 10),
        vramTotalMb: parseInt(total, 10),
      };
    } catch {
      return { percent: null, temp: null, vramUsedMb: null, vramTotalMb: null };
    }
  }

  private async pollOllama() {
    try {
      const res = await fetch(`${OLLAMA_API_URL}/api/ps`, {
        signal: AbortSignal.timeout(2000),
      });
      if (!res.ok) return { model: null, tokensPerSec: null };
      const data = (await res.json()) as {
        models: Array<{ name: string; details?: { tokens_per_second?: number } }>;
      };
      if (data.models.length === 0) return { model: null, tokensPerSec: null };
      const active = data.models[0];
      return {
        model: active.name,
        tokensPerSec: active.details?.tokens_per_second ?? null,
      };
    } catch {
      return { model: null, tokensPerSec: null };
    }
  }
}
