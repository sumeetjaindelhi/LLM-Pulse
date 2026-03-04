import { EventEmitter } from "node:events";
import si from "systeminformation";
import { execa } from "execa";
import { OLLAMA_API_URL, ALERT_THRESHOLDS } from "../core/constants.js";
import type { SessionStats, ModelUsage } from "../core/types.js";

export interface MonitorSnapshot {
  cpuPercent: number;
  cpuTemp: number | null;
  gpuPercent: number | null;
  gpuTemp: number | null;
  gpuVramUsedMb: number | null;
  gpuVramTotalMb: number | null;
  gpuPowerWatt: number | null;
  gpuClockMhz: number | null;
  ramUsedMb: number;
  ramTotalMb: number;
  ramPercent: number;
  activeModel: string | null;
  tokensPerSec: number | null;
  // Ollama detailed info
  modelSize: string | null; // e.g. "4.9 GB"
  modelQuantization: string | null; // e.g. "Q4_K_M"
  modelContextLength: number | null;
  modelMaxContext: number | null;
}

export class HardwareMonitor extends EventEmitter {
  private interval: ReturnType<typeof setInterval> | null = null;
  private running = false;

  // Sparkline history (ring buffer of last N snapshots)
  readonly cpuHistory: number[] = [];
  readonly gpuHistory: number[] = [];
  readonly tokHistory: number[] = [];
  private readonly maxHistory = ALERT_THRESHOLDS.sparklineHistory;

  // Session tracking
  readonly session: SessionStats = {
    totalTokens: 0,
    totalTimeMs: 0,
    totalRequests: 0,
    startedAt: Date.now(),
    modelHistory: new Map(),
    lastModelSwapAt: null,
  };

  private lastModel: string | null = null;
  private lastTokPerSec: number | null = null;
  private lastPollTime: number = Date.now();

  start(intervalMs = 1000): void {
    if (this.running) return;
    this.running = true;
    this.session.startedAt = Date.now();
    this.lastPollTime = Date.now();

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

  private pushHistory(arr: number[], value: number | null): void {
    arr.push(value ?? 0);
    if (arr.length > this.maxHistory) arr.shift();
  }

  private updateSession(snapshot: MonitorSnapshot): void {
    const now = Date.now();
    const elapsed = now - this.lastPollTime;
    this.lastPollTime = now;

    // Detect model swap
    if (snapshot.activeModel && snapshot.activeModel !== this.lastModel) {
      if (this.lastModel !== null) {
        this.session.lastModelSwapAt = now;
      }
      this.lastModel = snapshot.activeModel;
    }

    // Track tokens + time for active model
    if (snapshot.activeModel && snapshot.tokensPerSec !== null && snapshot.tokensPerSec > 0) {
      const tokensThisPoll = snapshot.tokensPerSec * (elapsed / 1000);
      this.session.totalTokens += tokensThisPoll;
      this.session.totalTimeMs += elapsed;

      let usage = this.session.modelHistory.get(snapshot.activeModel);
      if (!usage) {
        usage = {
          name: snapshot.activeModel,
          avgTokPerSec: 0,
          totalTokens: 0,
          totalTimeMs: 0,
          requests: 0,
          startedAt: now,
        };
        this.session.modelHistory.set(snapshot.activeModel, usage);
      }
      usage.totalTokens += tokensThisPoll;
      usage.totalTimeMs += elapsed;
      usage.avgTokPerSec = usage.totalTokens / (usage.totalTimeMs / 1000);
    }

    // Detect new request (tok/s goes from 0/null to positive)
    if (
      snapshot.tokensPerSec !== null &&
      snapshot.tokensPerSec > 0 &&
      (this.lastTokPerSec === null || this.lastTokPerSec === 0)
    ) {
      this.session.totalRequests++;
      const usage = this.session.modelHistory.get(snapshot.activeModel!);
      if (usage) usage.requests++;
    }

    this.lastTokPerSec = snapshot.tokensPerSec;
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
        gpuPowerWatt: gpu.powerWatt,
        gpuClockMhz: gpu.clockMhz,
        ramUsedMb: mem.usedMb,
        ramTotalMb: mem.totalMb,
        ramPercent: mem.percent,
        activeModel: ollama.model,
        tokensPerSec: ollama.tokensPerSec,
        modelSize: ollama.modelSize,
        modelQuantization: ollama.modelQuantization,
        modelContextLength: ollama.contextLength,
        modelMaxContext: ollama.maxContext,
      };

      // Update histories
      this.pushHistory(this.cpuHistory, snapshot.cpuPercent);
      this.pushHistory(this.gpuHistory, snapshot.gpuPercent);
      this.pushHistory(this.tokHistory, snapshot.tokensPerSec);

      // Update session stats
      this.updateSession(snapshot);

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
        "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total,power.draw,clocks.current.graphics",
        "--format=csv,noheader,nounits",
      ]);
      const parts = stdout.trim().split(",").map((s) => s.trim());
      return {
        percent: parseInt(parts[0], 10),
        temp: parseInt(parts[1], 10),
        vramUsedMb: parseInt(parts[2], 10),
        vramTotalMb: parseInt(parts[3], 10),
        powerWatt: parseFloat(parts[4]) || null,
        clockMhz: parseInt(parts[5], 10) || null,
      };
    } catch {
      return {
        percent: null,
        temp: null,
        vramUsedMb: null,
        vramTotalMb: null,
        powerWatt: null,
        clockMhz: null,
      };
    }
  }

  private async pollOllama() {
    try {
      const res = await fetch(`${OLLAMA_API_URL}/api/ps`, {
        signal: AbortSignal.timeout(2000),
      });
      if (!res.ok)
        return {
          model: null,
          tokensPerSec: null,
          modelSize: null,
          modelQuantization: null,
          contextLength: null,
          maxContext: null,
        };

      const data = (await res.json()) as {
        models: Array<{
          name: string;
          size?: number;
          details?: {
            tokens_per_second?: number;
            quantization_level?: string;
            parameter_size?: string;
          };
          size_vram?: number;
        }>;
      };

      if (data.models.length === 0)
        return {
          model: null,
          tokensPerSec: null,
          modelSize: null,
          modelQuantization: null,
          contextLength: null,
          maxContext: null,
        };

      const active = data.models[0];
      const sizeGb = active.size
        ? `${(active.size / (1024 * 1024 * 1024)).toFixed(1)} GB`
        : null;

      return {
        model: active.name,
        tokensPerSec: active.details?.tokens_per_second ?? null,
        modelSize: sizeGb,
        modelQuantization: active.details?.quantization_level ?? null,
        contextLength: null, // Ollama /api/ps doesn't expose this directly
        maxContext: null,
      };
    } catch {
      return {
        model: null,
        tokensPerSec: null,
        modelSize: null,
        modelQuantization: null,
        contextLength: null,
        maxContext: null,
      };
    }
  }
}
