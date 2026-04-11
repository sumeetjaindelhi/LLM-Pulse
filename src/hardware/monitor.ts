import { EventEmitter } from "node:events";
import si from "systeminformation";
import { execa } from "execa";
import { OLLAMA_API_URL, ALERT_THRESHOLDS } from "../core/constants.js";
import { OllamaPsSchema } from "../core/api-schemas.js";
import { parseRocmCsv } from "./gpu.js";
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
  gpuVendor: string | null;
  gpuModel: string | null;
  activeModel: string | null;
  tokensPerSec: number | null;
  // Ollama detailed info
  modelSize: string | null; // e.g. "4.9 GB"
  modelQuantization: string | null; // e.g. "Q4_K_M"
  modelContextLength: number | null;
  modelMaxContext: number | null;
}

export class HardwareMonitor extends EventEmitter {
  private timeout: ReturnType<typeof setTimeout> | null = null;
  private running = false;
  private polling = false; // guard against overlapping polls
  private intervalMs = 2000;
  private gpuVendor: "NVIDIA" | "AMD" | "Apple" | "unknown" = "unknown";
  private gpuModelName: string | null = null;
  // Cached on Apple Silicon only: total unified memory, used as the GPU's
  // effective "total VRAM" since ioreg's PerformanceStatistics doesn't expose
  // a total VRAM figure for Apple Silicon.
  private appleVramTotalMb: number | null = null;
  private ollamaBaseUrl: string;

  // Sparkline history (ring buffer of last N snapshots)
  readonly cpuHistory: number[] = [];
  readonly gpuHistory: number[] = [];
  readonly tokHistory: number[] = [];
  readonly gpuTempHistory: number[] = [];
  readonly gpuVramHistory: number[] = [];
  readonly gpuPowerHistory: number[] = [];
  private readonly maxHistory = ALERT_THRESHOLDS.sparklineHistory;
  // Cap on session.modelHistory entries — prevents unbounded growth over
  // long-running monitor sessions. 50 is roomy for realistic usage (active
  // developers typically cycle through 10-20 quants max) while keeping the
  // per-tick React state copy O(1) bounded. Eviction is FIFO via JS Map's
  // insertion-order iteration.
  private readonly maxModelHistory = 50;

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

  constructor(ollamaHost?: string) {
    super();
    this.ollamaBaseUrl = ollamaHost || OLLAMA_API_URL;
  }

  start(intervalMs = 2000): void {
    if (this.running) return;
    this.running = true;
    this.intervalMs = intervalMs;
    this.session.startedAt = Date.now();
    this.lastPollTime = Date.now();

    // Detect GPU vendor before first poll so AMD/Apple users get correct data immediately
    this.detectGpuVendor().then(() => {
      if (!this.running) return;
      this.poll();
    });
  }

  private async detectGpuVendor(): Promise<void> {
    try {
      const graphics = await si.graphics();
      for (const c of graphics.controllers) {
        if (!c.model || c.model.includes("Microsoft")) continue;
        const v = (c.vendor || "").toLowerCase();
        if (v.includes("nvidia")) { this.gpuVendor = "NVIDIA"; this.gpuModelName = c.model; return; }
        if (v.includes("amd") || v.includes("advanced micro")) { this.gpuVendor = "AMD"; this.gpuModelName = c.model; return; }
        if (v.includes("apple")) {
          this.gpuVendor = "Apple";
          this.gpuModelName = c.model;
          // Cache total system memory as the Apple GPU's effective VRAM.
          try {
            const mem = await si.mem();
            this.appleVramTotalMb = Math.round(mem.total / (1024 * 1024));
          } catch {
            // leave as null; pollGpuApple will report null vramTotalMb
          }
          return;
        }
      }
    } catch {
      // keep "unknown"
    }
  }

  stop(): void {
    this.running = false;
    if (this.timeout) {
      clearTimeout(this.timeout);
      this.timeout = null;
    }
    // Release the modelHistory Map so long-running processes that stop and
    // restart monitors don't accumulate state across sessions.
    this.session.modelHistory.clear();
  }

  private pushHistory(arr: number[], value: number | null): void {
    arr.push(value ?? 0);
    if (arr.length > this.maxHistory) arr.shift();
  }

  private updateSession(snapshot: MonitorSnapshot): void {
    const now = Date.now();
    // Cap elapsed at 1.5× the poll interval. Without this cap, a skipped poll
    // (gatherSnapshot took >interval due to slow ioreg/fetch) gives elapsed=4s+
    // and multiplying the instantaneous tokensPerSec by that inflates the
    // token count. Ollama's /api/ps reports the LAST-KNOWN rate, not a rolling
    // average, so over-counting also happens when the model goes idle between
    // prompts. The cap bounds the error to ~1 interval per incident.
    const elapsed = Math.min(now - this.lastPollTime, this.intervalMs * 1.5);
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
        // FIFO-evict the oldest entry if we're at the cap. JS Map preserves
        // insertion order, so the first key is the oldest.
        if (this.session.modelHistory.size >= this.maxModelHistory) {
          const oldest = this.session.modelHistory.keys().next().value;
          if (oldest !== undefined) this.session.modelHistory.delete(oldest);
        }
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
      snapshot.activeModel &&
      snapshot.tokensPerSec !== null &&
      snapshot.tokensPerSec > 0 &&
      (this.lastTokPerSec === null || this.lastTokPerSec === 0)
    ) {
      this.session.totalRequests++;
      const usage = this.session.modelHistory.get(snapshot.activeModel);
      if (usage) usage.requests++;
    }

    this.lastTokPerSec = snapshot.tokensPerSec;
  }

  private async gatherSnapshot(): Promise<MonitorSnapshot> {
    const [cpu, mem, gpu, ollama] = await Promise.all([
      this.pollCpu(),
      this.pollMemory(),
      this.pollGpu(),
      this.pollOllama(),
    ]);

    return {
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
      gpuVendor: this.gpuVendor === "unknown" ? null : this.gpuVendor,
      gpuModel: this.gpuModelName,
      activeModel: ollama.model,
      tokensPerSec: ollama.tokensPerSec,
      modelSize: ollama.modelSize,
      modelQuantization: ollama.modelQuantization,
      modelContextLength: ollama.contextLength,
      modelMaxContext: ollama.maxContext,
    };
  }

  /**
   * One-shot snapshot — for request/response consumers (e.g. MCP tools).
   * Does NOT start the polling loop, mutate session state, or emit events.
   */
  async takeSnapshot(): Promise<MonitorSnapshot> {
    if (this.gpuVendor === "unknown") {
      await this.detectGpuVendor();
    }
    return this.gatherSnapshot();
  }

  private async poll(): Promise<void> {
    if (!this.running) return;

    // Skip if previous poll is still running (dedup subprocess calls)
    if (this.polling) return;
    this.polling = true;

    try {
      const snapshot = await this.gatherSnapshot();

      // Update histories
      this.pushHistory(this.cpuHistory, snapshot.cpuPercent);
      this.pushHistory(this.gpuHistory, snapshot.gpuPercent);
      this.pushHistory(this.tokHistory, snapshot.tokensPerSec);
      this.pushHistory(this.gpuTempHistory, snapshot.gpuTemp);
      const vramPercent = snapshot.gpuVramUsedMb !== null && snapshot.gpuVramTotalMb !== null && snapshot.gpuVramTotalMb > 0
        ? (snapshot.gpuVramUsedMb / snapshot.gpuVramTotalMb) * 100
        : null;
      this.pushHistory(this.gpuVramHistory, vramPercent);
      this.pushHistory(this.gpuPowerHistory, snapshot.gpuPowerWatt);

      // Update session stats
      this.updateSession(snapshot);

      this.emit("snapshot", snapshot);
    } catch {
      // Swallow polling errors to keep running
    } finally {
      this.polling = false;
      if (this.running) {
        this.timeout = setTimeout(() => this.poll(), this.intervalMs);
      }
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
    const totalMb = Math.max(Math.round(mem.total / (1024 * 1024)), 1);
    const availableMb = Math.round(mem.available / (1024 * 1024));
    // Match detectMemory() semantics: exclude reclaimable buffcache from "used"
    // so the TUI RAM bar agrees with the scan output.
    const usedMb = Math.max(0, totalMb - availableMb);
    return {
      totalMb,
      usedMb,
      percent: Math.round((usedMb / totalMb) * 100),
    };
  }

  private async pollGpu() {
    const nullResult = { percent: null, temp: null, vramUsedMb: null, vramTotalMb: null, powerWatt: null, clockMhz: null };

    if (this.gpuVendor === "AMD") {
      return this.pollGpuAmd();
    }

    if (this.gpuVendor === "Apple") {
      return this.pollGpuApple();
    }

    // NVIDIA (default) — also used when vendor is unknown as a first attempt
    try {
      const { stdout } = await execa("nvidia-smi", [
        "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total,power.draw,clocks.current.graphics",
        "--format=csv,noheader,nounits",
      ], { timeout: 5000 });
      const parts = stdout.trim().split(",").map((s) => s.trim());
      const percent = parseInt(parts[0], 10);
      const vramUsedMb = parseInt(parts[2], 10);
      const vramTotalMb = parseInt(parts[3], 10);
      if (isNaN(percent) || isNaN(vramUsedMb) || isNaN(vramTotalMb)) {
        return nullResult;
      }
      return {
        percent,
        temp: parseInt(parts[1], 10) || null,
        vramUsedMb,
        vramTotalMb,
        powerWatt: parseFloat(parts[4]) || null,
        clockMhz: parseInt(parts[5], 10) || null,
      };
    } catch {
      return nullResult;
    }
  }

  private async pollGpuAmd() {
    const nullResult = { percent: null, temp: null, vramUsedMb: null, vramTotalMb: null, powerWatt: null, clockMhz: null };
    try {
      const { stdout } = await execa("rocm-smi", [
        "--showmeminfo", "vram",
        "--showtemp",
        "--showuse",
        "--csv",
      ], { timeout: 5000 });

      const stats = parseRocmCsv(stdout);
      if (stats.vramTotalMb === 0) return nullResult;

      return {
        percent: stats.utilizationPercent || null,
        temp: stats.temperatureCelsius || null,
        vramUsedMb: stats.vramUsedMb,
        vramTotalMb: stats.vramTotalMb,
        powerWatt: null,
        clockMhz: null,
      };
    } catch {
      return nullResult;
    }
  }

  /**
   * Apple Silicon GPU poll via `ioreg -c AGXAccelerator`. No sudo required.
   *
   * The `PerformanceStatistics` dict exposes live metrics like:
   *   "Device Utilization %"=27
   *   "In use system memory"=1439367168
   *   "Alloc system memory"=2774810624
   * (verified live on an M5 Pro, macOS 15.x — all on one line, no spaces
   * around `=`; the regex tolerates both).
   *
   * GPU temperature, clock, and power aren't in this dict — those live in the
   * `IOReportLegend` channels which require more elaborate parsing. We return
   * null for what we can't read (honest "unknown") rather than guess.
   */
  private async pollGpuApple() {
    const nullResult = { percent: null, temp: null, vramUsedMb: null, vramTotalMb: null, powerWatt: null, clockMhz: null };
    try {
      const { stdout } = await execa(
        "ioreg",
        ["-r", "-d", "1", "-w", "0", "-c", "AGXAccelerator"],
        { timeout: 3000 },
      );
      const statsMatch = stdout.match(/"PerformanceStatistics"\s*=\s*\{([\s\S]*?)\}/);
      if (!statsMatch) return { ...nullResult, vramTotalMb: this.appleVramTotalMb };
      const block = statsMatch[1];
      const readNum = (pattern: RegExp): number | null => {
        const m = block.match(pattern);
        if (!m) return null;
        const n = parseInt(m[1], 10);
        return Number.isNaN(n) ? null : n;
      };
      // `\s*%?` tolerates both "Device Utilization" and "Device Utilization %"
      // variants seen across macOS versions.
      const utilPercent = readNum(/"Device Utilization\s*%?"\s*=\s*(\d+)/);
      const inUseBytes = readNum(/"In use system memory"\s*=\s*(\d+)/);
      return {
        percent: utilPercent,
        temp: null,
        vramUsedMb: inUseBytes !== null ? Math.round(inUseBytes / (1024 * 1024)) : null,
        // Report the cached unified-memory total as the GPU's "total VRAM" —
        // architecturally accurate on Apple Silicon (Metal can address all
        // system RAM). If detectGpuVendor couldn't read si.mem(), this is null.
        vramTotalMb: this.appleVramTotalMb,
        powerWatt: null,
        clockMhz: null,
      };
    } catch {
      return { ...nullResult, vramTotalMb: this.appleVramTotalMb };
    }
  }

  private async pollOllama() {
    try {
      const res = await fetch(`${this.ollamaBaseUrl}/api/ps`, {
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

      const data = OllamaPsSchema.parse(await res.json());

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
