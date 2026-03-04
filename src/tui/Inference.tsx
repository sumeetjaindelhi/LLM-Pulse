import React from "react";
import { Text, Box } from "ink";
import { Sparkline } from "./Sparkline.js";
import type { MonitorSnapshot } from "../hardware/monitor.js";
import type { SessionStats } from "../core/types.js";

interface InferenceProps {
  snapshot: MonitorSnapshot;
  session: SessionStats;
  tokHistory: number[];
}

function formatDuration(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  if (m >= 60) {
    const h = Math.floor(m / 60);
    const rm = m % 60;
    return `${h}h ${rm}m ${s}s`;
  }
  return `${m}m ${String(s).padStart(2, "0")}s`;
}

function formatNumber(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return Math.round(n).toLocaleString();
}

function renderLargeChart(data: number[], width: number): string[] {
  if (data.length === 0) return ["  No throughput data yet"];

  const slice = data.slice(-width);
  const max = Math.max(...slice, 1);
  const lines: string[] = [];
  const height = 6;
  const blocks = [" ", "\u2581", "\u2582", "\u2583", "\u2584", "\u2585", "\u2586", "\u2587", "\u2588"];

  // Build chart rows from top to bottom
  for (let row = height; row >= 1; row--) {
    const threshold = (row / height) * max;
    const prevThreshold = ((row - 1) / height) * max;
    let line = "";

    for (const val of slice) {
      if (val >= threshold) {
        line += "\u2588";
      } else if (val > prevThreshold) {
        // Partial block
        const fraction = (val - prevThreshold) / (threshold - prevThreshold);
        const blockIdx = Math.round(fraction * 8);
        line += blocks[Math.min(blockIdx, 8)];
      } else {
        line += " ";
      }
    }

    // Pad if needed
    const padding = width - slice.length;
    if (padding > 0) line = " ".repeat(padding) + line;

    const label = row === height ? `${Math.round(max).toString().padStart(3)} \u2524` : "    \u2502";
    lines.push(`  ${label}${line}`);
  }

  // Bottom axis
  const axisLine = "\u2500".repeat(width);
  lines.push(`    \u2514${axisLine} tok/s`);

  return lines;
}

export function Inference({ snapshot, session, tokHistory }: InferenceProps) {
  const chartWidth = 50;
  const chartLines = renderLargeChart(tokHistory, chartWidth);
  const avgTokPerSec =
    session.totalTimeMs > 0 ? session.totalTokens / (session.totalTimeMs / 1000) : 0;
  const uptime = Date.now() - session.startedAt;

  // Sort model history by time spent (descending)
  const models = Array.from(session.modelHistory.values()).sort(
    (a, b) => b.totalTimeMs - a.totalTimeMs,
  );

  const maxModelTime = models.length > 0 ? Math.max(...models.map((m) => m.totalTimeMs)) : 1;

  return (
    <Box flexDirection="column">
      <Text bold dimColor>{"  Throughput (last 60s)"}</Text>

      {/* Large sparkline chart */}
      {chartLines.map((line, i) => (
        <Text key={`chart-${i}`} color="green">{line}</Text>
      ))}

      <Text>{""}</Text>

      {/* Session stats */}
      <Text bold dimColor>{"  Session Stats"}</Text>
      <Text dimColor>{"  \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"}</Text>

      <Text>
        <Text dimColor>{"  Total tokens:     "}</Text>
        <Text>{formatNumber(session.totalTokens)}</Text>
      </Text>
      <Text>
        <Text dimColor>{"  Total time:       "}</Text>
        <Text>{formatDuration(uptime)}</Text>
      </Text>
      <Text>
        <Text dimColor>{"  Avg tok/s:        "}</Text>
        <Text color="cyan">{avgTokPerSec > 0 ? avgTokPerSec.toFixed(1) : "\u2014"}</Text>
      </Text>
      <Text>
        <Text dimColor>{"  Requests:         "}</Text>
        <Text>{session.totalRequests}</Text>
      </Text>
      <Text>
        <Text dimColor>{"  Current model:    "}</Text>
        <Text color="cyan">{snapshot.activeModel ?? "none"}</Text>
      </Text>

      {/* Model history */}
      {models.length > 0 && (
        <Box flexDirection="column" marginTop={1}>
          <Text bold dimColor>{"  Model History (this session)"}</Text>
          {models.map((m, i) => {
            const barWidth = 20;
            const filled = Math.round((m.totalTimeMs / maxModelTime) * barWidth);
            const empty = barWidth - filled;
            const bar = "\u2588".repeat(filled) + "\u2591".repeat(empty);
            return (
              <Text key={m.name}>
                <Text dimColor>{"  "}</Text>
                <Text color="cyan">{m.name.padEnd(16)}</Text>
                <Text dimColor>{` ${m.avgTokPerSec.toFixed(1).padStart(5)} tok/s `}</Text>
                <Text dimColor>{formatDuration(m.totalTimeMs).padStart(8)}</Text>
                <Text dimColor>{"  ["}</Text>
                <Text color="green">{bar}</Text>
                <Text dimColor>{"]"}</Text>
              </Text>
            );
          })}
        </Box>
      )}
    </Box>
  );
}
