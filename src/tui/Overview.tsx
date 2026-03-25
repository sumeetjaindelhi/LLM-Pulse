import React from "react";
import { Text, Box } from "ink";
import { Sparkline } from "./Sparkline.js";
import { AlertBar } from "./AlertBar.js";
import type { MonitorSnapshot } from "../hardware/monitor.js";
import type { SessionStats } from "../core/types.js";

interface OverviewProps {
  snapshot: MonitorSnapshot;
  session: SessionStats;
  cpuHistory: number[];
  gpuHistory: number[];
  tokHistory: number[];
}

function barColor(percent: number): string {
  return percent >= 90 ? "red" : percent >= 70 ? "yellow" : "green";
}

function renderBar(percent: number, width = 20): string {
  const filled = Math.round((percent / 100) * width);
  const empty = width - filled;
  return "\u2588".repeat(filled) + "\u2591".repeat(empty);
}

function formatUptime(startedAt: number): string {
  const ms = Date.now() - startedAt;
  const totalSec = Math.floor(ms / 1000);
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  if (m >= 60) {
    const h = Math.floor(m / 60);
    const rm = m % 60;
    return `${h}h ${rm}m`;
  }
  return `${m}m ${String(s).padStart(2, "0")}s`;
}

export const Overview = React.memo(function Overview({ snapshot, session, cpuHistory, gpuHistory, tokHistory }: OverviewProps) {
  const vramPercent =
    snapshot.gpuVramUsedMb !== null && snapshot.gpuVramTotalMb !== null && snapshot.gpuVramTotalMb > 0
      ? Math.round((snapshot.gpuVramUsedMb / snapshot.gpuVramTotalMb) * 100)
      : null;

  const vramStr =
    snapshot.gpuVramUsedMb !== null && snapshot.gpuVramTotalMb !== null
      ? `${(snapshot.gpuVramUsedMb / 1024).toFixed(1)}/${(snapshot.gpuVramTotalMb / 1024).toFixed(1)} GB`
      : "";

  return (
    <Box flexDirection="column">
      {/* CPU bar + sparkline */}
      <Text>
        <Text dimColor>{"  CPU  "}</Text>
        <Text color={barColor(snapshot.cpuPercent)}>{renderBar(snapshot.cpuPercent)}</Text>
        <Text dimColor>{` ${String(snapshot.cpuPercent).padStart(3)}%`}</Text>
        <Text dimColor>{snapshot.cpuTemp !== null ? `  ${snapshot.cpuTemp}\u00B0C` : ""}</Text>
        <Text>{"     "}</Text>
        <Sparkline data={cpuHistory} width={10} color="cyan" />
      </Text>

      {/* GPU bar + sparkline */}
      {snapshot.gpuPercent !== null ? (
        <Text>
          <Text dimColor>{"  GPU  "}</Text>
          <Text color={barColor(snapshot.gpuPercent)}>{renderBar(snapshot.gpuPercent)}</Text>
          <Text dimColor>{` ${String(snapshot.gpuPercent).padStart(3)}%`}</Text>
          <Text dimColor>{snapshot.gpuTemp !== null ? `  ${snapshot.gpuTemp}\u00B0C` : ""}</Text>
          <Text>{"     "}</Text>
          <Sparkline data={gpuHistory} width={10} color="magenta" />
        </Text>
      ) : (
        <Text>
          <Text dimColor>{"  GPU  "}</Text>
          <Text color="gray">{"\u2591".repeat(20)}  No GPU data</Text>
        </Text>
      )}

      {/* RAM bar */}
      <Text>
        <Text dimColor>{"  RAM  "}</Text>
        <Text color={barColor(snapshot.ramPercent)}>{renderBar(snapshot.ramPercent)}</Text>
        <Text dimColor>
          {` ${String(snapshot.ramPercent).padStart(3)}%  ${(snapshot.ramUsedMb / 1024).toFixed(1)}/${(snapshot.ramTotalMb / 1024).toFixed(1)} GB`}
        </Text>
      </Text>

      {/* VRAM bar */}
      {vramPercent !== null && (
        <Text>
          <Text dimColor>{"  VRAM "}</Text>
          <Text color={barColor(vramPercent)}>{renderBar(vramPercent)}</Text>
          <Text dimColor>{` ${String(vramPercent).padStart(3)}%  ${vramStr}`}</Text>
        </Text>
      )}

      <Text>{""}</Text>

      {/* Model info */}
      {snapshot.activeModel ? (
        <Box flexDirection="column">
          <Text>
            <Text dimColor>{"  Model: "}</Text>
            <Text color="cyan" bold>{snapshot.activeModel}</Text>
            {snapshot.modelQuantization && (
              <Text dimColor>{` (${snapshot.modelQuantization}`}</Text>
            )}
            {snapshot.modelSize && (
              <Text dimColor>{` \u00B7 ${snapshot.modelSize}`}</Text>
            )}
            {snapshot.modelQuantization && <Text dimColor>{")"}</Text>}
            <Text dimColor>{"     Status: "}</Text>
            <Text color={snapshot.tokensPerSec !== null && snapshot.tokensPerSec > 0 ? "green" : "yellow"}>
              {snapshot.tokensPerSec !== null && snapshot.tokensPerSec > 0 ? "generating" : "loaded"}
            </Text>
          </Text>
          <Text>
            <Text dimColor>{"  Speed: "}</Text>
            <Text color="green">
              {snapshot.tokensPerSec !== null ? `${snapshot.tokensPerSec.toFixed(1)} tok/s` : "idle"}
            </Text>
            <Text dimColor>{"          Uptime: "}</Text>
            <Text>{formatUptime(session.startedAt)}</Text>
          </Text>
        </Box>
      ) : (
        <Text dimColor>{"  No active model (Ollama idle or not running)"}</Text>
      )}

      <Text>{""}</Text>

      {/* Smart alerts */}
      <AlertBar snapshot={snapshot} session={session} tokHistory={tokHistory} />
    </Box>
  );
});
