import React from "react";
import { Text, Box } from "ink";
import { Sparkline } from "./Sparkline.js";
import type { MonitorSnapshot } from "../hardware/monitor.js";

interface GpuDetailProps {
  snapshot: MonitorSnapshot;
  gpuHistory: number[];
  gpuTempHistory: number[];
  gpuVramHistory: number[];
  gpuPowerHistory: number[];
}

function barColor(percent: number): string {
  return percent >= 90 ? "red" : percent >= 70 ? "yellow" : "green";
}

function renderBar(percent: number, width = 20): string {
  const filled = Math.round((percent / 100) * width);
  const empty = width - filled;
  return "\u2588".repeat(filled) + "\u2591".repeat(empty);
}

function tempColor(temp: number): string {
  return temp >= 85 ? "red" : temp >= 70 ? "yellow" : "green";
}

export function GpuDetail({ snapshot, gpuHistory, gpuTempHistory, gpuVramHistory, gpuPowerHistory }: GpuDetailProps) {
  if (snapshot.gpuPercent === null) {
    return (
      <Box flexDirection="column">
        <Text dimColor>{"  No GPU detected"}</Text>
      </Box>
    );
  }

  const vramPercent =
    snapshot.gpuVramUsedMb !== null && snapshot.gpuVramTotalMb !== null && snapshot.gpuVramTotalMb > 0
      ? Math.round((snapshot.gpuVramUsedMb / snapshot.gpuVramTotalMb) * 100)
      : null;

  const vramStr =
    snapshot.gpuVramUsedMb !== null && snapshot.gpuVramTotalMb !== null
      ? `${(snapshot.gpuVramUsedMb / 1024).toFixed(1)}/${(snapshot.gpuVramTotalMb / 1024).toFixed(1)} GB`
      : "";

  const header = snapshot.gpuVendor || snapshot.gpuModel
    ? `GPU Detail \u2014 ${[snapshot.gpuVendor, snapshot.gpuModel].filter(Boolean).join(" ")}`
    : "GPU Detail";

  // Peak stats
  const peakUtil = gpuHistory.length > 0 ? Math.max(...gpuHistory) : 0;
  const peakTemp = gpuTempHistory.length > 0 ? Math.max(...gpuTempHistory) : 0;
  const peakVram = gpuVramHistory.length > 0 ? Math.max(...gpuVramHistory) : 0;
  const peakPower = gpuPowerHistory.length > 0 ? Math.max(...gpuPowerHistory) : 0;

  return (
    <Box flexDirection="column">
      <Text bold dimColor>{`  ${header}`}</Text>
      <Text dimColor>{"  \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"}</Text>

      <Text>{""}</Text>

      {/* Utilization */}
      <Text>
        <Text dimColor>{"  Util   "}</Text>
        <Text color={barColor(snapshot.gpuPercent)}>{renderBar(snapshot.gpuPercent)}</Text>
        <Text dimColor>{` ${String(snapshot.gpuPercent).padStart(3)}%`}</Text>
        <Text>{"   "}</Text>
        <Sparkline data={gpuHistory} width={12} color="magenta" />
      </Text>

      {/* Temperature */}
      {snapshot.gpuTemp !== null && (
        <Text>
          <Text dimColor>{"  Temp   "}</Text>
          <Text color={tempColor(snapshot.gpuTemp)}>{renderBar(snapshot.gpuTemp)}</Text>
          <Text dimColor>{` ${String(snapshot.gpuTemp).padStart(3)}\u00B0C`}</Text>
          <Text>{"  "}</Text>
          <Sparkline data={gpuTempHistory} width={12} color={tempColor(snapshot.gpuTemp)} />
        </Text>
      )}

      {/* VRAM */}
      {vramPercent !== null && (
        <Text>
          <Text dimColor>{"  VRAM   "}</Text>
          <Text color={barColor(vramPercent)}>{renderBar(vramPercent)}</Text>
          <Text dimColor>{` ${String(vramPercent).padStart(3)}%  ${vramStr}`}</Text>
          <Text>{"   "}</Text>
          <Sparkline data={gpuVramHistory} width={12} color="cyan" />
        </Text>
      )}

      {/* Power */}
      {snapshot.gpuPowerWatt !== null && (
        <Text>
          <Text dimColor>{"  Power  "}</Text>
          <Text>{`${snapshot.gpuPowerWatt.toFixed(0)}W`}</Text>
          <Text>{"                          "}</Text>
          <Sparkline data={gpuPowerHistory} width={12} color="cyan" />
        </Text>
      )}

      {/* Clock */}
      {snapshot.gpuClockMhz !== null && (
        <Text>
          <Text dimColor>{"  Clock  "}</Text>
          <Text>{`${snapshot.gpuClockMhz} MHz`}</Text>
        </Text>
      )}

      <Text>{""}</Text>

      {/* Peak stats */}
      <Text bold dimColor>{"  Peak Stats"}</Text>
      <Text>
        <Text dimColor>{"  Utilization: "}</Text>
        <Text>{`${Math.round(peakUtil)}%`}</Text>
        {peakTemp > 0 && (
          <>
            <Text dimColor>{"   Temp: "}</Text>
            <Text color={tempColor(peakTemp)}>{`${Math.round(peakTemp)}\u00B0C`}</Text>
          </>
        )}
        {peakVram > 0 && (
          <>
            <Text dimColor>{"   VRAM: "}</Text>
            <Text>{`${Math.round(peakVram)}%`}</Text>
          </>
        )}
        {peakPower > 0 && (
          <>
            <Text dimColor>{"   Power: "}</Text>
            <Text>{`${Math.round(peakPower)}W`}</Text>
          </>
        )}
      </Text>
    </Box>
  );
}
