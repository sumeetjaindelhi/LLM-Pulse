import React from "react";
import { Text, Box } from "ink";
import type { MonitorSnapshot } from "../hardware/monitor.js";
import type { VramBreakdown } from "../core/types.js";
import { formatMb } from "../cli/ui/progress.js";
import { vramBreakdown, fitsInFreeVram, FIT_CONTEXT_TOKENS } from "./vram-breakdown.js";

interface VramMapProps {
  snapshot: MonitorSnapshot;
}

function renderSegmentedBar(breakdown: VramBreakdown, width: number = 44): string {
  const { totalMb, modelVramMb, otherMb } = breakdown;
  if (totalMb === 0) return "\u2591".repeat(width);

  const mChars = Math.round((modelVramMb / totalMb) * width);
  const oChars = Math.round((otherMb / totalMb) * width);
  const fChars = Math.max(0, width - mChars - oChars);

  return "\u2588".repeat(mChars) + "\u2592".repeat(oChars) + "\u2591".repeat(fChars);
}

function segmentRow(label: string, glyph: string, color: string, mb: number, totalMb: number) {
  const barLen = Math.max(1, Math.round((mb / totalMb) * 16));
  return (
    <Text>
      <Text dimColor>{`  ${label.padEnd(16)}`}</Text>
      <Text color={color}>{`[${glyph.repeat(barLen)}]`.padEnd(18)}</Text>
      <Text dimColor>{`  ${formatMb(mb).padStart(9)}  ${Math.round((mb / totalMb) * 100).toString().padStart(2)}%`}</Text>
    </Text>
  );
}

export const VramMap = React.memo(function VramMap({ snapshot }: VramMapProps) {
  const breakdown = vramBreakdown(snapshot);

  if (!breakdown) {
    // A utilization reading with no memory figures is unified memory (GB10), not a missing GPU.
    const reason = snapshot.gpuPercent !== null
      ? "GPU reports no memory figures (unified memory \u2014 see RAM)"
      : "No VRAM data available (no supported GPU detected)";
    return (
      <Box flexDirection="column">
        <Text dimColor>{`  ${reason}`}</Text>
      </Box>
    );
  }

  const usedPercent = Math.round((breakdown.usedMb / breakdown.totalMb) * 100);
  const apuRamAvailableMb = snapshot.gpuVendor === "AMD" && snapshot.gpuUnifiedMemory ? snapshot.ramTotalMb - snapshot.ramUsedMb : null;
  const fits = fitsInFreeVram(breakdown.freeMb, snapshot.gpuVendor, apuRamAvailableMb);

  return (
    <Box flexDirection="column">
      <Text bold dimColor>{`  VRAM Map \u2014 ${formatMb(breakdown.totalMb)} Total`}</Text>
      <Text dimColor>{"  \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"}</Text>

      <Text>{""}</Text>

      {/* Overall usage bar */}
      <Text>
        <Text>{"  "}</Text>
        <Text color={usedPercent >= 85 ? "red" : usedPercent >= 70 ? "yellow" : "green"}>
          {renderSegmentedBar(breakdown)}
        </Text>
        <Text dimColor>{`  ${usedPercent}% used`}</Text>
      </Text>

      <Text>{""}</Text>

      {/* Breakdown table */}
      {breakdown.modelVramMb > 0 &&
        segmentRow("Loaded (Ollama)", "\u2588", "green", breakdown.modelVramMb, breakdown.totalMb)}
      {breakdown.otherMb > 0 &&
        segmentRow(breakdown.split ? "Other GPU use" : "Used", "\u2592", "gray", breakdown.otherMb, breakdown.totalMb)}
      {segmentRow("Free", "\u2591", "blue", breakdown.freeMb, breakdown.totalMb)}

      {/* GPU power info if available */}
      {snapshot.gpuPowerWatt !== null && (
        <Box marginTop={1}>
          <Text>
            <Text dimColor>{"  Power: "}</Text>
            <Text>{`${snapshot.gpuPowerWatt.toFixed(0)}W`}</Text>
            {snapshot.gpuClockMhz !== null && (
              <Text dimColor>{`   Clock: ${snapshot.gpuClockMhz} MHz`}</Text>
            )}
          </Text>
        </Box>
      )}

      {fits.length > 0 && (
        <Box flexDirection="column" marginTop={1}>
          <Text dimColor>{`  Can still fit (need at ${FIT_CONTEXT_TOKENS.toLocaleString()} tokens):`}</Text>
          {fits.map((f) => (
            <Text key={f.name}>
              <Text color="cyan">{`    ${f.name}`}</Text>
              <Text dimColor>{`  ${formatMb(f.needMb)}`}</Text>
            </Text>
          ))}
        </Box>
      )}

      {/* No model loaded */}
      {!snapshot.activeModel && (
        <Box marginTop={1}>
          <Text dimColor>{"  No model currently loaded \u2014 VRAM showing system baseline"}</Text>
        </Box>
      )}
    </Box>
  );
});
