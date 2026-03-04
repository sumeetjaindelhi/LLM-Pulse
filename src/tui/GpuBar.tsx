import React from "react";
import { Text } from "ink";

interface GpuBarProps {
  percent: number | null;
  temp: number | null;
  vramUsedMb: number | null;
  vramTotalMb: number | null;
  width?: number;
}

export function GpuBar({ percent, temp, vramUsedMb, vramTotalMb, width = 20 }: GpuBarProps) {
  if (percent === null) {
    return (
      <Text>
        <Text dimColor>{"  GPU  "}</Text>
        <Text color="gray">{"░".repeat(width)}  No GPU data</Text>
      </Text>
    );
  }

  const filled = Math.round((percent / 100) * width);
  const empty = width - filled;
  const bar = "█".repeat(filled) + "░".repeat(empty);

  const color = percent >= 90 ? "red" : percent >= 70 ? "yellow" : "green";
  const tempStr = temp !== null ? `  ${temp}°C` : "";
  const vramStr =
    vramUsedMb !== null && vramTotalMb !== null
      ? `  VRAM: ${(vramUsedMb / 1024).toFixed(1)}/${(vramTotalMb / 1024).toFixed(1)} GB`
      : "";

  return (
    <Text>
      <Text dimColor>{"  GPU  "}</Text>
      <Text color={color}>{bar}</Text>
      <Text dimColor>{` ${String(percent).padStart(3)}%${tempStr}${vramStr}`}</Text>
    </Text>
  );
}
