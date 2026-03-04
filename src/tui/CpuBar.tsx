import React from "react";
import { Text } from "ink";

interface CpuBarProps {
  percent: number;
  temp: number | null;
  width?: number;
}

export function CpuBar({ percent, temp, width = 20 }: CpuBarProps) {
  const filled = Math.round((percent / 100) * width);
  const empty = width - filled;
  const bar = "█".repeat(filled) + "░".repeat(empty);

  const color = percent >= 90 ? "red" : percent >= 70 ? "yellow" : "green";
  const tempStr = temp !== null ? `  ${temp}°C` : "";

  return (
    <Text>
      <Text dimColor>{"  CPU  "}</Text>
      <Text color={color}>{bar}</Text>
      <Text dimColor>{` ${String(percent).padStart(3)}%${tempStr}`}</Text>
    </Text>
  );
}
