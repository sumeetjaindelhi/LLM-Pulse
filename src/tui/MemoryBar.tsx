import React from "react";
import { Text } from "ink";

interface MemoryBarProps {
  percent: number;
  usedMb: number;
  totalMb: number;
  width?: number;
}

export function MemoryBar({ percent, usedMb, totalMb, width = 20 }: MemoryBarProps) {
  const filled = Math.round((percent / 100) * width);
  const empty = width - filled;
  const bar = "█".repeat(filled) + "░".repeat(empty);

  const color = percent >= 90 ? "red" : percent >= 70 ? "yellow" : "green";
  const usedGb = (usedMb / 1024).toFixed(1);
  const totalGb = (totalMb / 1024).toFixed(1);

  return (
    <Text>
      <Text dimColor>{"  RAM  "}</Text>
      <Text color={color}>{bar}</Text>
      <Text dimColor>{` ${String(percent).padStart(3)}%        ${usedGb}/${totalGb} GB`}</Text>
    </Text>
  );
}
