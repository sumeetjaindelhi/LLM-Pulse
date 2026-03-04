import React from "react";
import { Text } from "ink";

const BLOCKS = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];

interface SparklineProps {
  data: number[];
  width?: number;
  color?: string;
}

export function Sparkline({ data, width = 10, color = "cyan" }: SparklineProps) {
  if (data.length === 0) {
    return <Text dimColor>{"─".repeat(width)}</Text>;
  }

  // Take last `width` data points
  const slice = data.slice(-width);

  const min = Math.min(...slice);
  const max = Math.max(...slice);
  const range = max - min;

  const chars = slice.map((v) => {
    if (range === 0) return BLOCKS[4]; // flat line — mid-height block
    const idx = Math.round(((v - min) / range) * 7);
    return BLOCKS[Math.min(idx, 7)];
  });

  // Pad with spaces if we don't have enough data yet
  const padding = width - chars.length;
  const padStr = padding > 0 ? " ".repeat(padding) : "";

  return (
    <Text color={color}>{padStr}{chars.join("")}</Text>
  );
}
