import React from "react";
import { Text } from "ink";

interface InferenceStatsProps {
  activeModel: string | null;
  tokensPerSec: number | null;
}

export function InferenceStats({ activeModel, tokensPerSec }: InferenceStatsProps) {
  if (!activeModel) {
    return (
      <Text dimColor>{"  No active model (Ollama idle or not running)"}</Text>
    );
  }

  const tps = tokensPerSec !== null ? `${tokensPerSec.toFixed(1)} tok/s` : "measuring...";

  return (
    <Text>
      <Text dimColor>{"  Active: "}</Text>
      <Text color="cyan" bold>{activeModel}</Text>
      <Text dimColor>{"    "}</Text>
      <Text color="green">{tps}</Text>
    </Text>
  );
}
