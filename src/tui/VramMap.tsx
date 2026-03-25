import React from "react";
import { Text, Box } from "ink";
import type { MonitorSnapshot } from "../hardware/monitor.js";
import type { VramBreakdown } from "../core/types.js";

interface VramMapProps {
  snapshot: MonitorSnapshot;
}

// Estimate VRAM breakdown from available data
function estimateVramBreakdown(snapshot: MonitorSnapshot): VramBreakdown | null {
  if (snapshot.gpuVramUsedMb === null || snapshot.gpuVramTotalMb === null) {
    return null;
  }

  const totalMb = snapshot.gpuVramTotalMb;
  const usedMb = snapshot.gpuVramUsedMb;
  const freeMb = totalMb - usedMb;

  // Estimate model weights from model size string (e.g. "4.9 GB")
  let modelWeightsMb = 0;
  if (snapshot.modelSize) {
    const match = snapshot.modelSize.match(/([\d.]+)\s*GB/i);
    if (match) {
      modelWeightsMb = Math.round(parseFloat(match[1]) * 1024);
    }
  }

  // If we don't have model size data but a model is active, estimate from model name
  if (modelWeightsMb === 0 && snapshot.activeModel) {
    // Rough estimates based on common model sizes
    const name = snapshot.activeModel.toLowerCase();
    if (name.includes("70b")) modelWeightsMb = 40000;
    else if (name.includes("34b") || name.includes("33b")) modelWeightsMb = 20000;
    else if (name.includes("13b") || name.includes("14b")) modelWeightsMb = 8000;
    else if (name.includes("8b") || name.includes("7b")) modelWeightsMb = 5000;
    else if (name.includes("3b")) modelWeightsMb = 2000;
    else if (name.includes("1b") || name.includes("1.5b")) modelWeightsMb = 1000;
  }

  // Don't let weights exceed used VRAM
  modelWeightsMb = Math.min(modelWeightsMb, usedMb);

  // Estimate KV cache — rough heuristic based on remaining used VRAM
  // KV cache is typically 10-25% of used VRAM for active inference
  const remainingUsed = usedMb - modelWeightsMb;
  const kvCacheMb = snapshot.activeModel ? Math.max(0, Math.round(remainingUsed * 0.6)) : 0;

  // Overhead is the rest
  const overheadMb = Math.max(0, usedMb - modelWeightsMb - kvCacheMb);

  return { totalMb, usedMb, freeMb, modelWeightsMb, kvCacheMb, overheadMb };
}

// Find models that could fit in remaining VRAM
function canStillFit(freeMb: number): string[] {
  const fits: string[] = [];
  const candidates = [
    { name: "Phi-3 Mini Q4", vram: 2500 },
    { name: "Llama 3.2 3B Q4", vram: 2000 },
    { name: "Gemma 2B Q4", vram: 1800 },
    { name: "TinyLlama 1.1B", vram: 800 },
    { name: "Qwen2 1.5B Q4", vram: 1200 },
    { name: "Llama 3.1 8B Q4", vram: 5000 },
    { name: "Mistral 7B Q4", vram: 4500 },
    { name: "DeepSeek Coder 6.7B Q4", vram: 4200 },
  ];

  for (const c of candidates) {
    if (c.vram <= freeMb) {
      fits.push(`${c.name} (${(c.vram / 1024).toFixed(1)} GB)`);
    }
  }

  // Sort by size descending — show biggest that fits first
  fits.sort((a, b) => {
    const aMatch = a.match(/([\d.]+) GB/);
    const bMatch = b.match(/([\d.]+) GB/);
    return (bMatch ? parseFloat(bMatch[1]) : 0) - (aMatch ? parseFloat(aMatch[1]) : 0);
  });

  return fits.slice(0, 3);
}

function renderSegmentedBar(breakdown: VramBreakdown, width: number = 44): string {
  const { totalMb, modelWeightsMb, kvCacheMb, overheadMb, freeMb } = breakdown;
  if (totalMb === 0) return "\u2591".repeat(width);

  const wChars = Math.round((modelWeightsMb / totalMb) * width);
  const kChars = Math.round((kvCacheMb / totalMb) * width);
  const oChars = Math.round((overheadMb / totalMb) * width);
  const fChars = Math.max(0, width - wChars - kChars - oChars);

  return "\u2588".repeat(wChars) + "\u2593".repeat(kChars) + "\u2592".repeat(oChars) + "\u2591".repeat(fChars);
}

export const VramMap = React.memo(function VramMap({ snapshot }: VramMapProps) {
  const breakdown = estimateVramBreakdown(snapshot);

  const overheadLabel = snapshot.gpuVendor === "NVIDIA" ? "CUDA overhead"
    : snapshot.gpuVendor === "Apple" ? "Metal overhead"
    : snapshot.gpuVendor === "AMD" ? "ROCm overhead"
    : "GPU overhead";

  if (!breakdown) {
    return (
      <Box flexDirection="column">
        <Text dimColor>{"  No VRAM data available (no supported GPU detected)"}</Text>
      </Box>
    );
  }

  const usedPercent = Math.round((breakdown.usedMb / breakdown.totalMb) * 100);
  const fits = canStillFit(breakdown.freeMb);

  return (
    <Box flexDirection="column">
      <Text bold dimColor>{`  VRAM Map \u2014 ${breakdown.totalMb.toLocaleString()} MB Total`}</Text>
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
      {breakdown.modelWeightsMb > 0 && (
        <Text>
          <Text dimColor>{"  Model weights   "}</Text>
          <Text color="green">{"[\u2588".padEnd(Math.max(1, Math.round((breakdown.modelWeightsMb / breakdown.totalMb) * 16)) + 1, "\u2588") + "]"}</Text>
          <Text dimColor>{`  ${breakdown.modelWeightsMb.toLocaleString().padStart(7)} MB  ${Math.round((breakdown.modelWeightsMb / breakdown.totalMb) * 100).toString().padStart(2)}%`}</Text>
        </Text>
      )}

      {breakdown.kvCacheMb > 0 && (
        <Text>
          <Text dimColor>{"  KV Cache        "}</Text>
          <Text color="yellow">{"[\u2593".padEnd(Math.max(1, Math.round((breakdown.kvCacheMb / breakdown.totalMb) * 16)) + 1, "\u2593") + "]"}</Text>
          <Text dimColor>{`  ${breakdown.kvCacheMb.toLocaleString().padStart(7)} MB  ${Math.round((breakdown.kvCacheMb / breakdown.totalMb) * 100).toString().padStart(2)}%`}</Text>
        </Text>
      )}

      {breakdown.overheadMb > 0 && (
        <Text>
          <Text dimColor>{`  ${overheadLabel.padEnd(16)}`}</Text>
          <Text color="gray">{"[\u2592".padEnd(Math.max(1, Math.round((breakdown.overheadMb / breakdown.totalMb) * 16)) + 1, "\u2592") + "]"}</Text>
          <Text dimColor>{`  ${breakdown.overheadMb.toLocaleString().padStart(7)} MB  ${Math.round((breakdown.overheadMb / breakdown.totalMb) * 100).toString().padStart(2)}%`}</Text>
        </Text>
      )}

      <Text>
        <Text dimColor>{"  Free            "}</Text>
        <Text color="blue">{"[\u2591".padEnd(Math.max(1, Math.round((breakdown.freeMb / breakdown.totalMb) * 16)) + 1, "\u2591") + "]"}</Text>
        <Text dimColor>{`  ${breakdown.freeMb.toLocaleString().padStart(7)} MB  ${Math.round((breakdown.freeMb / breakdown.totalMb) * 100).toString().padStart(2)}%`}</Text>
      </Text>

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

      {/* Can still fit */}
      {fits.length > 0 && (
        <Box flexDirection="column" marginTop={1}>
          <Text>
            <Text dimColor>{"  Can still fit: "}</Text>
            <Text color="cyan">{fits[0]}</Text>
            {fits.length > 1 && <Text dimColor>{` (+${fits.length - 1} more)`}</Text>}
          </Text>
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
