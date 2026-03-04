import React, { useState, useEffect } from "react";
import { Text, Box, useInput, useApp } from "ink";
import { CpuBar } from "./CpuBar.js";
import { GpuBar } from "./GpuBar.js";
import { MemoryBar } from "./MemoryBar.js";
import { InferenceStats } from "./InferenceStats.js";
import { HardwareMonitor, type MonitorSnapshot } from "../hardware/monitor.js";

const EMPTY_SNAPSHOT: MonitorSnapshot = {
  cpuPercent: 0,
  cpuTemp: null,
  gpuPercent: null,
  gpuTemp: null,
  gpuVramUsedMb: null,
  gpuVramTotalMb: null,
  ramUsedMb: 0,
  ramTotalMb: 1,
  ramPercent: 0,
  activeModel: null,
  tokensPerSec: null,
};

export function App() {
  const { exit } = useApp();
  const [snapshot, setSnapshot] = useState<MonitorSnapshot>(EMPTY_SNAPSHOT);
  const [ticks, setTicks] = useState(0);

  useEffect(() => {
    const monitor = new HardwareMonitor();

    monitor.on("snapshot", (s: MonitorSnapshot) => {
      setSnapshot(s);
      setTicks((t) => t + 1);
    });

    monitor.start(1000);

    return () => {
      monitor.stop();
    };
  }, []);

  useInput((input) => {
    if (input === "q") {
      exit();
    }
  });

  return (
    <Box flexDirection="column" paddingTop={1}>
      <Text bold color="cyan">{"  LLM Pulse — Live Monitor"}</Text>
      <Text dimColor>{"  ─────────────────────────────────────────────"}</Text>
      <Text>{""}</Text>

      <CpuBar percent={snapshot.cpuPercent} temp={snapshot.cpuTemp} />
      <GpuBar
        percent={snapshot.gpuPercent}
        temp={snapshot.gpuTemp}
        vramUsedMb={snapshot.gpuVramUsedMb}
        vramTotalMb={snapshot.gpuVramTotalMb}
      />
      <MemoryBar
        percent={snapshot.ramPercent}
        usedMb={snapshot.ramUsedMb}
        totalMb={snapshot.ramTotalMb}
      />

      <Text>{""}</Text>
      <InferenceStats
        activeModel={snapshot.activeModel}
        tokensPerSec={snapshot.tokensPerSec}
      />

      <Text>{""}</Text>
      <Text dimColor>{"  ─────────────────────────────────────────────"}</Text>
      <Text dimColor>{`  [q] quit    Updates: ${ticks}`}</Text>
    </Box>
  );
}
