import React, { useState, useEffect, useRef } from "react";
import { Text, Box, useInput, useApp } from "ink";
import { Overview } from "./Overview.js";
import { Inference } from "./Inference.js";
import { VramMap } from "./VramMap.js";
import { HardwareMonitor, type MonitorSnapshot } from "../hardware/monitor.js";
import type { MonitorTab, SessionStats } from "../core/types.js";

const TABS: MonitorTab[] = ["overview", "inference", "vram"];
const TAB_LABELS: Record<MonitorTab, string> = {
  overview: "Overview",
  inference: "Inference",
  vram: "VRAM",
};

const EMPTY_SNAPSHOT: MonitorSnapshot = {
  cpuPercent: 0,
  cpuTemp: null,
  gpuPercent: null,
  gpuTemp: null,
  gpuVramUsedMb: null,
  gpuVramTotalMb: null,
  gpuPowerWatt: null,
  gpuClockMhz: null,
  ramUsedMb: 0,
  ramTotalMb: 1,
  ramPercent: 0,
  gpuVendor: null,
  activeModel: null,
  tokensPerSec: null,
  modelSize: null,
  modelQuantization: null,
  modelContextLength: null,
  modelMaxContext: null,
};

const EMPTY_SESSION: SessionStats = {
  totalTokens: 0,
  totalTimeMs: 0,
  totalRequests: 0,
  startedAt: Date.now(),
  modelHistory: new Map(),
  lastModelSwapAt: null,
};

export function App() {
  const { exit } = useApp();
  const [snapshot, setSnapshot] = useState<MonitorSnapshot>(EMPTY_SNAPSHOT);
  const [activeTab, setActiveTab] = useState<MonitorTab>("overview");
  const [ticks, setTicks] = useState(0);

  // Keep references to monitor data that updates outside React state
  const [session, setSession] = useState<SessionStats>(EMPTY_SESSION);
  const [cpuHistory, setCpuHistory] = useState<number[]>([]);
  const [gpuHistory, setGpuHistory] = useState<number[]>([]);
  const [tokHistory, setTokHistory] = useState<number[]>([]);

  const monitorRef = useRef<HardwareMonitor | null>(null);

  useEffect(() => {
    const monitor = new HardwareMonitor();
    monitorRef.current = monitor;

    const handler = (s: MonitorSnapshot) => {
      setSnapshot(s);
      setTicks((t) => t + 1);

      // Copy history arrays (shallow copy for React diffing)
      setCpuHistory([...monitor.cpuHistory]);
      setGpuHistory([...monitor.gpuHistory]);
      setTokHistory([...monitor.tokHistory]);

      // Copy session stats (create new object for React diffing)
      setSession({
        ...monitor.session,
        modelHistory: new Map(monitor.session.modelHistory),
      });
    };

    monitor.on("snapshot", handler);
    monitor.start(1000);

    return () => {
      monitor.removeListener("snapshot", handler);
      monitor.stop();
    };
  }, []);

  useInput((input, key) => {
    if (input === "q") {
      exit();
    }
    if (key.tab) {
      setActiveTab((current) => {
        const idx = TABS.indexOf(current);
        return TABS[(idx + 1) % TABS.length];
      });
    }
  });

  // Render tab header
  const tabHeader = TABS.map((tab) => {
    if (tab === activeTab) {
      return `[${TAB_LABELS[tab]}]`;
    }
    return ` ${TAB_LABELS[tab]} `;
  }).join(" ");

  return (
    <Box flexDirection="column" paddingTop={1}>
      {/* Title + tabs */}
      <Text>
        <Text bold color="cyan">{"  LLM Pulse \u2014 Live Monitor"}</Text>
        <Text>{"          "}</Text>
        <Text bold color="white">{tabHeader}</Text>
      </Text>
      <Text dimColor>{"  \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"}</Text>
      <Text>{""}</Text>

      {/* Active tab content */}
      {activeTab === "overview" && (
        <Overview
          snapshot={snapshot}
          session={session}
          cpuHistory={cpuHistory}
          gpuHistory={gpuHistory}
          tokHistory={tokHistory}
        />
      )}
      {activeTab === "inference" && (
        <Inference
          snapshot={snapshot}
          session={session}
          tokHistory={tokHistory}
        />
      )}
      {activeTab === "vram" && (
        <VramMap snapshot={snapshot} />
      )}

      {/* Footer */}
      <Text>{""}</Text>
      <Text dimColor>{"  \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"}</Text>
      <Text dimColor>{`  [Tab] switch view  [q] quit    Updates: ${ticks}`}</Text>
    </Box>
  );
}
