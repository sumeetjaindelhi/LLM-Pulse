import React, { useReducer, useEffect, useRef, useCallback } from "react";
import { Text, Box, useInput, useApp } from "ink";
import { Overview } from "./Overview.js";
import { Inference } from "./Inference.js";
import { GpuDetail } from "./GpuDetail.js";
import { VramMap } from "./VramMap.js";
import { ModelManager } from "./ModelManager.js";
import { HardwareMonitor, type MonitorSnapshot } from "../hardware/monitor.js";
import { OLLAMA_API_URL } from "../core/constants.js";
import type { MonitorTab, SessionStats } from "../core/types.js";

const TABS: MonitorTab[] = ["overview", "inference", "gpu", "vram", "models"];
const TAB_LABELS: Record<MonitorTab, string> = {
  overview: "Overview",
  inference: "Inference",
  gpu: "GPU",
  vram: "VRAM",
  models: "Models",
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
  gpuModel: null,
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

// ── Reducer ──────────────────────────────────

interface MonitorState {
  snapshot: MonitorSnapshot;
  session: SessionStats;
  cpuHistory: number[];
  gpuHistory: number[];
  tokHistory: number[];
  gpuTempHistory: number[];
  gpuVramHistory: number[];
  gpuPowerHistory: number[];
}

type MonitorAction = {
  type: "tick";
  snapshot: MonitorSnapshot;
  session: SessionStats;
  cpuHistory: number[];
  gpuHistory: number[];
  tokHistory: number[];
  gpuTempHistory: number[];
  gpuVramHistory: number[];
  gpuPowerHistory: number[];
};

const initialState: MonitorState = {
  snapshot: EMPTY_SNAPSHOT,
  session: EMPTY_SESSION,
  cpuHistory: [],
  gpuHistory: [],
  tokHistory: [],
  gpuTempHistory: [],
  gpuVramHistory: [],
  gpuPowerHistory: [],
};

function monitorReducer(_state: MonitorState, action: MonitorAction): MonitorState {
  switch (action.type) {
    case "tick":
      return {
        snapshot: action.snapshot,
        session: action.session,
        cpuHistory: action.cpuHistory,
        gpuHistory: action.gpuHistory,
        tokHistory: action.tokHistory,
        gpuTempHistory: action.gpuTempHistory,
        gpuVramHistory: action.gpuVramHistory,
        gpuPowerHistory: action.gpuPowerHistory,
      };
  }
}

// ── App ──────────────────────────────────────

export function App({ host }: { host?: string }) {
  const { exit } = useApp();
  const [state, dispatch] = useReducer(monitorReducer, initialState);
  const [activeTab, setActiveTab] = React.useState<MonitorTab>("overview");
  const ticksRef = useRef(0);
  const monitorRef = useRef<HardwareMonitor | null>(null);

  useEffect(() => {
    const monitor = new HardwareMonitor(host);
    monitorRef.current = monitor;

    const handler = (s: MonitorSnapshot) => {
      ticksRef.current += 1;

      // Single dispatch replaces 7 separate setState calls
      dispatch({
        type: "tick",
        snapshot: s,
        session: {
          ...monitor.session,
          modelHistory: new Map(monitor.session.modelHistory),
        },
        cpuHistory: [...monitor.cpuHistory],
        gpuHistory: [...monitor.gpuHistory],
        tokHistory: [...monitor.tokHistory],
        gpuTempHistory: [...monitor.gpuTempHistory],
        gpuVramHistory: [...monitor.gpuVramHistory],
        gpuPowerHistory: [...monitor.gpuPowerHistory],
      });
    };

    monitor.on("snapshot", handler);
    monitor.start(2000);

    return () => {
      monitor.removeListener("snapshot", handler);
      monitor.stop();
    };
  }, []);

  const handleTab = useCallback(() => {
    setActiveTab((current) => {
      const idx = TABS.indexOf(current);
      return TABS[(idx + 1) % TABS.length];
    });
  }, []);

  useInput((input, key) => {
    if (input === "q") {
      exit();
    }
    if (key.tab) {
      handleTab();
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
          snapshot={state.snapshot}
          session={state.session}
          cpuHistory={state.cpuHistory}
          gpuHistory={state.gpuHistory}
          tokHistory={state.tokHistory}
        />
      )}
      {activeTab === "inference" && (
        <Inference
          snapshot={state.snapshot}
          session={state.session}
          tokHistory={state.tokHistory}
        />
      )}
      {activeTab === "gpu" && (
        <GpuDetail
          snapshot={state.snapshot}
          gpuHistory={state.gpuHistory}
          gpuTempHistory={state.gpuTempHistory}
          gpuVramHistory={state.gpuVramHistory}
          gpuPowerHistory={state.gpuPowerHistory}
        />
      )}
      {activeTab === "vram" && (
        <VramMap snapshot={state.snapshot} />
      )}
      {activeTab === "models" && (
        <ModelManager snapshot={state.snapshot} ollamaHost={host || OLLAMA_API_URL} />
      )}

      {/* Footer */}
      <Text>{""}</Text>
      <Text dimColor>{"  \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"}</Text>
      <Text dimColor>{`  [Tab] switch view  [q] quit    Updates: ${ticksRef.current}`}</Text>
    </Box>
  );
}
