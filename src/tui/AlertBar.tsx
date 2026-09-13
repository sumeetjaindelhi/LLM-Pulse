import React from "react";
import { Text, Box } from "ink";
import type { MonitorSnapshot } from "../hardware/monitor.js";
import type { SmartAlert, SessionStats } from "../core/types.js";
import { ALERT_THRESHOLDS, pickGpuTempThreshold } from "../core/constants.js";

interface AlertBarProps {
  snapshot: MonitorSnapshot;
  session: SessionStats;
}

function computeAlerts(snapshot: MonitorSnapshot, session: SessionStats): SmartAlert[] {
  const alerts: SmartAlert[] = [];

  // VRAM nearly full. Guard on `> 0` for the denominator — without it, a GPU
  // that reports 0 total VRAM would surface "VRAM at Infinity%" alerts.
  if (
    snapshot.gpuVramUsedMb !== null &&
    snapshot.gpuVramTotalMb !== null &&
    snapshot.gpuVramTotalMb > 0
  ) {
    const vramPercent = (snapshot.gpuVramUsedMb / snapshot.gpuVramTotalMb) * 100;
    if (vramPercent >= ALERT_THRESHOLDS.vramHighPercent) {
      alerts.push({
        severity: "warning",
        icon: "\u26A0",
        message: `VRAM at ${Math.round(vramPercent)}% \u2014 next large prompt may cause swapping`,
      });
    }
  }

  // GPU high temperature. Threshold varies by vendor — Apple Silicon throttles
  // around 72°C (fanless + power-efficient), NVIDIA/AMD desktops tolerate 85°C,
  // mobile dGPUs around 78°C. "Mobile", "Laptop", "Max-Q", and "Max-P" in the
  // model name are strong signals of a laptop SKU.
  if (snapshot.gpuTemp !== null) {
    const modelLower = (snapshot.gpuModel ?? "").toLowerCase();
    const isMobile = /mobile|laptop|max-[qp]/.test(modelLower);
    const threshold = snapshot.gpuVendor
      ? pickGpuTempThreshold(snapshot.gpuVendor, isMobile)
      : ALERT_THRESHOLDS.gpuTempHighCelsius;
    if (snapshot.gpuTemp >= threshold) {
      alerts.push({
        severity: "warning",
        icon: "\u26A0",
        message: `GPU temperature at ${Math.round(snapshot.gpuTemp)}\u00B0C (threshold ${threshold}\u00B0C) \u2014 possible thermal throttling`,
      });
    }
  }

  // No model loaded for a while
  if (
    snapshot.activeModel === null &&
    Date.now() - session.startedAt > ALERT_THRESHOLDS.noModelTimeoutMs
  ) {
    alerts.push({
      severity: "info",
      icon: "\u2139",
      message: "No active model \u2014 run `ollama run llama3.1:8b` to start",
    });
  }

  return alerts;
}

export const AlertBar = React.memo(function AlertBar({ snapshot, session }: AlertBarProps) {
  const alerts = computeAlerts(snapshot, session);

  if (alerts.length === 0) return null;

  return (
    <Box flexDirection="column">
      {alerts.map((alert, i) => {
        const color = alert.severity === "warning" ? "yellow" : alert.severity === "success" ? "green" : "blue";
        return (
          <Text key={`alert-${alert.severity}-${i}`}>
            <Text color={color}>{`  ${alert.icon} `}</Text>
            <Text color={color}>{alert.message}</Text>
          </Text>
        );
      })}
    </Box>
  );
});
