import React from "react";
import { Text, Box } from "ink";
import type { MonitorSnapshot } from "../hardware/monitor.js";
import type { SmartAlert, SessionStats } from "../core/types.js";
import { ALERT_THRESHOLDS, EXPECTED_TOK_PER_SEC } from "../core/constants.js";

interface AlertBarProps {
  snapshot: MonitorSnapshot;
  session: SessionStats;
  tokHistory: number[];
}

function computeAlerts(snapshot: MonitorSnapshot, session: SessionStats, tokHistory: number[]): SmartAlert[] {
  const alerts: SmartAlert[] = [];

  // VRAM nearly full
  if (snapshot.gpuVramUsedMb !== null && snapshot.gpuVramTotalMb !== null) {
    const vramPercent = (snapshot.gpuVramUsedMb / snapshot.gpuVramTotalMb) * 100;
    if (vramPercent >= ALERT_THRESHOLDS.vramHighPercent) {
      alerts.push({
        severity: "warning",
        icon: "\u26A0",
        message: `VRAM at ${Math.round(vramPercent)}% \u2014 next large prompt may cause swapping`,
      });
    }
  }

  // GPU high temperature
  if (snapshot.gpuTemp !== null && snapshot.gpuTemp >= ALERT_THRESHOLDS.gpuTempHighCelsius) {
    alerts.push({
      severity: "warning",
      icon: "\u26A0",
      message: `GPU temperature at ${Math.round(snapshot.gpuTemp)}\u00B0C \u2014 possible thermal throttling`,
    });
  }

  // Speed drop detection (compare recent avg to earlier avg)
  if (tokHistory.length >= 20) {
    const recentSlice = tokHistory.slice(-10).filter((v) => v > 0);
    const earlierSlice = tokHistory.slice(-30, -10).filter((v) => v > 0);
    if (recentSlice.length >= 3 && earlierSlice.length >= 3) {
      const recentAvg = recentSlice.reduce((a, b) => a + b, 0) / recentSlice.length;
      const earlierAvg = earlierSlice.reduce((a, b) => a + b, 0) / earlierSlice.length;
      const dropPercent = ((earlierAvg - recentAvg) / earlierAvg) * 100;
      if (dropPercent >= ALERT_THRESHOLDS.tokSpeedDropPercent) {
        alerts.push({
          severity: "warning",
          icon: "\u26A0",
          message: `Speed dropped ${Math.round(dropPercent)}% \u2014 possible thermal throttling or context overflow`,
        });
      }
    }
  }

  // GPU underutilized during inference
  if (
    snapshot.gpuPercent !== null &&
    snapshot.gpuPercent < ALERT_THRESHOLDS.gpuUnderutilizedPercent &&
    snapshot.tokensPerSec !== null &&
    snapshot.tokensPerSec > 0
  ) {
    alerts.push({
      severity: "warning",
      icon: "\u26A0",
      message: `GPU at ${snapshot.gpuPercent}% during inference \u2014 model may be CPU-bound`,
    });
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

  // Performance vs expected (if model is running)
  if (snapshot.tokensPerSec !== null && snapshot.tokensPerSec > 0 && snapshot.gpuVramTotalMb !== null) {
    const vramTierGb = String(Math.round(snapshot.gpuVramTotalMb / 1024));
    const tierExpected = EXPECTED_TOK_PER_SEC[vramTierGb];
    if (tierExpected && snapshot.activeModel) {
      // Try to match model size from name (e.g. "llama3.1:8b" → 8)
      const sizeMatch = snapshot.activeModel.match(/(\d+)[bB]/);
      if (sizeMatch) {
        const paramSize = sizeMatch[1];
        const expected = tierExpected[paramSize];
        if (expected) {
          const diffPercent = ((snapshot.tokensPerSec - expected) / expected) * 100;
          if (diffPercent >= 20) {
            alerts.push({
              severity: "success",
              icon: "\u2713",
              message: `Running ${Math.round(diffPercent)}% faster than average for your hardware + ${paramSize}B model`,
            });
          } else if (diffPercent <= -25) {
            alerts.push({
              severity: "warning",
              icon: "\u26A0",
              message: `Running ${Math.round(Math.abs(diffPercent))}% slower than expected \u2014 check thermals or background load`,
            });
          }
        }
      }
    }
  }

  // Good state: inference running smoothly
  if (alerts.length === 0 && snapshot.activeModel && snapshot.tokensPerSec !== null && snapshot.tokensPerSec > 0) {
    alerts.push({
      severity: "success",
      icon: "\u2713",
      message: "Inference running smoothly",
    });
  }

  return alerts;
}

export const AlertBar = React.memo(function AlertBar({ snapshot, session, tokHistory }: AlertBarProps) {
  const alerts = computeAlerts(snapshot, session, tokHistory);

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
