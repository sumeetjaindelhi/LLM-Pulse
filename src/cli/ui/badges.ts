import { theme } from "./colors.js";
import type { FitLevel, CheckSeverity } from "../../core/types.js";

const FIT_BADGES: Record<FitLevel, string> = {
  excellent: "★★★★★",
  comfortable: "★★★★☆",
  tight: "★★★☆☆",
  barely: "★★☆☆☆",
  cannot_run: "✗",
};

export function fitBadge(level: FitLevel): string {
  const badge = FIT_BADGES[level];
  switch (level) {
    case "excellent": return theme.pass(badge);
    case "comfortable": return theme.pass(badge);
    case "tight": return theme.warning(badge);
    case "barely": return theme.fail(badge);
    case "cannot_run": return theme.fail(badge);
  }
}

const SEVERITY_ICONS: Record<CheckSeverity, string> = {
  pass: "✓",
  warning: "⚠",
  fail: "✗",
  info: "ℹ",
};

export function severityIcon(severity: CheckSeverity): string {
  const icon = SEVERITY_ICONS[severity];
  switch (severity) {
    case "pass": return theme.pass(icon);
    case "warning": return theme.warning(icon);
    case "fail": return theme.fail(icon);
    case "info": return theme.info(icon);
  }
}
