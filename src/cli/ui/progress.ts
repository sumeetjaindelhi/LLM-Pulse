import { theme } from "./colors.js";

export function progressBar(
  percent: number,
  width = 20,
  label?: string,
): string {
  const clamped = Math.max(0, Math.min(100, percent));
  const filled = Math.round((clamped / 100) * width);
  const empty = width - filled;

  const bar = "█".repeat(filled) + "░".repeat(empty);

  const coloredBar =
    clamped >= 90 ? theme.fail(bar)
    : clamped >= 70 ? theme.warning(bar)
    : theme.pass(bar);

  const suffix = label ? ` ${label}` : ` ${clamped}%`;
  return `${coloredBar}${theme.muted(suffix)}`;
}

export function formatMb(mb: number): string {
  if (mb >= 1024) {
    return `${(mb / 1024).toFixed(1)} GB`;
  }
  return `${mb.toLocaleString()} MB`;
}
