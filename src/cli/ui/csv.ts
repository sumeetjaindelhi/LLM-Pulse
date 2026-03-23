/**
 * RFC 4180 CSV formatting utilities.
 */

/** Escape a value per RFC 4180: quote if it contains commas, quotes, or newlines. */
export function escapeCsvValue(value: unknown): string {
  const str = value === null || value === undefined ? "" : String(value);
  if (str.includes(",") || str.includes('"') || str.includes("\n") || str.includes("\r")) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

/** Build a full CSV string from headers and rows. */
export function toCsv(headers: string[], rows: (string | number | boolean | null | undefined)[][]): string {
  const lines: string[] = [];
  lines.push(headers.map(escapeCsvValue).join(","));
  for (const row of rows) {
    lines.push(row.map(escapeCsvValue).join(","));
  }
  return lines.join("\n");
}
