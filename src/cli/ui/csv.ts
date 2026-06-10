/**
 * RFC 4180 CSV formatting utilities.
 */

// Leading characters that spreadsheet apps treat as formula starts (CWE-1236).
// Strings from outside the program (scraped library descriptions, Ollama API
// model names) end up in CSV exports — without this guard, a hostile value
// like `=cmd|...` executes when the file is opened in Excel/Sheets. The OWASP
// mitigation is a single-quote prefix, which spreadsheets strip on display.
// Applied to strings only, so numeric cells (e.g. -5) stay machine-readable.
const FORMULA_TRIGGER_RE = /^[=+\-@\t\r]/;

/** Escape a value per RFC 4180: quote if it contains commas, quotes, or
 *  newlines. String values that look like spreadsheet formulas get a
 *  single-quote prefix first. */
export function escapeCsvValue(value: unknown): string {
  let str = value === null || value === undefined ? "" : String(value);
  if (typeof value === "string" && FORMULA_TRIGGER_RE.test(str)) {
    str = `'${str}`;
  }
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
