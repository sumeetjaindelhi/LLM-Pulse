/**
 * GitHub-flavoured markdown formatting utilities.
 */

// Bidirectional-text controls reorder how text is displayed in terminals and on
// GitHub ("Trojan Source" spoofing), so they are removed outright.
const BIDI_CONTROL_RE = /[\u200e\u200f\u202a-\u202e\u2066-\u2069]/g;

// C0 controls (incl. TAB, LF, CR, ESC), DEL, C1 controls (incl. CSI) and the
// Unicode line/paragraph separators. A line break inside a value would end a
// table row and let a hostile string append fake rows; ESC/CSI would let it
// drive the terminal the report is first printed to.
const CONTROL_RUN_RE = /[\u0000-\u001f\u007f-\u009f\u2028\u2029]+/g;

// `|` splits table cells, `<` `>` open raw HTML and autolinks, `[` `]` open links
// and images. The backslash is escaped in the same pass so a value's own
// trailing backslash can never cancel the escape we add before a pipe.
const MARKDOWN_SPECIAL_RE = /[\\|<>[\]]/g;

/** Make an arbitrary value safe to interpolate into GitHub-flavoured markdown
 *  (table cell, list item, heading, quote). null/undefined render as an em dash. */
export function escapeMarkdownValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  return String(value)
    .replace(BIDI_CONTROL_RE, "")
    .replace(CONTROL_RUN_RE, " ")
    .replace(MARKDOWN_SPECIAL_RE, "\\$&");
}

/** Build a GFM table: header row, `| --- |` separator, one line per row. Every
 *  header and cell is escaped. */
export function toMarkdownTable(
  headers: string[],
  rows: (string | number | boolean | null | undefined)[][],
): string {
  const toLine = (cells: unknown[]): string => `| ${cells.map(escapeMarkdownValue).join(" | ")} |`;
  return [toLine(headers), `|${" --- |".repeat(headers.length)}`, ...rows.map(toLine)].join("\n");
}
