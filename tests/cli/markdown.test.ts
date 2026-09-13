import { describe, it, expect } from "vitest";
import { escapeMarkdownValue, toMarkdownTable } from "../../src/cli/ui/markdown.js";

describe("toMarkdownTable", () => {
  it("renders a header, a separator and one line per row", () => {
    expect(toMarkdownTable(["a", "b"], [[1, "x"], [2, "y"]])).toBe(
      "| a | b |\n| --- | --- |\n| 1 | x |\n| 2 | y |",
    );
  });

  it("renders only the header and separator when there are no rows", () => {
    expect(toMarkdownTable(["a", "b"], [])).toBe("| a | b |\n| --- | --- |");
  });
});

describe("escapeMarkdownValue", () => {
  it("renders null/undefined as an em dash and leaves plain values unchanged", () => {
    expect(escapeMarkdownValue(null)).toBe("—");
    expect(escapeMarkdownValue(undefined)).toBe("—");
    expect(escapeMarkdownValue(42)).toBe("42");
    expect(escapeMarkdownValue(true)).toBe("true");
    expect(escapeMarkdownValue("Q4_K_M")).toBe("Q4_K_M");
  });

  it("escapes a pipe so it cannot split a table cell", () => {
    expect(escapeMarkdownValue("a|b")).toBe("a\\|b");
  });

  it("escapes a backslash before a pipe so the pipe stays escaped", () => {
    // Input a\|b → output a\\\|b (escaped backslash, then escaped pipe).
    expect(escapeMarkdownValue("a\\|b")).toBe("a\\\\\\|b");
  });

  it("collapses line breaks and tab runs to a single space", () => {
    expect(escapeMarkdownValue("a\r\nb")).toBe("a b");
    expect(escapeMarkdownValue("a\nb")).toBe("a b");
    expect(escapeMarkdownValue("a\u2028b")).toBe("a b");
    expect(escapeMarkdownValue("a\t\tb")).toBe("a b");
  });

  it("strips ESC and C1 CSI so no terminal control sequence survives", () => {
    expect(escapeMarkdownValue("\u001b[31mred\u001b[0m")).not.toContain("\u001b");
    expect(escapeMarkdownValue("\u009b31m")).not.toContain("\u009b");
  });

  it("escapes raw HTML", () => {
    expect(escapeMarkdownValue("<img src=x>")).toBe("\\<img src=x\\>");
  });

  it("escapes link and image syntax", () => {
    expect(escapeMarkdownValue("[click](https://evil)")).toBe("\\[click\\](https://evil)");
    expect(escapeMarkdownValue("![](https://t)")).toBe("!\\[\\](https://t)");
  });

  it("removes bidirectional-text controls", () => {
    expect(escapeMarkdownValue("abc\u202edef")).toBe("abcdef");
  });
});
