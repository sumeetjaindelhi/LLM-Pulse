import { describe, it, expect } from "vitest";
import { escapeCsvValue, toCsv } from "../../src/cli/ui/csv.js";

describe("escapeCsvValue — formula injection guard", () => {
  it("neutralizes leading = + - @ in strings with a single-quote prefix", () => {
    expect(escapeCsvValue("=cmd|'/C calc'!A0")).toBe("'=cmd|'/C calc'!A0");
    expect(escapeCsvValue("+SUM(A1:A9)")).toBe("'+SUM(A1:A9)");
    expect(escapeCsvValue("-2+3")).toBe("'-2+3");
    expect(escapeCsvValue("@evil")).toBe("'@evil");
  });

  it("neutralizes leading tab and carriage return", () => {
    // \t and \r are alternate formula triggers in Excel; \r also forces quoting
    expect(escapeCsvValue("\t=1+1")).toBe("'\t=1+1");
    expect(escapeCsvValue("\r=1+1")).toBe(`"'\r=1+1"`);
  });

  it("leaves negative numbers untouched — guard applies to strings only", () => {
    expect(escapeCsvValue(-5)).toBe("-5");
    expect(escapeCsvValue(-273.15)).toBe("-273.15");
  });

  it("applies RFC 4180 quoting after the prefix", () => {
    expect(escapeCsvValue("=1,2")).toBe(`"'=1,2"`);
    expect(escapeCsvValue('=a"b')).toBe(`"'=a""b"`);
  });

  it("leaves ordinary strings unchanged", () => {
    expect(escapeCsvValue("llama3.1:8b")).toBe("llama3.1:8b");
    expect(escapeCsvValue("A fast, capable model")).toBe(`"A fast, capable model"`);
  });
});

describe("toCsv", () => {
  it("guards injected cells without disturbing numeric columns", () => {
    const csv = toCsv(["name", "vram"], [["=HYPERLINK(\"http://evil\")", -1]]);
    expect(csv).toBe('name,vram\n"\'=HYPERLINK(""http://evil"")",-1');
  });
});
