import { describe, it, expect } from "vitest";
import { parsePullLines } from "../../src/tui/pull-stream.js";

describe("parsePullLines", () => {
  it("parses complete lines and keeps the unfinished tail", () => {
    const { events, rest } = parsePullLines('{"status":"pulling manifest"}\n{"status":"downl');
    expect(events).toEqual([{ status: "pulling manifest" }]);
    expect(rest).toBe('{"status":"downl');
  });

  it("parses a line split across two chunks once it completes", () => {
    const first = parsePullLines('{"status":"downloading","total":100,');
    expect(first.events).toEqual([]);
    const second = parsePullLines(`${first.rest}"completed":50}\n`);
    expect(second.events).toEqual([{ status: "downloading", total: 100, completed: 50 }]);
    expect(second.rest).toBe("");
  });

  it("surfaces error lines", () => {
    const { events } = parsePullLines(
      '{"status":"pulling manifest"}\n{"error":"pull model manifest: file does not exist"}\n',
    );
    expect(events[1]).toEqual({ error: "pull model manifest: file does not exist" });
  });

  it("skips blank and malformed lines", () => {
    const { events } = parsePullLines('\nnot json\n{"status":"success"}\n');
    expect(events).toEqual([{ status: "success" }]);
  });
});
