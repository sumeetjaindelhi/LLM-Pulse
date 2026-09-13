import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { modelNotFoundPayload, renderModelNotFound } from "../../src/cli/ui/errors.js";

beforeEach(() => {
  vi.spyOn(console, "log").mockImplementation(() => {});
  process.exitCode = undefined;
});

afterEach(() => {
  vi.restoreAllMocks();
  process.exitCode = undefined;
});

describe("renderModelNotFound", () => {
  it("sets exit code 1 in silent (json/csv) mode", () => {
    renderModelNotFound("nosuchmodelxyz", { silent: true });
    expect(process.exitCode).toBe(1);
  });

  it("sets exit code 1 in table mode", () => {
    renderModelNotFound("nosuchmodelxyz");
    expect(process.exitCode).toBe(1);
  });
});

describe("modelNotFoundPayload", () => {
  // The MCP server builds its error responses from this; a long-lived server
  // process must not have its exit code flipped by a tool call.
  it("does not touch the exit code", () => {
    modelNotFoundPayload("nosuchmodelxyz");
    expect(process.exitCode).toBeUndefined();
  });
});
