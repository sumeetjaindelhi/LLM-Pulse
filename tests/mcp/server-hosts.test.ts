import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

// The MCP server only talks to local runtimes. Hosts that come from a
// .llmpulserc in the server's cwd or from OLLAMA_HOST must pass the same
// localhost validation as the tool's `host` input, or fall back to loopback.
function mockRcFile(rc: Record<string, unknown> | null) {
  vi.doMock("node:fs", async (importOriginal) => {
    const actual = await importOriginal<typeof import("node:fs")>();
    return {
      ...actual,
      readFileSync: vi.fn((path: Parameters<typeof actual.readFileSync>[0], ...rest: unknown[]) => {
        if (String(path).endsWith(".llmpulserc")) {
          if (rc === null) throw Object.assign(new Error("ENOENT"), { code: "ENOENT" });
          return JSON.stringify(rc);
        }
        return (actual.readFileSync as (...args: unknown[]) => unknown)(path, ...rest);
      }),
    };
  });
}

describe("MCP host resolution", () => {
  const originalEnv = process.env.OLLAMA_HOST;

  beforeEach(() => {
    vi.resetModules();
    delete process.env.OLLAMA_HOST;
  });

  afterEach(() => {
    if (originalEnv !== undefined) process.env.OLLAMA_HOST = originalEnv;
    else delete process.env.OLLAMA_HOST;
    vi.doUnmock("node:fs");
  });

  it("falls back to loopback when OLLAMA_HOST points off-host", async () => {
    process.env.OLLAMA_HOST = "10.0.0.5:11434";
    mockRcFile(null);
    const { resolveMcpOllamaHost } = await import("../../src/mcp/server.js");
    expect(resolveMcpOllamaHost()).toBe("http://127.0.0.1:11434");
  });

  it("falls back to loopback when .llmpulserc ollamaHost points off-host", async () => {
    mockRcFile({ ollamaHost: "http://169.254.169.254/x" });
    const { resolveMcpOllamaHost } = await import("../../src/mcp/server.js");
    expect(resolveMcpOllamaHost()).toBe("http://127.0.0.1:11434");
  });

  it("keeps a localhost .llmpulserc ollamaHost", async () => {
    mockRcFile({ ollamaHost: "http://localhost:11500" });
    const { resolveMcpOllamaHost } = await import("../../src/mcp/server.js");
    expect(resolveMcpOllamaHost()).toBe("http://localhost:11500");
  });

  it("keeps a validated localhost host input", async () => {
    mockRcFile({ ollamaHost: "http://10.0.0.5:11434" });
    const { resolveMcpOllamaHost } = await import("../../src/mcp/server.js");
    expect(resolveMcpOllamaHost("http://127.0.0.1:11600")).toBe("http://127.0.0.1:11600");
  });

  it("falls back to loopback when .llmpulserc lmstudioHost points off-host", async () => {
    mockRcFile({ lmstudioHost: "http://10.0.0.5:1234" });
    const { resolveMcpLmStudioHost } = await import("../../src/mcp/server.js");
    expect(resolveMcpLmStudioHost()).toBe("http://127.0.0.1:1234");
  });

  it("keeps a localhost .llmpulserc lmstudioHost", async () => {
    mockRcFile({ lmstudioHost: "http://localhost:4321" });
    const { resolveMcpLmStudioHost } = await import("../../src/mcp/server.js");
    expect(resolveMcpLmStudioHost()).toBe("http://localhost:4321");
  });
});
