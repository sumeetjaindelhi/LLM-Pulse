import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

// Each test mocks the config file state and OLLAMA_HOST env var, then
// re-imports resolveOllamaHost to exercise the precedence chain:
//   CLI > config > env > default
describe("resolveOllamaHost precedence", () => {
  const originalEnv = process.env.OLLAMA_HOST;

  beforeEach(() => {
    vi.resetModules();
    delete process.env.OLLAMA_HOST;
  });

  afterEach(() => {
    if (originalEnv !== undefined) process.env.OLLAMA_HOST = originalEnv;
    else delete process.env.OLLAMA_HOST;
    vi.restoreAllMocks();
  });

  it("uses the default constant when nothing else is set", async () => {
    vi.doMock("node:fs", async (importOriginal) => {
      const actual = await importOriginal<typeof import("node:fs")>();
      return {
        ...actual,
        readFileSync: vi.fn(() => {
          throw Object.assign(new Error("ENOENT"), { code: "ENOENT" });
        }),
      };
    });
    const { resolveOllamaHost } = await import("../../src/core/config.js");
    expect(resolveOllamaHost()).toBe("http://127.0.0.1:11434");
  });

  it("honors OLLAMA_HOST env var in host:port form", async () => {
    process.env.OLLAMA_HOST = "192.168.1.50:11434";
    vi.doMock("node:fs", async (importOriginal) => {
      const actual = await importOriginal<typeof import("node:fs")>();
      return {
        ...actual,
        readFileSync: vi.fn(() => {
          throw Object.assign(new Error("ENOENT"), { code: "ENOENT" });
        }),
      };
    });
    const { resolveOllamaHost } = await import("../../src/core/config.js");
    const result = resolveOllamaHost();
    expect(result).toContain("192.168.1.50");
    expect(result).toContain("11434");
    expect(result).toMatch(/^http:\/\//);
  });

  it("honors OLLAMA_HOST env var in full URL form", async () => {
    process.env.OLLAMA_HOST = "https://ollama.internal:443";
    vi.doMock("node:fs", async (importOriginal) => {
      const actual = await importOriginal<typeof import("node:fs")>();
      return {
        ...actual,
        readFileSync: vi.fn(() => {
          throw Object.assign(new Error("ENOENT"), { code: "ENOENT" });
        }),
      };
    });
    const { resolveOllamaHost } = await import("../../src/core/config.js");
    expect(resolveOllamaHost()).toContain("ollama.internal");
  });

  it("CLI flag beats env var", async () => {
    process.env.OLLAMA_HOST = "http://should-be-ignored:11434";
    vi.doMock("node:fs", async (importOriginal) => {
      const actual = await importOriginal<typeof import("node:fs")>();
      return {
        ...actual,
        readFileSync: vi.fn(() => {
          throw Object.assign(new Error("ENOENT"), { code: "ENOENT" });
        }),
      };
    });
    const { resolveOllamaHost } = await import("../../src/core/config.js");
    expect(resolveOllamaHost("http://cli-wins:11434")).toBe("http://cli-wins:11434");
  });

  it("ignores malformed env values and falls back to default", async () => {
    process.env.OLLAMA_HOST = "not a url at all!@#";
    vi.doMock("node:fs", async (importOriginal) => {
      const actual = await importOriginal<typeof import("node:fs")>();
      return {
        ...actual,
        readFileSync: vi.fn(() => {
          throw Object.assign(new Error("ENOENT"), { code: "ENOENT" });
        }),
      };
    });
    const { resolveOllamaHost } = await import("../../src/core/config.js");
    expect(resolveOllamaHost()).toBe("http://127.0.0.1:11434");
  });
});
