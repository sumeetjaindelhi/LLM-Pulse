import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

const notFound = () => Object.assign(new Error("not found"), { exitCode: 1 });

describe("runtime detection", () => {
  const originalFetch = globalThis.fetch;
  const originalPlatform = process.platform;
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.resetModules();
    fetchMock = vi.fn();
    globalThis.fetch = fetchMock as unknown as typeof fetch;
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
    Object.defineProperty(process, "platform", { value: originalPlatform });
    vi.restoreAllMocks();
  });

  describe("detectOllama", () => {
    it("reports running when the API answers even though the binary is not on PATH", async () => {
      const execaMock = vi.fn().mockRejectedValue(notFound());
      vi.doMock("execa", () => ({ execa: execaMock }));
      fetchMock.mockImplementation(async (url: string) => ({
        ok: true,
        json: async () =>
          url.endsWith("/api/version") ? { version: "0.12.0" } : { models: [{ name: "llama3.1:8b", size: 1 }] },
      }));

      const { detectOllama } = await import("../../src/runtimes/ollama.js");
      const info = await detectOllama("http://127.0.0.1:11434");

      expect(info).toEqual({
        name: "Ollama",
        status: "running",
        version: "0.12.0",
        path: null,
        models: ["llama3.1:8b"],
      });
      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:11434/api/version",
        expect.objectContaining({ redirect: "error" }),
      );
      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:11434/api/tags",
        expect.objectContaining({ redirect: "error" }),
      );
    });

    it("skips the CLI version fallback when neither the API nor the binary exists", async () => {
      const execaMock = vi.fn().mockRejectedValue(notFound());
      vi.doMock("execa", () => ({ execa: execaMock }));
      fetchMock.mockRejectedValue(new TypeError("fetch failed"));

      const { detectOllama } = await import("../../src/runtimes/ollama.js");
      const info = await detectOllama();

      expect(info.status).toBe("not_found");
      expect(execaMock.mock.calls.map((c) => c[0])).not.toContain("ollama");
    });

    it("reads the CLI version when the binary exists but the API is down", async () => {
      vi.doMock("execa", () => ({
        execa: vi.fn().mockImplementation(async (cmd: string) => {
          if (cmd === "ollama") return { stdout: "ollama version is 0.11.4" };
          return { stdout: "/usr/local/bin/ollama\n" };
        }),
      }));
      fetchMock.mockRejectedValue(new TypeError("fetch failed"));

      const { detectOllama } = await import("../../src/runtimes/ollama.js");
      const info = await detectOllama();

      expect(info).toMatchObject({ status: "installed", version: "0.11.4", path: "/usr/local/bin/ollama" });
    });
  });

  describe("detectLlamaCpp", () => {
    it("does not treat an unrelated binary named main as llama.cpp", async () => {
      vi.doMock("execa", () => ({
        execa: vi.fn().mockImplementation(async (cmd: string, args: string[]) => {
          if ((cmd === "which" || cmd === "where") && args[0] === "main") return { stdout: "/usr/bin/main\n" };
          throw notFound();
        }),
      }));

      const { detectLlamaCpp } = await import("../../src/runtimes/llamacpp.js");
      const info = await detectLlamaCpp();

      expect(info.status).toBe("not_found");
      expect(info.path).toBeNull();
    });
  });

  describe("detectLmStudio", () => {
    it("refuses redirects when probing the local server", async () => {
      Object.defineProperty(process, "platform", { value: "darwin" });
      vi.doMock("node:fs", () => ({ existsSync: vi.fn().mockReturnValue(true) }));
      fetchMock.mockResolvedValue({ ok: true, json: async () => ({ data: [{ id: "qwen2.5-7b" }] }) });

      const { detectLmStudio } = await import("../../src/runtimes/lmstudio.js");
      const info = await detectLmStudio("http://127.0.0.1:1234");

      expect(info.status).toBe("running");
      expect(fetchMock).toHaveBeenCalledWith(
        "http://127.0.0.1:1234/v1/models",
        expect.objectContaining({ redirect: "error" }),
      );
    });
  });
});
