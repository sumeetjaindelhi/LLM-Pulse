import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

const refresh = vi.hoisted(() => vi.fn(async () => {}));
vi.mock("../../src/models/library-cache.js", () => ({ refreshLibraryCatalogIfStale: refresh }));
vi.mock("../../src/core/config.js", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../src/core/config.js")>()),
  // Tests must not read the developer's real ~/.llmpulserc.
  isOffline: () => process.env.LLMPULSE_OFFLINE === "1",
}));

// Command bodies are not under test; only whether the hook fetched first.
const commandMocks = ["scan", "doctor", "models", "compare", "check", "quant-advice", "optimize", "context-fit", "report", "benchmark", "profile"];
for (const name of commandMocks) {
  const exportName = `${name.replace(/-(\w)/g, (_m, c: string) => c.toUpperCase())}Command`;
  vi.doMock(`../../src/cli/commands/${name}.js`, () => ({ [exportName]: vi.fn(async () => {}) }));
}

const { createProgram } = await import("../../src/cli/program.js");

async function run(...args: string[]): Promise<void> {
  const program = createProgram();
  for (const cmd of [program, ...program.commands]) cmd.exitOverride();
  await program.parseAsync(args, { from: "user" });
}

beforeEach(() => {
  refresh.mockClear();
  vi.stubEnv("LLMPULSE_OFFLINE", "");
});

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("catalog refresh hook", () => {
  it.each([
    [["scan"]],
    [[]],
    [["models"]],
    [["compare", "llama3.1:8b"]],
    [["check", "llama3.1:8b"]],
    [["quant-advice", "llama3.1:8b"]],
    [["optimize", "llama3.1:8b"]],
    [["context-fit", "llama3.1:8b", "--prompt-tokens", "100"]],
    [["report"]],
  ])("refreshes before %j", async (args) => {
    await run(...args);
    expect(refresh).toHaveBeenCalledTimes(1);
  });

  it.each([[["doctor"]], [["benchmark"]], [["profile"]]])("does not fetch for %j", async (args) => {
    await run(...args);
    expect(refresh).not.toHaveBeenCalled();
  });

  it.each([[["models", "--library"]], [["models", "--refresh"]]])(
    "leaves %j to the command's own explicit fetch",
    async (args) => {
      await run(...args);
      expect(refresh).not.toHaveBeenCalled();
    },
  );

  it.each([[["models", "--installed"]], [["models", "--live"]]])("does not fetch for %j, which never reads the live catalog", async (args) => {
    await run(...args);
    expect(refresh).not.toHaveBeenCalled();
  });

  it("does not fetch when offline", async () => {
    vi.stubEnv("LLMPULSE_OFFLINE", "1");
    await run("scan");
    expect(refresh).not.toHaveBeenCalled();
  });

  it("awaits the refresh before the command runs", async () => {
    const order: string[] = [];
    refresh.mockImplementationOnce(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
      order.push("refresh");
    });
    const { scanCommand } = await import("../../src/cli/commands/scan.js");
    vi.mocked(scanCommand).mockImplementationOnce(async () => {
      order.push("scan");
    });
    await run("scan");
    expect(order).toEqual(["refresh", "scan"]);
  });
});
