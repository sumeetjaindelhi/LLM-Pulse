import { resolveOllamaHost } from "../../core/config.js";

export async function monitorCommand(options?: { host?: string }): Promise<void> {
  // Resolve through the config chain (CLI flag > .llmpulserc > default). Before
  // this fix, the monitor TUI bypassed the config file and always polled the
  // hardcoded default — users who set `ollamaHost` in their config were
  // silently ignored.
  const ollamaHost = resolveOllamaHost(options?.host);

  const [{ default: React }, { render }, { App }] = await Promise.all([
    import("react"),
    import("ink"),
    import("../../tui/App.js"),
  ]);

  const { waitUntilExit } = render(React.createElement(App, { host: ollamaHost }));
  await waitUntilExit();
}
