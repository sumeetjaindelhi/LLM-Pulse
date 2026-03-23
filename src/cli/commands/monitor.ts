export async function monitorCommand(options?: { host?: string }): Promise<void> {
  const [{ default: React }, { render }, { App }] = await Promise.all([
    import("react"),
    import("ink"),
    import("../../tui/App.js"),
  ]);

  const { waitUntilExit } = render(React.createElement(App, { host: options?.host }));
  await waitUntilExit();
}
