export async function monitorCommand(): Promise<void> {
  const [{ default: React }, { render }, { App }] = await Promise.all([
    import("react"),
    import("ink"),
    import("../../tui/App.js"),
  ]);

  const { waitUntilExit } = render(React.createElement(App));
  await waitUntilExit();
}
