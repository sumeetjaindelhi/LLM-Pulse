import React from "react";
import { render } from "ink";
import { App } from "../../tui/App.js";

export async function monitorCommand(): Promise<void> {
  const { waitUntilExit } = render(React.createElement(App));
  await waitUntilExit();
}
