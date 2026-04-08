#!/usr/bin/env node

import { startServer } from "../src/mcp/server.js";

startServer().catch((err) => {
  process.stderr.write(`llm-pulse-mcp: ${err instanceof Error ? err.message : String(err)}\n`);
  process.exit(1);
});
