import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    globals: true,
    environment: "node",
    include: ["tests/**/*.test.ts"],
    // Tests never touch the live catalog or the user's cache; live-mode tests opt in with vi.stubEnv.
    env: { LLMPULSE_OFFLINE: "1" },
  },
});
