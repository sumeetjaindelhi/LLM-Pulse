# llm-pulse

Zero-config CLI tool for local LLM hardware monitoring, model recommendations, and system diagnostics.

## Quick Reference

```bash
npm run build          # TypeScript → dist/
npm test               # vitest run (19 tests)
npm run dev            # tsc --watch
npm run lint           # tsc --noEmit (type check only)
node dist/bin/llm-pulse.js           # default: scan + recommend
node dist/bin/llm-pulse.js monitor   # live TUI dashboard (Tab to switch views)
node dist/bin/llm-pulse.js doctor    # system health check
node dist/bin/llm-pulse.js models    # browse model database
node dist/bin/llm-pulse.js benchmark # inference benchmark via Ollama
```

## Architecture

```
bin/llm-pulse.ts              → CLI entry point (commander)
src/
  cli/
    program.ts                → Command definitions (scan, doctor, models, monitor, benchmark)
    commands/                 → Command implementations
    ui/                       → Terminal UI helpers (badges, boxes, colors, progress, tables)
  core/
    types.ts                  → All TypeScript types (hardware, models, scoring, monitor, CLI)
    constants.ts              → Thresholds, weights, URLs, expected tok/s benchmarks
  hardware/
    index.ts                  → Hardware detection orchestrator
    cpu.ts, gpu.ts, memory.ts, disk.ts  → Per-component detection
    monitor.ts                → Real-time polling (EventEmitter, 1s interval)
  models/
    database.ts               → Model database loader (data/models.json, Zod-validated)
    schema.ts                 → Zod schemas for model data
  analysis/
    doctor.ts                 → System diagnostics & health scoring
    recommender.ts            → Model recommendation engine
    scorer.ts                 → Hardware-to-model scoring (fit levels, composite scores)
  runtimes/
    index.ts, ollama.ts, llamacpp.ts, lmstudio.ts  → Runtime detection
  tui/                        → React + Ink terminal UI
    App.tsx                   → Main app: tab system (Overview / Inference / VRAM)
    Overview.tsx              → Tab 1: hardware bars + sparklines + model info + alerts
    Inference.tsx             → Tab 2: throughput chart + session stats + model history
    VramMap.tsx               → Tab 3: VRAM breakdown + power info + "can still fit"
    Sparkline.tsx             → Reusable sparkline component (▁▂▃▅▆▇█ chars)
    AlertBar.tsx              → Smart contextual LLM-aware alerts
    CpuBar.tsx, GpuBar.tsx, MemoryBar.tsx  → Individual bar components (used by Overview)
    InferenceStats.tsx        → Legacy inference display (kept for compatibility)
data/
  models.json                → 30+ model database with quantization variants
tests/
  analysis/
    doctor.test.ts            → 7 tests for diagnostics
    scorer.test.ts            → 12 tests for scoring/recommendations
  fixtures/hardware-profiles/ → Test fixtures (high-end-nvidia, cpu-only, apple-m2)
```

## Tech Stack

- **TypeScript** (ES2022, Node16 modules, strict mode, JSX react-jsx)
- **React + Ink** for terminal UI (TUI)
- **systeminformation** for CPU/RAM polling
- **execa** for nvidia-smi subprocess calls
- **commander** for CLI parsing
- **chalk/boxen/cli-table3/ora** for static CLI output
- **Zod** for runtime data validation
- **Vitest** for testing (globals enabled, node environment)

## Key Patterns

- **ES Modules** everywhere — all imports use `.js` extensions (even for .ts/.tsx files)
- **HardwareMonitor** extends EventEmitter, emits `"snapshot"` events
- Monitor stores sparkline history as arrays (`cpuHistory`, `gpuHistory`, `tokHistory`) capped at 60 entries
- Session tracking lives on the monitor instance (`monitor.session`), not persisted to disk
- GPU data comes from `nvidia-smi` CSV output — gracefully returns nulls if not available
- Ollama data comes from `http://127.0.0.1:11434/api/ps` — gracefully returns nulls if not running
- Model database is loaded once and cached (`cachedModels` in database.ts)
- Fit scoring uses VRAM ratio: `available / required` mapped to fit levels via `FIT_THRESHOLDS`

## Monitor TUI Details

- **Tab key** cycles through: Overview → Inference → VRAM → Overview
- **q** quits
- App.tsx creates a single `HardwareMonitor` instance, copies history/session into React state on each tick
- Alert logic in `AlertBar.tsx` checks: VRAM pressure, speed drops, GPU underutil, idle timeout, performance vs expected benchmarks
- VramMap estimates breakdown (weights / KV cache / overhead / free) — these are estimates, not exact
- `EXPECTED_TOK_PER_SEC` in constants.ts maps GPU VRAM tier → model param size → expected tok/s

## Testing

- Tests are in `tests/` directory (excluded from tsconfig compilation)
- Uses hardware profile fixtures in `tests/fixtures/hardware-profiles/`
- Test files match pattern `tests/**/*.test.ts`
- Run single: `npx vitest run tests/analysis/scorer.test.ts`

## Common Tasks

**Add a new model to the database**: Edit `data/models.json`, add entry matching `ModelEntry` interface (id, name, provider, parametersBillion, contextWindow, categories, qualityTier, qualityScore, quantizations, ollamaTag, releaseDate)

**Add a new monitor alert**: Add condition logic in `AlertBar.tsx`'s `computeAlerts()` function, optionally add new threshold to `ALERT_THRESHOLDS` in constants.ts

**Add a new monitor tab**: Create component in `src/tui/`, add to `MonitorTab` union in types.ts, add to `TABS` and `TAB_LABELS` in App.tsx, render in the tab switch block

**Modify hardware polling**: Edit `HardwareMonitor.poll()` in `src/hardware/monitor.ts`, update `MonitorSnapshot` interface in same file
