# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> Releases before `1.0.0` predate this changelog and were not consistently
> git-tagged, so their dates are omitted rather than reconstructed.

## [1.0.0] - 2026-06-28

First stable release. **No functional changes from 0.9.7** — this release marks
the public surface as stable under semantic versioning.

### Added
- `CHANGELOG.md` (this file).
- README **Stability** section documenting the semver guarantee.

### Stability
- The CLI commands and flags, the `table`/`json`/`csv` output shapes, the
  programmatic API (`detectHardware`, `getRecommendations`), and the 7 MCP tools
  are now stable. Breaking changes to any of them will bump the major version.

### Verified
- Full command smoke pass on Apple Silicon; build clean; 183/183 tests passing;
  `npm audit` clean; published tarball limited to `dist/` + `data/`.

## [0.9.7]
### Added
- `context-fit` CLI command and `context-fit-check` MCP tool (tool #7): does a
  prompt of N tokens fit, given `min(native context, VRAM-afforded KV cache)` —
  with a yes/tight/no verdict, whether the model or the hardware is the binding
  limit, and a remedy. Shared `maxContextTokensForVram` KV estimate extracted so
  the optimizer and fit-check agree. (183 tests.)

## [0.9.6]
### Added
- `doctor --fix --dry-run`: previews the exact commands `--fix` would run,
  classified against the same binary allowlist the runner enforces.
### Security
- CSV formula-injection guard (CWE-1236) in exports; `npm audit` 9 → 0
  vulnerabilities; `package-lock.json` now tracked.
### Fixed
- README MCP config for fresh installs (`npx -y -p llm-pulse llm-pulse-mcp`);
  documented `compare`/`profile`; removed a phantom `scan -v` flag. (164 tests.)

## [0.9.5]
### Added
- `optimize <model>`: recommends tuned Ollama runtime parameters
  (`num_ctx`/`num_gpu`/`num_thread`/`num_batch`) for the sweet-spot quant, with a
  conservative KV-cache-bounded context that won't risk OOM. (154 tests.)

## [0.9.3]
### Added
- `quant-advice <model>`: quality-vs-VRAM tradeoff table with a sweet-spot pick
  (the largest quant that still fits comfortably). (134 tests.)

## [0.9.2]
### Added
- GPU layer-offload guidance in `check`: when a model overflows VRAM, computes
  how many transformer blocks to put on the GPU (Ollama `num_gpu` /
  llama.cpp `--n-gpu-layers`). Hidden on Apple Silicon and CPU-only. (129 tests.)

## [0.9.0]
### Fixed
- Platform correctness pass: Apple Silicon wired-memory cap via
  `sysctl iogpu.wired_limit_mb` (was overcounting); Linux cgroups v1/v2 container
  RAM limits; Intel Arc VRAM via sysfs DRM; ROCm 6.x per-GPU JSON parsing;
  Windows `nvidia-smi` path resolver; `OLLAMA_HOST` honored; hybrid CPU P/E-core
  weighting.
### Security
- Cache symlink-escape + TOCTOU defense, prototype-pollution strip, 5 MB library
  HTML cap, dependency CVE fixes. (123 tests, up from 85.)

## [0.8.0]
### Added
- Dynamic model catalog: scrape ollama.com/library with a 24 h disk cache and
  `--library` / `--refresh` flags; catalog grows from 48 to 245+ models.

## [0.7.4]
### Security
- Production hardening: SSRF lockdown on the MCP `host` parameter, 60 s hardware
  cache, capped session history.

## [0.7.2] – [0.7.3]
### Added
- First-class Apple Silicon support (unified memory as VRAM, Metal GPU live stats
  via `ioreg`, accurate RAM/disk/CPU).
### Fixed
- Security & correctness hardening (`parseIntSafe`, subprocess retry wrappers,
  defensive guards).

## [0.7.0] – [0.7.1]
### Added
- MCP server for Claude Code / AI assistants, plus a `monitor` snapshot tool
  (6 tools at the time).

## [0.6.0]
### Added
- `check <model>` — the "Can I Run This?" pre-check.

## [0.5.0]
### Added
- `doctor --fix` auto-fix suggestions; model pull from the TUI monitor.
### Changed
- TUI performance: `useReducer`, `React.memo`, subprocess dedup.

## [0.4.0]
### Added
- JSON/CSV export for reports; `.llmpulserc` user config; remote Ollama support
  via `--host`.

## [0.3.0] – [0.3.1]
### Added
- `compare` command; model database expanded to 48 models / 109 quantizations;
  Zod schema validation; `setTimeout`-chain polling.

## [0.2.0]
### Added
- GPU detail tab and temperature alerts in the TUI.

## [0.1.0]
### Added
- Initial release: hardware scan, model recommendations, GPU utilization
  tracking, and the live `monitor` TUI.
