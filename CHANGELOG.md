# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> Releases before `1.0.0` predate this changelog and were not consistently
> git-tagged, so their dates are omitted rather than reconstructed.

## [1.1.0] - 2026-09-13

### Added
- `report` command: prints a paste-ready GitHub-flavoured markdown system report
  (hardware, runtimes, health score, top recommended models) for GitHub issues,
  Reddit and Discord. Flags `-c/--category`, `-t/--top`, `-H/--host`, same as
  `scan`. Private by default — runtime install paths, installed model names and
  the Ollama host URL are never printed — and untrusted strings (driver-reported
  hardware names, Ollama versions) are escaped so they cannot inject table rows,
  HTML, links, bidi spoofing or terminal control sequences. CLI only; the MCP
  server stays at 7 tools.
- Monitor Overview and the MCP `monitor` snapshot show the loaded model's context
  length (`modelContextLength`, from Ollama `/api/ps`).
- Model database: optional `kvMbPer1kTokens` field on the exported `ModelEntry`
  type (real fp16 KV-cache MB per 1K tokens), set for 14 curated models.
- `searchModels(query, category?)` accepts an optional category filter.

### Changed
- `engines.node` is now `>=20.5.0`. 1.0.0 already needed it in practice (its
  `ink` and `execa` dependencies require Node 20 / 20.5).

### Removed
- Monitor Inference tab, tok/s displays, the "generating" status and tok/s-based
  alerts (speed drop, GPU underutilized, faster/slower than expected). Ollama's
  `/api/ps` exposes no live throughput, so they never showed real data. The
  exported `MonitorTab` type no longer includes `"inference"`. `tokensPerSec`
  stays in JSON / MCP snapshots and is always `null`.

### Fixed
- **Model library scraper:** `models --library` / `--refresh` parse the current
  ollama.com/library markup again (it returned 0 models); an offline `--refresh`
  keeps the cached catalog instead of deleting it. Parsing is linear, responses
  declaring more than 5 MB are refused before download, and model cards over
  64 KB are skipped.
- **Model data and tags:** Ollama pull tags — Llama 4 Scout is `llama4:scout`,
  Command R 7B is `command-r7b` (release date 2024-12). `models --search` also
  applies `--category` (CLI and MCP `models` tool).
- **Analysis correctness:**
  - `optimize` / `check` / `quant-advice` / `context-fit`: when no quant fits
    comfortably, the auto-picked quant is a tight fit rather than a
    higher-quality fit that needs CPU offload (e.g. Q4_K_M instead of Q5_K_M on
    16 GB Apple M2 + deepseek-r1-7b).
  - `optimize` / `context-fit`: KV-cache estimates use the per-model figures, so
    recommended `num_ctx` and context ceilings are lower (and no longer risk OOM)
    for phi-3-mini, phi-4-mini, codellama-7b/13b, gemma-2-2b/9b,
    gemma-3-4b/12b/27b, qwen-3-0.6b/1.7b/4b, llama-3.2-3b and smollm2-1.7b.
  - `optimize`: when the KV budget cannot hold the 2048-token floor, `num_batch`
    is 256 and the note warns of CPU offload.
  - `context-fit` remedy no longer suggests trimming the prompt to 0 tokens.
  - `compare --quant` is case-insensitive, like the other commands.
  - `check` on Apple Silicon no longer claims "75% usable for inference".
- **CLI input validation and exit codes:**
  - Invalid `--format` / `--category` values are rejected with exit 1 instead of
    silently falling back. `--help` lists the allowed choices.
  - Numeric flags (`--top`, `--prompt-tokens`, `--response-tokens`, `--rounds`,
    `--context-size`) accept only whole decimal integers; `50k`, `1.5`, `-1` etc.
    are rejected with exit 1 instead of being truncated or replaced by the
    default.
  - Error outputs exit with code 1: model not found (`check`, `quant-advice`,
    `optimize`, `context-fit`, `compare`), unknown `--quant` in `context-fit`, no
    fitting quantization in `optimize`, `compare` with fewer than 2 models, and
    `benchmark`/`profile` when Ollama is not running, no models are installed or
    every run failed. Payload shapes are unchanged.
  - `compare --format json/csv` writes warnings to stderr, so stdout is always
    valid JSON/CSV.
  - Table borders no longer contain ANSI colour codes when piped or with
    `NO_COLOR`.
  - `profile` no longer hangs after a failed inference.
- **Hardware and runtime detection:**
  - Multi-NVIDIA: each card reports its own stats, and one card nvidia-smi cannot
    read no longer disables CUDA detection for the others.
  - AMD GPUs are detected from rocm-smi CSV output when `--json` is unavailable;
    the monitor shows AMD live stats.
  - Intel iGPUs report `vramMb` 0 so models are scored against system RAM;
    discrete Intel Arc cards keep their reported VRAM.
  - Containers (cgroups v1/v2): `memory.availableMb` is capped at the container's
    limit minus its current usage, excluding reclaimable page cache.
  - Apple Silicon `monitor`: `gpuVramTotalMb` is the Metal wired limit, matching
    `scan`, instead of total RAM.
  - Linux hybrid CPUs count performance cores from per-CPU topology instead of
    halving (wrong on hybrid CPUs without SMT).
  - Ollama is reported running when its API answers even if the binary is not on
    `PATH`; an unrelated `main` binary is no longer mistaken for llama.cpp.
- **TUI model manager:** `d` asks for confirmation before deleting; mid-stream
  pull failures show as errors instead of "Download complete!"; leaving the
  Models tab or quitting aborts an in-flight pull.

### Security
- MCP server: Ollama / LM Studio hosts resolved from `.llmpulserc` or
  `OLLAMA_HOST` are validated as loopback, otherwise the default local URL is
  used. The unused `host` input on the `check`, `recommend` and `models` tools
  is documented as ignored.
- Local Ollama / LM Studio API calls refuse HTTP redirects.
- Cache writes are atomic (temp file + rename) and replace, rather than follow,
  a planted symlink, even a dangling one; a failed write leaves no temp file
  behind.
- `npm audit fix` lockfile refresh: 0 production advisories.

### Verified
- 311 tests across 41 test files.

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
