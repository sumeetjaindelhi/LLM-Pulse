# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> Releases before `1.0.0` predate this changelog and were not consistently
> git-tagged, so their dates are omitted rather than reconstructed.

## [1.2.1] - 2026-11-19

1.2.0 was tagged but never published to npm, so 1.2.1 is the first npm
release carrying the 1.2.0 changes below, plus these dependency updates.

### Security
- Dependency floors raised past newly published advisories:
  - `@modelcontextprotocol/sdk` ^1.32.2: OAuth, session-binding and HTTP
    request-size advisories. The llm-pulse MCP server uses only the stdio
    transport.
  - `systeminformation` ^5.33.23: `networkStats()` and `services()`
    advisories. llm-pulse calls neither.
  - `zod` ^4.6.7: memory exhaustion while validating oversized constrained
    collections. The fix exists only in zod 4.
- `npm audit fix` refreshed the lockfile's transitive dependencies, and
  vitest is now 4.1.12 (dev only). `npm audit` reports 0 vulnerabilities.

### Changed
- Zod 4: the schemas use `z.url()`, `.loose()` and `error:` where zod 4
  deprecates `.url()` on strings, `.passthrough()` and `message:`. Validation
  rules and error messages are unchanged. The exported `LlmPulseConfig` type
  keeps its shape; it now resolves through zod 4's types, which support
  TypeScript 5.5 and later.

## [1.2.0] - 2026-11-18

Every memory figure is now exact (read from the model file or measured on your
machine), derived from the pinned llama.cpp and Ollama sources, or marked as an
estimate (`est.`, `≈`). The README section "How llm-pulse calculates fit" gives
each formula and its source.

### Added
- **Live model catalog:** recommendations, `models` and the single-model
  commands also use models from ollama.com/library, so new releases appear
  without a llm-pulse update. The CLI refreshes the catalog when the cached
  copy is more than a day old (5-second limit); the MCP server checks hourly.
  The exported `getAllModels` / `searchModels` / `filterByCategory` read the
  cached catalog but never fetch it. With the catalog in use, every model,
  curated ones included, is scored from size, release recency, pull velocity
  and a coding bonus; offline, curated models keep their hand-set scores.
  `LLMPULSE_OFFLINE=1` or `"offline": true` in `.llmpulserc` turns all of this
  off, including `models --library`.
- **Exact model facts:** `check`, `compare`, `quant-advice`, `optimize`,
  `context-fit` and the MCP `check` and `context-fit-check` tools read an
  installed GGUF model's architecture, training context and file size from
  your Ollama (`/api/show`, `/api/tags`). For a model that is neither bundled
  nor installed they read the same facts from Ollama's registry (manifest,
  config and the first 256 KiB of the GGUF file, cached in
  `~/.llmpulse/cache`), picking the build Ollama would pull on this platform.
  Vision projectors and draft models count toward the file size.
- **Memory breakdown in JSON:** `memory` (`contextTokens`, `weightsMb`,
  `weightsExact`, `kvCacheMb`, `kvExact`, `computeMb`, `totalMb`, `budgetMb`,
  `budgetExact`)
  in `check` (every quantization), `quant-advice`, `compare`, `optimize` (at
  `num_ctx`), `context-fit` (at the needed tokens), `scan`, and the MCP `scan`,
  `recommend`, `check` and `context-fit-check` tools. The `check`, `scan`,
  `context-fit`, `quant-advice` and `compare` CSVs append the same columns.
- More JSON fields:
  - `qualityMeasured` in `quant-advice` and `optimize`: false when
    `qualityRetention` is only a placeholder;
  - `pullQuantization` beside `pullCommand` in `check`, `quant-advice` and MCP
    `check`: the file that command installs;
  - `contextWindowEstimated` in `check`, `compare`, `optimize` and MCP
    `recommend`;
  - `gpuCount` in `check`, `quant-advice`, `optimize`, `scan` and the MCP
    `scan` and `recommend` tools;
  - `weightsExact` in `models` (JSON and CSV);
  - `nativeWindowEstimated` in `context-fit`;
  - `ollamaDefaultContext` and `kvQ8MaxContext` in `optimize`;
  - `catalog` in `scan`.
- `doctor`: a `Context` row, when a loaded model runs with less context than
  agents and coding tools need (64,000 tokens) while the machine could hold
  more, and a `Model catalog` row.
- MCP: `models` takes `limit` (default 100, max 500) and returns `total`;
  `check` now uses its `host` input; the `monitor` snapshot gains `modelVramMb` (Ollama
  `/api/ps` `size_vram`), `gpuCount` and `gpuUnifiedMemory`.
- Hardware: NVIDIA DGX Spark / GB10 unified memory; the AMD Strix Halo APU
  (gfx1151, its GTT pool); amdgpu `vram_total` instead of the lspci BAR size;
  Intel Arc discrete cards told apart from Arc-branded iGPUs;
  `OLLAMA_IGPU_ENABLE` honoured as Ollama does.
- Types: `ArchFacts`, `MemoryBreakdown`, `CatalogHealth`, `LoadedModel.arch`,
  `ModelEntry.arch` / `source` / `activeParametersBillion` /
  `contextWindowEstimated`, `QuantizationVariant.exact`,
  `GpuInfo.unifiedMemory` / `gfxTarget`, `ContextFitResult.memory` /
  `nativeWindowEstimated`.

### Changed
- **Memory model:** a fit is weights + KV cache + compute buffer at the
  context Ollama will use on this machine (4,096, 32,768 or 262,144 tokens by
  total VRAM, capped at the model's training context), checked against usable
  memory:
  - discrete GPUs: VRAM − 1 GiB (llama.cpp's fit margin) − 457 MiB (an
    estimate of what the driver and runtime context hold on an idle GPU),
    summed over the GPUs of one runtime (CUDA, ROCm or Vulkan), since Ollama
    loads a model onto one runtime's GPUs;
  - Apple Silicon: the Metal working set − 1 GiB;
  - GB10: available RAM − 1 GiB − 457 MiB; Strix Halo: min(GTT pool,
    available RAM) − 1 GiB − 457 MiB;
  - CPU only: available RAM.
  Integrated GPUs count only where Ollama uses them (NVIDIA, AMD gfx1151,
  Apple, or with `OLLAMA_IGPU_ENABLE`); others count as CPU only. The usable
  figure is marked `est.` where it includes the 457 MiB estimate. Fit levels
  are redefined on this basis (README table); the exported `classifyFit` keeps
  the 1.1.0 weights-ratio thresholds.
- KV cache sizes are exact wherever the architecture is known, including
  sliding-window, latent-attention and recurrent layers. They are marked
  `est.` for unknown architectures, models shipped with a draft model (when
  their facts come from the registry), parallel hybrids such as Falcon-H1,
  and MLX builds.
- The compute buffer (`≈`) adds the attention mask (context × batch × 2 bytes,
  one per GPU, four per GPU when several GPUs hold the whole model) to a
  vocabulary-based estimate of the graph's working memory.
- Quality per quantization is `exp(−KLD)` from llama.cpp's published KL
  divergence, shown with one decimal; formats without a measurement show
  `n/a`.
- Recommendations score the file `ollama pull <tag>` installs, rank by fit
  before score, show one model per family, and leave out embedding and
  natively ternary models. Partial fits are offered only with a discrete GPU.
- `check`, `compare` of named models and MCP `check` headline the same
  quantization (`compare --category` shows the pulled file, as `scan` does).
  MCP `check` returns the `check --format json` payload (an unknown
  quantization is an error there) and gains `gpuOffload`.
- `check` and `quant-advice` list quantizations smallest file first, in
  tables, JSON (`allQuantizations`, `quantizations`) and CSV (`check` listed
  them by score before).
- Table columns now say what they show: `scan`, `compare` and `report`
  `VRAM` → `Need` (weights + KV cache + compute buffer at the default
  context, not the weights alone); `check` `VRAM` / `Available` →
  `Need at N tokens` / `Usable`; `models` `VRAM` and `quant-advice` `Size` →
  `Weights`. These renames, and the MCP `monitor` `modelSize` unit (GB →
  GiB), change output the Stability section otherwise changes only in major
  releases. They correct labels: the old `VRAM` columns showed weights under
  the wrong name, and the old units were already binary.
- `optimize` tunes the file `FROM <tag>` loads, unless that file would not fit
  whole (then the sweet spot). For a quantization the tag does not install,
  the Modelfile reads `FROM <Q8_0 tag>` instead of silently loading another
  file. `num_batch` shows the batch Ollama picks automatically for `num_ctx`
  and is pinned (256) only on tight fits.
- Pull lines name the file `ollama pull <tag>` installs and its size, and
  point to the model's tag list when the recommended quantization is another
  file. `gpuOffload.ollamaCommand` is `null` when `ollama run <tag>` would
  load another quantization than the one sized. The suggested layer count is
  shown as an estimate (`≈`).
- Sizes are labelled GiB/MiB. The values were always binary (MiB ÷ 1024) but
  were labelled GB/MB: tables, `doctor` messages, the monitor TUI and the MCP
  `monitor` `modelSize` string.
- "Usable memory" replaces "VRAM" wherever the figure is the budget. Changed
  `optimize` notes: `automatic — Ollama uses up to this`, `smaller — eases
  tight-fit memory spike`, `KV cache won't fit at 2048 — expect CPU offload`,
  `capped at est. window — pull to confirm`.
- Several GPUs are named together (`2× NVIDIA GeForce RTX 3090`); a GPU
  Ollama does not use is labelled so instead of counting as usable VRAM.
- The `scan` tip ("you can run up to NB") is sized at a 4K context and marked
  as an estimate. The `doctor` GPU row no longer claims a model-size range.
- Monitor VRAM Map: measured figures only (the loaded model's GPU memory as
  Ollama reports it, other GPU use, free memory). "Can still fit" lists
  curated models whose need at 4,096 tokens fits in free memory after the
  runtime's reserves. The `VramBreakdown` type replaces `modelWeightsMb`,
  `kvCacheMb` and `overheadMb`, which held invented values, with
  `modelVramMb`, `otherMb` and `split`.
- `--quant` is matched after exact facts are applied; a live-catalog model's
  only quantization matches both `default` (or `ternary`) and its real type.

### Removed
- The flat 6 GB headroom subtracted from Apple Silicon GPU memory: the
  runtime's own margin replaces it.
- Invented figures in the monitor: model weights guessed from the model name,
  a 60% KV-cache share of used VRAM, and a fixed "can still fit" list.

### Fixed
- **Bundled model data, now read from Ollama's registry:**
  - Default files: the `gemma2`, `mixtral`, `phi3`, `codellama`,
    `deepseek-coder-v2`, `yi`, `starcoder2`, `tinyllama` and
    `mistral-small:22b` tags install Q4_0; `llama3.2:1b` and `smollm2:1.7b`
    install Q8_0; `nomic-embed-text` installs F16. Their sizes are the real
    files, vision projectors included.
  - Training contexts of 29 models, e.g. Qwen 2.5 is 32,768 tokens (not
    131,072), Qwen 3 8B–32B 40,960, Command R7B 8,192.
  - 12 models (Qwen 3 8B/14B/32B, DeepSeek-R1 7B/14B/32B, Gemma 3 12B/27B,
    Phi-4, Mistral Small 24B, QwQ, Llama 4 Scout) listed a Q5_K_M that Ollama
    does not publish; they now list the published Q8_0.
- Intel GPUs on Linux reported 16 or 256 MiB of VRAM (an lspci BAR size), so
  every model read as unable to run; they now count as CPU only.
- Model card badges from ollama.com are stripped of terminal control
  characters.

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
  - Apple Silicon usable GPU memory comes from Metal's
    `recommendedMaxWorkingSetSize` (74% of 24 GB on an M5 Pro) instead of a
    fixed 67% estimate, which understated it by about 1.7 GB there.
    `iogpu.wired_limit_mb` still wins when set; 67% remains the last resort
    when Metal cannot be queried.
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
- `npm audit fix` lockfile refresh: 0 production advisories. vitest was
  upgraded to 4 (dev only), so `npm audit` reports 0 vulnerabilities. Running
  the test suite now needs Node `^20.19.0 || >=22.12.0` (vite 8); the published
  package still runs on `>=20.5.0`.

### Verified
- 319 tests across 41 test files.

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
