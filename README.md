# llm-pulse

[![npm version](https://img.shields.io/npm/v/llm-pulse.svg)](https://www.npmjs.com/package/llm-pulse)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Node.js](https://img.shields.io/badge/node-%3E%3D20.5-brightgreen.svg)](https://nodejs.org/)

Zero-config CLI that tells you what LLMs your PC can run. Scans hardware, finds runtimes, recommends models.

```bash
npx llm-pulse
```

## Install

```bash
# Run directly (no install)
npx llm-pulse

# Or install globally
npm install -g llm-pulse
```

Requires Node.js 20.5+.

## Commands

### `llm-pulse` / `llm-pulse scan`

Hardware scan + model recommendations. Each pick shows the memory it needs at the context Ollama will use on your machine; `est.` marks an estimated part (see [How llm-pulse calculates fit](#how-llm-pulse-calculates-fit)).

```bash
llm-pulse                            # Full scan (default)
llm-pulse --format json              # JSON output
llm-pulse --category coding --top 3  # Top 3 coding models
```

| Flag | Description | Default |
|---|---|---|
| `-f, --format` | `table`, `json`, or `csv` | `table` |
| `-c, --category` | `general`, `coding`, `reasoning`, `creative` (no model is tagged creative yet), `multilingual` | `all` |
| `-t, --top <n>` | Number of recommendations | `5` |
| `-H, --host <url>` | Ollama API host URL | `http://127.0.0.1:11434` |

Invalid `--format`, `--category` and number values are rejected rather than silently replaced — e.g. `--format yaml`, `--category bogus`, or `--top 50k` print an error and exit 1. An unknown `--quant` is an error only for `context-fit`; `check` then shows every quantization and `compare` / `optimize` use the sweet spot, with a warning in table output. Commands that report an error exit 1 too — e.g. model not found, or Ollama not running for `benchmark` / `profile`.

### `llm-pulse check <model>`

"Can I run this model?" verdict with GPU layer-offload guidance when it doesn't fully fit.

```bash
llm-pulse check llama3.1:8b          # Check a specific model
llm-pulse check llama3.1:70b         # Overflow case — shows partial-offload tip
llm-pulse check qwen2.5-coder:14b --quant Q4_K_M
llm-pulse check llama3.1:70b --format json
```

Each quantization row shows the memory it needs at the context Ollama will use on your machine. The recommendation splits that into weights, KV cache and compute buffer. The pull line names the file `ollama pull <tag>` actually installs; when the recommended quantization is a different file, it points to the model's tag list on ollama.com. `--host` reads exact facts from that Ollama when the model is installed there.

When a model doesn't fit fully, the `GPU Layer Offload` section says how many transformer blocks to put on the GPU (Ollama `num_gpu` / llama.cpp `--n-gpu-layers`), with the rest on the CPU, e.g. "Put ≈44 of 80 layers on GPU (~22.0 GiB), rest on CPU". The count is an estimate (see below). It is hidden on Apple Silicon (unified memory) and CPU-only systems. The `/set parameter num_gpu` hint appears only when `ollama run <tag>` loads the quantization it was sized for.

### `llm-pulse compare [models...]`

Compare models side by side on your hardware: fit level, memory needed at Ollama's default context, and a rough speed class per model. Models you name are compared with their exact file facts (installed, or from Ollama's registry).

```bash
llm-pulse compare llama3.1:8b phi3:mini qwen2.5-coder:14b
llm-pulse compare --category coding --top 3    # Auto-pick top 3 coding models
llm-pulse compare llama3.1:8b phi3:mini --quant Q4_K_M
```

### `llm-pulse quant-advice <model>`

Which quantization should you actually pick? Shows a quality-vs-VRAM tradeoff table with the sweet-spot recommendation for your hardware — the largest quant that still fits comfortably.

```bash
llm-pulse quant-advice llama3.1:8b       # Sweet-spot pick + full tradeoff table
llm-pulse quant-advice llama3.1:70b      # "Nothing fits" → redirects to check for offload tips
llm-pulse quant-advice qwen2.5-coder:14b --format json
```

Each row gets a note: "Sweet spot — best quality you can fit", "Smaller — faster, slight quality drop", "Overkill — negligible quality gain over sweet spot", "Too big — overflows usable memory", etc. The recommendation follows the llama.cpp community heuristic: buy the most quality you can afford in memory, since gains at the high end are real but diminishing. Quality is llama.cpp's measured KL divergence (see [Quality per quantization](#how-llm-pulse-calculates-fit)); formats with no published measurement show `n/a`.

### `llm-pulse optimize <model>`

Recommends tuned Ollama runtime parameters (`num_ctx`, `num_gpu`, `num_thread`, `num_batch`) as a paste-ready Modelfile plus interactive `/set parameter` lines. It tunes the file `FROM <tag>` loads, the one `ollama pull <tag>` installs. When that file would not fit whole, it tunes the sweet-spot quantization instead, and `--quant` pins one. If the tuned quantization is not the tag's file, the Modelfile reads `FROM <Q8_0 tag>` (for example) and points to the model's tag list, so a pasted Modelfile cannot silently load a different file.

```bash
llm-pulse optimize llama3.1:8b                  # Balanced tuned profile + Modelfile
llm-pulse optimize llama3.1:8b --quant Q4_K_M   # Pin a specific quantization
llm-pulse optimize llama3.1:8b --format json
```

`num_thread` uses your physical performance cores (skipping efficiency cores and SMT); `num_ctx` is the largest standard size (2,048, 4,096, … 262,144) whose KV cache and compute buffer fit beside the weights, capped at the model's training context; `num_gpu` reuses the layer-offload math (omitted on Apple Silicon, where Ollama offloads all layers); `num_batch` shows the batch Ollama picks automatically for that context and stays out of the Modelfile, except on tight fits, where 256 is pinned to ease the prompt-eval memory spike. JSON output also carries `ollamaDefaultContext`, `kvQ8MaxContext` (the `num_ctx` ceiling with `OLLAMA_KV_CACHE_TYPE=q8_0`) and `memory` (the need at `num_ctx`).

### `llm-pulse context-fit <model>`

"Will this prompt fit in the context window?" — answers using the smaller of the model's native context window and the KV-cache ceiling your hardware can sustain. Returns a `yes` / `tight` / `no` verdict, which ceiling is binding (`model` vs `hardware`), and a remedy when it doesn't fit (smaller quant that does, or trim/offload suggestion).

```bash
llm-pulse context-fit llama3.1:8b --prompt-tokens 50000                          # will this prompt fit in context?
llm-pulse context-fit llama3.1:8b --prompt-tokens 50000 --response-tokens 1024   # reserve room to generate
llm-pulse context-fit llama3.1:8b --prompt-tokens 50000 --format json
```

The hardware ceiling uses the model's exact KV-cache size whenever its architecture is known (bundled, installed, or from Ollama's registry). JSON output carries `memory`, the need at the requested tokens.

### `llm-pulse report`

Paste-ready markdown summary of your machine — hardware, runtimes, health score, and the top recommended models — for GitHub issues, Reddit, or Discord.

```bash
llm-pulse report > my-rig.md                  # save to a file
llm-pulse report | pbcopy                     # copy to the clipboard (macOS)
llm-pulse report --category coding --top 3    # top 3 coding models
```

| Flag | Description | Default |
|---|---|---|
| `-c, --category` | `general`, `coding`, `reasoning`, `creative` (no model is tagged creative yet), `multilingual` | `all` |
| `-t, --top <n>` | Number of recommendations | `5` |
| `-H, --host <url>` | Ollama API host URL | `http://127.0.0.1:11434` |

Output is always GitHub-flavoured markdown on stdout — no colour, no spinner — so it redirects and pipes cleanly. Discord does not render markdown tables; wrap the paste in a ```` ``` ```` code fence there.

**Privacy:** the report shares your CPU model, cores and threads, GPU model, VRAM and usable memory, accelerator version (e.g. CUDA), RAM size/type/speed, disk type and free space, runtime versions and installed-model counts, the `doctor` health checks, and your OS/architecture. Installed models appear only as counts, plus a `yes` in the Installed column when a recommended model is installed. It never prints runtime install paths or the Ollama host URL.

### `llm-pulse doctor`

System health check — scores your setup and gives suggestions.

```bash
llm-pulse doctor
llm-pulse doctor --format json
llm-pulse doctor --fix --dry-run    # Preview the exact commands --fix would run
llm-pulse doctor --fix              # Auto-fix detected issues
```

`--dry-run` prints each planned fix with the exact command (e.g. `$ brew install ollama`) and changes nothing — review first, then run `--fix` to apply.

The `Context` row warns when a loaded model runs with less context than agents and coding tools need (64,000 tokens) while this machine could hold more. The `Model catalog` row says whether the live ollama.com catalog is in use.

### `llm-pulse models`

Browse the model database filtered for your hardware. Pulls in the live ollama.com/library catalog (cached 24 h) on top of the curated database. Sizes are each model's weights; `est.` marks an estimate from bits per weight rather than the real file size.

```bash
llm-pulse models                      # Curated + cached live catalog (the 48 curated when offline)
llm-pulse models --library            # Full Ollama library (240+ models)
llm-pulse models --refresh            # Force refresh library cache
llm-pulse models --search llama       # Search by name
llm-pulse models --category coding    # Filter by category
llm-pulse models --fits               # Only models that fit your VRAM
```

### `llm-pulse monitor`

Live TUI dashboard — like htop for LLMs. Press `Tab` to switch views, `q` to quit.

- **Overview** — CPU/GPU/RAM/VRAM bars with sparklines, the loaded model and its context length, plus smart alerts (VRAM pressure, GPU temperature, no active model)
- **GPU** — Per-GPU utilization, temperature, VRAM, and power sparklines with peak stats + temperature alerts
- **VRAM Map** — the loaded model's GPU memory as Ollama reports it (`size_vram`), other GPU use and free memory, plus curated models that still fit at 4,096 tokens
- **Models** — Browse installed Ollama models; pull new ones or delete (with a confirmation prompt), from inside the TUI

```bash
llm-pulse monitor
```

### `llm-pulse benchmark`

Quick inference benchmark via Ollama.

```bash
llm-pulse benchmark                  # Auto-picks smallest model
llm-pulse benchmark --model phi3     # Specific model
llm-pulse benchmark --rounds 5       # 5 rounds (default: 3)
```

### `llm-pulse profile`

Run inference with hardware profiling — latency breakdown (TTFT, generation), plus a VRAM and GPU-utilization timeline sampled during the run.

```bash
llm-pulse profile                          # Short/medium/long prompt set
llm-pulse profile --model phi3             # Specific model
llm-pulse profile --prompt "Explain DNS"   # Custom prompt
llm-pulse profile --context-size 4096
```

## Programmatic API

```typescript
import { detectHardware, getRecommendations } from "llm-pulse";

const hardware = await detectHardware();
const recs = getRecommendations(hardware, { category: "coding", top: 3 });

console.log(recs[0].score.model.name);  // "Qwen 2.5 Coder 14B"
console.log(recs[0].score.fitLevel);     // "comfortable"
console.log(recs[0].pullCommand);        // "ollama pull qwen2.5-coder:14b"
```

## MCP Server

Use llm-pulse as an MCP tool from Claude Code, Cursor, or any MCP-compatible AI assistant. The assistant can scan your hardware, check model compatibility, and snapshot live GPU/VRAM state — all without leaving the chat.

Add to your Claude Code config (`~/.claude.json` or your project's `.mcp.json`):

```json
{
  "mcpServers": {
    "llm-pulse": {
      "command": "npx",
      "args": ["-y", "-p", "llm-pulse", "llm-pulse-mcp"]
    }
  }
}
```

(`llm-pulse-mcp` is a binary inside the `llm-pulse` package, so npx needs `-p llm-pulse`. If you've installed globally with `npm install -g llm-pulse`, you can use `"command": "llm-pulse-mcp"` with no args instead.)

Exposed tools:

| Tool | What it does |
|---|---|
| `scan` | Full hardware scan + ranked model recommendations |
| `check` | "Can I run this model?" verdict (yes/maybe/no): the same JSON as `check --format json` (an unknown quantization is an error here), with every quantization's memory breakdown |
| `context-fit-check` | "Will a prompt of N tokens fit?" — verdict (yes/tight/no), which ceiling is binding (model vs hardware), and a remedy |
| `recommend` | Ranked model list for your hardware, filterable by category |
| `doctor` | System health score with actionable suggestions |
| `models` | Browse / search the model database, optionally filtered to models that fit; `limit` (default 100, max 500) caps the rows and `total` says how many matched |
| `monitor` | One-shot live snapshot — CPU/GPU%, VRAM, temp, power, active Ollama model + context length (`tokensPerSec` is always `null`) |

## How llm-pulse calculates fit

Every memory figure llm-pulse shows is one of three kinds:

- **Measured** on your machine, or read from the model file.
- **Derived** with a formula taken from the runtime's own source code.
- **Estimated**, and marked `est.` (or `≈`), with the method given below.

Sources are pinned to these commits:

- llama.cpp [`a300a9c`](https://github.com/ggml-org/llama.cpp/tree/a300a9c33462f23d9d61e951650d249486f2b0f6)
- Ollama [`b8db8d5`](https://github.com/ollama/ollama/tree/b8db8d5b5c8daedbbef7bb9ace2221a38bc384c2)

Ollama serves every GGUF model through llama.cpp's `llama-server` (`llm/server.go`), so llm-pulse uses llama.cpp's memory accounting. Sizes are binary: 1 GiB = 1024 MiB.

**Memory budget.** This is the memory available for the model, its KV cache and its compute buffer. llm-pulse calls it "usable" memory. llama.cpp fits a model into each device's free memory minus a 1 GiB margin (`common/fit.cpp`, `common/common.h`).

| Platform | Budget | Basis |
|---|---|---|
| NVIDIA / AMD / Intel discrete | VRAM − 1 GiB − 457 MiB, summed over the GPUs of one runtime | The 457 MiB is llm-pulse's estimate of what the driver and runtime context hold on an idle GPU. Ollama loads a model onto the GPUs of one runtime (CUDA, ROCm or Vulkan; `server/sched.go`), so llm-pulse uses the runtime with the most memory. |
| Apple Silicon | Metal working set − 1 GiB | llama.cpp's Metal backend reports `recommendedMaxWorkingSetSize`, less what it has already allocated, as free (`ggml-metal-device.m`); `iogpu.wired_limit_mb` overrides the working set |
| NVIDIA DGX Spark / GB10 | available RAM − 1 GiB − 457 MiB | llama.cpp reads `MemAvailable` on unified-memory CUDA devices (`ggml-cuda.cu`); 457 MiB as above |
| AMD Strix Halo (gfx1151) | min(GTT pool, available RAM) − 1 GiB − 457 MiB | amdgpu `mem_info_gtt_total`; APU detection follows Ollama `discover/amd.go` |
| CPU only | available RAM | llm-pulse's own rule: a model larger than available RAM would swap |

Ollama uses an integrated GPU only if it is an NVIDIA (GB10, Jetson), AMD gfx1151 (Strix Halo) or Apple GPU, unless `OLLAMA_IGPU_ENABLE=1` is set (`discover/runner.go`). llm-pulse follows the same rule, reading `OLLAMA_IGPU_ENABLE` from its own environment; any other integrated GPU counts as CPU only. On Linux the Intel driver does not report VRAM size, so llm-pulse counts Intel GPUs there as CPU only, although Ollama can run a discrete Arc card through Vulkan.

The usable figure is shown as `est.` where it includes the 457 MiB estimate. It assumes an idle GPU: VRAM that the desktop or other apps already use lowers it.

NPUs and neural engines are never counted. This covers the Apple Neural Engine, Intel NPU, AMD XDNA and Qualcomm Hexagon. Ollama runs models on GPUs and CPUs only, and llama.cpp's NPU backends are experimental.

On Windows, NVIDIA drivers 536.40 and later can spill VRAM into system RAM. A model that does not fit may therefore still load, but much slower.

**Memory a model needs** at context length *c*:

- **Weights.** The real file size when the model is installed or its file is known from Ollama's registry, including the vision projector and any draft model Ollama loads with it. Shown without a marker.
  - Otherwise `parameters × bits per weight / 8`, shown as `est.`:
    - llama.cpp's measured whole-file bits per weight (`tools/quantize/README.md`), e.g. 4.894 for Q4_K_M and 8.501 for Q8_0;
    - types it has not measured (Q4_0, Q5_0, MXFP4 and others): their nominal bits plus 0.4;
    - a live-catalog model whose file type is not known yet (its quant is shown as `default`): 5 bits, within the 4.90–5.28 that Ollama's default builds measure;
    - natively ternary models: 2 bits.
- **KV cache.** llama.cpp's formula (`src/llama-kv-cache.cpp`): for each attention layer, `context × KV heads × (key + value head size) × bytes per element`. The context is padded to a multiple of 256.
  - Sliding-window layers (Gemma 2/3, gpt-oss, Command R7B and others) cache only the window plus one batch (`src/llama-kv-cache-iswa.cpp`).
  - Latent-attention models (DeepSeek-V2/V3 style) have no V cache.
  - Recurrent layers (Qwen3-Next, Nemotron-H and others) hold a fixed state instead.
  - KV cache types: f16 takes 2 bytes per element, q8_0 34/32 and q4_0 18/32.
  - The architecture comes from the model file's GGUF header: bundled for the curated models (all except Llama 3.2 Vision, whose `mllama` architecture llama.cpp does not run), read from your local Ollama (`/api/show`) for installed GGUF models, or from the first 256 KiB of the file in Ollama's registry.
  - Shown as `est.` when that architecture is unknown (the KV cache is then estimated from the parameter count) and for setups llm-pulse does not model exactly: models shipped with a draft model, which Ollama loads by default and runs in a second context (detected from the registry; for an installed model llm-pulse cannot tell); parallel hybrids such as Falcon-H1; and MLX builds, which Ollama installs on Macs for some newer models.
  - Multi-token-prediction layers run only when a model's parameters set `draft_num_predict` (`server/routes.go`); llm-pulse assumes they are off.
- **Compute buffer (≈).** Two parts:
  - `vocabulary × batch × 4 bytes`, plus 10%: a conservative stand-in for the graph's working memory. It is not the logits buffer: `llama-server` reserves output rows for one token per sequence (`tools/server/server-context.cpp`). When the vocabulary size is unknown, 262,144 tokens are assumed.
  - The attention mask, exactly: `context × batch × 2 bytes`, plus a window-sized mask for sliding-window models (`src/llama-graph.cpp`). Each GPU holds a copy; when several GPUs hold the whole model, each keeps four (pipeline parallelism, `ggml/src/ggml-backend.cpp`).
  - The batch is the largest one Ollama picks automatically for the context (512 up to 4,096 tokens, 1,024 up to 32,768, then 2,048; `server/sched.go`), or a `num_batch` you pin. Ollama steps its automatic batch down when memory is short, so there the figure is an upper bound. With flash attention off (older GPUs, or `OLLAMA_FLASH_ATTENTION=0`) the buffer is larger than shown.

**Context Ollama will use.** Ollama picks a default context from the total VRAM of all the GPUs it uses (`server/routes.go`), capped at the model's training context (`llm/server.go`):

- at least 47 GiB: 262,144 tokens;
- at least 23 GiB: 32,768 tokens;
- otherwise: 4,096 tokens.

This holds while `OLLAMA_CONTEXT_LENGTH` and `num_ctx` are unset, `OLLAMA_NUM_PARALLEL` is 1 (each parallel slot gets its own context) and the KV cache is f16. If a model does not fit at that context, layers move to the CPU; if loading still fails for lack of memory, Ollama retries once with the next smaller automatic context (`server/sched.go`).

Context windows of live-catalog models are guesses (8,192 or 32,768 tokens, shown as `est.`) until exact facts give the training context.

**Fit levels:**

| Level | Meaning |
|---|---|
| excellent | Fits at Ollama's default context within 80% of the budget. The 80% line is llm-pulse's (Ollama uses 80% of a GPU's free memory when it chooses between GPUs, `server/sched.go`). |
| comfortable | Fits fully on the GPU at Ollama's default context. |
| tight | Fits fully on the GPU only with a smaller context. `optimize` and `context-fit` show the largest `num_ctx` that keeps it on the GPU. |
| barely | Does not fit even at 4,096 tokens, but at least 75% of it does. Part of the model runs on the CPU, so it is slow. GPU machines only; the 75% line is llm-pulse's own definition. |
| cannot run | Less than that. |

The number of layers `check` and `optimize` suggest putting on the GPU is an estimate (`≈`): it splits the file evenly across layers, while llama.cpp keeps the input embeddings on the CPU and places the output layer first.

**Quality per quantization.** llama.cpp's measured KL divergence against FP16 for Llama-3 8B (`tools/perplexity/README.md`), shown as `exp(−KLD)`:

| Quant | Score |
|---|---|
| Q8_0 | 0.999 |
| Q6_K | 0.995 |
| Q5_K_M | 0.989 |
| Q4_K_M | 0.969 |
| Q4_0 | 0.931 |
| Q3_K_M | 0.903 |
| Q2_K | 0.641 |

Formats with no published measurement, such as MXFP4, show `n/a` instead of a score. In JSON they carry `qualityMeasured: false`, and their `qualityRetention` is Q4_K_M's value, used only for ranking.

**Recommendations (estimate).** `scan`, `report`, `compare --category` and the MCP `scan` and `recommend` tools score the file that `ollama pull <tag>` installs:

- Models that fit whole at Ollama's default context come first. A model that fits whole only at a smaller context ranks with them when 32,768 tokens still fit.
- Partial fits follow, and only on a machine with a discrete GPU to share the load. Models that cannot run are left out (the MCP `recommend` tool with `onlyFitting: false`, and `getRecommendations` by default, list them last).
- One model per family is shown.
- Embedding models (they return vectors, not text) and natively ternary models (no quality measurement for ternary weights) are left out.

When the live catalog is in use, every model, curated ones included, gets a quality score from four things (offline, curated models keep their hand-set scores):

- Size: a quadratic in log₂ of effective parameters, fitted to the hand-scored curated models (R² 0.965). Mixture-of-experts models count as total^0.75 × active^0.25.
- Release recency, measured against the newest model in Ollama's library.
- Pull velocity.
- A bonus for coding models.

It is a ranking heuristic, not a benchmark.

**Speed class (rough).** Decoding is limited by memory bandwidth, so the speed class comes from the bytes read per token (only the active parameters for mixture-of-experts models) and from where the model runs. Run `llm-pulse benchmark` for measured speed.

## Network and privacy

Apart from `doctor --fix`, llm-pulse makes no network request beyond these:

- **Your local runtimes:** Ollama (`--host`, `ollamaHost` in `.llmpulserc`, or `OLLAMA_HOST`; default `http://127.0.0.1:11434`) for runtime status, installed models and loaded-model context, and LM Studio's local API (`/v1/models`, default `http://127.0.0.1:1234`) when it is installed.
- **ollama.com/library**: the public model catalog. Commands that list or look up models (`scan`, `models`, `check`, `compare`, `quant-advice`, `optimize`, `context-fit`, `report`) fetch it when the cached copy is more than a day old, with a 5-second limit. `models --library` fetches it when the cached copy is more than a day old and `--refresh` always, with a 10-second limit. The MCP server checks every hour. `doctor`, `monitor`, `benchmark` and `profile` never fetch it.
- **registry.ollama.ai**: the public model registry. Single-model commands (`check`, `compare`, `quant-advice`, `optimize`, `context-fit`, and the MCP `check` and `context-fit-check` tools) use it for models that are neither bundled nor installed. llm-pulse reads the model's manifest, its small config file and the first 256 KiB of the model file (the GGUF header), the same files `ollama pull` downloads. The registry serves these files from Cloudflare R2 (`*.r2.cloudflarestorage.com`). Results are cached in `~/.llmpulse/cache`.

They send no hardware details. Registry requests name the model being looked up.

`doctor --fix` runs only when you ask: it executes the install and pull commands it lists (e.g. `brew install ollama`, `ollama pull llama3.2:3b`), which download from the internet. Preview them with `--dry-run`.

To turn off both ollama.com and the registry, set `LLMPULSE_OFFLINE=1` or add `"offline": true` to `.llmpulserc`. llm-pulse then uses only its bundled model list and your local runtimes; `models --library` says the live catalog is off instead of fetching it.

## Supported

**Hardware:** NVIDIA GPUs (CUDA), NVIDIA DGX Spark / GB10 (unified memory), AMD GPUs (ROCm) and the Strix Halo APU (GTT pool), Intel Arc, Apple Silicon (Metal), any CPU (AVX2/NEON), DDR4/DDR5, NVMe/SSD/HDD. Other integrated GPUs count as CPU only, as Ollama runs them; so do Intel GPUs on Linux, whose VRAM size llm-pulse cannot read; NPUs and neural engines are not counted (see [How llm-pulse calculates fit](#how-llm-pulse-calculates-fit)).

**Runtimes:** [Ollama](https://ollama.com), [llama.cpp](https://github.com/ggerganov/llama.cpp), [LM Studio](https://lmstudio.ai)

**Models:** 48 curated + recent models from the live Ollama library (cached 24 h; `models --library` lists the whole catalog, 240+) — across general, coding, reasoning, multilingual. Curated models carry 117 quantization variants, each one Ollama publishes a tag for. For every curated model, the file its default tag installs (its real size and type) was read from Ollama's registry, and so was its architecture, except for Llama 3.2 Vision (`mllama`, which llama.cpp does not run).

## Stability

llm-pulse follows [semantic versioning](https://semver.org). As of **1.0.0**, the CLI commands and flags, the `table`/`json`/`csv` output shapes, the programmatic API (`detectHardware`, `getRecommendations`), and the 7 MCP tools are considered stable — any breaking change to them bumps the major version.

The markdown printed by `llm-pulse report` follows the same rule as `table` output: minor releases may only add to it. Scripts should parse `llm-pulse scan --format json` or `llm-pulse doctor --format json` instead.

## License

MIT
