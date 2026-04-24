# llm-pulse

[![npm version](https://img.shields.io/npm/v/llm-pulse.svg)](https://www.npmjs.com/package/llm-pulse)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Node.js](https://img.shields.io/badge/node-%3E%3D18-brightgreen.svg)](https://nodejs.org/)

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

Requires Node.js 18+.

## Commands

### `llm-pulse` / `llm-pulse scan`

Hardware scan + model recommendations.

```bash
llm-pulse                            # Full scan (default)
llm-pulse --format json              # JSON output
llm-pulse --category coding --top 3  # Top 3 coding models
```

| Flag | Description | Default |
|---|---|---|
| `-f, --format` | `table` or `json` | `table` |
| `-c, --category` | `general`, `coding`, `reasoning`, `creative`, `multilingual` | `all` |
| `-t, --top <n>` | Number of recommendations | `5` |
| `-v, --verbose` | Detailed output | `false` |

### `llm-pulse check <model>`

"Can I run this model?" verdict with GPU layer-offload guidance when it doesn't fully fit.

```bash
llm-pulse check llama3.1:8b          # Check a specific model
llm-pulse check llama3.1:70b         # Overflow case — shows partial-offload tip
llm-pulse check qwen2.5-coder:14b --quant Q4_K_M
llm-pulse check llama3.1:70b --format json
```

When a model overflows your VRAM, the `GPU Layer Offload` section tells you how many transformer blocks to put on the GPU (maps to Ollama `num_gpu` / llama.cpp `--n-gpu-layers`) with the rest on CPU — e.g. "Put 44 of 80 layers on GPU (~22 GB), rest on CPU". Hidden on Apple Silicon (unified memory) and CPU-only systems.

### `llm-pulse doctor`

System health check — scores your setup and gives suggestions.

```bash
llm-pulse doctor
llm-pulse doctor --format json
llm-pulse doctor --fix              # Auto-fix detected issues
```

### `llm-pulse models`

Browse the model database filtered for your hardware. Pulls in the live ollama.com/library catalog (cached 24 h) on top of the curated database.

```bash
llm-pulse models                      # Curated set (48 models)
llm-pulse models --library            # Full Ollama library (245+ models)
llm-pulse models --refresh            # Force refresh library cache
llm-pulse models --search llama       # Search by name
llm-pulse models --category coding    # Filter by category
llm-pulse models --fits               # Only models that fit your VRAM
```

### `llm-pulse monitor`

Live TUI dashboard — like htop for LLMs. Press `Tab` to switch views, `q` to quit.

- **Overview** — CPU/GPU/RAM/VRAM bars with sparklines + smart alerts
- **Inference** — Throughput chart + session stats
- **GPU** — Per-GPU utilization, temperature, VRAM, and power sparklines with peak stats + temperature alerts
- **VRAM Map** — Visual VRAM breakdown (model weights / KV cache / overhead / free)

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
      "args": ["-y", "llm-pulse-mcp"]
    }
  }
}
```

Exposed tools:

| Tool | What it does |
|---|---|
| `scan` | Full hardware scan + ranked model recommendations |
| `check` | "Can I run this model?" verdict (yes/maybe/no) with best quantization + speed estimate |
| `recommend` | Ranked model list for your hardware, filterable by category |
| `doctor` | System health score with actionable suggestions |
| `models` | Browse / search the model database, optionally filtered to models that fit |
| `monitor` | One-shot live snapshot — CPU/GPU%, VRAM, temp, power, active Ollama model + tok/s |

## Supported

**Hardware:** NVIDIA GPU (full CUDA/VRAM), AMD, Intel, Apple Silicon, any CPU (AVX2/NEON), DDR4/DDR5, NVMe/SSD/HDD

**Runtimes:** [Ollama](https://ollama.com), [llama.cpp](https://github.com/ggerganov/llama.cpp), [LM Studio](https://lmstudio.ai)

**Models:** 48 curated + 245+ via live Ollama library catalog (cached 24 h) — across general, coding, reasoning, creative, multilingual — each with Q4/Q5/Q8/F16 quantization variants

## License

MIT
