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

### `llm-pulse doctor`

System health check — scores your setup and gives suggestions.

```bash
llm-pulse doctor
llm-pulse doctor --format json
```

### `llm-pulse models`

Browse the model database filtered for your hardware.

```bash
llm-pulse models                      # All 45+ models
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

## Supported

**Hardware:** NVIDIA GPU (full CUDA/VRAM), AMD, Intel, Apple Silicon, any CPU (AVX2/NEON), DDR4/DDR5, NVMe/SSD/HDD

**Runtimes:** [Ollama](https://ollama.com), [llama.cpp](https://github.com/ggerganov/llama.cpp), [LM Studio](https://lmstudio.ai)

**Models:** 45+ models across general, coding, reasoning, creative, multilingual — each with Q4/Q5/Q8/F16 quantization variants

## License

MIT
