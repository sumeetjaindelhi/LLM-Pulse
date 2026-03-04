# llm-pulse

Zero-config CLI that tells you what LLMs your PC can run. Hardware detection, runtime discovery, and model recommendations — all in your terminal.

```
npx llm-pulse
```

No Docker. No browser. No config files.

## Why llm-pulse?

You want to run LLMs locally but face the same questions every time:
- **Can my PC handle it?** How much VRAM do I actually have?
- **Which model?** There are hundreds — which ones fit my hardware?
- **Is everything set up right?** Drivers, runtimes, disk space?

llm-pulse answers all three in seconds.

| Feature | llm-pulse | gpu-hot | OpenLLM Monitor | Prometheus+Grafana |
|---|---|---|---|---|
| Zero config | Yes | No (Docker) | No (MongoDB) | No (YAML configs) |
| Terminal-only | Yes | No (browser) | No (browser) | No (browser) |
| Cross-platform | Yes | NVIDIA only | Linux only | Yes |
| Model recommendations | Yes | No | No | No |
| Runtime detection | Yes | No | Partial | No |
| Install complexity | `npx` | Docker + pip | npm + MongoDB | 3+ services |

## Quick Start

```bash
# Run directly (no install needed)
npx llm-pulse

# Or install globally
npm install -g llm-pulse
llm-pulse
```

**Requirements:** Node.js 18+

## Commands

### `llm-pulse` (or `llm-pulse scan`)

Full hardware scan with model recommendations.

```
$ llm-pulse

╭ LLM Pulse v0.1.0 ──────────────────────────────────────╮
│                                                         │
│  Hardware                                               │
│                                                         │
│  CPU   AMD Ryzen 7 5800X                                │
│        16 threads · AVX2 ✓ · 3.8 GHz                   │
│  GPU   NVIDIA GeForce RTX 3080                          │
│        10.0 GB VRAM · CUDA 12.4                         │
│        ████████████████░░░░ 78% utilized                │
│  RAM   32.0 GB DDR4 @ 3200 MHz                          │
│        ████████░░░░░░░░░░░░ 42% used                    │
│  Disk  NVMe · 280 GB free                               │
│                                                         │
│  Runtimes Detected                                      │
│                                                         │
│  ✓ Ollama v0.5.1 (running)                              │
│    Models: llama3.1:8b, deepseek-r1:7b                  │
│  ✗ llama.cpp (not found)                                │
│  ✗ LM Studio (not found)                                │
│                                                         │
│  Recommended Models for Your Hardware                   │
│                                                         │
│  #  Model              Quant    Fit     VRAM   Speed    │
│  1  Llama 3.1 8B       Q4_K_M   ★★★★★  5 GB   fast     │
│  2  DeepSeek R1 7B     Q5_K_M   ★★★★☆  5 GB   fast     │
│  3  Qwen 2.5 14B       Q4_K_M   ★★★☆☆  8 GB   moderate │
│  4  Mistral 7B         Q5_K_M   ★★★★☆  5 GB   fast     │
│  5  Phi-3 Mini 3.8B    Q8_0     ★★★★★  4 GB   fast     │
│                                                         │
│  Run: ollama pull llama3.1:8b                           │
│                                                         │
│  Tip: You can comfortably run up to 14B parameter       │
│  models with Q4 quantization.                           │
│                                                         │
╰─────────────────────────────────────────────────────────╯
```

**Options:**

| Flag | Description | Default |
|---|---|---|
| `-f, --format <format>` | Output format: `table` or `json` | `table` |
| `-c, --category <cat>` | Filter: `general`, `coding`, `reasoning`, `creative`, `multilingual` | `all` |
| `-t, --top <n>` | Number of recommendations | `5` |
| `-v, --verbose` | Show detailed output | `false` |

```bash
llm-pulse --format json              # Machine-readable output
llm-pulse --category coding --top 3  # Top 3 coding models
```

### `llm-pulse doctor`

System health check with actionable advice.

```
$ llm-pulse doctor

  ✓ CPU supports AVX2 — great for inference
  ✓ GPU has 10 GB VRAM — can run 7B-14B models
  ✓ 32 GB RAM — good headroom for CPU offloading
  ✓ NVMe SSD — fast model loading
  ✓ Ollama installed and running
  ⚠ GPU driver 535.x — update to 550+ recommended

  Score: 85/100 — Great for local LLMs
  Suggestion: Update NVIDIA driver for better performance.
```

### `llm-pulse models`

Browse the full model database, filtered for your hardware.

```bash
llm-pulse models                      # List all 30+ models
llm-pulse models --search llama       # Search by name
llm-pulse models --category coding    # Filter by category
llm-pulse models --fits               # Only show models your hardware can run
llm-pulse models --fits --format json # JSON output
```

### `llm-pulse monitor`

Live-updating TUI dashboard with 3 tabs. Like htop, but for LLMs.

Press `Tab` to switch views, `q` to quit. Updates every second.

**Tab 1 — Overview** (default): Hardware bars with sparkline trends, active model info, and smart alerts.

```
  LLM Pulse — Live Monitor          [Overview] Inference  VRAM
  ─────────────────────────────────────────────────────────────
  CPU  ████████████░░░░░░░░  62%  45°C     ▁▂▃▅▆▅▃▂▃▅
  GPU  ██████████████░░░░░░  71%  68°C     ▃▅▆▇▇▆▅▆▇█
  RAM  ████████░░░░░░░░░░░░  42%  13.8/32 GB
  VRAM ██████████████░░░░░░  72%  7.2/10 GB

  Model: llama3.1:8b (Q4_K_M · 5 GB)     Status: generating
  Speed: 42.3 tok/s          Uptime: 12m 34s

  ⚠ VRAM at 72% — loading a 2nd model will cause swapping
  ✓ Running 30% faster than average for RTX 3080 + 8B Q4
  ─────────────────────────────────────────────────────────────
  [Tab] switch view  [q] quit
```

**Tab 2 — Inference**: Throughput chart over time, session stats, and per-model usage breakdown.

```
  Throughput (last 60s)
   50 ┤                          ╭─╮
   40 ┤      ╭──╮    ╭──╮   ╭──╯  ╰──╮
   30 ┤  ╭──╯   ╰──╮╯   ╰─╯          ╰──
   20 ┤──╯
    0 ┤─────────────────────────────────── tok/s

  Session Stats
  Total tokens:     12,847     Avg tok/s: 38.2
  Requests:         47         Uptime:    12m 34s

  Model History (this session)
  llama3.1:8b      38.2 tok/s  11m 20s  [████████████████░░░░]
  phi3:mini        52.1 tok/s   1m 14s  [███░░░░░░░░░░░░░░░░░]
```

**Tab 3 — VRAM Map**: Visual VRAM breakdown showing what's using your GPU memory.

```
  VRAM Map — 10,240 MB Total
  ████████████████████████████░░░░░░░░░░░░░░░░  72% used

  Model weights   [████████████████]  4,800 MB  47%
  KV Cache        [██████████]        2,100 MB  21%
  CUDA overhead   [██]                  400 MB   4%
  Free            [░░░░░░░░░░░░]      2,940 MB  29%

  Power: 180W   Clock: 1920 MHz
  Can still fit: Phi-3 Mini Q4 (2.5 GB)
```

**Smart alerts** provide LLM-specific insights no system monitor shows:
- VRAM pressure warnings before swapping happens
- Speed drop detection (thermal throttling, context overflow)
- GPU underutilization during inference (CPU-bound model)
- Performance comparison vs expected for your hardware

### `llm-pulse benchmark`

Quick inference benchmark via Ollama.

```bash
llm-pulse benchmark                  # Auto-picks smallest installed model
llm-pulse benchmark --model phi3     # Benchmark a specific model
llm-pulse benchmark --rounds 5       # Run 5 rounds (default: 3)
```

Measures tokens/sec, time-to-first-token, and gives a performance rating. Requires Ollama to be running.

## Supported Hardware

- **GPU:** NVIDIA (full support with CUDA/VRAM/utilization via nvidia-smi), AMD, Intel, Apple Silicon
- **CPU:** Intel, AMD, Apple Silicon — detects cores, threads, AVX2 support
- **RAM:** DDR4/DDR5/LPDDR5 — total, available, speed
- **Disk:** NVMe/SSD/HDD detection, free space check

## Supported Runtimes

| Runtime | Detection | Details |
|---|---|---|
| [Ollama](https://ollama.com) | Binary + API | Version, running status, installed models |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | Binary | Detects llama-server, llama-cli |
| [LM Studio](https://lmstudio.ai) | Install path + API | Checks default paths and local server |

## Model Database

30+ models across 5 categories:

| Category | Models |
|---|---|
| General | Llama 3.1/3.2/3.3, Qwen 2.5, Gemma 2, Mistral, Phi-3/4, Yi, Command R |
| Coding | Qwen 2.5 Coder, Code Llama, DeepSeek Coder V2, StarCoder2 |
| Reasoning | DeepSeek R1, Phi-4, Phi-3 |
| Multilingual | Qwen 2.5, Yi, Command R |

Each model includes multiple quantization variants (Q4_K_M, Q5_K_M, Q8_0, F16) with accurate VRAM requirements.

## How Scoring Works

Models are ranked by a composite score combining:

1. **Fit** — Does it fit in your VRAM? (★★★★★ = 50%+ headroom, down to ✗ = can't run)
2. **Quality** — Benchmark-derived quality score per model
3. **Quantization** — Higher-bit quantizations retain more quality
4. **Speed** — Estimated inference speed based on model size and your hardware

Category-specific weights: `--category coding` prioritizes quality over fit, while `general` balances all factors.

## Programmatic API

```typescript
import { detectHardware, getRecommendations, getAllModels } from "llm-pulse";

const hardware = await detectHardware();
const recs = getRecommendations(hardware, { category: "coding", top: 3 });

console.log(recs[0].score.model.name);     // "Qwen 2.5 Coder 14B"
console.log(recs[0].score.fitLevel);        // "comfortable"
console.log(recs[0].pullCommand);           // "ollama pull qwen2.5-coder:14b"
```

## Development

```bash
git clone https://github.com/sumeetjaindelhi/LLM-Pulse.git
cd llm-pulse
npm install
npm run build
npm test

# Run from source
node dist/bin/llm-pulse.js
node dist/bin/llm-pulse.js doctor
node dist/bin/llm-pulse.js models --search deepseek
```

## License

MIT
