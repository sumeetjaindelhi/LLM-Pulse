import React, { useState, useEffect, useCallback, useRef } from "react";
import { Text, Box, useInput } from "ink";
import type { MonitorSnapshot } from "../hardware/monitor.js";
import { parsePullLines, type PullStreamEvent } from "./pull-stream.js";

interface ModelInfo {
  name: string;
  size: string;
  family: string;
  quantization: string;
  parameterSize: string;
}

interface PullProgress {
  model: string;
  status: string;
  percent: number;
  downloading: boolean;
}

// Suggested models to pull if not installed
const SUGGESTED_MODELS = [
  { tag: "llama3.2:3b", desc: "Llama 3.2 3B — fast & capable, ~2 GB" },
  { tag: "llama3.1:8b", desc: "Llama 3.1 8B — great all-rounder, ~5 GB" },
  { tag: "mistral:7b", desc: "Mistral 7B — strong reasoning, ~4 GB" },
  { tag: "codellama:7b", desc: "Code Llama 7B — code generation, ~4 GB" },
  { tag: "phi3:mini", desc: "Phi-3 Mini — small but smart, ~2 GB" },
  { tag: "gemma2:2b", desc: "Gemma 2 2B — Google's compact model, ~2 GB" },
  { tag: "qwen2.5:7b", desc: "Qwen 2.5 7B — multilingual, ~4 GB" },
  { tag: "deepseek-coder-v2:16b", desc: "DeepSeek Coder V2 — coding beast, ~9 GB" },
];

interface ModelManagerProps {
  snapshot: MonitorSnapshot;
  ollamaHost: string;
}

export const ModelManager = React.memo(function ModelManager({ snapshot, ollamaHost }: ModelManagerProps) {
  const [installed, setInstalled] = useState<ModelInfo[]>([]);
  const [selectedIdx, setSelectedIdx] = useState(0);
  const [mode, setMode] = useState<"installed" | "pull">("installed");
  const [pullProgress, setPullProgress] = useState<PullProgress | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState(0);
  // Aborts an in-flight pull when the tab unmounts (Tab switch or quit), so the
  // streaming socket does not keep the process alive after the TUI exits.
  const unmountAbortRef = useRef(new AbortController());

  useEffect(() => () => unmountAbortRef.current.abort(), []);

  // Fetch installed models
  const fetchModels = useCallback(async () => {
    try {
      const res = await fetch(`${ollamaHost}/api/tags`, {
        redirect: "error",
        signal: AbortSignal.timeout(3000),
      });
      if (!res.ok) return;
      const data = await res.json();
      const models: ModelInfo[] = (data.models ?? []).map((m: Record<string, unknown>) => {
        const details = (m.details ?? {}) as Record<string, unknown>;
        const sizeBytes = typeof m.size === "number" ? m.size : 0;
        return {
          name: String(m.name ?? ""),
          size: `${(sizeBytes / (1024 * 1024 * 1024)).toFixed(1)} GB`,
          family: String(details.family ?? "—"),
          quantization: String(details.quantization_level ?? "—"),
          parameterSize: String(details.parameter_size ?? "—"),
        };
      });
      setInstalled(models);
      setLastRefresh(Date.now());
    } catch {
      // Ollama not running
    }
  }, [ollamaHost]);

  useEffect(() => {
    fetchModels();
    const interval = setInterval(fetchModels, 10000);
    return () => clearInterval(interval);
  }, [fetchModels]);

  // Available suggested models not yet installed
  const installedNames = new Set(installed.map((m) => m.name));
  const suggestions = SUGGESTED_MODELS.filter((s) => !installedNames.has(s.tag));

  const items = mode === "installed" ? installed : suggestions;
  const maxIdx = items.length - 1;

  // Keep the selection inside the list when it shrinks (delete or refresh).
  useEffect(() => {
    setSelectedIdx((i) => Math.min(i, Math.max(0, maxIdx)));
  }, [maxIdx]);

  useInput((input, key) => {
    if (pullProgress?.downloading) return; // block input during download

    if (pendingDelete !== null) {
      if (input === "y") deleteModel(pendingDelete);
      setPendingDelete(null);
      return;
    }

    if (key.upArrow) {
      setSelectedIdx((i) => Math.max(0, i - 1));
    }
    if (key.downArrow) {
      setSelectedIdx((i) => Math.min(maxIdx, i + 1));
    }
    if (input === "p") {
      setMode("pull");
      setSelectedIdx(0);
      setErrorMessage(null);
    }
    if (input === "i") {
      setMode("installed");
      setSelectedIdx(0);
      setErrorMessage(null);
    }
    if (input === "r") {
      fetchModels();
    }
    if (key.return && mode === "pull" && suggestions.length > 0) {
      const selected = suggestions[selectedIdx];
      if (selected) {
        pullModel(selected.tag);
      }
    }
    if (input === "d" && mode === "installed" && installed.length > 0) {
      const selected = installed[selectedIdx];
      if (selected) {
        setPendingDelete(selected.name);
      }
    }
  });

  async function pullModel(tag: string) {
    setPullProgress({ model: tag, status: "Starting download...", percent: 0, downloading: true });
    setErrorMessage(null);

    try {
      const res = await fetch(`${ollamaHost}/api/pull`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: tag, stream: true }),
        redirect: "error",
        signal: AbortSignal.any([unmountAbortRef.current.signal, AbortSignal.timeout(600000)]), // 10 min timeout
      });

      if (!res.ok || !res.body) {
        setPullProgress(null);
        setErrorMessage(`Failed to pull ${tag}: HTTP ${res.status}`);
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let succeeded = false;

      // Ollama reports pull failures as an {"error": ...} line on a 200 response.
      const applyEvents = (events: PullStreamEvent[]) => {
        for (const event of events) {
          if (event.error) throw new Error(event.error);
          if (event.status === "success") succeeded = true;
          const percent = event.total && event.completed ? Math.round((event.completed / event.total) * 100) : 0;
          setPullProgress({ model: tag, status: event.status ?? "", percent, downloading: true });
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const parsed = parsePullLines(buffer + decoder.decode(value, { stream: true }));
        buffer = parsed.rest;
        applyEvents(parsed.events);
      }
      applyEvents(parsePullLines(`${buffer}${decoder.decode()}\n`).events);

      if (!succeeded) throw new Error("stream ended without success");

      setPullProgress({ model: tag, status: "Download complete!", percent: 100, downloading: false });
      fetchModels();
    } catch (err) {
      setPullProgress(null);
      setErrorMessage(`Failed to pull ${tag}: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  async function deleteModel(name: string) {
    try {
      const res = await fetch(`${ollamaHost}/api/delete`, {
        method: "DELETE",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
        redirect: "error",
        signal: AbortSignal.timeout(10000),
      });
      if (!res.ok) {
        setErrorMessage(`Failed to delete ${name}: HTTP ${res.status}`);
        return;
      }
      await fetchModels();
    } catch (err) {
      setErrorMessage(`Failed to delete ${name}: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  function renderProgressBar(percent: number, width: number = 30): string {
    const filled = Math.round((percent / 100) * width);
    const empty = width - filled;
    return "\u2588".repeat(filled) + "\u2591".repeat(empty);
  }

  return (
    <Box flexDirection="column">
      <Text>
        <Text bold dimColor>{"  Models "}</Text>
        <Text dimColor>{"  "}</Text>
        <Text bold={mode === "installed"} color={mode === "installed" ? "cyan" : undefined} dimColor={mode !== "installed"}>
          {"[i] Installed"}
        </Text>
        <Text dimColor>{"  "}</Text>
        <Text bold={mode === "pull"} color={mode === "pull" ? "cyan" : undefined} dimColor={mode !== "pull"}>
          {"[p] Pull New"}
        </Text>
        <Text dimColor>{"  [r] Refresh"}</Text>
      </Text>
      <Text dimColor>{"  \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"}</Text>

      <Text>{""}</Text>

      {/* Pull progress */}
      {pullProgress && (
        <Box flexDirection="column">
          <Text>
            <Text dimColor>{"  "}</Text>
            <Text color="cyan">{pullProgress.downloading ? "\u21BB" : "\u2713"}</Text>
            <Text>{` ${pullProgress.model}: ${pullProgress.status}`}</Text>
          </Text>
          {pullProgress.percent > 0 && (
            <Text>
              <Text dimColor>{"    "}</Text>
              <Text color="green">{renderProgressBar(pullProgress.percent)}</Text>
              <Text dimColor>{` ${pullProgress.percent}%`}</Text>
            </Text>
          )}
          <Text>{""}</Text>
        </Box>
      )}

      {/* Error */}
      {errorMessage && (
        <Text>
          <Text dimColor>{"  "}</Text>
          <Text color="red">{`\u2717 ${errorMessage}`}</Text>
        </Text>
      )}

      {/* Delete confirmation */}
      {pendingDelete !== null && (
        <Text>
          <Text dimColor>{"  "}</Text>
          <Text color="yellow">{`Delete ${pendingDelete}? [y/N]`}</Text>
        </Text>
      )}

      {/* Installed models list */}
      {mode === "installed" && (
        <Box flexDirection="column">
          {installed.length === 0 ? (
            <Box flexDirection="column">
              <Text dimColor>{"  No models installed. Press [p] to pull one."}</Text>
            </Box>
          ) : (
            installed.map((m, i) => (
              <Text key={m.name}>
                <Text color={i === selectedIdx ? "cyan" : undefined}>
                  {i === selectedIdx ? "  \u25B6 " : "    "}
                </Text>
                <Text bold={i === selectedIdx} color={i === selectedIdx ? "white" : undefined}>
                  {m.name.padEnd(24)}
                </Text>
                <Text dimColor>{`${m.parameterSize.padEnd(8)} ${m.quantization.padEnd(10)} ${m.size.padStart(8)}`}</Text>
              </Text>
            ))
          )}
          {installed.length > 0 && (
            <Box marginTop={1}>
              <Text dimColor>{"  [\u2191\u2193] navigate  [d] delete selected  [p] pull new model"}</Text>
            </Box>
          )}
        </Box>
      )}

      {/* Pull suggestions list */}
      {mode === "pull" && (
        <Box flexDirection="column">
          {suggestions.length === 0 ? (
            <Text dimColor>{"  All suggested models are already installed!"}</Text>
          ) : (
            suggestions.map((s, i) => (
              <Text key={s.tag}>
                <Text color={i === selectedIdx ? "cyan" : undefined}>
                  {i === selectedIdx ? "  \u25B6 " : "    "}
                </Text>
                <Text bold={i === selectedIdx} color={i === selectedIdx ? "white" : undefined}>
                  {s.desc}
                </Text>
              </Text>
            ))
          )}
          {suggestions.length > 0 && (
            <Box marginTop={1}>
              <Text dimColor>{"  [\u2191\u2193] navigate  [Enter] pull selected  [i] back to installed"}</Text>
            </Box>
          )}
        </Box>
      )}

      {/* Active model info */}
      {snapshot.activeModel && (
        <Box marginTop={1}>
          <Text>
            <Text dimColor>{"  Active: "}</Text>
            <Text color="green" bold>{snapshot.activeModel}</Text>
          </Text>
        </Box>
      )}
    </Box>
  );
});
