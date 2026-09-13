export interface PullStreamEvent {
  status?: string;
  completed?: number;
  total?: number;
  error?: string;
}

/**
 * Parse the newline-delimited JSON body of Ollama's streaming /api/pull.
 * Network chunks can end mid-line, so only complete lines are parsed and the
 * unfinished tail is returned for the caller to prepend to the next chunk.
 */
export function parsePullLines(buffer: string): { events: PullStreamEvent[]; rest: string } {
  const lines = buffer.split("\n");
  const rest = lines.pop() ?? "";
  const events: PullStreamEvent[] = [];
  for (const line of lines) {
    if (!line.trim()) continue;
    try {
      events.push(JSON.parse(line) as PullStreamEvent);
    } catch {
      // Ignore malformed lines; a real failure arrives as an explicit error line.
    }
  }
  return { events, rest };
}
