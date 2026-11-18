import type { ModelEntry } from "../../src/core/types.js";

// Not the Q8_0 estimate of any row the tests use, so a test can tell which model a command used.
export const EXACT_Q8_MB = 8000;

/** Stand-in for withExactFacts: the model as if its tag installed a Q8_0 file, the one exact quant, at that file's real size. */
export async function withExactQ8(_host: string, model: ModelEntry): Promise<ModelEntry> {
  return {
    ...model,
    quantizations: model.quantizations.map(({ exact: _exact, ...q }) => (q.name === "Q8_0" ? { ...q, vramMb: EXACT_Q8_MB, exact: true } : q)),
  };
}
