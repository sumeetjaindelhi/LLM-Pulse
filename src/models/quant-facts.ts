// Quantization facts: bits per weight and quality (KL divergence vs FP16) from llama.cpp docs.

interface QuantFact {
  // File bpw measured on Llama-3.1-8B (includes tensors kept at higher precision).
  fileBpw?: number;
  // Block-layout bpw; used only when fileBpw is not measured (kept on measured rows as documentation).
  nominalBpw?: number;
  // Mean KLD vs FP16 on Llama-3 8B; absent when unpublished.
  meanKld?: number;
}

// Source of every row: llama.cpp a300a9c33462f23d9d61e951650d249486f2b0f6.
// fileBpw: tools/quantize/README.md:145-174; nominalBpw: ggml/src/ggml-common.h block static_asserts
// (the pinned spec also lists 4.5/5.0/5.5/6.5625/8.5 there); meanKld: tools/perplexity/README.md:38-125.
// F16 meanKld 0 is a definition (F16 is the reference); BF16 shares F16's 16-bit layout and its measured fileBpw.
// IQ4_XS and IQ4_NL KLDs are imatrix rows (perplexity README.md:61-62); the others are no-imatrix rows.
const QUANT_FACTS: Record<string, QuantFact> = {
  Q8_0: { nominalBpw: 8.5, fileBpw: 8.5008, meanKld: 0.001355 },
  Q6_K: { nominalBpw: 6.5625, fileBpw: 6.5633, meanKld: 0.005452 },
  Q5_K_M: { nominalBpw: 5.5, fileBpw: 5.7036, meanKld: 0.010762 },
  Q5_K_S: { nominalBpw: 5.5, fileBpw: 5.5704, meanKld: 0.016595 },
  Q5_0: { nominalBpw: 5.5, meanKld: 0.022239 },
  Q5_1: { nominalBpw: 6.0, meanKld: 0.018045 },
  Q4_K_M: { nominalBpw: 4.5, fileBpw: 4.8944, meanKld: 0.031273 },
  Q4_K_S: { nominalBpw: 4.5, fileBpw: 4.6672, meanKld: 0.043136 },
  Q4_0: { nominalBpw: 4.5, meanKld: 0.07194 },
  Q4_1: { nominalBpw: 5.0, meanKld: 0.071683 },
  IQ4_XS: { nominalBpw: 4.25, fileBpw: 4.4597, meanKld: 0.036334 },
  IQ4_NL: { nominalBpw: 4.5, fileBpw: 4.6818, meanKld: 0.035742 },
  Q3_K_M: { fileBpw: 3.996, meanKld: 0.101913 },
  Q3_K_S: { nominalBpw: 3.4375, fileBpw: 3.6429, meanKld: 0.228217 },
  Q2_K: { nominalBpw: 2.625, fileBpw: 3.1593, meanKld: 0.445132 },
  F16: { nominalBpw: 16, fileBpw: 16.0005, meanKld: 0 },
  BF16: { nominalBpw: 16, fileBpw: 16.0005, meanKld: 0 },
  // Quality not published.
  MXFP4: { nominalBpw: 4.25 },
};

// Estimate ~ Q4_K_M's measured excess of 0.39 (tools/quantize/README.md:167); output.weight is forced to Q6_K
// (src/llama-quant.cpp:572-592). Against measured Llama-3.1-8B file bpw, nominal + 0.4 errs -4.3% (Q2_K) to +6.1% (Q6_K)
// (README.md:145-174); MXFP4 is unmeasured.
export const ESTIMATED_BPW_EXCESS = 0.4;

function lookup(quant: string): QuantFact | undefined {
  return QUANT_FACTS[quant.toUpperCase()];
}

/** Bits per weight of the GGUF file; `estimated` when derived from nominal bpw. Undefined for unknown types. */
export function quantFileBpw(quant: string): { bpw: number; estimated: boolean } | undefined {
  const fact = lookup(quant);
  if (!fact) return undefined;
  if (fact.fileBpw !== undefined) return { bpw: fact.fileBpw, estimated: false };
  if (fact.nominalBpw === undefined) return undefined;
  return { bpw: fact.nominalBpw + ESTIMATED_BPW_EXCESS, estimated: true };
}

/**
 * Geometric-mean probability ratio vs FP16, exp(-meanKLD). Model-specific (Llama-3 8B).
 * Unpublished types report 1 with `published: false` so callers can say "no published quality data".
 */
export function quantRetention(quant: string): { retention: number; published: boolean } {
  const kld = lookup(quant)?.meanKld;
  return kld === undefined ? { retention: 1, published: false } : { retention: Math.exp(-kld), published: true };
}
