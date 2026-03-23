// Public API exports
export { detectHardware } from "./hardware/index.js";
export { getAllModels, searchModels, filterByCategory } from "./models/database.js";
export { fetchOllamaModels, clearOllamaCache } from "./models/ollama-models.js";
export { getMergedModels } from "./models/merged-models.js";
export { getRecommendations } from "./analysis/recommender.js";
export { scoreModel, classifyFit } from "./analysis/scorer.js";
export { getConfig, loadConfig, resolveOllamaHost } from "./core/config.js";
export type { LlmPulseConfig } from "./core/config.js";
export type * from "./core/types.js";
