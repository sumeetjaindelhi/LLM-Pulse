import { describe, it, expect } from "vitest";
import { OllamaTagsSchema, OllamaPickModelSchema } from "../../src/core/api-schemas.js";

const local = { name: "llama3.1:8b", size: 4_900_000_000 };
const cloudByHost = { name: "kimi-k3:cloud", size: 0, remote_host: "https://ollama.com:443", remote_model: "kimi-k3" };
const cloudByModelOnly = { name: "glm-5:cloud", size: 0, remote_model: "glm-5" };
const cloudByHostOnly = { name: "qwen3-coder:480b-cloud", size: 0, remote_host: "https://ollama.com:443" };

describe.each([
  ["OllamaTagsSchema", OllamaTagsSchema],
  ["OllamaPickModelSchema", OllamaPickModelSchema],
])("%s", (_name, schema) => {
  it("drops cloud entries carrying remote_host or remote_model and keeps local ones", () => {
    const parsed = schema.parse({ models: [local, cloudByHost, cloudByModelOnly, cloudByHostOnly] });
    expect(parsed.models.map((m) => m.name)).toEqual(["llama3.1:8b"]);
  });

  it("still parses a response with no cloud fields", () => {
    expect(schema.parse({ models: [local] }).models).toHaveLength(1);
  });

  it("defaults to an empty list", () => {
    expect(schema.parse({}).models).toEqual([]);
  });
});
