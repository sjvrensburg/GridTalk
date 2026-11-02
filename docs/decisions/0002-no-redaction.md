# 0002: No value-redaction in GridTalk

## Decision

GridTalk contains no redaction layer: no privacy modes, no redacted
tool calls, no refusal logic. ExcelGRAG's `off|passage|values`
machinery is explicitly not ported.

## Rationale

Privacy-conscious users have two honest options: run a local model
(nothing leaves the PC), or place a redacting proxy between GridTalk
and an external LLM. That proxy is a separate project at a later
stage. GridTalk supports it unchanged because its external backend is
a plain OpenAI-compatible client: point `base_url` at the proxy.

## Consequences

The settings UI and README must state plainly that an external
endpoint receives cell values verbatim. A contract test asserts the
external backend sends the passage unmodified, so redaction cannot
creep back in silently.
