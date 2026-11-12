# Adapters (Phase D)

A GridTalk **adapter** is versioned, human-readable harness
configuration — learned rules and knobs — that improves the
local agent on a workbook shape *without retraining the model*.

## Layout

```
<config>/gridtalk/adapters/<name>/
  ADAPTER.md     learned rules (plain markdown, human-reviewable)
  adapter.toml   version pointer + knobs: top_k, max_steps, ref_context
  CARD.md        eval record appended every optimise round
  skills/*.md    optional skills; the first `Triggers: a, b` line
                 scopes a skill to matching questions (no line =
                 always). Reserved forms: an empty `Triggers:` line
                 means never auto-apply; `Triggers: (always)` is the
                 explicit always-apply marker — a literal `(always)`
                 trigger word is reserved and rejected at commit.
  v1/ v2/ …      committed versions; adapter.toml picks the active one
```

Knobs the harness honours today: `top_k` (search hits),
`max_steps` (investigate budget), `ref_context` (when a question
names a cell like `Sheet1!D2`, include that cell directly even if
retrieval missed it; on by default, as with no adapter, so set
`ref_context = false` to turn it off).

Loading order: built-in prompt → adapter rules → matched skills
→ per-workbook sidecar. The sidecar is `<workbook>.gridtalk.md`,
an AGENT.md-style file you (or an external harness) can edit next
to the spreadsheet; it always applies last.

## Using an adapter

```bash
gridtalk adapter list
gridtalk adapter show demo        # rules + CARD
gridtalk adapter rollback demo    # move the pointer back one version
gridtalk investigate book.csv --adapter demo "what does Sheet1!D2 calculate to?"
gridtalk bench run book.csv --adapter demo
```

In the desktop UI, the **Adapter** button in the toolbar picks
the active adapter (persisted) and shows its CARD.md.

## The optimise loop

```bash
gridtalk optimise <workbook> [--adapter <name>] [--rounds 3]
```

Each round:

1. Scores the current adapter on the workbook's bench tasks
   (train and dev; see `docs/bench.md`).
2. Proposes candidates from failure clusters — e.g. formula
   failures produce a "just the final number" rule plus the
   `ref_context` knob; citation misses raise `top_k`.
3. Keeps the best train improvement only if **dev does not
   regress** (dev is a gate, never a proposer input).
4. Commits it as `vN+1` and appends the outcome — accepted or
   rejected, with scores — to CARD.md.

Ground truth is always the evaluator, so the whole loop runs
locally with any backend, including the echo floor. With an
external proposer/backend the usual decision-0002 posture
applies: context leaves the machine verbatim.

Current proposer is deterministic and rule-based. Phase E can
drive the same propose/eval/commit steps from external evolvers
(e.g. hermit) over the CLI/MCP surface.
