# Future development: agentic interfaces + self-optimisation

Status: investigation, not started. Companion to `docs/plan.md`.
Main CI at time of writing: run 41423800953 (post-merge) — check
`gh run list --branch main` for the terminal result.

## Avenue 1: Agentic interfaces to GridTalk

### Where we are

GridTalk already has the *inside* of an agent surface, but no
outside one:

- `crates/gridtalk-app/src/investigate.rs`: `Tools::execute(tool, args)`
  with tools `search`, `expand`, `cells`, `formulas`, `what_if`.
  Args are stringly-typed `key=value;key=value`; results are
  human-formatted strings; the planner in `investigate()` is
  deterministic, not model-driven.
- CLI (`gridtalk ask|investigate|graph|view|…`) prints text, not
  JSON. There is no server, no tool schema, no session state.
- The local chat path uses whichever `ChatBackend` is configured.

An external harness (Claude Code, Codex, Cursor, …) today can only
shell out to the CLI and scrape text. That is the gap.

### The interface: MCP first, JSON CLI second

The Model Context Protocol is the standard these harnesses already
speak for tools. Build `gridtalk mcp` as an MCP server over stdio:

**Tools (schema-typed, JSON in/out):**

| Tool | Maps to today | Notes |
|---|---|---|
| `open_workbook` | `load_path` + snapshot/graph/index build | Returns workbook id, sheets, regions |
| `search` | `HybridIndex::search` | Returns ranges, scores, passages, `vector_only` |
| `expand` | graph neighbours | 1–2 hops, edge kinds |
| `get_cells` | `cells` tool | Typed values + formulas, not display strings |
| `list_formulas` | `formulas` tool | Formula text + evaluated value |
| `what_if` | `eval::what_if` | Read-only simulation; never writes the file |
| `investigate` | `investigate()` | Full cited answer + trail |
| `graph_overview` | graph nodes/edges | For harness-side reasoning |
| `close_workbook` | — | Frees session state |

**Design decisions:**

1. **Crate:** use the Rust MCP SDK (`rmcp`) in a new
   `gridtalk-mcp` binary target (or `gridtalk mcp` subcommand).
   Tool arguments become serde structs — this forces the
   stringly-typed `key=value` parsing in `Tools::execute` to be
   replaced by typed calls, which the internal planner also uses.
2. **Sessions:** the server holds open workbooks keyed by an id:
   snapshot + graph + index cached; invalidated when the file's
   mtime changes (re-snapshot via the incremental path).
3. **Safety/privacy defaults:** `open_workbook` only opens paths
   the calling harness supplies. `what_if` is simulation-only.
   Any future write tool (`set_cell`, `save`) is behind
   `--allow-write`, off by default, and the MCP tool list simply
   omits it otherwise. The server is local stdio; a loopback
   streamable-HTTP transport is a later option, never default-on.
4. **JSON CLI as the fallback surface:** `gridtalk ask --json`,
   `gridtalk tools call <tool> --json '{…}'` with stable exit
   codes, so harnesses that prefer subprocess + JSON (and scripts,
   and hermit-style drivers — see Avenue 2) are not forced into
   MCP. One typed core, two transports.
5. **Harness recipes shipped in-repo:** `.mcp.json` example for
   Claude Code, a Codex `config.toml` snippet, and a worked
   example ("drive GridTalk from Claude Code to audit this
   workbook") in docs.

**Acceptance:** an MCP client (rmcp test client, then a real
Claude Code / Codex session) can open the demo workbook, search,
run a what-if, and get a cited investigate answer without the
GridTalk UI running.

## Avenue 2: Optimising the local agent without retraining (hermit)

### What hermit is

Hermit (Infini-AI-Lab) is a framework for *harness adapter
evolution*: the model and the existing harness stay frozen; an
external, trainable **harness adapter** learns from task feedback.
Its loop is RL-style: run tasks → a judge scores them →
`evolver.step` proposes an adapter update → the proposal is
checked and committed as the next adapter version. Methods
include GEPA, SkillOpt, Meta-Harness Adapter, ReHarness and
others. Concretely, an adapter carries human-readable learned
rules — the walkthrough shows a learner writing
`Output format: uppercase.` into `ADAPTER.md`, later runs loading
it, and the adapter exporting an `AGENTS.md` that Codex picks up
in a workspace; each adapter records evaluation details in
`.hermit/CARD.md`. It is a Python package (`pip install
hermit-harness`).

### The GridTalk analogue: adapters for spreadsheet QA

Our "harness" is exactly the surface hermit adapts: the
investigate planner, the system prompt, retrieval knobs, and
tool policy. None of it needs model weights to improve. Define a
**GridTalk adapter** as a versioned directory:

```
~/.config/gridtalk/adapters/<name>/
  ADAPTER.md     # learned rules / instructions for the local model
  adapter.toml   # knobs: retrieval top_k, graph hops, max_steps,
                 # temperature, tool order/policy, few-shot on/off
  skills/*.md    # domain skills (e.g. "auditing provisions",
                 # "reconciling totals") in SKILL.md style
  CARD.md        # eval record per version (hermit convention)
  v1/ v2/ …      # committed versions; active pointer in adapter.toml
```

Loading order at chat/investigate time: built-in prompt →
global adapter → per-workbook sidecar (`<workbook>.gridtalk.md`,
the AGENT.md-style file a user or a harness can edit next to the
spreadsheet) → skills whose triggers match. This answers the
assumption in the brief: yes, our harness can employ AGENT.md /
SKILL.md-style files — we define their schema and honour them,
and the optimiser writes them.

### Why GridTalk is unusually well placed for this

Most agent optimisation needs an LLM judge. Spreadsheet QA often
does not: **formulas are ground truth.** A benchmark can be
*generated from the workbook itself*:

- Value questions ("what is Qty for Provision?") — exact cell.
- What-if questions ("what if B2 is 10?") — exact number from
  `eval::what_if`.
- Formula questions ("how is Total calculated?") — judge by
  whether the cited range and formula cells match.
- Citation correctness is mechanically checkable against the
  retrieved ranges.

So the fitness function is deterministic, local, and free — no
judge model, no data leaving the machine, and small local models
can be scored honestly. This is the strongest argument that
optimisation can lift small-model performance here.

### The optimisation loop (native first, hermit-compatible)

1. `gridtalk bench generate <workbook>` — synthesise tasks with
   ground-truth answers; split train/dev (dev is never optimised
   against directly).
2. `gridtalk bench run --adapter <name>` — run the local model +
   investigate harness over tasks; record accuracy, citation
   precision, latency, tokens into `CARD.md`.
3. **Proposer:** reflective edit of `ADAPTER.md` + knobs. The
   proposer can be the local model itself, a bigger model via the
   existing external backend, or the user. Failed tasks (with
   their trails) are the proposer's input — the trail design from
   Phase 3b pays off here.
4. Re-run dev split; commit as `vN+1` only if it beats `vN` on
   dev (else keep as a rejected proposal in CARD.md). Rollback is
   switching the active pointer.
5. **Hermit interop (optional, later):** because Avenue 1 gives
   GridTalk an MCP/JSON-CLI surface, hermit's Python evolvers
   (GEPA/SkillOpt) can drive GridTalk as the harness under
   optimisation, and GridTalk adapters export `AGENTS.md` so
   Codex benefits from what the local loop learned. Native loop
   first — it needs no Python runtime — hermit as an advanced
   driver second.

### Small-model levers the adapter controls

- Output discipline rules (cite-first format, one-sentence
  answers) — cheap wins for 4B-class models.
- Retrieval knobs per workbook shape (top_k, hops) instead of
  global constants.
- Few-shot exemplars selected from skills.
- Tool policy (when to what-if vs answer directly), max steps.
- Sampling (temperature) per task type.

### Risks and guardrails

- **Overfitting** to generated tasks → dev-split gating, task
  regeneration with different phrasings, periodic human review
  of ADAPTER.md diffs (they are plain markdown on purpose).
- **Adapter drift / bad commits** → versioned, CARD.md always
  shows score deltas, one-click rollback in the UI.
- **Privacy** → tasks derive from the user's workbook; the whole
  loop must run locally by default. If the proposer is an
  external model, that is the existing verbatim-send posture and
  must be labelled as such (decision 0002 applies unchanged).
- **Judge gaming** → ground truth from formulas/eval only for
  scored tasks; free-text quality is reported, not scored.

## Phased plan

**Phase A (foundation, ~1 wk):** typed tool core. Refactor
`Tools::execute` to serde-typed calls returning structured
results; string rendering becomes a presentation layer. Add
`gridtalk tools call --json`. Tests: contract tests per tool.

**Phase B (MCP server, ~1–2 wks):** `gridtalk mcp` (rmcp, stdio),
session/cache layer, `.mcp.json` + Codex recipes, MCP contract
tests, one real Claude Code and one Codex session driven
end-to-end as acceptance.

**Phase C (benchmark generator, ~1 wk):** `gridtalk bench
generate|run`, deterministic judges, CARD.md writer. Acceptance:
bench on the demo workbook produces scored results for the echo
backend (floor) and a real local model.

**Phase D (adapter format + loop, ~2 wks):** adapter directories,
loading into investigate/chat, `gridtalk optimise` (propose →
eval → commit/reject), adapter picker + CARD viewer in the UI.
Acceptance: on a synthetic workbook with known weaknesses, one
optimise cycle measurably improves dev accuracy and the
improvement is visible as an ADAPTER.md diff.

**Phase E (hermit interop, optional):** document/validate
driving GridTalk from hermit's Python evolvers via the JSON CLI;
`AGENTS.md` export from an adapter.

Total: ~5–6 weeks for A–D, E as stretch. Sequencing note: A is
also the fix for several stringly-typed warts the code review
already flagged, so it pays for itself even if B–E slip.

## Sources

- Hermit (framework, ADAPTER.md/AGENTS.md/CARD.md conventions,
  evolution methods): https://github.com/Infini-AI-Lab/hermit
  and its docs at https://infini-ai-lab.github.io/hermit/docs/
- Model Context Protocol: https://modelcontextprotocol.io/docs/getting-started/intro
- GridTalk tool surface today: `crates/gridtalk-app/src/investigate.rs`
