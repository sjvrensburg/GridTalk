# 0006: The model chooses its own tool calls

Decided 2026-11-13. It reverses the plan's earlier note that a
model-driven agent loop was deprioritised as a product decision.

## Decision

When the backend can take tool turns, `investigate`, `bench` and the
desktop chat answer with a tool loop: the deterministic planner
opens (search, then what the question names, at most four calls and
no model turn), and the model then chooses its own calls until it
answers or its budget is spent. The fixed pipeline stays for the
echo backend, and for a caller or adapter that asks for it
(`--no-tool-loop`, `tool_loop = false`).

The loop is model-agnostic. It depends on no model's own tool
template:

- The OpenAI-compatible backend sends the tools as function
  definitions (`tools` / `tool_calls`).
- The local llama.cpp backend describes the tools in the system
  prompt and holds each reply to a grammar: one JSON object naming a
  tool and its arguments, or `ANSWER:` and the answer. A first turn
  that must be a call is a property of the grammar, not a retry.

The harness is what is tuned, not the model: the loop's behaviour is
adapter knobs (`max_steps`, `tool_loop`, `first_call`) and adapter
rules, which `gridtalk optimise` and hermit evolve. A model
fine-tuned for GridTalk later slots in behind the same protocol, and
`investigate --trace` writes the requests and replies such training
would start from.

The tools, the planner and the loop live in one library crate,
`gridtalk-agent`, so the CLI, the MCP server, bench and the desktop
UI share one implementation.

## Why

- A measurement on 30 synthetic workbooks
  ([#43](https://github.com/sjvrensburg/GridTalk/issues/43)): the
  fixed pipeline's tool results carried the answer to 526 of 1,334
  questions; the same tools, called per question, carried 1,126. The
  gap is the pipeline, not the tools or the graph. Each of #34, #39,
  #41 and #42 widened what the pipeline reads and converged on this.
- Questions need different calls. Following an input to a result
  needs an `expand` at each step of the chain; a question about one
  column needs that column read and little else. A fixed budget
  spent the same way every time serves neither.
- The desktop chat had its own, weaker path (search hits and the
  selection in one prompt). It now answers the way the CLI does.

## How the loop is kept in hand

- The first model turn must be a tool call (`first_call`), so a
  small model cannot answer from the question's own words.
- `max_steps` bounds the model's calls; the turn after the last one
  can only answer. The planner's opening reads come on top.
- A result is cut to a share of the context when it is added. Older
  results are shortened only when the conversation outgrows the
  context, because a rewritten history costs the local backend its
  cached prefix: a local conversation keeps one llama.cpp context
  and evaluates only the new part of each prompt.
- A call repeated with the same arguments is not run again, and a
  call that cannot run (no such tool, bad arguments, not JSON) is
  answered with what was wrong.
- The answer's citations are checked: a cited range no tool result
  read or named is reported (`unread_citations`), and an answer that
  cites nothing it read is given its sources.

## Consequences

- An answer costs several model turns instead of one. On the local
  backend the turns after the first are cheap (the prefix is
  reused); on an external endpoint each turn is a request that
  carries the whole conversation, verbatim (0002).
- Scores now depend on the model's choices, so bench reports, per
  kind of question, how often the tool results carried the answer
  next to how often the answer was right. The echo backend still
  scores the fixed pipeline deterministically.
- The measurement that motivated this used scripted calls as an
  oracle. A real model does worse than that ceiling, and the
  questions were eight templates over synthetic workbooks: tune
  adapters on a held-out split.

## Not done

- A local model whose chat template llama.cpp's built-in applier
  does not recognise cannot be used at all (the limit predates the
  loop and applies to `ask` too).
- The desktop chat shows the tool call in progress but does not
  stream the answer's text, and it builds no trace.
- The local grammar takes one call per turn; an external endpoint
  may return several.
