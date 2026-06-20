# Changelog

## Unreleased

## 0.2.0 (2026-06-19)

Reframed as **multi-model orchestration**: failover (resilience) is now joined by
tier-split (cost/latency). The package adds an agent loop that gathers tool data
on a cheap/local model and composes the final answer on a frontier model — fully
backward compatible, still `langchain-core`-only.

- `TieredChatAgent` — runs the tool-gathering loop on a `gatherer` and composes the
  answer on a `composer`. Binds tools for you, executes tool calls (errors become
  result text, never raised), nudges once on a premature "ready" signal, and has a
  `max_rounds` runaway backstop. Either tier can be a `FailoverChatModel`.
- `synthesize_answer(composer, query, messages)` — the compose step on its own, for
  callers running their own loop. Flattens `ToolMessage`s (and `{"role":"tool"}`
  dicts) into a clean, model-portable prompt; strips `<think>…</think>` blocks.
- `is_premature_marker(content, tools_bound, tools_called)` — the structural safety
  invariant (tools available ↔ none called) so the composer never answers from
  zero data; real direct answers (e.g. greetings) pass through.
- `create_tiered_agent` convenience constructor; exported prompt defaults
  (`READY_MARKER`, `DEFAULT_GATHER_DIRECTIVE`, `DEFAULT_SYNTH_SYSTEM`).

## 0.1.1 (2026-05-30)
- **Fix (important):** bound tools now actually reach the model. The wrapper
  delegated to the inner model's `_generate`/`_stream` directly, which bypassed
  the `RunnableBinding` kwargs that `bind_tools` produces for real chat models
  (e.g. `ChatOpenAI`) and silently dropped `tool_calls`. It now delegates via
  `invoke`/`stream`, so tool-calling genuinely survives failover. Added a
  regression test that exercises the `RunnableBinding` path (the 0.1.0 test used
  a fake whose `bind_tools` returned a plain model, so it missed this).

## 0.1.0 (2026-05-30)

- Initial release.
- `FailoverChatModel`: primary/secondary failover with stateful recovery.
- Connection-aware failover that walks the exception cause/context chain.
- `bind_tools` preserved across failover (binds both legs).
- Mid-stream safety: only fails over before the first streamed token.
- `create_failover_llm` convenience constructor with `/models` auto-discovery.
- `extract_token_metrics` helper for OpenAI-compatible and Ollama metadata.
