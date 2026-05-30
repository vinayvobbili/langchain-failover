# Changelog

## Unreleased

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
