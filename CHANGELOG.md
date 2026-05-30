# Changelog

## 0.1.0 (unreleased)

- Initial release.
- `FailoverChatModel`: primary/secondary failover with stateful recovery.
- Connection-aware failover that walks the exception cause/context chain.
- `bind_tools` preserved across failover (binds both legs).
- Mid-stream safety: only fails over before the first streamed token.
- `create_failover_llm` convenience constructor with `/models` auto-discovery.
- `extract_token_metrics` helper for OpenAI-compatible and Ollama metadata.
