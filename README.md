# langchain-failover

[![CI](https://github.com/vinayvobbili/langchain-failover/actions/workflows/ci.yml/badge.svg)](https://github.com/vinayvobbili/langchain-failover/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/langchain-failover)](https://pypi.org/project/langchain-failover/)
[![Python](https://img.shields.io/pypi/pyversions/langchain-failover)](https://pypi.org/project/langchain-failover/)
[![License](https://img.shields.io/pypi/l/langchain-failover)](https://github.com/vinayvobbili/langchain-failover/blob/main/LICENSE)

A tiny, dependency-light **primary/secondary failover wrapper** for LangChain chat
models. Point it at two chat models; it serves from the primary, transparently
falls back to the secondary on connection errors, and switches back the moment the
primary recovers — **and tool-calling keeps working across the failover.**

> **Background:** [SOC-in-a-Box: One LLM, Eight Hats](https://vinayvobbili.github.io/posts/building-soc-in-a-box/) — the production AI SOC this was extracted from, where it fails a local LLM over to a backup mid-incident.

```python
from langchain_openai import ChatOpenAI
from langchain_failover import FailoverChatModel

primary = ChatOpenAI(base_url="http://gpu-box:8001/v1", api_key="x", model="local")
backup  = ChatOpenAI(base_url="http://cpu-box:8002/v1", api_key="x", model="local")

llm = FailoverChatModel(primary=primary, secondary=backup)

llm.invoke("Summarise this incident…")   # served by primary
# …primary host dies…
llm.invoke("And the next one?")           # transparently served by backup
# …primary comes back…
llm.invoke("One more")                     # back on primary, logged as recovered
```

## Install

```bash
pip install langchain-failover            # core
pip install "langchain-failover[openai]"  # + langchain-openai for create_failover_llm
```

## Why not `RunnableWithFallbacks` / `.with_fallbacks()`?

LangChain ships per-invocation fallbacks, and they're great for what they do. This
package exists for the cases they don't cover well:

- **Stateful recovery.** `FailoverChatModel` remembers which leg it's on and logs
  the transition both ways (`active` property tells you). `.with_fallbacks()` is
  stateless — every call re-tries the (possibly still-dead) primary first.
- **Tool-calling survives failover.** `bind_tools` is overridden to bind on *both*
  legs and return another `FailoverChatModel`. With strict langchain-core
  (`>=1.4`, where `BaseChatModel.bind_tools` raises by default) naïve wrappers
  break at bind time; agents using this one keep working.
- **Connection-aware, not blanket.** It only fails over on connection/network
  errors (walking the exception's `__cause__`/`__context__` chain, so a socket
  error wrapped three layers deep still counts). A `ValueError` from a bad prompt
  propagates instead of being silently retried on a second endpoint.
- **Mid-stream safety.** During `stream()`, it only fails over if the primary dies
  *before* the first token — so you never get duplicated, half-streamed output.

## Local-model convenience

If you run local OpenAI-compatible servers (vLLM, mlx-lm, Ollama, LM Studio) and
don't want to hardcode model names, `create_failover_llm` auto-discovers the served
model id from each endpoint's `/models`:

```python
from langchain_failover import create_failover_llm

llm = create_failover_llm(
    primary_url="http://localhost:8001/v1",
    secondary_url="http://localhost:8002/v1",
)
```

## Bonus helper

`extract_token_metrics(response.response_metadata)` normalises token counts and
timings across OpenAI-compatible and Ollama metadata shapes into a single
`{input_tokens, output_tokens, prompt_time, generation_time}` dict.

## License

MIT
