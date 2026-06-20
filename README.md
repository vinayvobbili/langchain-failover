# langchain-failover

[![CI](https://github.com/vinayvobbili/langchain-failover/actions/workflows/ci.yml/badge.svg)](https://github.com/vinayvobbili/langchain-failover/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/langchain-failover)](https://pypi.org/project/langchain-failover/)
[![Python](https://img.shields.io/pypi/pyversions/langchain-failover)](https://pypi.org/project/langchain-failover/)
[![License](https://img.shields.io/pypi/l/langchain-failover)](https://github.com/vinayvobbili/langchain-failover/blob/main/LICENSE)

Tiny, dependency-light **multi-model orchestration** for LangChain chat models —
two strategies for running more than one model behind one interface:

- **Failover** (`FailoverChatModel`) — for **resilience**. Serve from a primary,
  transparently fall back to a secondary on connection errors, switch back the
  moment the primary recovers. Tool-calling keeps working across the failover.
- **Tier-split** (`TieredChatAgent`) — for **cost/latency**. Run the tool-gathering
  loop on a cheap/local model, then compose the final answer on a frontier model.
  The long generation moves off the contended box.

They compose: either tier of a `TieredChatAgent` can itself be a
`FailoverChatModel`. Depends only on `langchain-core`.

> **Background:** [SOC-in-a-Box: One LLM, Eight Hats](https://vinayvobbili.github.io/posts/building-soc-in-a-box/) — the production AI SOC this was extracted from, where it fails a local LLM over to a backup mid-incident and offloads final-answer synthesis to a frontier model.

## Failover — for resilience

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

## Tier-split — for cost/latency

A tool-calling agent spends almost all of its tokens and wall-clock on the *loop*
(decide a call, read the result, decide the next) — cheap reasoning. Writing the
final answer is the part that wants a stronger model. They don't have to be the
same model. `TieredChatAgent` runs the gathering loop on a cheap/local `gatherer`
and composes the answer on a frontier `composer`:

```python
from langchain_failover import TieredChatAgent

agent = TieredChatAgent(
    gatherer=local_llm,      # cheap/local — drives the tool loop (tools are bound for you)
    composer=frontier_llm,   # frontier — writes the final answer from gathered data
    tools=[search, lookup_host, get_timeline],
)

agent.invoke("What changed in the incident overnight?").content
```

The gatherer is told to *gather, then stop* — it doesn't write the prose answer.
A structural guard (`is_premature_marker`) catches the model trying to answer
before calling any tool and nudges it to gather first, so the composer never
writes an answer from zero data. On a contended local GPU this routinely turns a
multi-minute final turn into a couple of seconds, because the long generation
moves off the busy box. Running your own loop? `synthesize_answer(composer, query,
messages)` is the compose step on its own.

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
