"""langchain-failover — multi-model orchestration for LangChain chat models.

Two orchestration strategies over plain LangChain chat models:

* **Failover** (:class:`FailoverChatModel`) — serve from a primary, transparently
  fall back to a secondary on connection errors, switch back on recovery. For
  *resilience*. Tool-calling survives the failover.
* **Tier-split** (:class:`TieredChatAgent`) — run the tool-gathering loop on a
  cheap/local model, compose the final answer on a frontier model. For
  *cost/latency*: the long generation moves off the contended box.

The two compose — either tier of a :class:`TieredChatAgent` can itself be a
:class:`FailoverChatModel`.
"""
from langchain_failover.failover import (
    FailoverChatModel,
    create_failover_llm,
    extract_token_metrics,
    is_connection_error,
)
from langchain_failover.tiered import (
    DEFAULT_GATHER_DIRECTIVE,
    DEFAULT_SYNTH_SYSTEM,
    READY_MARKER,
    TieredChatAgent,
    create_tiered_agent,
    is_premature_marker,
    synthesize_answer,
)

__version__ = "0.2.0"

__all__ = [
    # Failover (resilience)
    "FailoverChatModel",
    "create_failover_llm",
    "is_connection_error",
    "extract_token_metrics",
    # Tier-split (cost/latency)
    "TieredChatAgent",
    "create_tiered_agent",
    "synthesize_answer",
    "is_premature_marker",
    "READY_MARKER",
    "DEFAULT_GATHER_DIRECTIVE",
    "DEFAULT_SYNTH_SYSTEM",
    "__version__",
]
