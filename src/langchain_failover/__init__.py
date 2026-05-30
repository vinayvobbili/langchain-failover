"""langchain-failover — a primary/secondary failover wrapper for LangChain chat models."""
from langchain_failover.failover import (
    FailoverChatModel,
    create_failover_llm,
    extract_token_metrics,
    is_connection_error,
)

__version__ = "0.1.1"

__all__ = [
    "FailoverChatModel",
    "create_failover_llm",
    "extract_token_metrics",
    "is_connection_error",
    "__version__",
]
