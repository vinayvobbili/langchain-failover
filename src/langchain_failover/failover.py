"""A LangChain chat model that fails over between a primary and a secondary model.

The wrapper delegates every call to the primary chat model. If the primary raises
a connection-related error it transparently retries on the secondary, and it
switches back to the primary the moment the primary answers again. ``bind_tools``
is preserved across the failover so tool-calling agents keep working when either
leg is the one serving the request.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from pydantic import ConfigDict

logger = logging.getLogger(__name__)

# Exception *type names* (not classes) that we treat as "the endpoint is
# unreachable, try the other one." Matching by name keeps us independent of
# which HTTP/client library raised it (httpx, requests, urllib, openai, ...).
_CONNECTION_ERROR_NAMES = (
    "ConnectionError",
    "ConnectError",
    "RemoteProtocolError",
    "ConnectionRefusedError",
    "TimeoutError",
    "ReadTimeout",
    "APIConnectionError",
)


def is_connection_error(exc: BaseException) -> bool:
    """Return True if ``exc`` (or anything in its cause/context chain) looks like
    a connection/network failure worth failing over on.

    We walk ``__cause__``/``__context__`` because client libraries routinely wrap
    the original socket error inside a higher-level exception, so the interesting
    type is often several links down the chain.
    """
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        name = type(current).__name__
        if name in _CONNECTION_ERROR_NAMES:
            return True
        if "connection" in name.lower():
            return True
        if "connection" in str(current).lower()[:200]:
            return True
        current = current.__cause__ or current.__context__
    return False


class FailoverChatModel(BaseChatModel):
    """Wraps two chat models — tries ``primary``, falls back to ``secondary``.

    All calls (``invoke``, ``stream``, ``generate``, tool-bound variants) are
    delegated to ``primary``. On a connection-related error the call is retried
    on ``secondary`` and the model remembers it is running degraded; the next
    successful ``primary`` call flips it back and logs the recovery.

    Example
    -------
    >>> from langchain_openai import ChatOpenAI
    >>> from langchain_failover import FailoverChatModel
    >>> primary = ChatOpenAI(base_url="http://localhost:8001/v1", api_key="x", model="local")
    >>> backup = ChatOpenAI(base_url="http://localhost:8002/v1", api_key="x", model="local")
    >>> llm = FailoverChatModel(primary=primary, secondary=backup)
    >>> llm.invoke("hello").content  # doctest: +SKIP
    """

    # ``Any`` rather than ``BaseChatModel`` on purpose: ``ChatModel.bind_tools``
    # returns a ``Runnable`` binding (e.g. langchain's ``_ChatModelBinding``),
    # not a ``BaseChatModel``. The binding still exposes ``_generate``/``_stream``,
    # so wrapping it works — but a strict type would reject it.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    primary: Any
    secondary: Any
    _active: str = "primary"

    @property
    def _llm_type(self) -> str:
        return "failover"

    def _mark_primary_recovered(self) -> None:
        if self._active != "primary":
            logger.info("Failover: primary recovered, switching back")
            self._active = "primary"

    def _switch_to_secondary(self, exc: BaseException) -> None:
        logger.warning(
            "Failover: primary down (%s), switching to secondary",
            type(exc).__name__,
        )
        self._active = "secondary"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        try:
            result = self.primary._generate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            self._mark_primary_recovered()
            return result
        except Exception as exc:
            if is_connection_error(exc):
                self._switch_to_secondary(exc)
                return self.secondary._generate(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
            raise

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        # Only fail over if the primary dies *before* emitting its first chunk;
        # once tokens are flowing a mid-stream error is a real error, not a
        # connect failure, and retrying would duplicate already-yielded output.
        try:
            started = False
            for chunk in self.primary._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                if not started:
                    started = True
                    self._mark_primary_recovered()
                yield chunk
        except Exception as exc:
            if is_connection_error(exc) and not started:
                self._switch_to_secondary(exc)
                yield from self.secondary._stream(
                    messages, stop=stop, run_manager=run_manager, **kwargs
                )
            else:
                raise

    def bind_tools(self, tools, **kwargs) -> "FailoverChatModel":
        """Bind tools on both legs so failover preserves tool-calling.

        langchain-core >=1.4 made ``BaseChatModel.bind_tools`` strict (it raises
        ``NotImplementedError`` by default), so without this override any agent
        that binds tools to a ``FailoverChatModel`` would fail at bind time. Each
        bound leg is a ``Runnable`` that still exposes ``_generate``/``_stream``,
        so the delegation above keeps working on the returned wrapper.
        """
        return FailoverChatModel(
            primary=self.primary.bind_tools(tools, **kwargs),
            secondary=self.secondary.bind_tools(tools, **kwargs),
        )

    @property
    def active(self) -> str:
        """Which leg served the most recent call: ``"primary"`` or ``"secondary"``."""
        return self._active


def create_failover_llm(
    primary_url: str,
    secondary_url: str,
    temperature: float = 0.1,
    api_key: str = "not-needed",
    **kwargs: Any,
) -> FailoverChatModel:
    """Build a :class:`FailoverChatModel` from two OpenAI-compatible base URLs.

    The served model id is auto-discovered from each endpoint's ``/models`` list
    (handy for local servers like vLLM, mlx-lm, Ollama, or LM Studio, where you
    often don't want to hardcode the model name). Extra ``kwargs`` are forwarded
    to both underlying ``ChatOpenAI`` instances.

    Args:
        primary_url: Primary endpoint base URL, e.g. ``http://localhost:8001/v1``.
        secondary_url: Fallback endpoint base URL.
        temperature: Sampling temperature for both legs.
        api_key: Bearer token sent to both endpoints (many local servers ignore it).
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "create_failover_llm requires langchain-openai. "
            "Install it with `pip install langchain-failover[openai]`."
        ) from exc

    import urllib.request
    import json

    def _discover_model(base_url: str) -> str:
        try:
            req = urllib.request.Request(
                f"{base_url.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read()).get("data", [])
            if data:
                return data[0]["id"]
        except Exception:
            pass
        return "default"

    def _make_client(base_url: str) -> Any:
        return ChatOpenAI(
            model=_discover_model(base_url),
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
            **kwargs,
        )

    logger.info("Failover LLM: primary=%s, secondary=%s", primary_url, secondary_url)
    return FailoverChatModel(
        primary=_make_client(primary_url),
        secondary=_make_client(secondary_url),
    )


def extract_token_metrics(meta: Optional[dict]) -> dict:
    """Pull token counts and timings out of a LangChain ``response_metadata`` dict.

    Handles both OpenAI-compatible servers (``token_usage``/``usage``; no timing)
    and Ollama (``prompt_eval_count``/``eval_count`` plus nanosecond durations).
    Every field defaults to 0 when absent so callers never have to guard.

    Returns:
        ``{"input_tokens", "output_tokens", "prompt_time", "generation_time"}``.
    """
    if not meta:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "prompt_time": 0.0,
            "generation_time": 0.0,
        }

    usage = meta.get("token_usage") or meta.get("usage") or {}
    input_tokens = usage.get("prompt_tokens", 0) or meta.get("prompt_eval_count", 0)
    output_tokens = usage.get("completion_tokens", 0) or meta.get("eval_count", 0)

    prompt_time = meta.get("prompt_eval_duration", 0) / 1e9 if "prompt_eval_duration" in meta else 0.0
    generation_time = meta.get("eval_duration", 0) / 1e9 if "eval_duration" in meta else 0.0

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "prompt_time": prompt_time,
        "generation_time": generation_time,
    }
