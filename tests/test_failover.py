"""Tests for FailoverChatModel — no network required.

Uses tiny fake chat models that either answer or raise, so the failover,
recovery, streaming, and bind_tools behaviour can be exercised deterministically.
"""
from typing import Any, List, Optional

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.messages import AIMessageChunk
from pydantic import ConfigDict

from langchain_failover import FailoverChatModel, is_connection_error


class _FakeChat(BaseChatModel):
    """Answers with a fixed reply, or raises a chosen exception on every call."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    reply: str = "ok"
    raises: Any = None
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(
        self,
        messages,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        object.__setattr__(self, "calls", self.calls + 1)
        if self.raises is not None:
            raise self.raises
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        object.__setattr__(self, "calls", self.calls + 1)
        if self.raises is not None:
            raise self.raises
        yield ChatGenerationChunk(message=AIMessageChunk(content=self.reply))

    def bind_tools(self, tools, **kwargs):
        # Mirror the reply so a bound model is still identifiable in tests.
        return _FakeChat(reply=f"bound:{self.reply}", raises=self.raises)


def test_primary_serves_when_healthy():
    llm = FailoverChatModel(primary=_FakeChat(reply="primary"), secondary=_FakeChat(reply="secondary"))
    assert llm.invoke("hi").content == "primary"
    assert llm.active == "primary"


def test_fails_over_on_connection_error():
    primary = _FakeChat(raises=ConnectionError("refused"))
    llm = FailoverChatModel(primary=primary, secondary=_FakeChat(reply="secondary"))
    assert llm.invoke("hi").content == "secondary"
    assert llm.active == "secondary"


def test_non_connection_error_propagates():
    primary = _FakeChat(raises=ValueError("bad prompt"))
    llm = FailoverChatModel(primary=primary, secondary=_FakeChat(reply="secondary"))
    with pytest.raises(ValueError):
        llm.invoke("hi")


def test_recovers_back_to_primary():
    primary = _FakeChat(raises=ConnectionError("down"))
    secondary = _FakeChat(reply="secondary")
    llm = FailoverChatModel(primary=primary, secondary=secondary)
    assert llm.invoke("hi").content == "secondary"
    assert llm.active == "secondary"
    # Primary heals.
    object.__setattr__(primary, "raises", None)
    object.__setattr__(primary, "reply", "primary-back")
    assert llm.invoke("hi").content == "primary-back"
    assert llm.active == "primary"


def test_streaming_fails_over():
    primary = _FakeChat(raises=ConnectionError("refused"))
    llm = FailoverChatModel(primary=primary, secondary=_FakeChat(reply="streamed"))
    chunks = list(llm.stream("hi"))
    assert "".join(c.content for c in chunks) == "streamed"


def test_bind_tools_preserved_on_both_legs():
    llm = FailoverChatModel(primary=_FakeChat(reply="p"), secondary=_FakeChat(reply="s"))
    bound = llm.bind_tools([])
    assert isinstance(bound, FailoverChatModel)
    assert bound.invoke("hi").content == "bound:p"


def test_is_connection_error_walks_cause_chain():
    inner = ConnectionRefusedError("nope")
    outer = RuntimeError("wrapper")
    outer.__cause__ = inner
    assert is_connection_error(outer)
    assert not is_connection_error(ValueError("totally unrelated"))


class _ToolAwareChat(BaseChatModel):
    """Echoes how many tools actually reached ``_generate`` as a kwarg.

    ``bind_tools`` returns a ``RunnableBinding`` (via ``Runnable.bind``), exactly
    like ``ChatOpenAI`` does — the bound tools are applied only inside
    ``invoke()``. A wrapper that delegates to the inner model's ``_generate()``
    directly drops them; one that delegates via ``invoke()`` preserves them.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return "toolaware"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        n = len(kwargs.get("tools") or [])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"tools={n}"))])

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=list(tools))


def test_bound_tools_actually_reach_the_model():
    # Regression for the 0.1.0 bug: FailoverChatModel delegated to the inner
    # model's _generate() directly, which bypassed the RunnableBinding kwargs and
    # silently dropped the bound tools (and thus tool_calls). The wrapper must
    # delegate via invoke() so the tools actually reach the model.
    base = _ToolAwareChat()
    llm = FailoverChatModel(primary=base, secondary=base)
    bound = llm.bind_tools([{"name": "a"}, {"name": "b"}])
    assert isinstance(bound, FailoverChatModel)
    assert bound.invoke("hi").content == "tools=2"
