"""Tests for TieredChatAgent and the synthesis helpers — no network required.

Scripted fake chat models drive the gather loop deterministically: the gatherer
returns a fixed sequence of AIMessages (with or without tool_calls), the composer
captures what it was asked to compose so we can assert the gathered tool data
actually reached it.
"""
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict

from langchain_failover import (
    READY_MARKER,
    TieredChatAgent,
    is_premature_marker,
    synthesize_answer,
)


def _tc(name: str, args: dict, cid: str = "call_1") -> dict:
    return {"name": name, "args": args, "id": cid, "type": "tool_call"}


class ScriptedChat(BaseChatModel):
    """Returns a fixed sequence of AIMessages on successive invocations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    responses: Any = None       # list[AIMessage]
    idx: int = 0
    bound: bool = False

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None,
                  **kwargs: Any) -> ChatResult:
        i = min(self.idx, len(self.responses) - 1)
        object.__setattr__(self, "idx", self.idx + 1)
        return ChatResult(generations=[ChatGeneration(message=self.responses[i])])

    def bind_tools(self, tools, **kwargs):
        object.__setattr__(self, "bound", True)
        return self  # same object so the scripted sequence continues


class CapturingChat(BaseChatModel):
    """Records the last human prompt it was asked to compose, returns a fixed reply."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    reply: str = "COMPOSED ANSWER"
    last_user: str = ""
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "capturing"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        object.__setattr__(self, "calls", self.calls + 1)
        humans = [m for m in messages if isinstance(m, HumanMessage)]
        object.__setattr__(self, "last_user", humans[-1].content if humans else "")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])


class FakeTool:
    def __init__(self, name: str, result: str = "", raises: Exception = None):
        self.name = name
        self._result = result
        self._raises = raises

    def invoke(self, args):
        if self._raises is not None:
            raise self._raises
        return self._result


# --- is_premature_marker ----------------------------------------------------

def test_premature_marker_true_on_marker_without_tools():
    assert is_premature_marker(READY_MARKER, tools_bound=True, tools_called=False)


def test_premature_marker_true_on_empty_without_tools():
    assert is_premature_marker("", tools_bound=True, tools_called=False)


def test_premature_marker_false_for_real_direct_answer():
    assert not is_premature_marker("hello there", tools_bound=True, tools_called=False)


def test_premature_marker_false_once_a_tool_ran():
    assert not is_premature_marker(READY_MARKER, tools_bound=True, tools_called=True)


def test_premature_marker_false_when_no_tools_bound():
    assert not is_premature_marker("", tools_bound=False, tools_called=False)


# --- TieredChatAgent.invoke -------------------------------------------------

def test_gather_then_compose():
    gatherer = ScriptedChat(responses=[
        AIMessage(content="", tool_calls=[_tc("search", {"q": "incident"})]),
        AIMessage(content=READY_MARKER),
    ])
    composer = CapturingChat()
    tool = FakeTool("search", result="RESULT-42")
    agent = TieredChatAgent(gatherer=gatherer, composer=composer, tools=[tool])

    out = agent.invoke("what happened?")
    assert out.content == "COMPOSED ANSWER"
    assert composer.calls == 1
    assert "RESULT-42" in composer.last_user      # tool data reached the composer
    assert gatherer.bound                          # tools were bound to the gatherer


def test_direct_answer_skips_composer():
    # Real content, no tool calls, tools available but not needed → returned as-is.
    gatherer = ScriptedChat(responses=[AIMessage(content="Hello! How can I help?")])
    composer = CapturingChat()
    agent = TieredChatAgent(gatherer=gatherer, composer=composer,
                            tools=[FakeTool("search", "x")])
    out = agent.invoke("hi")
    assert out.content == "Hello! How can I help?"
    assert composer.calls == 0                     # composer never invoked


def test_premature_marker_nudges_then_gathers():
    gatherer = ScriptedChat(responses=[
        AIMessage(content=READY_MARKER),                                  # premature
        AIMessage(content="", tool_calls=[_tc("search", {"q": "x"})]),    # after nudge
        AIMessage(content=""),                                            # done
    ])
    composer = CapturingChat()
    tool = FakeTool("search", result="GATHERED")
    agent = TieredChatAgent(gatherer=gatherer, composer=composer, tools=[tool])

    out = agent.invoke("look it up")
    assert out.content == "COMPOSED ANSWER"
    assert "GATHERED" in composer.last_user
    assert gatherer.idx == 3                        # premature + nudge + done


def test_tool_error_is_captured_not_raised():
    gatherer = ScriptedChat(responses=[
        AIMessage(content="", tool_calls=[_tc("boom", {})]),
        AIMessage(content=READY_MARKER),
    ])
    composer = CapturingChat()
    tool = FakeTool("boom", raises=RuntimeError("kaboom"))
    agent = TieredChatAgent(gatherer=gatherer, composer=composer, tools=[tool])

    out = agent.invoke("call the broken tool")
    assert out.content == "COMPOSED ANSWER"
    assert "[error]" in composer.last_user and "kaboom" in composer.last_user


def test_unknown_tool_is_reported():
    gatherer = ScriptedChat(responses=[
        AIMessage(content="", tool_calls=[_tc("ghost", {})]),
        AIMessage(content=READY_MARKER),
    ])
    composer = CapturingChat()
    agent = TieredChatAgent(gatherer=gatherer, composer=composer,
                            tools=[FakeTool("search", "x")])
    agent.invoke("call a tool that isn't there")
    assert "unknown tool" in composer.last_user


def test_max_rounds_backstop_still_composes():
    # Gatherer never stops calling tools — the backstop must compose anyway.
    looping = [AIMessage(content="", tool_calls=[_tc("search", {"q": str(i)}, f"c{i}")])
               for i in range(10)]
    gatherer = ScriptedChat(responses=looping)
    composer = CapturingChat()
    agent = TieredChatAgent(gatherer=gatherer, composer=composer,
                            tools=[FakeTool("search", "loop")], max_rounds=3)
    out = agent.invoke("loop forever")
    assert out.content == "COMPOSED ANSWER"
    assert gatherer.idx == 3                         # stopped at max_rounds
    assert composer.calls == 1


# --- synthesize_answer ------------------------------------------------------

def test_synthesize_flattens_tool_messages():
    composer = CapturingChat()
    messages = [
        HumanMessage(content="q"),
        ToolMessage(content="ALPHA", tool_call_id="1"),
        ToolMessage(content="BETA", tool_call_id="2"),
    ]
    out = synthesize_answer(composer, "the question", messages)
    assert out.content == "COMPOSED ANSWER"
    assert "ALPHA" in composer.last_user and "BETA" in composer.last_user
    assert "the question" in composer.last_user


def test_synthesize_handles_tool_dicts():
    composer = CapturingChat()
    messages = [{"role": "tool", "content": "DICTDATA"}]
    synthesize_answer(composer, "q", messages)
    assert "DICTDATA" in composer.last_user


def test_synthesize_reports_no_data():
    composer = CapturingChat()
    synthesize_answer(composer, "q", [HumanMessage(content="q")])
    assert "No tool data" in composer.last_user


def test_synthesize_strips_think_blocks():
    composer = CapturingChat(reply="<think>let me reason</think>The real answer.")
    out = synthesize_answer(composer, "q", [ToolMessage(content="d", tool_call_id="1")])
    assert out.content == "The real answer."


def test_pure_compose_without_tools_returns_direct_answer():
    gatherer = ScriptedChat(responses=[AIMessage(content="no tools needed")])
    composer = CapturingChat()
    agent = TieredChatAgent(gatherer=gatherer, composer=composer, tools=[])
    out = agent.invoke("just answer")
    assert out.content == "no tools needed"
    assert not gatherer.bound                        # nothing to bind
