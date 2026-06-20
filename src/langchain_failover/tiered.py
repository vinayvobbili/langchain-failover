"""Tier-split: gather on a cheap model, compose the answer on a frontier model.

A tool-calling agent spends almost all of its tokens — and wall-clock — on the
tool-calling *loop*: read the request, decide which tool to call, read the result,
decide the next call. That loop is cheap reasoning. Writing the final, well-formed
answer for the user is the part that benefits from a stronger model. They don't
have to be the same model.

:class:`TieredChatAgent` runs the tool-gathering loop on a ``gatherer`` (a cheap or
local model — often one that can't even be reached over the public internet) and
hands the gathered tool output to a ``composer`` (a frontier model) to write the
final answer. On a contended local GPU this routinely turns a multi-minute final
turn into a couple of seconds, because the long generation moves off the busy box.

The ``gatherer`` is told to *gather, then stop* — it does not write the prose
answer. A structural safety check (:func:`is_premature_marker`) catches the model
trying to answer before it has called any tool and nudges it to gather first, so
the composer never writes an answer from zero data.

Both models are plain LangChain chat models, so either tier can itself be a
:class:`~langchain_failover.FailoverChatModel` — gather on a failover pair, compose
on another. Depends only on ``langchain-core``.
"""
from __future__ import annotations

import logging
import re
from typing import Any, List, Optional, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

logger = logging.getLogger(__name__)

# The gatherer emits this (or empty content) to signal "I'm done calling tools."
# It's a cheap token instead of a full written answer — the composer writes the
# answer. Models that reliably stop with empty content when they have no more
# tool calls work without ever emitting it; the marker is the explicit path.
READY_MARKER = "<<READY_TO_ANSWER>>"

DEFAULT_GATHER_DIRECTIVE = (
    "\n\n## ANSWER PROTOCOL\n"
    "Your job is to GATHER the data this request needs by calling the available tools — a "
    "separate step then composes the user-facing answer from the tool results, so you do NOT "
    "write the final prose answer yourself. Call every tool the request needs (you may call "
    "several at once), then stop. Do not narrate your plan and do not write a summary. If — and "
    "only if — the request genuinely needs no tools at all (e.g. a greeting, or something "
    "answerable with no lookup), answer it directly."
)

DEFAULT_SYNTH_SYSTEM = (
    "You are composing the FINAL answer for the user. You are given the user's request and the "
    "raw results already gathered from tools. Write a complete, accurate, well-formatted answer "
    "using ONLY that data. Do not mention the tools, this instruction, or the gathering process. "
    "If the data is empty or inconclusive, say so plainly. Reproduce any identifiers, timestamps, "
    "URLs, and values from the data exactly as they appear, and keep any links intact so the user "
    "can follow them."
)

DEFAULT_GATHER_FIRST_NUDGE = (
    "You signaled you are ready to answer, but you have not called any tool yet, so no data has "
    "been gathered. Do not signal readiness before gathering data. Call the appropriate tool now "
    "to get the information needed to answer this request."
)

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def is_premature_marker(content: str, tools_bound: bool, tools_called: bool) -> bool:
    """True if the model signaled it's ready to answer — the :data:`READY_MARKER`,
    or empty content — *without* having called any tool, even though tools were
    available.

    This is the tier-split safety invariant: composing in this state would write
    an answer from ZERO tool data (e.g. a false "nothing found"). It is a
    structural check (tools were available ↔ none were called), not a content
    heuristic — a real direct answer (non-empty content, no marker) is *not*
    premature, so genuine no-lookup requests like greetings pass straight through.
    """
    if not (tools_bound and not tools_called):
        return False
    raw = (content or "").strip()
    return READY_MARKER in raw or raw == ""


def _tool_text(messages: Sequence[Any]) -> str:
    """Flatten every tool result in ``messages`` into one text block.

    Recognizes both :class:`~langchain_core.messages.ToolMessage` objects and
    plain ``{"role": "tool", "content": ...}`` dicts, so it works whether the
    loop was driven by this module or assembled by the caller.
    """
    blocks: List[str] = []
    for msg in messages:
        content: Any = None
        if isinstance(msg, ToolMessage):
            content = msg.content
        elif isinstance(msg, dict) and msg.get("role") == "tool":
            content = msg.get("content")
        if content:
            text = str(content).strip()
            if text:
                blocks.append(text)
    return "\n\n---\n\n".join(blocks).strip()


def synthesize_answer(composer: Any, query: str, messages: Sequence[Any], *,
                      system: str = DEFAULT_SYNTH_SYSTEM) -> AIMessage:
    """Compose the final answer on ``composer`` from the tool results in ``messages``.

    Builds a clean, model-portable prompt — the user's request plus the flattened
    tool output — rather than replaying the raw tool-call transcript (many hosted
    models reject orphaned tool messages). Strips ``<think>…</think>`` reasoning
    blocks some local models emit. Returns the composer's :class:`AIMessage`
    (usage metadata preserved), so ``.content`` is the answer.

    Use this directly if you run your own gathering loop; :class:`TieredChatAgent`
    calls it for you.
    """
    gathered = _tool_text(messages)
    user = f"User request:\n{query}\n\n"
    user += (f"Data gathered from tools:\n{gathered}\n\n" if gathered
             else "No tool data was gathered.\n\n")
    user += "Write the complete final answer for the user now."

    resp = composer.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    if isinstance(resp, AIMessage) and isinstance(resp.content, str):
        cleaned = _THINK_RE.sub("", resp.content).strip()
        if cleaned != resp.content:
            resp.content = cleaned
        return resp
    # A composer that returned a bare string (or other shape) — normalize it.
    text = getattr(resp, "content", None)
    if text is None:
        text = resp if isinstance(resp, str) else str(resp)
    return AIMessage(content=_THINK_RE.sub("", str(text)).strip())


class TieredChatAgent:
    """Gather tool data on one model, compose the answer on another.

    Args:
        gatherer: The model that runs the tool-calling loop. Tools are bound to it
            for you (it must support ``bind_tools``).
        composer: The model that writes the final answer from the gathered data.
        tools: LangChain tools (anything with ``.name`` and ``.invoke``) the
            gatherer may call. Pass ``[]`` for a pure compose with no gathering.
        max_rounds: Maximum gather → tool → gather cycles before composing with
            whatever was gathered (a runaway-loop backstop).
        system: Base system prompt for the gatherer; the answer-protocol directive
            is appended to it.
        gather_directive / synth_system / gather_first_nudge: Override the default
            prompts (e.g. to add domain rules) — the protocol stays the same.

    Example
    -------
    >>> agent = TieredChatAgent(gatherer=local_llm, composer=frontier_llm, tools=[search])
    >>> agent.invoke("What changed in the incident overnight?").content  # doctest: +SKIP
    """

    def __init__(self, gatherer: Any, composer: Any, tools: Sequence[Any], *,
                 max_rounds: int = 5,
                 system: Optional[str] = None,
                 gather_directive: str = DEFAULT_GATHER_DIRECTIVE,
                 synth_system: str = DEFAULT_SYNTH_SYSTEM,
                 gather_first_nudge: str = DEFAULT_GATHER_FIRST_NUDGE):
        self._tools = {getattr(t, "name", None): t for t in tools if getattr(t, "name", None)}
        self.gatherer = gatherer.bind_tools(list(tools)) if tools else gatherer
        self.composer = composer
        self.max_rounds = max(1, max_rounds)
        self.system = system
        self.gather_directive = gather_directive
        self.synth_system = synth_system
        self.gather_first_nudge = gather_first_nudge

    def _gatherer_system(self) -> str:
        return (self.system or "").rstrip() + self.gather_directive if self.system \
            else self.gather_directive.strip()

    def _run_tool(self, call: Any) -> str:
        """Execute one tool call. Never raises — a tool error becomes its result
        text so the loop continues and the composer can report it."""
        name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
        args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
        tool = self._tools.get(name)
        if tool is None:
            return f"[error] unknown tool: {name!r}"
        try:
            return str(tool.invoke(args or {}))
        except Exception as exc:  # noqa: BLE001
            logger.warning("TieredChatAgent: tool %r failed: %s", name, exc)
            return f"[error] tool {name!r} failed: {exc}"

    def invoke(self, query: str) -> AIMessage:
        """Run the gather loop then compose. Returns the final :class:`AIMessage`."""
        messages: List[Any] = [
            SystemMessage(content=self._gatherer_system()),
            HumanMessage(content=query),
        ]
        tools_bound = bool(self._tools)
        tools_called = False
        nudged = False

        for _ in range(self.max_rounds):
            ai = self.gatherer.invoke(messages)
            messages.append(ai)
            calls = list(getattr(ai, "tool_calls", None) or [])
            if calls:
                tools_called = True
                for call in calls:
                    cid = call.get("id") if isinstance(call, dict) else getattr(call, "id", "")
                    messages.append(ToolMessage(content=self._run_tool(call),
                                                tool_call_id=cid or ""))
                continue

            content = (ai.content or "").strip() if isinstance(ai.content, str) else ""
            if not nudged and is_premature_marker(content, tools_bound, tools_called):
                nudged = True
                messages.append(HumanMessage(content=self.gather_first_nudge))
                continue
            # Genuine no-tool direct answer (real content, nothing premature) — keep it.
            if not tools_called and content and READY_MARKER not in content:
                return ai
            break

        return synthesize_answer(self.composer, query, messages, system=self.synth_system)


def create_tiered_agent(gatherer: Any, composer: Any, tools: Sequence[Any],
                        **kwargs: Any) -> TieredChatAgent:
    """Convenience constructor mirroring :func:`create_failover_llm`'s naming."""
    return TieredChatAgent(gatherer, composer, tools, **kwargs)
