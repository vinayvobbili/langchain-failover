"""The LangGraph example runs end to end against scripted hosts, with no network."""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import ToolMessage  # noqa: E402

from langchain_failover import FailoverChatModel  # noqa: E402

_PATH = Path(__file__).resolve().parent.parent / "examples" / "langgraph_agent.py"
_spec = importlib.util.spec_from_file_location("langgraph_agent_example", _PATH)
example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(example)


def _run(llm, question="How many widgets does Initech have?"):
    return example.build_graph(llm).invoke({"messages": [("user", question)]})["messages"]


def test_graph_runs_tool_loop_on_healthy_primary():
    llm = FailoverChatModel(
        primary=example.ScriptedHost(name="primary"),
        secondary=example.ScriptedHost(name="backup"),
    )
    messages = _run(llm)
    assert any(isinstance(m, ToolMessage) and "120 units" in m.content for m in messages)
    assert messages[-1].content == "[primary] 120 units of widget in stock"


def test_graph_survives_primary_outage(caplog):
    llm = FailoverChatModel(
        primary=example.ScriptedHost(name="primary", down=True),
        secondary=example.ScriptedHost(name="backup", item="sprocket"),
    )
    messages = _run(llm)
    assert messages[-1].content == "[backup] 37 units of sprocket in stock"
    # bind_tools returns a new FailoverChatModel, so the switch shows in the log, not llm.active.
    assert "switching to secondary" in caplog.text
