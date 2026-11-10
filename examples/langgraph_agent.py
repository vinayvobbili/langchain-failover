"""A two-node LangGraph agent whose chat model is a FailoverChatModel.

The graph is the classic loop: an ``agent`` node calls the model, a ``tools`` node
runs whatever tool calls came back, and the edge between them repeats until the
model answers in plain text. Run it as is, with no servers: the two "hosts" are
scripted stand-ins, and the primary is down, so the backup serves every turn.

    pip install "langchain-failover[langgraph]"
    python examples/langgraph_agent.py

To use real hosts, build ``llm`` from two ChatOpenAI endpoints instead (see
``main``) and drop the scripted classes.
"""
from typing import Annotated, Any, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import ConfigDict
from typing_extensions import TypedDict

from langchain_failover import FailoverChatModel

STOCK = {"widget": 120, "gasket": 0, "sprocket": 37}  # made-up Initech inventory


@tool
def check_stock(item: str) -> str:
    """Return how many units of an item Initech has in the warehouse."""
    count = STOCK.get(item.lower())
    return f"unknown item: {item}" if count is None else f"{count} units of {item} in stock"


class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


def build_graph(llm: BaseChatModel):
    """Agent node (model with tools bound) plus a tool node, looped until no tool calls."""
    model = llm.bind_tools([check_stock])  # a FailoverChatModel binds tools on both legs

    def agent(state: State) -> dict:
        return {"messages": [model.invoke(state["messages"])]}

    graph = StateGraph(State)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode([check_stock]))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


class ScriptedHost(BaseChatModel):
    """Offline stand-in for a model host: asks for the tool, then answers from its result.

    With ``down=True`` it behaves like an unreachable host and raises ConnectionError.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "host"
    item: str = "widget"
    down: bool = False

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop: Optional[List[str]] = None, run_manager=None,
                  **kwargs: Any) -> ChatResult:
        if self.down:
            raise ConnectionError(f"{self.name} is unreachable")
        last = messages[-1]
        if isinstance(last, ToolMessage):
            msg = AIMessage(content=f"[{self.name}] {last.content}")
        else:
            msg = AIMessage(
                content="",
                tool_calls=[{"name": "check_stock", "args": {"item": self.item}, "id": "call_1"}],
            )
        return ChatResult(generations=[ChatGeneration(message=msg)])


def main() -> None:
    llm = FailoverChatModel(
        primary=ScriptedHost(name="primary", down=True),
        secondary=ScriptedHost(name="backup"),
    )
    # With real hosts:
    #   from langchain_openai import ChatOpenAI
    #   llm = FailoverChatModel(
    #       primary=ChatOpenAI(base_url="http://gpu-box:8001/v1", api_key="x", model="local"),
    #       secondary=ChatOpenAI(base_url="http://cpu-box:8002/v1", api_key="x", model="local"),
    #   )
    app = build_graph(llm)
    result = app.invoke({"messages": [("user", "How many widgets does Initech have?")]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
