"""Human-in-the-Loop Agent with LangGraph.

Allows interrupting execution for human assistance in decision-making.
"""

from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command

from src.agents.base import AgentState, get_llm, get_memory_checkpointer
from config.settings import settings
from tools import search_web, human_assistance


def build_human_in_loop_graph(model: str | None = None, checkpointer=None):
    """Build and compile the human-in-loop agent graph.
    
    Args:
        model: Optional model override. Uses settings.default_model if None.
        checkpointer: Optional state persister. Uses MemorySaver if None.
    
    Returns:
        Compiled LangGraph with human interruption capability.
    """
    if model is None:
        model = settings.default_model
    
    llm = get_llm(model)
    llm_with_tools = llm.bind_tools([search_web, human_assistance])

    def chatbot(state: AgentState):
        """Chatbot node that invokes the LLM with tools."""
        message = llm_with_tools.invoke(state["messages"])
        return {"messages": [message]}

    # Graph construction
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("chatbot", chatbot)

    tool_node = ToolNode(tools=[search_web, human_assistance])
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    if checkpointer is None:
        checkpointer = get_memory_checkpointer()
    
    return graph_builder.compile(checkpointer=checkpointer)


# Convenience export
graph = build_human_in_loop_graph()


if __name__ == "__main__":
    # Demo/test code
    try:
        from IPython.display import Image, display
        display(Image(graph.get_graph().draw_mermaid_png()))
    except Exception:
        pass

    # Example usage
    user_input = "I need some expert guidance and assistance for building an AI agent. Could you request assistance for me?"
    config: RunnableConfig = {"configurable": {"thread_id": "1"}}

    events = graph.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config,
        stream_mode="values",
    )
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()

    # Human assistance response
    human_response = (
        "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
        " It's much more reliable and extensible than simple autonomous agents."
    )

    human_command = Command(resume={"data": human_response})

    events = graph.stream(human_command, config, stream_mode="values")
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()