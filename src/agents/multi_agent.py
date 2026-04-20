"""Multi-Agent Workflow using LangGraph.

Implements a researcher-writer agent architecture where researchers find information
and writers summarize findings.
"""

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode

from src.agents.base import get_llm, get_memory_checkpointer
from config.settings import settings
from tools import search_web, write_summary


def build_multi_agent_graph(model: str | None = None, checkpointer=None):
    """Build and compile the multi-agent workflow.
    
    Args:
        model: Optional model override. Uses settings.default_model if None.
        checkpointer: Optional state persister for thread management.
    
    Returns:
        Compiled LangGraph with researcher and writer agents.
    """
    if model is None:
        model = settings.default_model
    
    llm = get_llm(model)

    def researcher_agent(state: MessagesState):
        """Researcher agent that searches for information."""
        messages = state["messages"]
        system_msg = SystemMessage(
            content="You are a research assistant. Use the search_web tool to find information about the user's request"
        )
        researcher_llm = llm.bind_tools([search_web])
        response = researcher_llm.invoke([system_msg] + messages)
        return {"messages": [response]}

    def writer_agent(state: MessagesState):
        """Writer agent that creates summaries."""
        messages = state["messages"]
        system_msg = SystemMessage(
            content="You are a technical writer. Review the conversation and create a clear, concise summary of the findings."
        )
        response = llm.invoke([system_msg] + messages)
        return {"messages": [response]}

    # Build graph
    workflow = StateGraph(MessagesState)
    
    # Add nodes
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("writer", writer_agent)
    
    # Define flow
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", END)
    
    if checkpointer is None:
        checkpointer = get_memory_checkpointer()
    
    return workflow.compile(checkpointer=checkpointer)


# Convenience export
graph = build_multi_agent_graph()


if __name__ == "__main__":
    # Demo
    from langchain_core.messages import HumanMessage
    
    graph = build_multi_agent_graph()
    response = graph.invoke({"messages": [HumanMessage(content="Research about the usecase of agentic ai in business")]})
    print(response["messages"][-1].content)