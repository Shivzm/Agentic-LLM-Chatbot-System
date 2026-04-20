"""Simple Chatbot Agent using LangGraph.

This agent demonstrates basic chatbot functionality with a single LLM node.
"""

from src.agents.base import AgentState, get_llm, get_chatbot_model
from langgraph.graph import StateGraph, START, END
from config.settings import settings


def build_chatbot_graph():
    """Build and compile the basic chatbot graph.
    
    Returns:
        Compiled LangGraph with a single LLM node.
    """
    llm = get_llm(get_chatbot_model())

    def chatbot(state: AgentState):
        """Node function that invokes the LLM with messages."""
        return {"messages": [llm.invoke(state["messages"])]}

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("llmchatbot", chatbot)
    graph_builder.add_edge(START, "llmchatbot")
    graph_builder.add_edge("llmchatbot", END)

    return graph_builder.compile()


# Convenience export for direct usage
graph = build_chatbot_graph()


if __name__ == "__main__":
    # Demo/test code
    from langchain_core.messages import HumanMessage

    response = graph.invoke({"messages": [HumanMessage(content="Hello!")]})
    print(response["messages"][-1].content)