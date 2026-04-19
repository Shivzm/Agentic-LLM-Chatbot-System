from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from agents.base import AgentState, get_llm
from tools.search import search_tool
from tools.custom import multiply

def build_react_agent(checkpointer=None):
    llm = get_llm()
    tools = [search_tool, multiply]
    llm_with_tools = llm.bind_tools(tools)

    def call_model(state: AgentState):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=checkpointer)