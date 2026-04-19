from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import START, END
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition


load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY") # type: ignore
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY") # type: ignore
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "Chat-Bot"

from langchain.chat_models import init_chat_model
llm=init_chat_model("groq:llama-3.3-70b-versatile")

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
def make_tool_graph():    
    ##Graph with tool call
    from langchain_core.tools import tool

    @tool
    def add(a:float, b:float):
        """Add two numbers"""
        return a+b

    tools = [add]
    tool_node = ToolNode([add])

    llm_with_tools = llm.bind_tools([add]) # bind_tools is the new method for providing tools to the llm and had replaced llm_functions.

    def call_llm_model(state:State):
        return{"messages":[llm_with_tools.invoke(state['messages'])]}    
    
    ##Graph
    builder = StateGraph(State)
    builder.add_node("tool_calling_llm", call_llm_model)  # Add the function reference
    builder.add_node("tools", ToolNode(tools))

    ##Add Edges
    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges(
        "tool_calling_llm",
        tools_condition
    )
    builder.add_edge("tools", "tool_calling_llm")

    ##Compile the graph
    graph = builder.compile()
    return graph

tool_agent = make_tool_graph()