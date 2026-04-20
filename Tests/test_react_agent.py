from src.agents import build_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.runnables.config import RunnableConfig

def test_react_agent_builds():
    graph = build_react_agent()
    assert graph is not None

def test_react_agent_responds():
    graph = build_react_agent()
    config: RunnableConfig = {"configurable": {"thread_id": "test"}}
    result = graph.invoke({"messages": [HumanMessage(content="What is 2 + 2?")]}, config=config)
    assert len(result["messages"]) > 0