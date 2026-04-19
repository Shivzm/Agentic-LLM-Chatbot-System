from src.agents.react_agent import build_react_agent

# tests/test_react_agent.py
def test_react_agent_builds():
    graph = build_react_agent()
    assert graph is not None

def test_react_agent_responds():
    graph = build_react_agent()
    config = {"configurable": {"thread_id": "test"}}
    result = graph.invoke({"messages": "What is 2 + 2?"}, config=config)
    assert len(result["messages"]) > 0