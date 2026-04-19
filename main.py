import argparse
from src.agents.react_agent import build_react_agent
from langgraph.checkpoint.memory import MemorySaver

def main():
    parser = argparse.ArgumentParser(description="LangGraph Agent CLI")
    parser.add_argument("--agent", choices=["react", "chatbot", "multi"], 
                        default="react")
    parser.add_argument("--thread-id", default="1")
    args = parser.parse_args()

    memory = MemorySaver()
    graph = build_react_agent(checkpointer=memory)
    config = {"configurable": {"thread_id": args.thread_id}}

    print(f"Running {args.agent} agent. Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        response = graph.invoke(
            {"messages": user_input}, config=config
        )
        print(f"Agent: {response['messages'][-1].content}\n")

if __name__ == "__main__":
    main()