"""Human assistance tool for LangGraph agents.

Allows agents to interrupt execution and request human input or assistance.
"""

from langchain_core.tools import tool
from langgraph.types import interrupt


@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human during agent execution.
    
    Interrupts the graph execution and pauses for human input.
    The human response is returned to the agent for further processing.
    
    Args:
        query: The question or request for human assistance.
    
    Returns:
        The human's response to the query.
    
    Raises:
        KeyError: If the interrupt response doesn't contain 'data' key.
    """
    try:
        human_response = interrupt({"query": query})
        return human_response["data"]
    except (KeyError, TypeError) as e:
        raise ValueError(f"Invalid human response format: {e}")
