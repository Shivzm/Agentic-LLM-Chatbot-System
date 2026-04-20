"""Custom utility tools for LangGraph agents.

Provides custom math and utility functions that can be used by agents.
"""

from langchain_core.tools import tool


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.
    
    Args:
        a: First integer to multiply
        b: Second integer to multiply
    
    Returns:
        Product of a and b
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add two numbers.
    
    Args:
        a: First integer to add
        b: Second integer to add
    
    Returns:
        Sum of a and b
    """
    return a + b


@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers.
    
    Args:
        a: First integer (minuend)
        b: Second integer (subtrahend)
    
    Returns:
        Difference of a and b
    """
    return a - b
