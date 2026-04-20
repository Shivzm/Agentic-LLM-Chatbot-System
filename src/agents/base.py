"""Base configurations and utilities for LangGraph agents.

This module provides shared utilities and base classes used across all agents.
"""

import os
from typing import Annotated

from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver

from config.settings import settings


class AgentState(TypedDict):
    """Base state for all agents.
    
    Attributes:
        messages: List of messages with automatic appending behavior
    """
    messages: Annotated[list[BaseMessage], add_messages]


def get_llm(model: str = ""):
    """Initialize and return a language model.
    
    Args:
        model: Model identifier (e.g., "groq:llama-3.3-70b-versatile").
               If empty, uses default_model from settings.
    
    Returns:
        Initialized chat model instance.
    """
    return init_chat_model(model or settings.default_model)


def get_memory_checkpointer():
    """Get a memory-based checkpointer for graph state persistence.
    
    Returns:
        MemorySaver instance for thread-based state management.
    """
    return MemorySaver()


def get_chatbot_model():
    """Get the configured chatbot model from settings.
    
    Returns:
        Model identifier string for chatbot (e.g., "groq:llama-3.1-8b-instant").
    """
    return settings.chatbot_model


def get_react_model():
    """Get the configured react agent model from settings.
    
    Returns:
        Model identifier string for react agent.
    """
    return settings.react_model