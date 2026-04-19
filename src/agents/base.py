from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from config.settings import settings

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def get_llm(model: str = ""):
    return init_chat_model(model or settings.default_model)