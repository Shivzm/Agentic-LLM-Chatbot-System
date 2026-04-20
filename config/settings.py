from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    groq_api_key: str = ""
    tavily_api_key: str = ""
    langsmith_api_key: str = ""
    langsmith_tracing: bool = False
    langsmith_project: str = "langgraph-agents"
    
    # Model configurations
    default_model: str = "groq:llama-3.3-70b-versatile"
    chatbot_model: str = "groq:llama-3.1-8b-instant"  # NEW
    react_model: str = "groq:llama-3.3-70b-versatile"  # NEW
    
    class Config:
        env_file = ".env"

settings = Settings()