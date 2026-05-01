from pydantic_settings import BaseSettings

from pydantic_settings import BaseSettings
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")  # ← ignores unknown env vars

    default_model:  str = "groq:llama-3.3-70b-versatile"
    chatbot_model: str = "groq:llama-3.1-8b-instant"  # NEW
    react_model: str = "groq:llama-3.3-70b-versatile"  # NEW
    
    class Config:
        env_file = ".env"

settings = Settings()