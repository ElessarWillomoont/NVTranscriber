from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    HF_TOKEN: Optional[str] = None
    DATABASE_URL: str = "sqlite:///./transcriber.db"
    
    class Config:
        env_file = ".env"

settings = Settings()
