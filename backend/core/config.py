from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/knowflow"

    class Config:
        env_file = ".env"

settings = Settings()
