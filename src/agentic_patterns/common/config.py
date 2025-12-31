from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env from project root
# __file__ = .../src/agentic_patterns/common/config.py
# Go up 3 levels: common -> agentic_patterns -> src -> project_root
project_root = Path(__file__).parent.parent.parent.parent
load_dotenv(project_root / '.env')

class Settings(BaseSettings):
    OPENAI_API_KEY: str
    ANTHROPIC_API_KEY: str
    GOOGLE_API_KEY: str | None = None

    # Optional settings
    LOG_LEVEL: str = "INFO"
    RESULTS_DIR: Path = project_root / "experiments" / "results"

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        extra = 'ignore'  # Allow extra fields in .env

settings = Settings()