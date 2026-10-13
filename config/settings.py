"""Application configuration loaded from environment variables."""
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root directory
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    """Application settings schema."""
    gemini_api_key: str
    gemini_model: str
    llm_temperature: float
    max_image_size_mb: int
    max_history_messages: int

    @property
    def max_image_size_bytes(self) -> int:
        """Return max image size in bytes."""
        return self.max_image_size_mb * 1024 * 1024

    @property
    def has_api_key(self) -> bool:
        """Check if an API key is set."""
        return bool(self.gemini_api_key and self.gemini_api_key.strip() and self.gemini_api_key != "your_gemini_api_key_here")


def get_settings() -> Settings:
    """Retrieve settings from environment variables with sensible defaults."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip() or "gemini-3.8-flash"

    temperature_value = os.getenv("LLM_TEMPERATURE")
    if temperature_value is None:
        temperature_value = os.getenv("GEMINI_TEMPERATURE", "0.1")
    try:
        temperature = float(temperature_value)
    except ValueError as exc:
        raise ValueError("LLM_TEMPERATURE must be a number between 0 and 2.") from exc
    if not 0 <= temperature <= 2:
        raise ValueError("LLM_TEMPERATURE must be between 0 and 2.")
    
    try:
        max_size_mb = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
    except ValueError:
        max_size_mb = 10

    try:
        max_history = int(os.getenv("MAX_HISTORY_MESSAGES", "12"))
    except ValueError:
        max_history = 12

    return Settings(
        gemini_api_key=api_key,
        gemini_model=model,
        llm_temperature=temperature,
        max_image_size_mb=max_size_mb,
        max_history_messages=max_history,
    )
