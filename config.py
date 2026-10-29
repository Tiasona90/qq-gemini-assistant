"""
Configuration management for the QQ Gemini Assistant.
Loads settings from environment variables and optional .env file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Set
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load .env if present in current working directory or script directory
env_path = Path(".env")
if env_path.exists():
    load_dotenv(dotenv_path=env_path)


class Settings(BaseModel):
    # Gemini API Settings
    gemini_api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.8-flash"))
    default_thinking_level: str = Field(
        default_factory=lambda: os.getenv("DEFAULT_THINKING_LEVEL", "LOW").upper()
    )
    default_context_messages: int = Field(
        default_factory=lambda: int(os.getenv("DEFAULT_CONTEXT_MESSAGES", "4"))
    )
    max_output_tokens: int = Field(
        default_factory=lambda: int(os.getenv("MAX_OUTPUT_TOKENS", "2048"))
    )
    enable_search_default: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_SEARCH_DEFAULT", "true").lower() in ("true", "1", "yes")
    )
    gemini_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("GEMINI_TIMEOUT_SECONDS", "45.0"))
    )
    gemini_max_retries: int = Field(
        default_factory=lambda: int(os.getenv("GEMINI_MAX_RETRIES", "3"))
    )

    # Dynamic Session Memory & Context Compression Settings
    session_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("SESSION_TIMEOUT_SECONDS", "600.0"))
    )
    session_compression_threshold_rounds: int = Field(
        default_factory=lambda: int(os.getenv("SESSION_COMPRESSION_THRESHOLD_ROUNDS", "10"))
    )
    session_keep_recent_rounds: int = Field(
        default_factory=lambda: int(os.getenv("SESSION_KEEP_RECENT_ROUNDS", "4"))
    )

    # QQ Whitelist & Access Control
    # Stores QQ IDs as strings to handle arbitrary lengths cleanly
    qq_whitelist_raw: str = Field(default_factory=lambda: os.getenv("QQ_WHITELIST", ""))

    # OneBot 11 Adapter Settings
    onebot_mode: str = Field(
        default_factory=lambda: os.getenv("ONEBOT_MODE", "reverse_ws").lower()
    )
    reverse_ws_host: str = Field(
        default_factory=lambda: os.getenv("REVERSE_WS_HOST", "127.0.0.1")
    )
    reverse_ws_port: int = Field(
        default_factory=lambda: int(os.getenv("REVERSE_WS_PORT", "8080"))
    )
    reverse_ws_path: str = Field(
        default_factory=lambda: os.getenv("REVERSE_WS_PATH", "/onebot/v11/ws")
    )
    forward_ws_url: str = Field(
        default_factory=lambda: os.getenv("FORWARD_WS_URL", "ws://127.0.0.1:3001")
    )
    onebot_access_token: str = Field(
        default_factory=lambda: os.getenv("ONEBOT_ACCESS_TOKEN", "")
    )

    # Concurrency and Queue Limits
    max_gemini_concurrency: int = Field(
        default_factory=lambda: int(os.getenv("MAX_GEMINI_CONCURRENCY", "2"))
    )
    user_request_timeout: float = Field(
        default_factory=lambda: float(os.getenv("USER_REQUEST_TIMEOUT", "60.0"))
    )

    # Anti-Detection Random Command Jitter (3.0 ~ 6.0s by default)
    command_jitter_min_seconds: float = Field(
        default_factory=lambda: float(os.getenv("COMMAND_JITTER_MIN_SECONDS", "3.0"))
    )
    command_jitter_max_seconds: float = Field(
        default_factory=lambda: float(os.getenv("COMMAND_JITTER_MAX_SECONDS", "6.0"))
    )

    # Persistence
    sqlite_db_path: str = Field(
        default_factory=lambda: os.getenv("SQLITE_DB_PATH", "data/assistant.db")
    )
    log_level: str = Field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper()
    )

    @property
    def qq_whitelist(self) -> Set[str]:
        """Parsed set of permitted QQ IDs."""
        if not self.qq_whitelist_raw.strip():
            return set()
        return {
            part.strip()
            for part in self.qq_whitelist_raw.split(",")
            if part.strip()
        }

    def is_user_allowed(self, user_id: str | int) -> bool:
        """Check if a QQ user is present in the whitelist."""
        uid = str(user_id).strip()
        # If whitelist is empty, reject all users for security
        whitelist = self.qq_whitelist
        if not whitelist:
            return False
        return uid in whitelist


def get_settings() -> Settings:
    """Return a fresh Settings instance."""
    return Settings()
