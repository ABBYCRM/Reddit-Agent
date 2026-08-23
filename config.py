"""
CaseClosedFL Reddit Agent - Configuration
Centralized settings with Pydantic Settings v2
"""
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # App
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    secret_key: str = Field(default="dev-secret", alias="SECRET_KEY")

    # Reddit API
    reddit_client_id: Optional[str] = Field(default=None, alias="REDDIT_CLIENT_ID")
    reddit_client_secret: Optional[str] = Field(default=None, alias="REDDIT_CLIENT_SECRET")
    reddit_user_agent: str = Field(
        default="caseclosedfl-agent/1.0 (by /u/caseclosedfl)",
        alias="REDDIT_USER_AGENT"
    )
    reddit_username: Optional[str] = Field(default=None, alias="REDDIT_USERNAME")
    reddit_password: Optional[str] = Field(default=None, alias="REDDIT_PASSWORD")
    reddit_max_qpm: int = Field(default=90, alias="REDDIT_MAX_QPM")
    reddit_burst_buffer: int = Field(default=5, alias="REDDIT_BURST_BUFFER")

    # NVIDIA NIM
    nvidia_api_key: Optional[str] = Field(default=None, alias="NVIDIA_API_KEY")
    nvidia_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        alias="NVIDIA_BASE_URL"
    )
    nvidia_model: str = Field(
        default="meta/llama-3.3-70b-instruct",
        alias="NVIDIA_MODEL"
    )
    nvidia_embedding_model: str = Field(
        default="nvidia/nv-embedqa-e5-v5",
        alias="NVIDIA_EMBEDDING_MODEL"
    )

    # Composio
    composio_api_key: Optional[str] = Field(default=None, alias="COMPOSIO_API_KEY")

    # Database
    database_url: str = Field(
        default="sqlite:///./data/caseclosed.db",
        alias="DATABASE_URL"
    )
    redis_url: Optional[str] = Field(default=None, alias="REDIS_URL")

    # ChromaDB
    chroma_persist_dir: str = Field(default="./chroma_db", alias="CHROMA_PERSIST_DIR")

    # Safety / Compliance
    max_daily_engagements: int = Field(default=50, alias="MAX_DAILY_ENGAGEMENTS")
    min_account_age_days: int = Field(default=14, alias="MIN_ACCOUNT_AGE_DAYS")
    enable_auto_reply: bool = Field(default=False, alias="ENABLE_AUTO_REPLY")
    enable_dm_outreach: bool = Field(default=False, alias="ENABLE_DM_OUTREACH")
    florida_bar_compliant: bool = Field(default=True, alias="FLORIDA_BAR_COMPLIANT")

    # Targeting
    target_states: str = Field(default="Florida", alias="TARGET_STATES")
    target_cities: str = Field(
        default="Miami,Orlando,Tampa,Jacksonville,Fort Lauderdale,West Palm Beach",
        alias="TARGET_CITIES"
    )
    target_subreddits: str = Field(
        default="legaladvice,florida,Miami,Orlando,tampa,insurance,personalfinance,caraccidents",
        alias="TARGET_SUBREDDITS"
    )

    # Lead Scoring
    lead_score_threshold: int = Field(default=75, alias="LEAD_SCORE_THRESHOLD")
    auto_qualify_threshold: int = Field(default=90, alias="AUTO_QUALIFY_THRESHOLD")

    # Scheduling
    discovery_interval_minutes: int = Field(default=30, alias="DISCOVERY_INTERVAL_MINUTES")
    monitor_interval_minutes: int = Field(default=15, alias="MONITOR_INTERVAL_MINUTES")
    heartbeat_interval_seconds: int = Field(default=60, alias="HEARTBEAT_INTERVAL_SECONDS")
    cleanup_interval_hours: int = Field(default=24, alias="CLEANUP_INTERVAL_HOURS")

    @staticmethod
    def parse_comma_list(value: str) -> List[str]:
        """Parse a comma-separated setting while discarding empty entries."""
        return [item.strip() for item in value.split(",") if item.strip()]

    @property
    def target_states_list(self) -> List[str]:
        return self.parse_comma_list(self.target_states)

    @property
    def target_cities_list(self) -> List[str]:
        return self.parse_comma_list(self.target_cities)

    @property
    def target_subreddits_list(self) -> List[str]:
        return self.parse_comma_list(self.target_subreddits)

# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
