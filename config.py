"""Validated application settings for the CaseClosedFL agent."""
from functools import lru_cache
from typing import List, Optional

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: str = Field(default="development", validation_alias=AliasChoices("APP_ENV", "app_env"))
    log_level: str = Field(default="INFO", validation_alias=AliasChoices("LOG_LEVEL", "log_level"))
    secret_key: str = Field(default="dev-only-secret", validation_alias=AliasChoices("SECRET_KEY", "secret_key"))
    operator_api_key: Optional[str] = Field(default=None, validation_alias=AliasChoices("OPERATOR_API_KEY", "operator_api_key"))

    # Composio is the production Reddit transport. Raw Reddit credentials are
    # retained only for an explicitly selected local read-only fallback.
    reddit_client_id: Optional[str] = Field(default=None, validation_alias=AliasChoices("REDDIT_CLIENT_ID", "reddit_client_id"))
    reddit_client_secret: Optional[str] = Field(default=None, validation_alias=AliasChoices("REDDIT_CLIENT_SECRET", "reddit_client_secret"))
    reddit_user_agent: str = Field(default="caseclosedfl-agent/1.0", validation_alias=AliasChoices("REDDIT_USER_AGENT", "reddit_user_agent"))
    reddit_username: Optional[str] = Field(default=None, validation_alias=AliasChoices("REDDIT_USERNAME", "reddit_username"))
    reddit_password: Optional[str] = Field(default=None, validation_alias=AliasChoices("REDDIT_PASSWORD", "reddit_password"))
    reddit_max_qpm: int = Field(default=90, ge=1, le=600, validation_alias=AliasChoices("REDDIT_MAX_QPM", "reddit_max_qpm"))
    reddit_burst_buffer: int = Field(default=5, ge=0, validation_alias=AliasChoices("REDDIT_BURST_BUFFER", "reddit_burst_buffer"))
    reddit_transport: str = Field(default="composio", validation_alias=AliasChoices("REDDIT_TRANSPORT", "reddit_transport"))

    nvidia_api_key: Optional[str] = Field(default=None, validation_alias=AliasChoices("NVIDIA_API_KEY", "nvidia_api_key"))
    nvidia_base_url: str = Field(default="https://integrate.api.nvidia.com/v1", validation_alias=AliasChoices("NVIDIA_BASE_URL", "nvidia_base_url"))
    nvidia_model: str = Field(default="meta/llama-3.3-70b-instruct", validation_alias=AliasChoices("NVIDIA_MODEL", "nvidia_model"))
    nvidia_embedding_model: str = Field(default="nvidia/nv-embedqa-e5-v5", validation_alias=AliasChoices("NVIDIA_EMBEDDING_MODEL", "nvidia_embedding_model"))

    composio_api_key: Optional[str] = Field(default=None, validation_alias=AliasChoices("COMPOSIO_API_KEY", "composio_api_key"))
    composio_base_url: str = Field(default="https://backend.composio.dev/api/v3", validation_alias=AliasChoices("COMPOSIO_BASE_URL", "composio_base_url"))
    composio_reddit_connected_account_id: Optional[str] = Field(default=None, validation_alias=AliasChoices("COMPOSIO_REDDIT_CONNECTED_ACCOUNT_ID", "composio_reddit_connected_account_id"))
    composio_reddit_user_id: Optional[str] = Field(default=None, validation_alias=AliasChoices("COMPOSIO_REDDIT_USER_ID", "composio_reddit_user_id"))
    composio_request_timeout_seconds: float = Field(default=30.0, gt=0, le=120, validation_alias=AliasChoices("COMPOSIO_REQUEST_TIMEOUT_SECONDS", "composio_request_timeout_seconds"))

    database_url: str = Field(default="sqlite:///./data/caseclosed_agent.db", validation_alias=AliasChoices("DATABASE_URL", "database_url"))
    chroma_persist_dir: str = Field(default="./data/chroma_db", validation_alias=AliasChoices("CHROMA_PERSIST_DIR", "chroma_persist_dir"))

    max_daily_engagements: int = Field(default=50, ge=0, validation_alias=AliasChoices("MAX_DAILY_ENGAGEMENTS", "max_daily_engagements"))
    min_account_age_days: int = Field(default=14, ge=0, validation_alias=AliasChoices("MIN_ACCOUNT_AGE_DAYS", "min_account_age_days"))
    enable_auto_reply: bool = Field(default=False, validation_alias=AliasChoices("ENABLE_AUTO_REPLY", "enable_auto_reply"))
    enable_dm_outreach: bool = Field(default=False, validation_alias=AliasChoices("ENABLE_DM_OUTREACH", "enable_dm_outreach"))
    florida_bar_compliant: bool = Field(default=True, validation_alias=AliasChoices("FLORIDA_BAR_COMPLIANT", "florida_bar_compliant"))

    # Keep these environment-backed fields as strings. Pydantic Settings 2.6
    # tries to JSON-decode List[str] before validators run, which rejects the
    # documented comma-delimited deployment values.
    target_states_raw: str = Field(default="Florida", validation_alias=AliasChoices("TARGET_STATES", "target_states", "target_states_raw"))
    target_cities_raw: str = Field(default="Miami, Orlando, Tampa, Jacksonville, Fort Lauderdale, West Palm Beach", validation_alias=AliasChoices("TARGET_CITIES", "target_cities", "target_cities_raw"))
    target_subreddits_raw: str = Field(default="legaladvice, florida, Miami, Orlando, tampa, insurance, personalfinance, caraccidents", validation_alias=AliasChoices("TARGET_SUBREDDITS", "target_subreddits", "target_subreddits_raw"))

    lead_score_threshold: int = Field(default=75, ge=0, le=100, validation_alias=AliasChoices("LEAD_SCORE_THRESHOLD", "lead_score_threshold"))
    auto_qualify_threshold: int = Field(default=90, ge=0, le=100, validation_alias=AliasChoices("AUTO_QUALIFY_THRESHOLD", "auto_qualify_threshold"))

    discovery_interval_minutes: int = Field(default=30, ge=1, validation_alias=AliasChoices("DISCOVERY_INTERVAL_MINUTES", "discovery_interval_minutes"))
    monitor_interval_minutes: int = Field(default=15, ge=1, validation_alias=AliasChoices("MONITOR_INTERVAL_MINUTES", "monitor_interval_minutes"))
    heartbeat_interval_seconds: int = Field(default=60, ge=5, validation_alias=AliasChoices("HEARTBEAT_INTERVAL_SECONDS", "heartbeat_interval_seconds"))
    cleanup_interval_hours: int = Field(default=24, ge=1, validation_alias=AliasChoices("CLEANUP_INTERVAL_HOURS", "cleanup_interval_hours"))
    run_scheduler: bool = Field(default=False, validation_alias=AliasChoices("RUN_SCHEDULER", "run_scheduler"))

    @model_validator(mode="after")
    def validate_production_safety(self):
        if self.app_env.lower() == "production":
            if not self.operator_api_key:
                raise ValueError("OPERATOR_API_KEY is required in production")
            if self.secret_key == "dev-only-secret":
                raise ValueError("SECRET_KEY must be changed in production")
            if self.enable_auto_reply or self.enable_dm_outreach:
                raise ValueError("Automated Reddit outreach must remain disabled during production startup")
        if self.reddit_transport not in {"composio", "praw"}:
            raise ValueError("REDDIT_TRANSPORT must be 'composio' or 'praw'")
        if self.app_env.lower() == "production" and self.reddit_transport == "composio" and not self.composio_api_key:
            raise ValueError("COMPOSIO_API_KEY is required for the Composio production transport")
        return self

    @property
    def target_states(self) -> List[str]:
        return self._parse_comma_list(self.target_states_raw)

    @property
    def target_cities(self) -> List[str]:
        return self._parse_comma_list(self.target_cities_raw)

    @property
    def target_subreddits(self) -> List[str]:
        return self._parse_comma_list(self.target_subreddits_raw)

    @staticmethod
    def _parse_comma_list(value: str) -> List[str]:
        return [item.strip() for item in value.split(",") if item.strip()]

    @property
    def target_states_list(self) -> List[str]:
        return self.target_states

    @property
    def target_cities_list(self) -> List[str]:
        return self.target_cities

    @property
    def target_subreddits_list(self) -> List[str]:
        return self.target_subreddits


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()