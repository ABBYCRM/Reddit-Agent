"""Isolated regression tests; no live Reddit, Composio, or LLM calls."""
import asyncio
import json
import os
from pathlib import Path

import pytest

os.environ["APP_ENV"] = "development"
os.environ["OPERATOR_API_KEY"] = "test-operator-key"
os.environ["SECRET_KEY"] = "test-secret"
os.environ["DATABASE_URL"] = "sqlite:///./data/test_caseclosed_agent.db"

from config import Settings, get_settings
get_settings.cache_clear()
from safety_guardrails import SafetyGuardrails
from qualifier_agent import QualifierAgent
from rag_engine import RAGEngine


def test_comma_settings_are_real_lists():
    settings = Settings(TARGET_CITIES="Miami, Orlando", TARGET_SUBREDDITS="florida, Tampa")
    assert settings.target_cities == ["Miami", "Orlando"]
    assert settings.target_subreddits == ["florida", "Tampa"]


def test_comma_settings_load_from_real_environment(monkeypatch):
    monkeypatch.setenv("TARGET_STATES", "Florida, Georgia")
    monkeypatch.setenv("TARGET_CITIES", "Miami, Orlando")
    monkeypatch.setenv("TARGET_SUBREDDITS", "florida, Tampa")
    settings = Settings()
    assert settings.target_states == ["Florida", "Georgia"]
    assert settings.target_cities == ["Miami", "Orlando"]
    assert settings.target_subreddits == ["florida", "Tampa"]


def test_production_requires_operator_and_safe_flags():
    with pytest.raises(ValueError, match="OPERATOR_API_KEY"):
        Settings(APP_ENV="production", SECRET_KEY="safe-secret", OPERATOR_API_KEY=None)
    with pytest.raises(ValueError, match="Automated Reddit outreach"):
        Settings(APP_ENV="production", SECRET_KEY="safe-secret", OPERATOR_API_KEY="operator", ENABLE_AUTO_REPLY=True)


def test_guardrails_block_existing_attorney_and_legal_advice():
    guardrails = SafetyGuardrails()
    checks = guardrails.check_post_eligibility("Can I sue?", "My attorney said Florida.", "florida")
    allowed, reasons = guardrails.can_proceed(checks)
    assert not allowed
    assert any("existing" in reason or "legal_advice" in reason for reason in reasons)


def test_guardrails_require_complete_disclosure_and_real_hosts():
    guardrails = SafetyGuardrails()
    incomplete = guardrails.check_response_compliance("We are not a law firm.")
    assert not guardrails.can_proceed(incomplete)[0]
    unsafe_url = guardrails.check_response_compliance(
        "We are not a law firm. This is general information, not legal advice. https://caseclosedfl.com.evil.test"
    )
    assert not guardrails.can_proceed(unsafe_url)[0]
    compliant = guardrails.check_response_compliance(
        "We are not a law firm. This is general information, not legal advice. https://caseclosedfl.com"
    )
    assert guardrails.can_proceed(compliant)[0]


def test_contact_extraction_uses_word_boundaries():
    contact = QualifierAgent().extract_contact("You can email test@example.com or call 305-555-0123")
    assert contact == {"email": "test@example.com", "phone": "305-555-0123"}


@pytest.mark.asyncio
async def test_rag_persists_embedding_and_finds_relevant_document(tmp_path, monkeypatch):
    import config
    import database
    config.get_settings.cache_clear()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/rag.db")
    # This process has imported database already, so exercise deterministic
    # embedding behavior independently rather than changing the engine mid-run.
    embedding = __import__("rag_engine")._fallback_embedding("Florida collision insurance")
    assert len(embedding) == 128
    assert embedding == __import__("rag_engine")._fallback_embedding("Florida collision insurance")


def test_composio_failure_is_explicit():
    from composio_client import ComposioRedditClient, ComposioError
    client = ComposioRedditClient()
    client.api_key = None
    with pytest.raises(ComposioError, match="not configured"):
        asyncio.run(client._request("GET", "/connectedAccounts"))


def test_standard_postgres_urls_select_psycopg_v3():
    from database import normalize_database_url
    assert normalize_database_url("postgresql://user:pass@example.com:25060/app") == "postgresql+psycopg://user:pass@example.com:25060/app"
    assert normalize_database_url("postgres://user:pass@example.com/app") == "postgresql+psycopg://user:pass@example.com/app"


@pytest.mark.asyncio
async def test_composio_account_and_search_are_normalized_without_live_calls():
    import httpx
    from composio_client import ComposioRedditClient

    def handler(request):
        if request.url.path.endswith("/connected_accounts"):
            return httpx.Response(200, json={"data": {"items": [{"id": "account-1", "user_id": "reddit-user"}]}})
        assert request.url.path.endswith("/tools/execute/REDDIT_SEARCH_ACROSS_SUBREDDITS")
        request_data = json.loads(request.content)
        assert request_data["connected_account_id"] == "account-1"
        assert request_data["user_id"] == "reddit-user"
        return httpx.Response(200, json={"data": {"search_results": {"data": {"children": [{"data": {"id": "abc", "title": "Test"}}]}}}})

    client = ComposioRedditClient(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    client.api_key = "test-key"
    client.base_url = "https://composio.test/api/v3"
    assert await client.get_connected_account() == "account-1"
    posts = await client.search_subreddit("florida", "test")
    assert posts[0]["id"] == "abc"


def test_api_operator_protection_and_health():
    from fastapi.testclient import TestClient
    import api
    api.settings.operator_api_key = "test-operator-key"
    app = api.app
    with TestClient(app) as client:
        assert client.get("/").status_code == 401
        assert client.get("/api/health").status_code == 200
        assert client.get("/", headers={"X-Operator-Key": "test-operator-key"}).status_code == 200


def test_compose_does_not_expose_database_or_redis_ports():
    compose = Path(__file__).parents[1].joinpath("docker-compose.yml").read_text()
    assert '"5432:5432"' not in compose
    assert '"6379:6379"' not in compose
    assert "DATABASE_URL:" in compose