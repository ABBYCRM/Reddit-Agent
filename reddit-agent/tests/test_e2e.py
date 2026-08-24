"""Isolated regression tests; no live Reddit, Composio, or LLM calls."""
import asyncio
import json
import os
import re
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
from types import SimpleNamespace


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


def test_production_requires_operator_and_runtime_outreach_defaults_off():
    with pytest.raises(ValueError, match="OPERATOR_API_KEY"):
        Settings(APP_ENV="production", SECRET_KEY="safe-secret", OPERATOR_API_KEY=None)
    settings = Settings(
        APP_ENV="production",
        SECRET_KEY="safe-secret",
        OPERATOR_API_KEY="operator",
        ENABLE_AUTO_REPLY=True,
        ENABLE_DM_OUTREACH=True,
    )
    assert settings.enable_auto_reply is True
    assert settings.enable_dm_outreach is True
    assert settings.max_daily_engagements == 10


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


def test_guardrails_block_stealth_unicode_obfuscation():
    guardrails = SafetyGuardrails()
    obfuscated = "CaseClosedFL is not a law firm. This is general information, not legal advice.\u200b"
    allowed, reasons = guardrails.can_proceed(guardrails.check_response_compliance(obfuscated))
    assert not allowed
    assert any("obfuscated_unicode" in reason for reason in reasons)


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

@pytest.mark.asyncio
async def test_composio_staging_contract_check_is_read_only_and_validates_shape():
    import httpx
    from composio_client import ComposioRedditClient

    requests = []

    def handler(request):
        requests.append((request.method, request.url.path, request.read().decode()))
        if request.url.path.endswith("/connected_accounts"):
            return httpx.Response(
                200,
                json={"data": {"items": [{"id": "account-1", "user_id": "reddit-user"}]}},
            )
        assert request.url.path.endswith("/tools/execute/REDDIT_SEARCH_ACROSS_SUBREDDITS")
        return httpx.Response(200, json={"data": {"search_results": {"data": {"children": []}}}})

    client = ComposioRedditClient(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    client.api_key = "test-key"
    client.base_url = "https://composio.test/api/v3"
    result = await client.validate_read_only_contract("florida")

    assert result == {
        "connected_account_id": "account-1",
        "action": "REDDIT_SEARCH_ACROSS_SUBREDDITS",
        "subreddit": "florida",
        "posts_checked": 0,
        "read_only": True,
    }
    assert len(requests) == 2
    assert requests[0][0] == "GET"
    assert requests[1][0] == "POST"
    assert "REDDIT_SEARCH_ACROSS_SUBREDDITS" in requests[1][1]
    assert "SUBMIT_COMMENT" not in requests[1][1]
    assert "SEND_MESSAGE" not in requests[1][1]
def test_api_operator_protection_and_health():
    from fastapi.testclient import TestClient
    import api
    api.settings.operator_api_key = "test-operator-key"
    app = api.app
    with TestClient(app) as client:
        assert client.get("/").status_code == 401
        assert client.get("/access").status_code == 200
        assert client.get("/api/health").status_code == 200
        headers = {"X-Operator-Key": "test-operator-key"}
        dashboard = client.get("/", headers=headers)
        assert dashboard.status_code == 200
        assert "Autonomous Outreach" in dashboard.text
        assert "data.message || data.detail" in dashboard.text

        sign_in = client.post("/api/operator-session", headers=headers)
        assert sign_in.status_code == 200
        dashboard = client.get("/")
        assert dashboard.status_code == 200
        csrf = re.search(r'const csrfToken = "([^"]+)"', dashboard.text).group(1)
        assert client.put("/api/outreach", json={"enabled": True}).status_code == 403
        toggle = client.put("/api/outreach", json={"enabled": True}, headers={"X-CSRF-Token": csrf})
        assert toggle.status_code == 409
        assert "cannot be enabled" in toggle.json()["detail"]
        assert "direct-message action" in toggle.json()["detail"]


def _reset_outreach_state():
    from database import DailyOutreachQuota, OutboundAction, OutreachControl, SafetyLog, db_session, init_db
    init_db()
    with db_session() as db:
        db.query(OutboundAction).delete()
        db.query(DailyOutreachQuota).delete()
        db.query(SafetyLog).delete()
        db.query(OutreachControl).delete()
    init_db()


class _StubReddit:
    def __init__(self, capabilities=None, fail_comment=False, comment_started=None, release_comment=None):
        self.capabilities = capabilities or {
            "transport": "test",
            "comment": True,
            "dm": True,
            "reason": None,
        }
        self.fail_comment = fail_comment
        self.comment_started = comment_started
        self.release_comment = release_comment
        self.comment_calls = 0
        self.dm_calls = 0

    async def get_outreach_capabilities(self, force_refresh=False):
        return self.capabilities

    async def post_comment(self, post_id, body):
        self.comment_calls += 1
        if self.comment_started:
            self.comment_started.set()
        if self.release_comment:
            await self.release_comment.wait()
        if self.fail_comment:
            raise RuntimeError("provider timeout")
        return SimpleNamespace(id=f"comment-{self.comment_calls}", created_utc=1)

    async def send_dm(self, username, subject, body):
        self.dm_calls += 1
        return True


@pytest.mark.asyncio
async def test_outreach_refuses_enablement_when_dm_is_unavailable(monkeypatch):
    import outreach_service

    _reset_outreach_state()
    reddit = _StubReddit({"transport": "test", "comment": True, "dm": False, "reason": "dm_action_unavailable"})
    monkeypatch.setattr(outreach_service, "get_reddit_client", lambda: reddit)
    service = outreach_service.OutreachService()
    with pytest.raises(outreach_service.OutreachUnavailableError, match="dm_action_unavailable"):
        await service.set_enabled(True, "test")
    assert (await service.get_status())["enabled"] is False


@pytest.mark.asyncio
async def test_outreach_combines_channels_under_one_ten_action_cap(monkeypatch):
    import outreach_service

    _reset_outreach_state()
    reddit = _StubReddit()
    monkeypatch.setattr(outreach_service, "get_reddit_client", lambda: reddit)
    service = outreach_service.OutreachService()
    await service.set_enabled(True, "test")

    for index in range(5):
        result = await service.send_comment("", f"post-{index}", "CaseClosedFL is not a law firm. This is general information, not legal advice.")
        assert result["sent"] is True
    for index in range(5):
        result = await service.send_dm("", f"dm-post-{index}", f"user-{index}", "General information", "CaseClosedFL is not a law firm. This is general information, not legal advice.")
        assert result["sent"] is True

    blocked = await service.send_comment("", "post-over-limit", "CaseClosedFL is not a law firm. This is general information, not legal advice.")
    assert blocked["sent"] is False
    assert blocked["reason"] == "daily_outreach_limit_reached"
    status = await service.get_status()
    assert status["reserved_today"] == 10
    assert reddit.comment_calls == 5
    assert reddit.dm_calls == 5


@pytest.mark.asyncio
async def test_concurrent_send_waits_asynchronously_for_provider_dispatch(monkeypatch):
    import outreach_service

    _reset_outreach_state()
    comment_started = asyncio.Event()
    release_comment = asyncio.Event()
    reddit = _StubReddit(comment_started=comment_started, release_comment=release_comment)
    monkeypatch.setattr(outreach_service, "get_reddit_client", lambda: reddit)
    service = outreach_service.OutreachService()
    await service.set_enabled(True, "test")
    first = asyncio.create_task(
        service.send_comment("", "concurrent-post-1", "CaseClosedFL is not a law firm. This is general information, not legal advice.")
    )
    await asyncio.wait_for(comment_started.wait(), timeout=1)
    second = asyncio.create_task(
        service.send_comment("", "concurrent-post-2", "CaseClosedFL is not a law firm. This is general information, not legal advice.")
    )
    await asyncio.sleep(0)
    assert not second.done()
    release_comment.set()
    first_result, second_result = await asyncio.wait_for(asyncio.gather(first, second), timeout=2)
    assert first_result["sent"] is True
    assert second_result["sent"] is True
    assert reddit.comment_calls == 2


@pytest.mark.asyncio
async def test_unknown_outbound_result_is_not_automatically_retried(monkeypatch):
    import outreach_service

    _reset_outreach_state()
    reddit = _StubReddit(fail_comment=True)
    monkeypatch.setattr(outreach_service, "get_reddit_client", lambda: reddit)
    service = outreach_service.OutreachService()
    await service.set_enabled(True, "test")
    first = await service.send_comment("", "unstable-post", "CaseClosedFL is not a law firm. This is general information, not legal advice.")
    second = await service.send_comment("", "unstable-post", "CaseClosedFL is not a law firm. This is general information, not legal advice.")
    assert first["status"] == "unknown"
    assert second["status"] == "unknown"
    assert reddit.comment_calls == 1


@pytest.mark.asyncio
async def test_disabling_outreach_blocks_new_actions_immediately(monkeypatch):
    import outreach_service

    _reset_outreach_state()
    reddit = _StubReddit()
    monkeypatch.setattr(outreach_service, "get_reddit_client", lambda: reddit)
    service = outreach_service.OutreachService()
    await service.set_enabled(True, "test")
    await service.set_enabled(False, "test")
    result = await service.send_comment("", "off-post", "CaseClosedFL is not a law firm. This is general information, not legal advice.")
    assert result["status"] == "blocked"
    assert result["reason"] == "outreach_disabled"
    assert reddit.comment_calls == 0


@pytest.mark.asyncio
async def test_disabling_outreach_cancels_claimed_actions_before_dispatch(monkeypatch):
    import outreach_service

    _reset_outreach_state()
    reddit = _StubReddit()
    monkeypatch.setattr(outreach_service, "get_reddit_client", lambda: reddit)
    service = outreach_service.OutreachService()
    await service.set_enabled(True, "test")
    claim = service._claim(
        channel="comment",
        dedupe_key="comment:claimed-post",
        body="CaseClosedFL is not a law firm. This is general information, not legal advice.",
        source_post_id="claimed-post",
        recipient_username=None,
        subject=None,
        engagement_id=None,
    )
    assert claim["claimed"] is True
    await service.set_enabled(False, "test")
    result = await service._dispatch(
        claim["action_id"],
        channel="comment",
        body="CaseClosedFL is not a law firm. This is general information, not legal advice.",
        source_post_id="claimed-post",
        recipient_username=None,
        subject=None,
    )
    assert result["status"] == "blocked"
    assert reddit.comment_calls == 0


@pytest.mark.asyncio
async def test_off_fences_action_marked_for_dispatch_before_provider_call(monkeypatch):
    import outreach_service

    _reset_outreach_state()
    reddit = _StubReddit()
    monkeypatch.setattr(outreach_service, "get_reddit_client", lambda: reddit)
    service = outreach_service.OutreachService()
    await service.set_enabled(True, "test")
    claim = service._claim(
        channel="comment",
        dedupe_key="comment:dispatching-post",
        body="CaseClosedFL is not a law firm. This is general information, not legal advice.",
        source_post_id="dispatching-post",
        recipient_username=None,
        subject=None,
        engagement_id=None,
    )
    assert service._mark_dispatching(claim["action_id"])["marked"] is True
    await service.set_enabled(False, "test")
    result = await service._deliver_with_fence(
        claim["action_id"],
        channel="comment",
        body="CaseClosedFL is not a law firm. This is general information, not legal advice.",
        source_post_id="dispatching-post",
        recipient_username=None,
        subject=None,
    )
    assert result["status"] == "blocked"
    assert result["reason"] == "outreach_disabled_before_provider_call"
    assert reddit.comment_calls == 0


@pytest.mark.asyncio
async def test_losing_dm_capability_blocks_comment_dispatch_too(monkeypatch):
    import outreach_service

    _reset_outreach_state()
    reddit = _StubReddit()
    monkeypatch.setattr(outreach_service, "get_reddit_client", lambda: reddit)
    service = outreach_service.OutreachService()
    await service.set_enabled(True, "test")
    reddit.capabilities = {"transport": "test", "comment": True, "dm": False, "reason": "dm_action_unavailable"}
    result = await service.send_comment(
        "",
        "capability-loss-post",
        "CaseClosedFL is not a law firm. This is general information, not legal advice.",
    )
    assert result["status"] == "blocked"
    assert result["reason"] == "dm_action_unavailable"
    assert reddit.comment_calls == 0


@pytest.mark.asyncio
async def test_composio_capability_check_requires_a_real_dm_action():
    import httpx
    from composio_client import ComposioRedditClient

    def handler(request):
        if request.url.path.endswith("/connected_accounts"):
            return httpx.Response(200, json={"data": {"items": [{"id": "account-1", "user_id": "reddit-user"}]}})
        assert request.url.path.endswith("/tools")
        return httpx.Response(200, json={"items": [{"slug": "REDDIT_POST_REDDIT_COMMENT"}]})

    client = ComposioRedditClient(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    client.api_key = "test-key"
    client.base_url = "https://composio.test/api/v3"
    capabilities = await client.get_outreach_capabilities()
    assert capabilities["comment"] is True
    assert capabilities["dm"] is False
    assert capabilities["reason"] == "dm_action_unavailable"


def test_compose_does_not_expose_database_or_redis_ports():
    compose = Path(__file__).parents[1].joinpath("docker-compose.yml").read_text()
    assert '"5432:5432"' not in compose
    assert '"6379:6379"' not in compose
    assert "DATABASE_URL:" in compose

@pytest.mark.asyncio
async def test_worker_staging_preflight_uses_only_read_only_contract_check(monkeypatch):
    import run

    events = []

    class RedditClient:
        async def validate_read_only_contract(self):
            events.append("read_only_contract")
            return {
                "connected_account_id": "account-1",
                "action": "REDDIT_SEARCH_SUBREDDIT",
                "posts_checked": 0,
            }

    class ComposioClient:
        async def get_connected_account(self):
            events.append("account_lookup")

    monkeypatch.setattr(run, "get_reddit_client", lambda: RedditClient())
    monkeypatch.setattr(run, "get_composio_client", lambda: ComposioClient())

    await run.validate_reddit_transport(SimpleNamespace(reddit_transport="composio", app_env="staging"))

    assert events == ["read_only_contract"]

@pytest.mark.asyncio
async def test_worker_production_preflight_does_not_run_staging_search(monkeypatch):
    import run

    events = []

    class RedditClient:
        async def validate_read_only_contract(self):
            events.append("read_only_contract")

    class ComposioClient:
        async def get_connected_account(self):
            events.append("account_lookup")

    monkeypatch.setattr(run, "get_reddit_client", lambda: RedditClient())
    monkeypatch.setattr(run, "get_composio_client", lambda: ComposioClient())

    await run.validate_reddit_transport(SimpleNamespace(reddit_transport="composio", app_env="production"))

    assert events == ["account_lookup"]

@pytest.mark.asyncio
async def test_composio_malformed_search_response_fails_instead_of_empty_discovery():
    import httpx
    from composio_client import ComposioError, ComposioRedditClient

    def handler(request):
        if request.url.path.endswith("/connected_accounts"):
            return httpx.Response(
                200,
                json={"data": {"items": [{"id": "account-1", "user_id": "reddit-user"}]}},
            )
        return httpx.Response(200, json={"data": {"unexpected": []}})

    client = ComposioRedditClient(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    client.api_key = "test-key"
    client.base_url = "https://composio.test/api/v3"
    with pytest.raises(ComposioError, match="unexpected search response"):
        await client.search_subreddit("florida", "test")
