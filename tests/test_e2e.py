"""Deterministic tests for the Reddit agent's core and orchestration flows."""
import math
import os
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("CHROMA_PERSIST_DIR", "./data/test-chroma")

from agent_orchestrator import AgentOrchestrator
from config import Settings
from database import Engagement, Lead, SessionLocal, init_db
from nvidia_llm import LLMResponse, NVIDIAClient
from qualifier_agent import QualifierAgent
from rag_engine import RAGEngine
from reddit_client import RedditRateLimiter
from safety_guardrails import SafetyCheck, SafetyGuardrails


class TestSettings:
    def test_csv_settings_are_parsed_and_credentials_are_optional(self):
        settings = Settings(
            _env_file=None,
            TARGET_SUBREDDITS="florida, Miami, ,tampa",
            TARGET_CITIES="Miami,Orlando",
        )

        assert settings.reddit_client_id is None
        assert settings.nvidia_api_key is None
        assert settings.target_subreddits_list == ["florida", "Miami", "tampa"]
        assert settings.target_cities_list == ["Miami", "Orlando"]


class TestRateLimiter:
    def test_token_refill(self, monkeypatch):
        limiter = RedditRateLimiter(max_qpm=60, burst_buffer=5)
        limiter.tokens = 0
        limiter.last_update = 1_000.0
        monkeypatch.setattr("reddit_client.time.time", lambda: 1_060.0)

        limiter._refill_tokens()

        assert limiter.tokens == 60

    def test_acquire_consumes_tokens(self, monkeypatch):
        limiter = RedditRateLimiter(max_qpm=60, burst_buffer=5)
        limiter.tokens = 10
        limiter.last_update = 1_000.0
        monkeypatch.setattr("reddit_client.time.time", lambda: 1_000.0)

        assert limiter.acquire(tokens=1) is True
        assert limiter.tokens == 9

    def test_rate_limit_state_near_limit(self):
        limiter = RedditRateLimiter(max_qpm=60, burst_buffer=2)
        limiter.rate_limit_state.remaining = 3

        assert limiter.rate_limit_state.is_near_limit is False

        limiter.rate_limit_state.remaining = 2
        assert limiter.rate_limit_state.is_near_limit is True


class TestSafetyGuardrails:
    def test_blocked_subreddit(self):
        checks = SafetyGuardrails().check_post_eligibility(
            post_title="Help after car crash",
            post_body="I was rear ended in Miami",
            subreddit="suicidewatch",
        )

        assert any(
            check.rule_name == "blocked_subreddit" and not check.passed
            for check in checks
        )

    def test_existing_attorney_detection(self):
        checks = SafetyGuardrails().check_post_eligibility(
            post_title="My attorney says...",
            post_body="My lawyer told me to file by Friday",
            subreddit="legaladvice",
        )

        assert any(
            check.rule_name == "existing_attorney" and not check.passed
            for check in checks
        )

    def test_response_compliance_missing_disclaimer(self):
        checks = SafetyGuardrails().check_response_compliance(
            "You should definitely sue them for 1 million dollars. Click here."
        )

        rules = {check.rule_name for check in checks if not check.passed}
        assert {"missing_disclaimer", "gives_legal_advice", "spam_detected"} <= rules

    def test_response_requires_every_disclaimer_phrase(self):
        checks = SafetyGuardrails().check_response_compliance(
            "CaseClosedFL is not a law firm. This is general information."
        )

        assert any(
            check.rule_name == "missing_disclaimer" and not check.passed
            for check in checks
        )

    def test_url_allowlist_checks_hostname_not_substring(self):
        checks = SafetyGuardrails().check_response_compliance(
            "CaseClosedFL is not a law firm. This is general information, not legal "
            "advice. Visit https://caseclosedfl.com.evil.example/collect."
        )

        assert any(
            check.rule_name == "unauthorized_url" and not check.passed
            for check in checks
        )

    def test_florida_abbreviation_uses_word_boundaries(self):
        checks = SafetyGuardrails().check_post_eligibility(
            post_title="Flowers after an accident",
            post_body="I need general support",
            subreddit="accidents",
        )

        assert any(
            check.rule_name == "florida_relevance"
            and check.severity == "warning"
            for check in checks
        )

    def test_can_proceed_logic(self):
        checks = [
            SafetyCheck(True, "ok", "info", "ok", "allow"),
            SafetyCheck(False, "bad", "critical", "bad", "block"),
        ]

        can_go, reasons = SafetyGuardrails().can_proceed(checks)

        assert can_go is False
        assert reasons == ["bad: bad"]


class TestQualifier:
    def test_extract_contact_uses_real_word_boundaries(self):
        contact = QualifierAgent().extract_contact(
            "You can reach me at person@example.com or 305-555-0199."
        )

        assert contact == {
            "email": "person@example.com",
            "phone": "305-555-0199",
        }


class TestDatabaseModels:
    def test_lead_defaults_and_engagement_relationship(self):
        init_db()
        with SessionLocal() as db:
            lead = Lead(
                reddit_username="testuser",
                reddit_post_id="abc123",
                subreddit="florida",
                intent_score=85.0,
                source_url="https://reddit.com/r/florida/comments/abc123",
            )
            engagement = Engagement(
                lead=lead,
                engagement_type="comment",
                reddit_post_id="abc123",
                subreddit="florida",
                our_response="General information, not legal advice.",
                compliance_check_passed=True,
            )
            db.add(engagement)
            db.flush()

            assert lead.id is not None
            assert engagement.lead is lead
            assert engagement.lead_id == lead.id


class TestNVIDIAClient:
    @pytest.mark.asyncio
    async def test_analyze_post_intent_parses_mocked_json(self):
        client = NVIDIAClient()
        client.chat = AsyncMock(
            return_value=LLMResponse(
                content='{"intent_score":85,"lead_temperature":"warm","recommended_action":"engage"}',
                model="test-model",
                usage={},
                finish_reason="stop",
            )
        )

        result = await client.analyze_post_intent(
            "Car accident in Miami", "I was rear ended", "florida"
        )

        assert result["intent_score"] == 85
        assert result["lead_temperature"] == "warm"
        client.chat.assert_awaited_once()


class TestRAGEngine:
    def test_local_embeddings_are_deterministic_and_normalized(self):
        first, second = RAGEngine._local_embed(["car crash Miami", "car crash Miami"])

        assert first == second
        assert len(first) == 384
        assert math.isclose(sum(value * value for value in first), 1.0)

    @pytest.mark.asyncio
    async def test_kb_initialization_is_idempotent(self):
        reddit_collection = MagicMock()
        kb_collection = MagicMock()
        persistent_client = MagicMock()
        persistent_client.get_or_create_collection.side_effect = [
            reddit_collection,
            kb_collection,
        ]

        with patch("rag_engine.chromadb.PersistentClient", return_value=persistent_client):
            engine = RAGEngine()
            engine._embed = AsyncMock(return_value=[[0.1] * 384 for _ in range(7)])
            await engine.initialize_kb()
            await engine.initialize_kb()

        assert kb_collection.upsert.call_count == 2
        assert len(kb_collection.upsert.call_args.kwargs["ids"]) == 7


class TestDiscoveryAgent:
    @pytest.mark.asyncio
    async def test_run_with_isolated_async_mocks(self):
        mock_post = SimpleNamespace(
            id="test123",
            title="Car crash help",
            selftext="I was hit in Miami",
            score=10,
            num_comments=5,
            author="testuser",
            subreddit="florida",
            created_utc=datetime.utcnow().timestamp(),
            permalink="/r/florida/comments/test123",
        )
        reddit = MagicMock()
        reddit.search_subreddit.return_value = [mock_post]
        llm = MagicMock()
        llm.generate_search_queries = AsyncMock(return_value=["car accident"])
        llm.analyze_post_intent = AsyncMock(
            return_value={
                "intent_score": 90,
                "qualification_score": 85,
                "lead_temperature": "hot",
                "accident_type": "car",
                "key_phrases": ["rear ended"],
                "location_hint": "Miami",
                "recommended_action": "engage",
            }
        )
        rag = MagicMock()
        rag.add_reddit_post = AsyncMock()
        guardrails = MagicMock()
        guardrails.check_post_eligibility.return_value = [
            SafetyCheck(True, "subreddit", "info", "ok", "allow")
        ]
        guardrails.can_proceed.return_value = (True, [])
        db = MagicMock()

        with (
            patch("discovery_agent.get_reddit_client", return_value=reddit),
            patch("discovery_agent.get_nvidia_client", return_value=llm),
            patch("discovery_agent.get_rag_engine", return_value=rag),
            patch("discovery_agent.get_guardrails", return_value=guardrails),
            patch("discovery_agent.get_db_session", return_value=db),
        ):
            from discovery_agent import DiscoveryAgent

            result = await DiscoveryAgent().run()

        assert result["posts_found"] == 1
        assert result["high_intent_posts"][0]["body"] == "I was hit in Miami"
        assert result["high_intent_posts"][0]["author"] == "testuser"
        rag.add_reddit_post.assert_awaited_once()
        db.close.assert_called_once()


class TestEndToEndPipeline:
    @pytest.mark.asyncio
    async def test_full_orchestration_flow(self):
        post = {
            "reddit_id": "abc123",
            "subreddit": "florida",
            "title": "Hit by truck in Orlando",
            "intent_score": 95,
            "lead_temperature": "hot",
        }
        discovery = MagicMock()
        discovery.run = AsyncMock(return_value={"high_intent_posts": [post]})
        engagement = MagicMock()
        engagement.run = AsyncMock(return_value={"processed": 1})
        monitor = MagicMock()
        monitor.add_monitor = AsyncMock(return_value=True)
        monitor.run = AsyncMock(
            return_value={"posts_checked": 1, "replies_requiring_action": []}
        )
        qualifier = MagicMock()
        qualifier.run_daily_qualification = AsyncMock(return_value={"leads_scored": 1})

        with (
            patch("agent_orchestrator.get_discovery_agent", return_value=discovery),
            patch("agent_orchestrator.get_engagement_agent", return_value=engagement),
            patch("agent_orchestrator.get_monitor_agent", return_value=monitor),
            patch("agent_orchestrator.get_qualifier_agent", return_value=qualifier),
        ):
            result = await AgentOrchestrator().run_full_cycle()

        assert result["status"] == "success"
        assert result["errors"] == []
        engagement.run.assert_awaited_once_with([post])
        monitor.add_monitor.assert_awaited_once()
        qualifier.run_daily_qualification.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_errors_keep_pipeline_status_failed(self):
        discovery = MagicMock()
        discovery.run = AsyncMock(side_effect=RuntimeError("discovery unavailable"))
        engagement = MagicMock()
        engagement.run = AsyncMock()
        monitor = MagicMock()
        monitor.run = AsyncMock(
            return_value={"posts_checked": 0, "replies_requiring_action": []}
        )
        qualifier = MagicMock()
        qualifier.run_daily_qualification = AsyncMock(return_value={"leads_scored": 0})

        with (
            patch("agent_orchestrator.get_discovery_agent", return_value=discovery),
            patch("agent_orchestrator.get_engagement_agent", return_value=engagement),
            patch("agent_orchestrator.get_monitor_agent", return_value=monitor),
            patch("agent_orchestrator.get_qualifier_agent", return_value=qualifier),
        ):
            result = await AgentOrchestrator().run_full_cycle()

        assert result["status"] == "failed"
        assert result["errors"] == ["discovery: discovery unavailable"]
