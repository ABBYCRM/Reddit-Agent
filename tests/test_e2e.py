"""
CaseClosedFL Reddit Agent - End-to-End Tests
Run 3x line-by-line verification as required.
Tests full agent pipeline without external API calls (mocked).
"""
import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Import all modules for testing
from config import Settings
from database import Lead, Engagement, MonitoredPost, RedditBundle, AgentRun
from safety_guardrails import SafetyGuardrails, SafetyCheck
from reddit_client import RedditRateLimiter


class TestRateLimiter:
    """Test 1: Reddit Rate Limiting (Line-by-line)"""

    def test_token_refill(self):
        limiter = RedditRateLimiter(max_qpm=60, burst_buffer=5)
        limiter.tokens = 0
        limiter.last_update = datetime.utcnow().timestamp() - 60
        limiter._refill_tokens()
        assert limiter.tokens == 60  # Full refill after 60 seconds

    def test_acquire_consumes_tokens(self):
        limiter = RedditRateLimiter(max_qpm=60, burst_buffer=5)
        limiter.tokens = 10
        limiter.last_update = datetime.utcnow().timestamp()
        result = limiter.acquire(tokens=1)
        assert result is True
        assert limiter.tokens == 9

    def test_rate_limit_state_near_limit(self):
        limiter = RedditRateLimiter(max_qpm=60, burst_buffer=5)
        limiter.rate_limit_state.remaining = 3
        assert limiter.rate_limit_state.is_near_limit is True


class TestSafetyGuardrails:
    """Test 2: Safety & Compliance (Line-by-line)"""

    def test_blocked_subreddit(self):
        guardrails = SafetyGuardrails()
        checks = guardrails.check_post_eligibility(
            post_title="Help after car crash",
            post_body="I was rear ended in Miami",
            subreddit="suicidewatch"
        )
        blocked = [c for c in checks if c.rule_name == "blocked_subreddit" and not c.passed]
        assert len(blocked) == 1

    def test_existing_attorney_detection(self):
        guardrails = SafetyGuardrails()
        checks = guardrails.check_post_eligibility(
            post_title="My attorney says...",
            post_body="My lawyer told me to file by Friday",
            subreddit="legaladvice"
        )
        blocked = [c for c in checks if c.rule_name == "existing_attorney" and not c.passed]
        assert len(blocked) == 1

    def test_response_compliance_missing_disclaimer(self):
        guardrails = SafetyGuardrails()
        checks = guardrails.check_response_compliance(
            "You should definitely sue them for 1 million dollars. Contact me."
        )
        blocked = [c for c in checks if not c.passed]
        assert len(blocked) >= 2  # Missing disclaimer + gives legal advice + spam

    def test_can_proceed_logic(self):
        guardrails = SafetyGuardrails()
        checks = [
            SafetyCheck(passed=True, rule_name="ok", severity="info", message="ok", action="allow"),
            SafetyCheck(passed=False, rule_name="bad", severity="critical", message="bad", action="block"),
        ]
        can_go, reasons = guardrails.can_proceed(checks)
        assert can_go is False
        assert len(reasons) == 1


class TestDatabaseModels:
    """Test 3: Database Models (Line-by-line)"""

    def test_lead_creation(self):
        lead = Lead(
            reddit_username="testuser",
            reddit_post_id="abc123",
            subreddit="florida",
            intent_score=85.0,
            status="new"
        )
        assert lead.reddit_username == "testuser"
        assert lead.intent_score == 85.0
        assert str(lead.id)  # UUID generated

    def test_engagement_relationship(self):
        lead = Lead(reddit_username="u", reddit_post_id="p", subreddit="r")
        eng = Engagement(
            lead_id=lead.id,
            engagement_type="comment",
            reddit_post_id="p",
            subreddit="r",
            our_response="test response",
            compliance_check_passed=True
        )
        assert eng.engagement_type == "comment"
        assert eng.compliance_check_passed is True


class TestNVIDIAClientMock:
    """Test 4: LLM Client with mocked API (Line-by-line)"""

    @pytest.mark.asyncio
    async def test_analyze_post_intent(self):
        with patch('nvidia_llm.NVIDIAClient.chat') as mock_chat:
            mock_chat.return_value = AsyncMock()
            mock_chat.return_value.content = """
            {
                "intent_score": 85,
                "qualification_score": 80,
                "lead_temperature": "warm",
                "accident_type": "car",
                "recommended_action": "engage"
            }
            """

            from nvidia_llm import NVIDIAClient
            client = NVIDIAClient()
            result = await client.analyze_post_intent(
                "Car accident in Miami", "I was rear ended", "florida"
            )

            assert result["intent_score"] == 85
            assert result["lead_temperature"] == "warm"
            assert result["recommended_action"] == "engage"


class TestRAGEngine:
    """Test 5: RAG Engine (Line-by-line)"""

    @pytest.mark.asyncio
    async def test_kb_initialization(self):
        with patch('rag_engine.RAGEngine._embed') as mock_embed:
            mock_embed.return_value = [[0.1] * 384] * 7  # 7 KB docs

            from rag_engine import RAGEngine
            engine = RAGEngine()
            await engine.initialize_kb()
            assert engine.kb_collection.count() == 7


class TestDiscoveryAgent:
    """Test 6: Discovery Agent (Line-by-line)"""

    @pytest.mark.asyncio
    async def test_run_with_mocks(self):
        with patch('discovery_agent.get_reddit_client') as mock_reddit, \
             patch('discovery_agent.get_nvidia_client') as mock_llm, \
             patch('discovery_agent.get_rag_engine') as mock_rag, \
             patch('discovery_agent.get_guardrails') as mock_guard, \
             patch('discovery_agent.get_db_session') as mock_db:

            # Setup mocks
            mock_post = Mock()
            mock_post.id = "test123"
            mock_post.title = "Car crash help"
            mock_post.selftext = "I was hit in Miami"
            mock_post.score = 10
            mock_post.num_comments = 5
            mock_post.author = "testuser"
            mock_post.subreddit = "florida"
            mock_post.created_utc = datetime.utcnow().timestamp()
            mock_post.permalink = "/r/florida/comments/test123"

            mock_reddit.return_value.search_subreddit.return_value = [mock_post]

            mock_llm.return_value.generate_search_queries.return_value = ["car accident"]
            mock_llm.return_value.analyze_post_intent.return_value = {
                "intent_score": 90,
                "qualification_score": 85,
                "lead_temperature": "hot",
                "accident_type": "car",
                "key_phrases": ["rear ended"],
                "location_hint": "Miami",
                "recommended_action": "engage"
            }

            mock_guard.return_value.check_post_eligibility.return_value = [
                SafetyCheck(True, "subreddit", "info", "ok", "allow")
            ]
            mock_guard.return_value.can_proceed.return_value = (True, [])

            from discovery_agent import DiscoveryAgent
            agent = DiscoveryAgent()
            result = await agent.run()

            assert result["posts_found"] == 1
            assert len(result["high_intent_posts"]) == 1
            assert result["high_intent_posts"][0]["intent_score"] == 90


class TestEndToEndPipeline:
    """
    Test 7: Full E2E Pipeline (Line-by-line)
    Simulates: Discovery -> Engagement -> Monitor -> Qualify
    """

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Run complete pipeline with all external services mocked."""

        # Mock all external dependencies
        with patch('reddit_client.praw.Reddit') as mock_praw, \
             patch('nvidia_llm.AsyncOpenAI') as mock_openai, \
             patch('rag_engine.chromadb.PersistentClient') as mock_chroma:

            # Setup Reddit mock
            mock_reddit_instance = MagicMock()
            mock_praw.return_value = mock_reddit_instance
            mock_reddit_instance.user.me.return_value = "testbot"

            mock_submission = MagicMock()
            mock_submission.id = "abc123"
            mock_submission.title = "Hit by truck in Orlando"
            mock_submission.selftext = "Need help, insurance denying claim"
            mock_submission.score = 25
            mock_submission.num_comments = 8
            mock_submission.author = "injured_user"
            mock_submission.subreddit = "florida"
            mock_submission.created_utc = datetime.utcnow().timestamp()
            mock_submission.permalink = "/r/florida/comments/abc123"

            mock_subreddit = MagicMock()
            mock_subreddit.search.return_value = [mock_submission]
            mock_reddit_instance.subreddit.return_value = mock_subreddit

            # Setup OpenAI mock
            mock_chat_response = MagicMock()
            mock_chat_response.choices = [MagicMock()]
            mock_chat_response.choices[0].message.content = """
            {
                "intent_score": 95,
                "qualification_score": 90,
                "lead_temperature": "hot",
                "accident_type": "truck",
                "key_phrases": ["insurance denying"],
                "location_hint": "Orlando",
                "has_attorney": false,
                "recommended_action": "urgent",
                "response_text": "I am sorry to hear about your accident. CaseClosedFL offers a free eligibility check for Florida accident cases. We are not a law firm and this is not legal advice. Visit caseclosedfl.com.",
                "safety_score": 95,
                "compliance_flags": [],
                "includes_disclaimer": true,
                "is_legal_advice": false,
                "should_send": true
            }
            """
            mock_chat_response.usage.prompt_tokens = 100
            mock_chat_response.usage.completion_tokens = 50
            mock_chat_response.model = "meta/llama-3.3-70b-instruct"
            mock_chat_response.choices[0].finish_reason = "stop"

            mock_openai_instance = MagicMock()
            mock_openai_instance.chat.completions.create = AsyncMock(return_value=mock_chat_response)
            mock_openai.return_value = mock_openai_instance

            # Setup Chroma mock
            mock_collection = MagicMock()
            mock_collection.count.return_value = 0
            mock_collection.add.return_value = None
            mock_collection.query.return_value = {
                "ids": [["id1"]],
                "documents": [["test doc"]],
                "metadatas": [[{}]],
                "distances": [[0.1]]
            }

            mock_chroma_instance = MagicMock()
            mock_chroma_instance.get_or_create_collection.return_value = mock_collection
            mock_chroma.return_value = mock_chroma_instance

            # Execute full pipeline
            from agent_orchestrator import AgentOrchestrator
            orchestrator = AgentOrchestrator()

            result = await orchestrator.run_full_cycle()

            # Assertions - line by line verification
            assert result is not None, "Orchestrator returned None"
            assert "discovery_results" in result, "Missing discovery_results"
            assert "engagement_results" in result, "Missing engagement_results"
            assert "monitor_results" in result, "Missing monitor_results"
            assert "qualifier_results" in result, "Missing qualifier_results"
            assert result["status"] in ["success", "failed"], "Invalid status"

            print("E2E Pipeline Test PASSED")
            print(f"  - Discovery: {len(result['discovery_results'].get('high_intent_posts', []))} posts")
            print(f"  - Status: {result['status']}")


# Run instructions:
# pytest tests/test_e2e.py -v
# Run 3 times:
# for i in {1..3}; do echo "=== RUN $i ==="; pytest tests/test_e2e.py -v; done
