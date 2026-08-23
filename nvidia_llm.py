"""
CaseClosedFL Reddit Agent - NVIDIA NIM LLM Client
Uses NVIDIA's free tier (build.nvidia.com) with OpenAI-compatible endpoints.
Supports chat completion, embeddings, and structured output.
"""
import json
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class LLMResponse:
    """Structured LLM response."""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    raw_response: Any = None


class NVIDIAClient:
    """
    NVIDIA NIM API client for CaseClosedFL agents.
    Free tier: ~1,000 inference credits, 40 RPM.
    Production: Upgrade at build.nvidia.com
    """

    def __init__(self):
        self.client = AsyncOpenAI(
            base_url=settings.nvidia_base_url,
            api_key=settings.nvidia_api_key,
        )
        self.model = settings.nvidia_model
        self.embedding_model = settings.nvidia_embedding_model
        self._request_count = 0
        self._error_count = 0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        json_mode: bool = False,
        model: Optional[str] = None
    ) -> LLMResponse:
        """
        Send chat completion request to NVIDIA NIM.

        Args:
            messages: List of {"role": "system|user|assistant", "content": "..."}
            temperature: 0.0-1.0, lower = more deterministic
            max_tokens: Max response length
            json_mode: Force JSON output
            model: Override default model
        """
        model = model or self.model

        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = await self.client.chat.completions.create(**kwargs)
            self._request_count += 1

            content = response.choices[0].message.content

            return LLMResponse(
                content=content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                finish_reason=response.choices[0].finish_reason,
                raw_response=response
            )
        except Exception as e:
            self._error_count += 1
            logger.error(f"NVIDIA NIM chat error: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts using NVIDIA embedding model."""
        try:
            response = await self.client.embeddings.create(
                model=self.embedding_model,
                input=texts,
                encoding_format="float"
            )
            self._request_count += 1
            return [item.embedding for item in response.data]
        except Exception as e:
            self._error_count += 1
            logger.error(f"NVIDIA NIM embedding error: {e}")
            raise

    async def analyze_post_intent(
        self,
        post_title: str,
        post_body: str,
        subreddit: str
    ) -> Dict[str, Any]:
        """
        Analyze a Reddit post for personal injury lead intent.
        Returns structured analysis with scores and tags.
        """
        system_prompt = """You are a lead qualification analyst for CaseClosedFL, 
a Florida accident intake service. Analyze Reddit posts to identify potential 
personal injury leads. 

Respond ONLY with valid JSON in this exact format:
{
  "intent_score": 0-100,
  "qualification_score": 0-100,
  "lead_temperature": "cold|warm|hot",
  "accident_type": "car|truck|motorcycle|pedestrian|rideshare|slip_fall|other|null",
  "injury_mentioned": true|false,
  "injury_severity": "minor|moderate|severe|unknown",
  "location_hint": "city or region mentioned, or null",
  "fault_discussed": true|false,
  "has_attorney": true|false,
  "time_since_accident": "days|weeks|months|years|unknown",
  "key_phrases": ["phrase1", "phrase2"],
  "recommended_action": "ignore|monitor|engage|urgent",
  "engagement_angle": "brief suggestion for helpful response",
  "compliance_flags": ["flag1"]
}

Rules:
- intent_score: How likely this person needs legal help (0-100)
- qualification_score: How well they match CaseClosedFL criteria (0-100)
- Never make legal claims or give legal advice
- Flag any post that seems to already have an attorney
- Be conservative - better to miss a lead than spam"""

        user_prompt = f"""Subreddit: r/{subreddit}
Post Title: {post_title}
Post Body: {post_body[:3000]}

Analyze this post for personal injury lead potential."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = await self.chat(messages, temperature=0.1, json_mode=True)

        try:
            result = json.loads(response.content)
            return result
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from LLM: {response.content[:500]}")
            return {
                "intent_score": 0,
                "qualification_score": 0,
                "lead_temperature": "cold",
                "recommended_action": "ignore",
                "compliance_flags": ["parse_error"]
            }

    async def craft_response(
        self,
        post_title: str,
        post_body: str,
        subreddit: str,
        context_docs: List[str],
        tone: str = "helpful_professional"
    ) -> Dict[str, Any]:
        """
        Craft a compliant, helpful Reddit response.
        Returns response text + safety metadata.
        """
        system_prompt = f"""You are a helpful, empathetic representative of CaseClosedFL, 
a Florida accident intake and eligibility screening service. You are NOT a lawyer and 
cannot give legal advice. Your job is to provide general information and offer our 
free eligibility check if it seems appropriate.

TONE: {tone}

CRITICAL COMPLIANCE RULES:
1. NEVER give legal advice or predict case outcomes
2. NEVER claim to be a law firm or attorney
3. ALWAYS disclose: "CaseClosedFL is not a law firm and does not provide legal advice"
4. NEVER solicit DMs aggressively - only offer help if asked
5. NEVER mention specific dollar amounts or settlements
6. ALWAYS be empathetic and genuinely helpful first
7. NEVER spam - only respond when you can add real value
8. Include: "This is general information, not legal advice"
9. If user already has an attorney, do NOT engage further
10. Keep responses under 150 words

CaseClosedFL offers:
- Free accident eligibility check (caseclosedfl.com)
- Florida statewide coverage
- No obligation intake screening
- Connection to participating attorneys if qualified

RELEVANT CONTEXT FROM SIMILAR CASES:
{chr(10).join(f"- {doc[:300]}" for doc in context_docs[:3])}

Respond with JSON:
{{
  "response_text": "the actual Reddit comment text",
  "safety_score": 0-100,
  "compliance_flags": ["flag1"],
  "includes_disclaimer": true|false,
  "is_legal_advice": true|false,
  "should_send": true|false
}}"""

        user_prompt = f"""Subreddit: r/{subreddit}
Post Title: {post_title}
Post Body: {post_body[:2000]}

Craft a helpful, compliant response."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = await self.chat(messages, temperature=0.4, max_tokens=500, json_mode=True)

        try:
            result = json.loads(response.content)
            return result
        except json.JSONDecodeError:
            return {
                "response_text": "I understand you're going through a difficult situation. CaseClosedFL offers a free eligibility check for Florida accident cases at caseclosedfl.com. We're not a law firm, but we can help connect you with resources. This is general information, not legal advice.",
                "safety_score": 80,
                "compliance_flags": ["parse_error_fallback"],
                "includes_disclaimer": True,
                "is_legal_advice": False,
                "should_send": True
            }

    async def generate_search_queries(self, topics: List[str]) -> List[str]:
        """Generate optimized Reddit search queries from topics."""
        system_prompt = """Generate 5-10 Reddit search queries to find people 
who may need personal injury help in Florida. Each query should be a simple 
Reddit search string (no boolean operators needed). Focus on natural language 
phrases people actually use.

Respond with JSON: {"queries": ["query1", "query2", ...]}"""

        user_prompt = f"""Topics: {', '.join(topics)}
Target: Florida accident victims needing legal help screening
Generate search queries."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = await self.chat(messages, temperature=0.7, json_mode=True)

        try:
            result = json.loads(response.content)
            return result.get("queries", topics)
        except json.JSONDecodeError:
            return topics

    def get_stats(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "embedding_model": self.embedding_model,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(self._request_count, 1)
        }


# Singleton
_nvidia_client: Optional[NVIDIAClient] = None


def get_nvidia_client() -> NVIDIAClient:
    global _nvidia_client
    if _nvidia_client is None:
        _nvidia_client = NVIDIAClient()
    return _nvidia_client
