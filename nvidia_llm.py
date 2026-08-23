"""Defensive NVIDIA NIM wrapper with safe, non-sending fallbacks."""
import json
import logging
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from config import get_settings

logger = logging.getLogger(__name__)


class NVIDIAClient:
    def __init__(self):
        self.settings = get_settings()
        self.client = AsyncOpenAI(base_url=self.settings.nvidia_base_url, api_key=self.settings.nvidia_api_key) if self.settings.nvidia_api_key else None
        self.model = self.settings.nvidia_model
        self.embedding_model = self.settings.nvidia_embedding_model
        self._request_count = 0
        self._error_count = 0

    async def _json(self, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> Optional[Dict[str, Any]]:
        if not self.client:
            return None
        try:
            response = await self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature,
                max_tokens=max_tokens, response_format={"type": "json_object"},
            )
            self._request_count += 1
            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else None
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            self._error_count += 1
            logger.warning("NVIDIA returned an invalid structured response: %s", exc)
        except Exception as exc:
            self._error_count += 1
            logger.warning("NVIDIA request failed: %s", exc)
        return None

    async def embed(self, texts: List[str]) -> List[List[float]]:
        if not self.client:
            raise RuntimeError("NVIDIA_API_KEY is not configured")
        response = await self.client.embeddings.create(model=self.embedding_model, input=texts, encoding_format="float")
        self._request_count += 1
        return [item.embedding for item in response.data]

    async def analyze_post_intent(self, post_title: str, post_body: str, subreddit: str) -> Dict[str, Any]:
        fallback = {
            "intent_score": 0, "qualification_score": 0, "lead_temperature": "cold",
            "accident_type": None, "injury_mentioned": False, "injury_severity": "unknown",
            "location_hint": None, "fault_discussed": False, "has_attorney": False,
            "time_since_accident": "unknown", "key_phrases": [], "recommended_action": "monitor",
            "engagement_angle": "", "compliance_flags": ["llm_unavailable_or_invalid"],
        }
        parsed = await self._json(
            [
                {"role": "system", "content": "Return JSON only. Assess accident-intake relevance conservatively. Never recommend legal advice or automated outreach. Required keys: intent_score, qualification_score, lead_temperature, accident_type, injury_mentioned, injury_severity, location_hint, fault_discussed, has_attorney, time_since_accident, key_phrases, recommended_action, engagement_angle, compliance_flags."},
                {"role": "user", "content": f"Subreddit: {subreddit}\nTitle: {post_title}\nBody: {post_body[:3000]}"},
            ], 0.1, 600,
        )
        if not parsed:
            return fallback
        result = {**fallback, **parsed}
        try:
            result["intent_score"] = max(0, min(100, int(result["intent_score"])))
            result["qualification_score"] = max(0, min(100, int(result["qualification_score"])))
        except (TypeError, ValueError):
            return fallback
        if result["recommended_action"] not in {"ignore", "monitor", "engage", "urgent"}:
            result["recommended_action"] = "monitor"
        if result["lead_temperature"] not in {"cold", "warm", "hot"}:
            result["lead_temperature"] = "cold"
        if not isinstance(result["key_phrases"], list):
            result["key_phrases"] = []
        return result

    async def craft_response(self, post_title: str, post_body: str, subreddit: str,
                             context_docs: List[str], tone: str = "helpful_professional") -> Dict[str, Any]:
        fallback = {
            "response_text": "CaseClosedFL is not a law firm. This is general information, not legal advice.",
            "safety_score": 0, "compliance_flags": ["llm_unavailable_or_invalid"],
            "includes_disclaimer": True, "is_legal_advice": False, "should_send": False,
        }
        parsed = await self._json(
            [
                {"role": "system", "content": "Return JSON only. Draft no more than 150 words. Include exactly these disclosure concepts: not a law firm, not legal advice, general information. Do not advise, promise results, solicit DMs, or ask for contact details. Required keys: response_text, safety_score, compliance_flags, includes_disclaimer, is_legal_advice, should_send. Set should_send false."},
                {"role": "user", "content": f"r/{subreddit}\n{post_title}\n{post_body[:2000]}\nContext: {' '.join(context_docs[:2])[:600]}"},
            ], 0.2, 400,
        )
        if not parsed or not isinstance(parsed.get("response_text"), str):
            return fallback
        result = {**fallback, **parsed}
        result["should_send"] = False
        result["is_legal_advice"] = bool(result.get("is_legal_advice", False))
        try:
            result["safety_score"] = max(0, min(100, float(result["safety_score"])))
        except (TypeError, ValueError):
            result["safety_score"] = 0
        return result

    async def generate_search_queries(self, topics: List[str]) -> List[str]:
        parsed = await self._json(
            [{"role": "system", "content": "Return JSON only with a 'queries' array of up to 10 neutral Reddit search terms."},
             {"role": "user", "content": ", ".join(topics)}], 0.3, 200,
        )
        queries = parsed.get("queries") if parsed else None
        return [str(query)[:120] for query in queries[:10]] if isinstance(queries, list) else topics

    def get_stats(self) -> Dict[str, Any]:
        return {"model": self.model, "configured": bool(self.client), "request_count": self._request_count, "error_count": self._error_count}


_nvidia_client: Optional[NVIDIAClient] = None


def get_nvidia_client() -> NVIDIAClient:
    global _nvidia_client
    if _nvidia_client is None:
        _nvidia_client = NVIDIAClient()
    return _nvidia_client