"""Generate humanized drafts, then route every send through the durable outbox."""
import hashlib
import logging
from typing import Any, Dict, Optional

from config import get_settings
from database import Engagement, Lead, db_session
from nvidia_llm import get_nvidia_client
from outreach_service import get_outreach_service
from rag_engine import get_rag_engine
from safety_guardrails import get_guardrails

logger = logging.getLogger(__name__)


class EngagementAgent:
    def __init__(self):
        self.settings = get_settings()
        self.llm = get_nvidia_client()
        self.rag = get_rag_engine()
        self.guardrails = get_guardrails()
        self.outreach = get_outreach_service()

    @staticmethod
    def _key(post_id: str, response: str) -> str:
        return f"comment:{post_id}:{hashlib.sha256(response.encode()).hexdigest()[:24]}"

    async def process_post(self, post_data: Dict[str, Any]) -> Dict[str, Any]:
        post_id = post_data["reddit_id"]
        result = {"post_id": post_id, "subreddit": post_data["subreddit"], "response_created": False,
                  "response_sent": False, "response_queued": False, "safety_passed": False,
                  "lead_created": False, "errors": []}
        try:
            context = await self.rag.get_context_for_response(post_data["title"], post_data.get("body", ""))
            context += await self.rag.get_kb_context(post_data["title"])
            response_data = await self.llm.craft_response(
                post_data["title"], post_data.get("body", ""), post_data["subreddit"], context,
            )
            response_text = response_data.get("response_text", "")
            result["response_created"] = bool(response_text)
            checks = self.guardrails.check_response_compliance(response_text)
            allowed, reasons = self.guardrails.can_proceed(checks)
            if not allowed:
                result["errors"].extend(reasons)
                return result
            result["safety_passed"] = True
            key = self._key(post_id, response_text)
            should_send_comment = bool(
                response_data.get("should_send_comment")
                and not response_data.get("is_legal_advice")
            )
            should_send_dm = bool(
                response_data.get("should_send_dm")
                and response_data.get("dm_text")
                and post_data.get("author")
                and post_data.get("author") != "deleted"
                and not response_data.get("is_legal_advice")
            )
            dm_text = str(response_data.get("dm_text") or "")
            if should_send_dm:
                dm_checks = self.guardrails.check_response_compliance(dm_text)
                dm_allowed, dm_reasons = self.guardrails.can_proceed(dm_checks)
                if not dm_allowed:
                    should_send_dm = False
                    result["errors"].extend(dm_reasons)
            with db_session() as db:
                existing = db.query(Engagement).filter(
                    (Engagement.idempotency_key == key) |
                    ((Engagement.reddit_post_id == post_id) & (Engagement.engagement_type == "comment"))
                ).first()
                if existing:
                    result["response_sent"] = existing.outbound_status == "sent"
                    result["response_queued"] = existing.outbound_status in {"queued", "pending"}
                    result["duplicate"] = True
                    return result

                lead = db.query(Lead).filter_by(reddit_post_id=post_id).first()
                if not lead:
                    analysis = post_data.get("analysis", {})
                    lead = Lead(
                        reddit_username=post_data.get("author") or "unknown",
                        reddit_post_id=post_id, subreddit=post_data["subreddit"],
                        intent_score=post_data.get("intent_score", 0),
                        qualification_score=post_data.get("qualification_score", 0),
                        lead_temperature=post_data.get("lead_temperature", "cold"),
                        accident_type=analysis.get("accident_type"),
                        injury_description=str(analysis.get("injury_severity", "")),
                        location_hint=analysis.get("location_hint"),
                        has_attorney=analysis.get("has_attorney"),
                        fault_indicated=str(analysis.get("fault_discussed", "")),
                        source_url=post_data.get("url", f"https://www.reddit.com/comments/{post_id}"),
                    )
                    db.add(lead)
                    db.flush()
                    result["lead_created"] = True
                engagement = Engagement(
                    lead_id=lead.id, engagement_type="comment", reddit_post_id=post_id,
                    subreddit=post_data["subreddit"], original_post_title=post_data["title"],
                    original_post_body=post_data.get("body", ""), our_response=response_text,
                    compliance_check_passed=True, compliance_flags=response_data.get("compliance_flags", []),
                    safety_score=response_data.get("safety_score", 0), outbound_status="queued",
                    idempotency_key=key,
                )
                db.add(engagement)
                db.flush()
                engagement_id = engagement.id
            # The outreach service creates an action row and reserves the
            # cross-process quota before touching Reddit. Unknown outcomes are
            # intentionally never retried automatically.
            if not should_send_comment and not should_send_dm:
                result["response_queued"] = True
                return result

            if should_send_comment:
                comment_result = await self.outreach.send_comment(engagement_id, post_id, response_text)
                result["response_sent"] = bool(comment_result.get("sent"))
                result["response_queued"] = comment_result.get("status") == "queued"
                if comment_result.get("reason"):
                    result["errors"].append(str(comment_result["reason"]))
            if should_send_dm:
                dm_result = await self.outreach.send_dm(
                    engagement_id,
                    post_id,
                    str(post_data["author"]),
                    str(response_data.get("dm_subject") or "General information from CaseClosedFL"),
                    dm_text,
                )
                result["dm_sent"] = bool(dm_result.get("sent"))
                result["response_sent"] = result["response_sent"] or result["dm_sent"]
                result["response_queued"] = result["response_queued"] or dm_result.get("status") == "queued"
                if dm_result.get("reason"):
                    result["errors"].append(str(dm_result["reason"]))
            return result
        except Exception as exc:
            logger.exception("Engagement failed for %s", post_id)
            result["errors"].append(str(exc)[:500])
            return result

    async def run(self, high_intent_posts: list) -> Dict[str, Any]:
        results = {"processed": 0, "responses_created": 0, "responses_sent": 0, "leads_created": 0, "errors": 0}
        for post in high_intent_posts:
            if post.get("recommended_action") not in {"engage", "urgent"}:
                continue
            item = await self.process_post(post)
            results["processed"] += 1
            results["responses_created"] += int(item["response_created"])
            results["responses_sent"] += int(item["response_sent"])
            results["leads_created"] += int(item["lead_created"])
            results["errors"] += int(bool(item["errors"]))
        return results


_engagement_agent: Optional[EngagementAgent] = None


def get_engagement_agent() -> EngagementAgent:
    global _engagement_agent
    if _engagement_agent is None:
        _engagement_agent = EngagementAgent()
    return _engagement_agent