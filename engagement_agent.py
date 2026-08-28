"""
CaseClosedFL Reddit Agent - Engagement Agent
Crafts and sends compliant Reddit responses.
Uses RAG context + safety checks before every engagement.
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from config import get_settings
from reddit_client import get_reddit_client
from nvidia_llm import get_nvidia_client
from rag_engine import get_rag_engine
from safety_guardrails import get_guardrails
from database import get_db_session, Engagement, Lead

logger = logging.getLogger(__name__)
settings = get_settings()


class EngagementAgent:
    """
    Agent responsible for engaging with qualified Reddit posts.
    NEVER sends without human approval unless AUTO_REPLY is explicitly enabled.
    """

    def __init__(self):
        self.reddit = get_reddit_client()
        self.llm = get_nvidia_client()
        self.rag = get_rag_engine()
        self.guardrails = get_guardrails()

    async def process_post(self, post_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single high-intent post.
        1. Get RAG context
        2. Craft response with LLM
        3. Run safety checks
        4. Store in DB (send only if auto-reply enabled)
        """
        result = {
            "post_id": post_data["reddit_id"],
            "subreddit": post_data["subreddit"],
            "response_created": False,
            "response_sent": False,
            "safety_passed": False,
            "lead_created": False,
            "errors": []
        }

        try:
            # Step 1: Get context from RAG
            context_docs = await self.rag.get_context_for_response(
                post_title=post_data["title"],
                post_body=post_data.get("body", "")
            )
            kb_context = await self.rag.get_kb_context(post_data["title"])

            all_context = context_docs + kb_context

            # Step 2: Craft response
            response_data = await self.llm.craft_response(
                post_title=post_data["title"],
                post_body=post_data.get("body", ""),
                subreddit=post_data["subreddit"],
                context_docs=all_context,
                tone="helpful_empathetic"
            )

            response_text = response_data.get("response_text", "")
            result["response_created"] = True

            # Step 3: Safety compliance check
            safety_checks = self.guardrails.check_response_compliance(response_text)
            can_send, block_reasons = self.guardrails.can_proceed(safety_checks)

            if not can_send:
                logger.warning(f"Response blocked for {post_data['reddit_id']}: {block_reasons}")
                result["errors"].extend(block_reasons)
                return result

            result["safety_passed"] = True

            # Step 4: Create or update lead record
            db = get_db_session()
            lead = db.query(Lead).filter(Lead.reddit_post_id == post_data["reddit_id"]).first()

            if not lead:
                lead = Lead(
                    reddit_username=post_data.get("author", "unknown"),
                    reddit_post_id=post_data["reddit_id"],
                    subreddit=post_data["subreddit"],
                    intent_score=post_data.get("intent_score", 0),
                    qualification_score=post_data.get("qualification_score", 0),
                    lead_temperature=post_data.get("lead_temperature", "cold"),
                    accident_type=post_data.get("analysis", {}).get("accident_type"),
                    injury_description=str(post_data.get("analysis", {}).get("injury_severity", "")),
                    location_hint=post_data.get("analysis", {}).get("location_hint"),
                    has_attorney=post_data.get("analysis", {}).get("has_attorney"),
                    fault_indicated=post_data.get("analysis", {}).get("fault_discussed"),
                    source_url=f"https://reddit.com/r/{post_data['subreddit']}/comments/{post_data['reddit_id']}",
                    status="new"
                )
                db.add(lead)
                db.commit()
                result["lead_created"] = True

            # Step 5: Store engagement record
            engagement = Engagement(
                lead_id=lead.id,
                engagement_type="comment",
                reddit_post_id=post_data["reddit_id"],
                subreddit=post_data["subreddit"],
                original_post_title=post_data["title"],
                original_post_body=post_data.get("body", ""),
                our_response=response_text,
                compliance_check_passed=True,
                safety_score=response_data.get("safety_score", 0),
                created_at=datetime.utcnow()
            )
            db.add(engagement)
            db.commit()

            # Step 6: Send ONLY if auto-reply is enabled
            if settings.enable_auto_reply and response_data.get("should_send", False):
                comment = self.reddit.post_comment(
                    submission_id=post_data["reddit_id"],
                    text=response_text
                )
                if comment:
                    engagement.reddit_comment_id = comment.id
                    engagement.reddit_created_utc = datetime.utcfromtimestamp(comment.created_utc)
                    db.commit()
                    result["response_sent"] = True
                    lead.last_engaged_at = datetime.utcnow()
                    db.commit()
                    logger.info(f"Comment sent to {post_data['reddit_id']}")
            else:
                logger.info(f"Response queued (auto-reply disabled): {post_data['reddit_id']}")
                result["response_queued"] = True

            return result

        except Exception as e:
            logger.error(f"Engagement failed for {post_data['reddit_id']}: {e}")
            result["errors"].append(str(e))
            return result

    async def run(self, high_intent_posts: list) -> Dict[str, Any]:
        """Run engagement cycle on a batch of posts."""
        results = {
            "processed": 0,
            "responses_created": 0,
            "responses_sent": 0,
            "leads_created": 0,
            "errors": 0
        }

        for post in high_intent_posts:
            if post.get("recommended_action") not in ["engage", "urgent"]:
                continue

            res = await self.process_post(post)
            results["processed"] += 1
            if res["response_created"]:
                results["responses_created"] += 1
            if res.get("response_sent"):
                results["responses_sent"] += 1
            if res["lead_created"]:
                results["leads_created"] += 1
            if res["errors"]:
                results["errors"] += 1

        return results


engagement_agent_instance: Optional[EngagementAgent] = None


def get_engagement_agent() -> EngagementAgent:
    global engagement_agent_instance
    if engagement_agent_instance is None:
        engagement_agent_instance = EngagementAgent()
    return engagement_agent_instance
