"""Read-only Reddit discovery and lead scoring."""
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from config import get_settings
from database import AgentRun, db_session
from nvidia_llm import get_nvidia_client
from rag_engine import get_rag_engine
from reddit_client import RedditPost, get_reddit_client
from safety_guardrails import get_guardrails

logger = logging.getLogger(__name__)


class DiscoveryAgent:
    BASE_QUERIES = [
        "car accident Florida", "rear ended Miami", "injured in crash Orlando",
        "hit by car Tampa", "truck accident Jacksonville", "motorcycle crash Florida",
        "insurance denying claim Florida", "medical bills after crash Florida",
    ]

    def __init__(self):
        self.settings = get_settings()
        self.reddit = get_reddit_client()
        self.llm = get_nvidia_client()
        self.rag = get_rag_engine()
        self.guardrails = get_guardrails()

    async def run(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "posts_found": 0, "posts_scored": 0, "posts_indexed": 0, "posts_blocked": 0,
            "high_intent_posts": [], "errors": [],
        }
        with db_session() as db:
            run = AgentRun(agent_name="discovery_agent", run_type="discovery", status="running")
            db.add(run)
            try:
                generated = await self.llm.generate_search_queries(["Florida car accident", "insurance problems"])
                queries = list(dict.fromkeys(self.BASE_QUERIES + generated))[:12]
                unique: Dict[str, RedditPost] = {}
                for subreddit in self.settings.target_subreddits:
                    for query in queries:
                        try:
                            for post in await self.reddit.search_subreddit(subreddit, query, limit=10):
                                unique.setdefault(post.id, post)
                        except Exception as exc:
                            logger.warning("Discovery failed for r/%s: %s", subreddit, exc)
                            results["errors"].append(f"search:{subreddit}:{type(exc).__name__}")

                results["posts_found"] = len(unique)
                for post in unique.values():
                    try:
                        if post.created_utc and datetime.now(timezone.utc) - datetime.fromtimestamp(post.created_utc, timezone.utc) > timedelta(days=7):
                            continue
                        checks = self.guardrails.check_post_eligibility(post.title, post.body, post.subreddit)
                        allowed, reasons = self.guardrails.can_proceed(checks)
                        if not allowed:
                            results["posts_blocked"] += 1
                            continue
                        analysis = await self.llm.analyze_post_intent(post.title, post.body, post.subreddit)
                        results["posts_scored"] += 1
                        await self.rag.add_reddit_post(
                            reddit_id=post.id, subreddit=post.subreddit, content_type="post",
                            title=post.title, body=post.body, author=post.author, score=post.score,
                            num_comments=post.num_comments, intent_tags=analysis.get("key_phrases", []),
                            location_tags=[analysis["location_hint"]] if analysis.get("location_hint") else [],
                            accident_type_tags=[analysis["accident_type"]] if analysis.get("accident_type") else [],
                            reddit_created_utc=datetime.fromtimestamp(post.created_utc, timezone.utc) if post.created_utc else None,
                        )
                        results["posts_indexed"] += 1
                        if analysis["intent_score"] >= self.settings.lead_score_threshold:
                            results["high_intent_posts"].append({
                                "reddit_id": post.id, "subreddit": post.subreddit, "title": post.title,
                                "body": post.body, "author": post.author or "deleted", "intent_score": analysis["intent_score"],
                                "qualification_score": analysis["qualification_score"], "lead_temperature": analysis["lead_temperature"],
                                "recommended_action": analysis["recommended_action"], "url": f"https://www.reddit.com{post.permalink}",
                                "analysis": analysis,
                            })
                    except Exception as exc:
                        logger.exception("Unable to process Reddit post %s", post.id)
                        results["errors"].append(f"post:{post.id}:{type(exc).__name__}")
                results["high_intent_posts"].sort(key=lambda item: item["intent_score"], reverse=True)
                results["high_intent_posts"] = results["high_intent_posts"][:50]
                run.status = "success" if not results["errors"] else "partial"
                run.items_processed = results["posts_found"]
                run.items_created = len(results["high_intent_posts"])
                run.errors_count = len(results["errors"])
                run.log_output = str({key: value for key, value in results.items() if key != "high_intent_posts"})[:10000]
            except Exception as exc:
                run.status = "failed"
                run.error_details = str(exc)[:5000]
                results["errors"].append(f"fatal:{type(exc).__name__}")
            finally:
                run.completed_at = datetime.utcnow()
        return results


_discovery_agent: Optional[DiscoveryAgent] = None


def get_discovery_agent() -> DiscoveryAgent:
    global _discovery_agent
    if _discovery_agent is None:
        _discovery_agent = DiscoveryAgent()
    return _discovery_agent