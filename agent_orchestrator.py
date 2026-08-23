"""Coordinates discovery and response drafting; scheduling owns monitor jobs."""
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from database import AgentRun, db_session
from discovery_agent import get_discovery_agent
from engagement_agent import get_engagement_agent
from monitor_agent import get_monitor_agent

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    def __init__(self):
        self.discovery = get_discovery_agent()
        self.engagement = get_engagement_agent()

    async def run_full_cycle(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"status": "running", "errors": [], "discovery_results": {}, "engagement_results": {}}
        with db_session() as db:
            run = AgentRun(agent_name="orchestrator", run_type="discovery_and_drafting", status="running")
            db.add(run)
            try:
                discovery = await self.discovery.run()
                result["discovery_results"] = discovery
                result["errors"].extend(discovery.get("errors", []))
                posts = discovery.get("high_intent_posts", [])
                if posts:
                    engagement = await self.engagement.run(posts)
                    result["engagement_results"] = engagement
                    if engagement.get("errors"):
                        result["errors"].append(f"engagement_errors:{engagement['errors']}")
                    # Monitor drafted/eligible posts so a later reply can be
                    # reviewed; the monitor itself never sends a response.
                    monitor = get_monitor_agent()
                    for post in posts:
                        if post.get("recommended_action") in {"engage", "urgent"}:
                            await monitor.add_monitor(
                                post["reddit_id"], post["subreddit"], post["title"],
                                priority=3 if post.get("lead_temperature") == "hot" else 2,
                                reason=f"intent_score={post.get('intent_score', 0)}",
                            )
                result["status"] = "success" if not result["errors"] else "partial"
                run.items_processed = discovery.get("posts_found", 0)
                run.items_created = result["engagement_results"].get("leads_created", 0)
                run.errors_count = len(result["errors"])
            except Exception as exc:
                logger.exception("Agent cycle crashed")
                result["status"] = "failed"
                result["errors"].append(str(exc)[:500])
                run.error_details = str(exc)[:5000]
            finally:
                run.status = result["status"]
                run.completed_at = datetime.utcnow()
        return result


_orchestrator: Optional[AgentOrchestrator] = None


def get_orchestrator() -> AgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator