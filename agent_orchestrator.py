"""
CaseClosedFL Reddit Agent - Agent Orchestrator
Simple async pipeline without langgraph dependency.
"""
import logging
from typing import Dict, Any
from datetime import datetime

from config import get_settings
from discovery_agent import get_discovery_agent
from engagement_agent import get_engagement_agent
from monitor_agent import get_monitor_agent
from qualifier_agent import get_qualifier_agent
from database import get_db_session, AgentRun

logger = logging.getLogger(__name__)
settings = get_settings()


class AgentOrchestrator:
    """Orchestrates the full agent pipeline. Simple async, no langgraph."""

    def __init__(self):
        self.discovery = get_discovery_agent()
        self.engagement = get_engagement_agent()
        self.monitor = get_monitor_agent()
        self.qualifier = get_qualifier_agent()

    async def run_full_cycle(self) -> Dict[str, Any]:
        """Execute complete agent cycle: Discover -> Engage -> Monitor -> Qualify."""
        run_id = None
        try:
            db = get_db_session()
            run_record = AgentRun(
                agent_name="orchestrator",
                run_type="full_cycle",
                status="running",
                started_at=datetime.utcnow()
            )
            db.add(run_record)
            db.commit()
            run_id = run_record.id

            result = {
                "status": "running",
                "errors": [],
                "discovery_results": {},
                "engagement_results": {},
                "monitor_results": {},
                "qualifier_results": {}
            }

            # Phase 1: Discovery
            logger.info("=== PHASE 1: DISCOVERY ===")
            try:
                disc = await self.discovery.run()
                result["discovery_results"] = disc
                logger.info(f"Discovery: {len(disc.get('high_intent_posts', []))} high-intent posts")
            except Exception as e:
                logger.error(f"Discovery failed: {e}")
                result["errors"].append(f"discovery: {str(e)}")

            # Phase 2: Engagement
            logger.info("=== PHASE 2: ENGAGEMENT ===")
            try:
                posts = result["discovery_results"].get("high_intent_posts", [])
                if posts:
                    eng = await self.engagement.run(posts)
                    result["engagement_results"] = eng

                    # Add monitors for engaged posts
                    for post in posts[:10]:
                        await self.monitor.add_monitor(
                            post_id=post["reddit_id"],
                            subreddit=post["subreddit"],
                            title=post["title"],
                            priority=3 if post.get("lead_temperature") == "hot" else 2,
                            reason=f"Intent: {post.get('intent_score')}"
                        )
            except Exception as e:
                logger.error(f"Engagement failed: {e}")
                result["errors"].append(f"engagement: {str(e)}")

            # Phase 3: Monitor
            logger.info("=== PHASE 3: MONITOR ===")
            try:
                mon = await self.monitor.run()
                result["monitor_results"] = mon

                if mon.get("replies_requiring_action"):
                    qual = await self.qualifier.process_new_replies(mon["replies_requiring_action"])
                    result["qualifier_results"] = qual
            except Exception as e:
                logger.error(f"Monitor failed: {e}")
                result["errors"].append(f"monitor: {str(e)}")

            # Phase 4: Qualify
            logger.info("=== PHASE 4: QUALIFY ===")
            try:
                qual = await self.qualifier.run_daily_qualification()
                result["qualifier_results"] = qual
            except Exception as e:
                logger.error(f"Qualify failed: {e}")
                result["errors"].append(f"qualify: {str(e)}")

            # Finalize
            result["status"] = "success" if not result["errors"] else "partial"

            run_record.status = result["status"]
            run_record.completed_at = datetime.utcnow()
            db.commit()

            return result

        except Exception as e:
            logger.error(f"Orchestrator crashed: {e}")
            if run_id:
                db = get_db_session()
                run_record = db.query(AgentRun).filter(AgentRun.id == run_id).first()
                if run_record:
                    run_record.status = "failed"
                    run_record.completed_at = datetime.utcnow()
                    run_record.error_details = str(e)
                    db.commit()
            return {"status": "failed", "errors": [str(e)]}


orchestrator_instance = None


def get_orchestrator() -> AgentOrchestrator:
    global orchestrator_instance
    if orchestrator_instance is None:
        orchestrator_instance = AgentOrchestrator()
    return orchestrator_instance
