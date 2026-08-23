"""
CaseClosedFL Reddit Agent - Agent Orchestrator
LangGraph-based state machine coordinating all sub-agents.
"""
import logging
from typing import Dict, Any, TypedDict, Annotated
from datetime import datetime

from langgraph.graph import StateGraph, END

from config import get_settings
from discovery_agent import get_discovery_agent
from engagement_agent import get_engagement_agent
from monitor_agent import get_monitor_agent
from qualifier_agent import get_qualifier_agent
from database import get_db_session, AgentRun

logger = logging.getLogger(__name__)
settings = get_settings()


class AgentState(TypedDict):
    """Shared state across agent workflow."""
    cycle: int
    discovery_results: Dict[str, Any]
    engagement_results: Dict[str, Any]
    monitor_results: Dict[str, Any]
    qualifier_results: Dict[str, Any]
    errors: list
    status: str  # running, success, failed


class AgentOrchestrator:
    """
    Orchestrates the full agent pipeline using LangGraph.
    State machine: Discover -> Engage -> Monitor -> Qualify
    """

    def __init__(self):
        self.discovery = get_discovery_agent()
        self.engagement = get_engagement_agent()
        self.monitor = get_monitor_agent()
        self.qualifier = get_qualifier_agent()

        # Build LangGraph
        self.workflow = StateGraph(AgentState)

        # Add nodes
        self.workflow.add_node("discover", self._run_discovery)
        self.workflow.add_node("engage", self._run_engagement)
        self.workflow.add_node("monitor", self._run_monitor)
        self.workflow.add_node("qualify", self._run_qualify)

        # Add edges
        self.workflow.set_entry_point("discover")
        self.workflow.add_edge("discover", "engage")
        self.workflow.add_edge("engage", "monitor")
        self.workflow.add_edge("monitor", "qualify")
        self.workflow.add_edge("qualify", END)

        self.app = self.workflow.compile()

    async def _run_discovery(self, state: AgentState) -> AgentState:
        """Discovery phase."""
        try:
            results = await self.discovery.run()
            state["discovery_results"] = results
            logger.info(f"Discovery: {len(results.get('high_intent_posts', []))} high-intent posts")
        except Exception as e:
            state["errors"].append(f"discovery: {str(e)}")
            state["status"] = "failed"
        return state

    async def _run_engagement(self, state: AgentState) -> AgentState:
        """Engagement phase."""
        try:
            posts = state["discovery_results"].get("high_intent_posts", [])
            if posts:
                results = await self.engagement.run(posts)
                state["engagement_results"] = results

                # Add monitors for engaged posts
                for post in posts[:10]:  # Monitor top 10
                    await self.monitor.add_monitor(
                        post_id=post["reddit_id"],
                        subreddit=post["subreddit"],
                        title=post["title"],
                        priority=3 if post.get("lead_temperature") == "hot" else 2,
                        reason=f"Intent score: {post.get('intent_score')}"
                    )
        except Exception as e:
            state["errors"].append(f"engagement: {str(e)}")
        return state

    async def _run_monitor(self, state: AgentState) -> AgentState:
        """Monitor phase."""
        try:
            results = await self.monitor.run()
            state["monitor_results"] = results

            # Process any replies through qualifier
            if results.get("replies_requiring_action"):
                qual_results = await self.qualifier.process_new_replies(
                    results["replies_requiring_action"]
                )
                state["qualifier_results"] = qual_results
        except Exception as e:
            state["errors"].append(f"monitor: {str(e)}")
        return state

    async def _run_qualify(self, state: AgentState) -> AgentState:
        """Qualification phase."""
        try:
            results = await self.qualifier.run_daily_qualification()
            state["qualifier_results"] = results
            state["status"] = "success"
        except Exception as e:
            state["errors"].append(f"qualify: {str(e)}")
            state["status"] = "failed"
        return state

    async def run_full_cycle(self) -> AgentState:
        """Execute complete agent cycle."""
        initial_state = AgentState(
            cycle=1,
            discovery_results={},
            engagement_results={},
            monitor_results={},
            qualifier_results={},
            errors=[],
            status="running"
        )

        return await self.app.ainvoke(initial_state)


orchestrator_instance = None


def get_orchestrator() -> AgentOrchestrator:
    global orchestrator_instance
    if orchestrator_instance is None:
        orchestrator_instance = AgentOrchestrator()
    return orchestrator_instance
