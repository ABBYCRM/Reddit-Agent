"""
CaseClosedFL Reddit Agent - Scheduler
APScheduler for cron-like execution of agent cycles.
"""
import logging
import asyncio
from typing import Dict, Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from config import get_settings
from agent_orchestrator import get_orchestrator
from heartbeat import HeartbeatMonitor

logger = logging.getLogger(__name__)
settings = get_settings()


class AgentScheduler:
    """
    Production scheduler for 24/7 autonomous operation.
    Manages all cron jobs, polling intervals, and heartbeats.
    """

    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.orchestrator = get_orchestrator()
        self.heartbeat = HeartbeatMonitor()

    def setup_jobs(self):
        """Configure all scheduled jobs."""

        # Main discovery cycle - every 30 minutes
        self.scheduler.add_job(
            self._run_discovery_cycle,
            IntervalTrigger(minutes=settings.discovery_interval_minutes),
            id="discovery_cycle",
            name="Reddit Discovery & Engagement",
            replace_existing=True,
            max_instances=1
        )

        # Monitor cycle - every 15 minutes
        self.scheduler.add_job(
            self._run_monitor_cycle,
            IntervalTrigger(minutes=settings.monitor_interval_minutes),
            id="monitor_cycle",
            name="Reply Monitoring",
            replace_existing=True,
            max_instances=1
        )

        # Daily qualification - once per day at 9 AM
        self.scheduler.add_job(
            self._run_qualification_cycle,
            CronTrigger(hour=9, minute=0),
            id="daily_qualification",
            name="Daily Lead Re-qualification",
            replace_existing=True
        )

        # Heartbeat - every 60 seconds
        self.scheduler.add_job(
            self._heartbeat,
            IntervalTrigger(seconds=settings.heartbeat_interval_seconds),
            id="heartbeat",
            name="System Heartbeat",
            replace_existing=True
        )

        # Cleanup - daily at 3 AM
        self.scheduler.add_job(
            self._cleanup,
            CronTrigger(hour=3, minute=0),
            id="cleanup",
            name="Daily Cleanup",
            replace_existing=True
        )

        logger.info("Scheduler configured with 5 jobs")

    async def _run_discovery_cycle(self):
        """Run full discovery + engagement cycle."""
        logger.info("=== STARTING DISCOVERY CYCLE ===")
        try:
            result = await self.orchestrator.run_full_cycle()
            logger.info(f"Cycle complete. Status: {result['status']}")
            if result['errors']:
                logger.warning(f"Cycle errors: {result['errors']}")
        except Exception as e:
            logger.error(f"Discovery cycle failed: {e}")

    async def _run_monitor_cycle(self):
        """Run monitoring cycle."""
        from monitor_agent import get_monitor_agent
        from qualifier_agent import get_qualifier_agent

        try:
            monitor = get_monitor_agent()
            results = await monitor.run()

            if results.get("replies_requiring_action"):
                qualifier = get_qualifier_agent()
                await qualifier.process_new_replies(results["replies_requiring_action"])

            logger.info(f"Monitor cycle: {results['posts_checked']} posts checked")
        except Exception as e:
            logger.error(f"Monitor cycle failed: {e}")

    async def _run_qualification_cycle(self):
        """Run daily qualification."""
        from qualifier_agent import get_qualifier_agent
        try:
            qualifier = get_qualifier_agent()
            results = await qualifier.run_daily_qualification()
            logger.info(f"Qualification: {results['leads_scored']} leads scored")
        except Exception as e:
            logger.error(f"Qualification cycle failed: {e}")

    async def _heartbeat(self):
        """System heartbeat."""
        self.heartbeat.ping()

    async def _cleanup(self):
        """Daily cleanup tasks."""
        logger.info("Running daily cleanup...")
        # Archive old data, reset counters, etc.
        pass

    def start(self):
        """Start the scheduler."""
        self.setup_jobs()
        self.scheduler.start()
        logger.info("Scheduler started - Agent is running 24/7")

    def shutdown(self):
        """Graceful shutdown."""
        self.scheduler.shutdown()
        logger.info("Scheduler shutdown complete")

    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": str(job.next_run_time)
            })

        return {
            "running": self.scheduler.running,
            "jobs": jobs,
            "heartbeat": self.heartbeat.get_status()
        }
