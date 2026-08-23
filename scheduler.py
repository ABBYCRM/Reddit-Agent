"""Single-owner async schedule for worker processes only."""
import logging
from typing import Any, Dict

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from agent_orchestrator import get_orchestrator
from config import get_settings
from heartbeat import HeartbeatMonitor

logger = logging.getLogger(__name__)


class AgentScheduler:
    def __init__(self):
        self.settings = get_settings()
        self.scheduler = AsyncIOScheduler()
        self.orchestrator = get_orchestrator()
        self.heartbeat = HeartbeatMonitor()
        self._configured = False

    def _job(self, func, trigger, job_id: str, name: str):
        self.scheduler.add_job(func, trigger, id=job_id, name=name, replace_existing=True,
                               max_instances=1, coalesce=True, misfire_grace_time=300)

    def setup_jobs(self):
        if self._configured:
            return
        self._job(self._run_discovery_cycle, IntervalTrigger(minutes=self.settings.discovery_interval_minutes), "discovery_cycle", "Reddit discovery and drafting")
        self._job(self._run_monitor_cycle, IntervalTrigger(minutes=self.settings.monitor_interval_minutes), "monitor_cycle", "Reply monitoring")
        self._job(self._run_qualification_cycle, CronTrigger(hour=9, minute=0), "daily_qualification", "Daily qualification")
        self._job(self._heartbeat, IntervalTrigger(seconds=self.settings.heartbeat_interval_seconds), "heartbeat", "System heartbeat")
        self._job(self._cleanup, CronTrigger(hour=3, minute=0), "cleanup", "Daily cleanup")
        self._configured = True

    async def _run_discovery_cycle(self):
        await self.orchestrator.run_full_cycle()

    async def _run_monitor_cycle(self):
        from monitor_agent import get_monitor_agent
        from qualifier_agent import get_qualifier_agent
        results = await get_monitor_agent().run()
        if results.get("replies_requiring_action"):
            await get_qualifier_agent().process_new_replies(results["replies_requiring_action"])

    async def _run_qualification_cycle(self):
        from qualifier_agent import get_qualifier_agent
        await get_qualifier_agent().run_daily_qualification()

    async def _heartbeat(self):
        self.heartbeat.ping()

    async def _cleanup(self):
        logger.info("Daily cleanup complete; retained records require explicit operator retention policy.")

    def start(self):
        if self.scheduler.running:
            return
        self.setup_jobs()
        self.scheduler.start()
        logger.info("Worker scheduler started")

    def shutdown(self):
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)

    def get_status(self) -> Dict[str, Any]:
        return {"running": self.scheduler.running, "jobs": [
            {"id": job.id, "name": job.name, "next_run": str(job.next_run_time)} for job in self.scheduler.get_jobs()
        ], "heartbeat": self.heartbeat.get_status()}