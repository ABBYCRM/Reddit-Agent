#!/usr/bin/env python3
"""Worker entry point. The web process never owns the scheduler."""
import asyncio
import logging
import signal
import sys

from config import get_settings
from composio_client import get_composio_client
from database import init_db
from outreach_service import get_outreach_service
from rag_engine import get_rag_engine
from reddit_client import get_reddit_client
from scheduler import AgentScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("worker")


async def validate_reddit_transport(settings) -> None:
    """Fail startup when the selected Reddit provider is unavailable or incompatible."""
    if settings.reddit_transport == "composio":
        # Fail the worker rather than silently processing no Reddit data when
        # the configured Composio connection or action contract is unavailable.
        if settings.app_env.lower() == "staging":
            contract = await get_reddit_client().validate_read_only_contract()
            logger.info(
                "Validated read-only Reddit staging contract: account=%s action=%s posts_checked=%s",
                contract["connected_account_id"],
                contract["action"],
                contract["posts_checked"],
            )
        else:
            await get_composio_client().get_connected_account()


async def main():
    settings = get_settings()
    init_db()
    await get_rag_engine().initialize_kb()
    await validate_reddit_transport(settings)
    outreach = await get_outreach_service().get_status()
    if outreach["enabled"] and not outreach["readiness"]["ready"]:
        raise RuntimeError(
            "Autonomous outreach is enabled but required comment and DM capabilities are unavailable"
        )
    scheduler = AgentScheduler()
    scheduler.start()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    logger.info(
        "Worker started; Reddit transport=%s autonomous_outreach=%s readiness=%s",
        settings.reddit_transport,
        outreach["enabled"],
        outreach["readiness"]["ready"],
    )
    await stop.wait()
    scheduler.shutdown()
    logger.info("Worker stopped")


if __name__ == "__main__":
    asyncio.run(main())