#!/usr/bin/env python3
"""Worker entry point. The web process never owns the scheduler."""
import asyncio
import logging
import signal
import sys

from config import get_settings
from composio_client import get_composio_client
from database import init_db
from rag_engine import get_rag_engine
from scheduler import AgentScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("worker")


async def main():
    settings = get_settings()
    init_db()
    await get_rag_engine().initialize_kb()
    if settings.reddit_transport == "composio":
        # Fail the worker rather than silently processing no Reddit data when
        # the configured Composio connection is unavailable.
        await get_composio_client().get_connected_account()
    scheduler = AgentScheduler()
    scheduler.start()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    logger.info("Worker started; Reddit transport=%s auto_reply=%s", settings.reddit_transport, settings.enable_auto_reply)
    await stop.wait()
    scheduler.shutdown()
    logger.info("Worker stopped")


if __name__ == "__main__":
    asyncio.run(main())