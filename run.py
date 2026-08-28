#!/usr/bin/env python3
"""
CaseClosedFL Reddit Agent - Main Entry Point
Initializes all systems and starts the 24/7 scheduler.
"""
import logging
import sys
import asyncio
import warnings

from config import get_settings
from database import init_db
from rag_engine import get_rag_engine
from scheduler import AgentScheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("main")


async def startup_sequence():
    """Initialize all systems before starting scheduler."""
    logger.info("=" * 60)
    logger.info("CaseClosedFL Reddit Agent - Startup Sequence")
    logger.info("=" * 60)

    # 1. Database
    logger.info("[1/4] Initializing database...")
    init_db()
    logger.info("    OK Database ready")

    # 2. RAG Knowledge Base
    logger.info("[2/4] Initializing RAG engine...")
    rag = get_rag_engine()
    await rag.initialize_kb()
    logger.info("    OK RAG KB initialized")

    # 3. Reddit client validation (soft fail - log only)
    logger.info("[3/4] Validating Reddit API...")
    try:
        from reddit_client import get_reddit_client
        reddit = get_reddit_client()
        stats = reddit.get_stats()
        logger.info(f"    OK Reddit client ready (daily: {stats['daily_engagements']}/{stats['daily_limit']})")
    except Exception as e:
        logger.warning(f"    Reddit client not ready: {e}. Will retry on first cycle.")

    # 4. Safety guardrails
    logger.info("[4/4] Loading safety guardrails...")
    from safety_guardrails import get_guardrails
    guardrails = get_guardrails()
    safety_stats = guardrails.get_stats()
    logger.info(f"    OK Guardrails active (FL Bar: {safety_stats['florida_bar_compliant']})")

    logger.info("=" * 60)
    logger.info("All systems operational. Starting 24/7 scheduler...")
    logger.info("=" * 60)


def main():
    """Main entry point."""
    settings = get_settings()

    # Soft warnings for missing credentials (no hard exits)
    if not settings.reddit_client_id or settings.reddit_client_id == "your_reddit_client_id":
        logger.warning("REDDIT_CLIENT_ID not configured. Agent will run in monitoring-only mode.")
    if not settings.nvidia_api_key or settings.nvidia_api_key == "nvapi-your-nvidia-key":
        logger.warning("NVIDIA_API_KEY not configured. LLM features disabled.")

    # Run startup
    asyncio.run(startup_sequence())

    # Start scheduler (blocks forever)
    scheduler = AgentScheduler()
    scheduler.start()

    # Keep main thread alive with proper event loop handling
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If APScheduler already started the loop, just block
            import time
            while True:
                time.sleep(1)
        else:
            loop.run_forever()
    except KeyboardInterrupt:
        logger.info("Shutdown signal received...")
        scheduler.shutdown()
        logger.info("Agent stopped gracefully")


if __name__ == "__main__":
    main()
