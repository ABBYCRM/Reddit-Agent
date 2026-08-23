#!/usr/bin/env python3
"""
CaseClosedFL Reddit Agent - Main Entry Point
Initializes all systems and starts the 24/7 scheduler.
"""
import logging
import sys
import asyncio

from config import get_settings
from database import init_db
from rag_engine import get_rag_engine
from scheduler import AgentScheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('agent.log')
    ]
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

    # 3. Reddit client validation
    logger.info("[3/4] Validating Reddit API...")
    from reddit_client import get_reddit_client
    reddit = get_reddit_client()
    stats = reddit.get_stats()
    logger.info(f"    OK Reddit client ready (daily: {stats['daily_engagements']}/{stats['daily_limit']})")

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

    # Validate critical config
    if not settings.reddit_client_id or settings.reddit_client_id == "your_reddit_client_id":
        logger.error("CRITICAL: Reddit credentials not configured. Set REDDIT_CLIENT_ID in .env")
        sys.exit(1)

    if not settings.nvidia_api_key or settings.nvidia_api_key == "nvapi-your-nvidia-key":
        logger.error("CRITICAL: NVIDIA API key not configured. Get one at build.nvidia.com")
        sys.exit(1)

    # Run startup
    asyncio.run(startup_sequence())

    # Start scheduler (blocks forever)
    scheduler = AgentScheduler()
    scheduler.start()

    try:
        # Keep main thread alive
        while True:
            asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        logger.info("Shutdown signal received...")
        scheduler.shutdown()
        logger.info("Agent stopped gracefully")


if __name__ == "__main__":
    main()
