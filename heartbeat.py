"""
CaseClosedFL Reddit Agent - Heartbeat Monitor
Tracks system health, Reddit API status, and agent vitals.
"""
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class HeartbeatMonitor:
    """
    Simple heartbeat system to ensure agent is alive.
    Logs vitals every interval. Can trigger alerts if missed.
    """

    def __init__(self):
        self.last_ping = time.time()
        self.ping_count = 0
        self.start_time = time.time()
        self.errors_since_start = 0

    def ping(self):
        """Record heartbeat."""
        self.last_ping = time.time()
        self.ping_count += 1

        # Log vitals every 10 pings (~10 minutes)
        if self.ping_count % 10 == 0:
            uptime = timedelta(seconds=int(time.time() - self.start_time))
            logger.info(f"Heartbeat #{self.ping_count} | Uptime: {uptime}")

    def get_status(self) -> Dict[str, Any]:
        """Get current heartbeat status."""
        now = time.time()
        seconds_since_ping = now - self.last_ping

        status = "healthy"
        if seconds_since_ping > settings.heartbeat_interval_seconds * 3:
            status = "warning"
        if seconds_since_ping > settings.heartbeat_interval_seconds * 5:
            status = "critical"

        return {
            "status": status,
            "last_ping_seconds_ago": int(seconds_since_ping),
            "total_pings": self.ping_count,
            "uptime_seconds": int(now - self.start_time),
            "errors_since_start": self.errors_since_start
        }
