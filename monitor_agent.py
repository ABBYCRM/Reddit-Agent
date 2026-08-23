"""Monitor already-reviewed posts without provider-specific reply flags."""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from database import MonitoredPost, db_session
from reddit_client import get_reddit_client

logger = logging.getLogger(__name__)


class MonitorAgent:
    def __init__(self):
        self.reddit = get_reddit_client()

    async def check_monitored_posts(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {"posts_checked": 0, "new_replies_found": 0, "replies_requiring_action": [], "errors": []}
        with db_session() as db:
            post_ids = [post.reddit_post_id for post in db.query(MonitoredPost).filter_by(is_active=True).all()]
        for post_id in post_ids:
            try:
                comments = await self.reddit.get_submission_comments(post_id)
                with db_session() as db:
                    monitored = db.query(MonitoredPost).filter_by(reddit_post_id=post_id).first()
                    if not monitored or not monitored.is_active:
                        continue
                    processed = set(monitored.processed_reply_ids or [])
                    new = [comment for comment in comments if comment.id not in processed]
                    for comment in new:
                        results["replies_requiring_action"].append({
                            "post_id": post_id, "reply_id": comment.id, "author": comment.author or "deleted",
                            "body": comment.body[:500], "created_utc": comment.created_utc, "action": "review_for_qualifier",
                        })
                    monitored.processed_reply_ids = list((processed | {comment.id for comment in new}))[-500:]
                    monitored.current_reply_count = len(comments)
                    monitored.last_checked_at = datetime.utcnow()
                    results["posts_checked"] += 1
                    results["new_replies_found"] += len(new)
            except Exception as exc:
                logger.warning("Unable to monitor Reddit post %s: %s", post_id, exc)
                results["errors"].append(f"{post_id}:{type(exc).__name__}")
        return results

    async def add_monitor(self, post_id: str, subreddit: str, title: str, priority: int = 1, reason: str = "") -> bool:
        with db_session() as db:
            if db.query(MonitoredPost).filter_by(reddit_post_id=post_id).first():
                return False
        try:
            comments = await self.reddit.get_submission_comments(post_id)
        except Exception as exc:
            logger.warning("Cannot initialize monitoring for %s: %s", post_id, exc)
            return False
        with db_session() as db:
            if db.query(MonitoredPost).filter_by(reddit_post_id=post_id).first():
                return False
            db.add(MonitoredPost(
                reddit_post_id=post_id, subreddit=subreddit, post_title=title, priority=priority,
                monitoring_reason=reason, reply_count_at_start=len(comments), current_reply_count=len(comments),
                processed_reply_ids=[comment.id for comment in comments], last_checked_at=datetime.utcnow(),
            ))
        return True

    async def run(self) -> Dict[str, Any]:
        return await self.check_monitored_posts()


_monitor_agent: Optional[MonitorAgent] = None


def get_monitor_agent() -> MonitorAgent:
    global _monitor_agent
    if _monitor_agent is None:
        _monitor_agent = MonitorAgent()
    return _monitor_agent