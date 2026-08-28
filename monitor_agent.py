"""
CaseClosedFL Reddit Agent - Monitor Agent
Watches engaged posts for replies, upvotes, and DMs.
Triggers follow-up actions and lead qualification.
"""
import logging
from typing import Dict, Any
from datetime import datetime

from config import get_settings
from reddit_client import get_reddit_client
from database import get_db_session, MonitoredPost

logger = logging.getLogger(__name__)
settings = get_settings()


class MonitorAgent:
    """
    Monitors posts we have engaged with for:
    - Replies to our comments
    - Direct messages
    - Upvote patterns
    - User activity changes
    """

    def __init__(self):
        self.reddit = get_reddit_client()

    async def check_monitored_posts(self) -> Dict[str, Any]:
        """Check all active monitored posts for new replies."""
        db = get_db_session()
        results = {
            "posts_checked": 0,
            "new_replies_found": 0,
            "replies_requiring_action": [],
            "errors": []
        }

        try:
            active_posts = db.query(MonitoredPost).filter(
                MonitoredPost.is_active == True
            ).all()

            for mp in active_posts:
                try:
                    mp.last_checked_at = datetime.utcnow()
                    db.commit()

                    # Get current state
                    submission = self.reddit.client.submission(id=mp.reddit_post_id)
                    submission.comments.replace_more(limit=0)

                    current_reply_count = len(submission.comments.list())
                    mp.current_reply_count = current_reply_count

                    # Check for new replies since our engagement
                    if current_reply_count > mp.reply_count_at_start:
                        new_replies = current_reply_count - mp.reply_count_at_start
                        results["new_replies_found"] += new_replies

                        # Look for replies to our comments
                        for comment in submission.comments.list():
                            if hasattr(comment, 'replies'):
                                for reply in comment.replies:
                                    if reply.new:  # PRAW marks new replies
                                        results["replies_requiring_action"].append({
                                            "post_id": mp.reddit_post_id,
                                            "reply_id": reply.id,
                                            "author": str(reply.author),
                                            "body": reply.body[:500],
                                            "created_utc": reply.created_utc,
                                            "action": "review_for_qualifier"
                                        })

                    db.commit()
                    results["posts_checked"] += 1

                except Exception as e:
                    logger.error(f"Error checking post {mp.reddit_post_id}: {e}")
                    results["errors"].append(str(e))

            return results

        except Exception as e:
            logger.error(f"Monitor agent failed: {e}")
            raise

    async def add_monitor(self, post_id: str, subreddit: str, title: str,
                         priority: int = 1, reason: str = "") -> bool:
        """Add a post to the monitoring list."""
        db = get_db_session()
        try:
            existing = db.query(MonitoredPost).filter(
                MonitoredPost.reddit_post_id == post_id
            ).first()

            if existing:
                return False

            # Get current reply count
            submission = self.reddit.client.submission(id=post_id)
            submission.comments.replace_more(limit=0)
            reply_count = len(submission.comments.list())

            mp = MonitoredPost(
                reddit_post_id=post_id,
                subreddit=subreddit,
                post_title=title,
                priority=priority,
                monitoring_reason=reason,
                reply_count_at_start=reply_count,
                current_reply_count=reply_count,
                last_checked_at=datetime.utcnow()
            )
            db.add(mp)
            db.commit()
            return True

        except Exception as e:
            logger.error(f"Failed to add monitor for {post_id}: {e}")
            return False

    async def run(self) -> Dict[str, Any]:
        """Full monitoring cycle."""
        return await self.check_monitored_posts()


monitor_agent_instance = None


def get_monitor_agent() -> MonitorAgent:
    global monitor_agent_instance
    if monitor_agent_instance is None:
        monitor_agent_instance = MonitorAgent()
    return monitor_agent_instance
