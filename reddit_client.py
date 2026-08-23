"""
CaseClosedFL Reddit Agent - Reddit API Client
Production-grade PRAW wrapper with token bucket rate limiting,
header-aware pacing, and exponential backoff.
"""
import time
import random
import logging
from datetime import datetime, timedelta
from typing import Iterator, List, Optional, Dict, Any
from dataclasses import dataclass

import praw
from praw.models import Submission, Comment, Subreddit
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class RateLimitState:
    """Tracks Reddit API rate limit headers."""
    used: int = 0
    remaining: int = 100
    reset_timestamp: float = 0.0
    last_updated: float = 0.0
    burst_buffer: int = 5

    @property
    def is_near_limit(self) -> bool:
        return self.remaining <= self.burst_buffer

    @property
    def seconds_until_reset(self) -> float:
        return max(0.0, self.reset_timestamp - time.time())


class RedditRateLimiter:
    """Token bucket + header-aware rate limiter for Reddit API."""

    def __init__(self, max_qpm: int = 90, burst_buffer: int = 5):
        self.max_qpm = max_qpm
        self.burst_buffer = burst_buffer
        self.tokens = float(max_qpm)
        self.last_update = time.time()
        self.rate_limit_state = RateLimitState(burst_buffer=burst_buffer)
        self._lock_acquired = False

    def _refill_tokens(self):
        now = time.time()
        elapsed = now - self.last_update
        # Refill based on per-minute rate
        self.tokens = min(
            self.max_qpm,
            self.tokens + (elapsed * self.max_qpm / 60.0)
        )
        self.last_update = now

    def acquire(self, tokens: int = 1) -> bool:
        """Attempt to acquire tokens. Blocks if necessary."""
        self._refill_tokens()

        # Respect Reddit's header-based limits first
        if self.rate_limit_state.is_near_limit:
            sleep_time = self.rate_limit_state.seconds_until_reset + random.uniform(1, 3)
            logger.warning(f"Rate limit near exhaustion. Sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
            self._refill_tokens()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True

        # Calculate wait time
        needed = tokens - self.tokens
        wait_seconds = (needed * 60.0) / self.max_qpm + random.uniform(0.5, 1.5)
        logger.info(f"Rate limit: waiting {wait_seconds:.1f}s for token refill")
        time.sleep(wait_seconds)
        self._refill_tokens()
        self.tokens -= tokens
        return True

    def update_from_headers(self, headers: Dict[str, str]):
        """Update state from Reddit API response headers."""
        try:
            self.rate_limit_state.used = int(headers.get("x-ratelimit-used", 0))
            self.rate_limit_state.remaining = int(headers.get("x-ratelimit-remaining", 100))
            reset_seconds = int(headers.get("x-ratelimit-reset", 600))
            self.rate_limit_state.reset_timestamp = time.time() + reset_seconds
            self.rate_limit_state.last_updated = time.time()
        except (ValueError, TypeError):
            pass


class RedditClient:
    """
    Production Reddit client with safety guardrails.
    Wraps PRAW with rate limiting, caching, and compliance checks.
    """

    def __init__(self):
        self.rate_limiter = RedditRateLimiter(
            max_qpm=settings.reddit_max_qpm,
            burst_buffer=settings.reddit_burst_buffer
        )
        self._client: Optional[praw.Reddit] = None
        self._last_request_time: Optional[datetime] = None
        self._daily_engagement_count: int = 0
        self._daily_reset: datetime = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)

    @property
    def client(self) -> praw.Reddit:
        if self._client is None:
            self._client = praw.Reddit(
                client_id=settings.reddit_client_id,
                client_secret=settings.reddit_client_secret,
                user_agent=settings.reddit_user_agent,
                username=settings.reddit_username,
                password=settings.reddit_password,
            )
            logger.info(f"Reddit client initialized. User: {self._client.user.me()}")
        return self._client

    def _check_daily_limit(self) -> bool:
        """Check if we've hit daily engagement cap."""
        now = datetime.utcnow()
        if now >= self._daily_reset:
            self._daily_engagement_count = 0
            self._daily_reset = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

        if self._daily_engagement_count >= settings.max_daily_engagements:
            logger.warning(f"Daily engagement limit reached: {self._daily_engagement_count}/{settings.max_daily_engagements}")
            return False
        return True

    def _pre_request(self):
        """Execute before every API call."""
        self.rate_limiter.acquire()
        self._last_request_time = datetime.utcnow()

    # ─── Discovery Methods ───────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((praw.exceptions.APIException, praw.exceptions.ClientException))
    )
    def search_subreddit(
        self, 
        subreddit_name: str, 
        query: str,
        sort: str = "new",
        time_filter: str = "week",
        limit: int = 25
    ) -> List[Submission]:
        """Search posts in a subreddit with rate limiting."""
        self._pre_request()
        subreddit = self.client.subreddit(subreddit_name)
        results = list(subreddit.search(query, sort=sort, time_filter=time_filter, limit=limit))
        logger.info(f"Found {len(results)} posts in r/{subreddit_name} for query: {query}")
        return results

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((praw.exceptions.APIException, praw.exceptions.ClientException))
    )
    def get_hot_posts(self, subreddit_name: str, limit: int = 25) -> List[Submission]:
        """Get hot posts from a subreddit."""
        self._pre_request()
        subreddit = self.client.subreddit(subreddit_name)
        results = list(subreddit.hot(limit=limit))
        return results

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((praw.exceptions.APIException, praw.exceptions.ClientException))
    )
    def get_new_posts(self, subreddit_name: str, limit: int = 25) -> List[Submission]:
        """Get new posts from a subreddit."""
        self._pre_request()
        subreddit = self.client.subreddit(subreddit_name)
        results = list(subreddit.new(limit=limit))
        return results

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((praw.exceptions.APIException, praw.exceptions.ClientException))
    )
    def get_comments(self, submission_id: str, limit: int = 50) -> List[Comment]:
        """Get comments from a submission."""
        self._pre_request()
        submission = self.client.submission(id=submission_id)
        submission.comments.replace_more(limit=0)
        comments = list(submission.comments.list()[:limit])
        return comments

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((praw.exceptions.APIException, praw.exceptions.ClientException))
    )
    def get_subreddit_info(self, subreddit_name: str) -> Subreddit:
        """Get subreddit metadata."""
        self._pre_request()
        return self.client.subreddit(subreddit_name)

    # ─── Engagement Methods (with safety checks) ─────────────────────────

    def can_engage(self) -> bool:
        """Check if engagement is allowed (daily limits, safety)."""
        if not settings.enable_auto_reply:
            logger.info("Auto-reply is disabled in settings")
            return False
        if not self._check_daily_limit():
            return False
        return True

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=5, max=60),
        retry=retry_if_exception_type((praw.exceptions.APIException,))
    )
    def post_comment(self, submission_id: str, text: str) -> Optional[Comment]:
        """Post a comment with safety checks."""
        if not self.can_engage():
            return None

        self._pre_request()
        submission = self.client.submission(id=submission_id)
        comment = submission.reply(text)
        self._daily_engagement_count += 1
        logger.info(f"Comment posted to {submission_id}: {comment.id}")
        return comment

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=5, max=60),
        retry=retry_if_exception_type((praw.exceptions.APIException,))
    )
    def send_dm(self, username: str, subject: str, message: str) -> bool:
        """Send a DM with safety checks."""
        if not self.can_engage() or not settings.enable_dm_outreach:
            return False

        self._pre_request()
        redditor = self.client.redditor(username)
        redditor.message(subject=subject, message=message)
        self._daily_engagement_count += 1
        logger.info(f"DM sent to u/{username}")
        return True

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((praw.exceptions.APIException, praw.exceptions.ClientException))
    )
    def get_user_activity(self, username: str, limit: int = 10) -> Dict[str, Any]:
        """Get recent user activity for qualification."""
        self._pre_request()
        redditor = self.client.redditor(username)

        # Check account age
        created_utc = datetime.utcfromtimestamp(redditor.created_utc)
        account_age_days = (datetime.utcnow() - created_utc).days

        # Get recent submissions/comments
        submissions = list(redditor.submissions.new(limit=limit))
        comments = list(redditor.comments.new(limit=limit))

        return {
            "username": username,
            "account_age_days": account_age_days,
            "link_karma": redditor.link_karma,
            "comment_karma": redditor.comment_karma,
            "recent_submissions": [
                {"title": s.title, "subreddit": str(s.subreddit), "created_utc": s.created_utc}
                for s in submissions
            ],
            "recent_comments": [
                {"body": c.body[:200], "subreddit": str(c.subreddit), "created_utc": c.created_utc}
                for c in comments
            ],
            "is_qualified_account": account_age_days >= settings.min_account_age_days
        }

    # ─── Utility ─────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            "daily_engagements": self._daily_engagement_count,
            "daily_limit": settings.max_daily_engagements,
            "rate_limit_remaining": self.rate_limiter.rate_limit_state.remaining,
            "last_request": self._last_request_time.isoformat() if self._last_request_time else None,
            "auto_reply_enabled": settings.enable_auto_reply,
            "dm_enabled": settings.enable_dm_outreach,
        }


# Singleton
_reddit_client: Optional[RedditClient] = None


def get_reddit_client() -> RedditClient:
    global _reddit_client
    if _reddit_client is None:
        _reddit_client = RedditClient()
    return _reddit_client
