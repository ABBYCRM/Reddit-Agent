"""Provider-neutral Reddit gateway with Composio as the production transport."""
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import praw

from composio_client import ComposioError, get_composio_client
from config import get_settings

logger = logging.getLogger(__name__)


class RedditTransportError(RuntimeError):
    """Raised when the selected Reddit provider cannot complete an operation."""


@dataclass(frozen=True)
class RedditPost:
    id: str
    subreddit: str
    title: str
    body: str
    author: Optional[str]
    created_utc: float
    score: int
    num_comments: int
    permalink: str


@dataclass(frozen=True)
class RedditComment:
    id: str
    author: Optional[str]
    body: str
    created_utc: float


def _timestamp(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return 0.0
    return 0.0


def _read(item: Any, *keys: str, default=None):
    for key in keys:
        value = item.get(key) if isinstance(item, dict) else getattr(item, key, None)
        if value is not None:
            return value
    return default


def normalize_post(item: Any, fallback_subreddit: str = "") -> RedditPost:
    item_id = str(_read(item, "id", "post_id", "submission_id", "name", default="")).removeprefix("t3_")
    if not item_id:
        raise RedditTransportError("Reddit post did not include an ID")
    subreddit = str(_read(item, "subreddit", "subreddit_name", default=fallback_subreddit)).removeprefix("r/")
    title = str(_read(item, "title", default=""))
    body = str(_read(item, "selftext", "body", "text", default=""))
    author = _read(item, "author", "author_name", "username")
    author = str(author) if author else None
    permalink = str(_read(item, "permalink", "url", default=f"/comments/{item_id}"))
    if permalink.startswith("https://www.reddit.com"):
        permalink = permalink.removeprefix("https://www.reddit.com")
    return RedditPost(
        id=item_id, subreddit=subreddit, title=title, body=body, author=author,
        created_utc=_timestamp(_read(item, "created_utc", "created_at", "createdAt")),
        score=int(_read(item, "score", "ups", default=0) or 0),
        num_comments=int(_read(item, "num_comments", "comment_count", default=0) or 0),
        permalink=permalink,
    )


def normalize_comment(item: Any) -> RedditComment:
    item_id = str(_read(item, "id", "comment_id", "name", default="")).removeprefix("t1_")
    if not item_id:
        raise RedditTransportError("Reddit comment did not include an ID")
    author = _read(item, "author", "author_name", "username")
    return RedditComment(
        id=item_id,
        author=str(author) if author else None,
        body=str(_read(item, "body", "text", default="")),
        created_utc=_timestamp(_read(item, "created_utc", "created_at", "createdAt")),
    )


class RedditClient:
    """Async client with no implicit provider fallback."""

    def __init__(self):
        self.settings = get_settings()
        self.transport = self.settings.reddit_transport
        self._client: Optional[praw.Reddit] = None
        self._daily_engagement_count = 0
        self._daily_reset = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

    @property
    def configured(self) -> bool:
        if self.transport == "composio":
            return bool(self.settings.composio_api_key)
        return bool(self.settings.reddit_client_id and self.settings.reddit_client_secret)

    def _praw(self) -> praw.Reddit:
        if not self.configured:
            raise RedditTransportError("PRAW credentials are not configured")
        if self._client is None:
            self._client = praw.Reddit(
                client_id=self.settings.reddit_client_id,
                client_secret=self.settings.reddit_client_secret,
                user_agent=self.settings.reddit_user_agent,
                username=self.settings.reddit_username,
                password=self.settings.reddit_password,
            )
        return self._client

    async def search_subreddit(self, subreddit_name: str, query: str, sort: str = "new",
                               time_filter: str = "week", limit: int = 25) -> List[RedditPost]:
        try:
            if self.transport == "composio":
                items = await get_composio_client().search_subreddit(subreddit_name, query, sort, time_filter, limit)
                return [normalize_post(item, subreddit_name) for item in items]
            items = await asyncio.to_thread(
                lambda: list(self._praw().subreddit(subreddit_name).search(query, sort=sort, time_filter=time_filter, limit=limit))
            )
            return [normalize_post(item, subreddit_name) for item in items]
        except ComposioError as exc:
            raise RedditTransportError(str(exc)) from exc

    async def get_submission_comments(self, submission_id: str, limit: int = 100) -> List[RedditComment]:
        try:
            if self.transport == "composio":
                items = await get_composio_client().get_submission_comments(submission_id, limit)
                return [normalize_comment(item) for item in items]
            def fetch():
                submission = self._praw().submission(id=submission_id)
                submission.comments.replace_more(limit=0)
                return list(submission.comments.list()[:limit])
            return [normalize_comment(item) for item in await asyncio.to_thread(fetch)]
        except ComposioError as exc:
            raise RedditTransportError(str(exc)) from exc

    def _can_engage(self) -> bool:
        now = datetime.now(timezone.utc)
        if now >= self._daily_reset:
            self._daily_engagement_count = 0
            self._daily_reset = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        return self.settings.enable_auto_reply and self._daily_engagement_count < self.settings.max_daily_engagements

    async def post_comment(self, submission_id: str, text: str) -> Optional[RedditComment]:
        if not self._can_engage():
            return None
        try:
            if self.transport == "composio":
                item = await get_composio_client().post_comment(submission_id, text)
                comment = normalize_comment(item) if item else None
            else:
                comment = normalize_comment(await asyncio.to_thread(lambda: self._praw().submission(id=submission_id).reply(text)))
            if comment:
                self._daily_engagement_count += 1
            return comment
        except ComposioError as exc:
            raise RedditTransportError(str(exc)) from exc

    async def send_dm(self, username: str, subject: str, message: str) -> bool:
        if not self._can_engage() or not self.settings.enable_dm_outreach:
            return False
        try:
            if self.transport == "composio":
                sent = await get_composio_client().send_dm(username, subject, message)
            else:
                await asyncio.to_thread(lambda: self._praw().redditor(username).message(subject=subject, message=message))
                sent = True
            if sent:
                self._daily_engagement_count += 1
            return sent
        except ComposioError as exc:
            raise RedditTransportError(str(exc)) from exc

    def get_stats(self) -> Dict[str, Any]:
        return {
            "provider": self.transport,
            "configured": self.configured,
            "daily_engagements": self._daily_engagement_count,
            "daily_limit": self.settings.max_daily_engagements,
            "auto_reply_enabled": self.settings.enable_auto_reply,
            "dm_enabled": self.settings.enable_dm_outreach,
        }


_reddit_client: Optional[RedditClient] = None


def get_reddit_client() -> RedditClient:
    global _reddit_client
    if _reddit_client is None:
        _reddit_client = RedditClient()
    return _reddit_client