"""
CaseClosedFL Reddit Agent - Composio Reddit Client
Uses Composio's managed OAuth for Reddit instead of raw PRAW credentials.
Falls back to PRAW if Composio is unavailable.
"""
import logging
from typing import List, Dict, Any, Optional

import requests

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ComposioRedditClient:
    """
    Reddit client using Composio's tool layer.
    Requires Composio API key and connected Reddit app.
    """

    BASE_URL = "https://backend.composio.dev/api/v1"

    def __init__(self):
        self.api_key = settings.composio_api_key
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        self._connected_entity = None
        self._fallback_to_praw = False

    def _request(self, method: str, endpoint: str, data: dict = None) -> dict:
        """Make request to Composio API."""
        url = f"{self.BASE_URL}{endpoint}"
        try:
            if method == "GET":
                resp = requests.get(url, headers=self.headers, timeout=30)
            elif method == "POST":
                resp = requests.post(url, headers=self.headers, json=data, timeout=30)
            else:
                resp = requests.request(method, url, headers=self.headers, json=data, timeout=30)

            if resp.status_code == 200:
                return resp.json()
            else:
                logger.warning(f"Composio API error {resp.status_code}: {resp.text[:200]}")
                return {}
        except Exception as e:
            logger.error(f"Composio request failed: {e}")
            return {}

    def get_connected_account(self) -> Optional[str]:
        """Get the connected Reddit entity ID."""
        if self._connected_entity:
            return self._connected_entity

        # List connected accounts
        result = self._request("GET", "/connectedAccounts")
        accounts = result.get("items", [])

        for account in accounts:
            if account.get("appName", "").lower() == "reddit":
                self._connected_entity = account.get("id")
                logger.info(f"Found Composio Reddit connection: {self._connected_entity}")
                return self._connected_entity

        logger.warning("No Composio Reddit connection found. Falling back to PRAW.")
        self._fallback_to_praw = True
        return None

    def search_subreddit(self, subreddit: str, query: str, sort: str = "new", 
                        time_filter: str = "week", limit: int = 25) -> List[Dict]:
        """Search posts in a subreddit via Composio."""
        entity = self.get_connected_account()
        if not entity:
            return []

        result = self._request("POST", "/actions/REDDIT_SEARCH_SUBREDDIT/execute", {
            "connectedAccountId": entity,
            "data": {
                "subreddit": subreddit,
                "query": query,
                "sort": sort,
                "time": time_filter,
                "limit": limit
            }
        })

        posts = result.get("data", {}).get("posts", [])
        return posts

    def get_hot_posts(self, subreddit: str, limit: int = 25) -> List[Dict]:
        """Get hot posts from a subreddit."""
        entity = self.get_connected_account()
        if not entity:
            return []

        result = self._request("POST", "/actions/REDDIT_GET_HOT_POSTS/execute", {
            "connectedAccountId": entity,
            "data": {
                "subreddit": subreddit,
                "limit": limit
            }
        })

        return result.get("data", {}).get("posts", [])

    def post_comment(self, submission_id: str, text: str) -> Optional[Dict]:
        """Post a comment via Composio."""
        entity = self.get_connected_account()
        if not entity:
            return None

        result = self._request("POST", "/actions/REDDIT_SUBMIT_COMMENT/execute", {
            "connectedAccountId": entity,
            "data": {
                "submission_id": submission_id,
                "text": text
            }
        })

        return result.get("data")

    def send_dm(self, username: str, subject: str, message: str) -> bool:
        """Send a DM via Composio."""
        entity = self.get_connected_account()
        if not entity:
            return False

        result = self._request("POST", "/actions/REDDIT_SEND_MESSAGE/execute", {
            "connectedAccountId": entity,
            "data": {
                "recipient": username,
                "subject": subject,
                "message": message
            }
        })

        return result.get("success", False)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "provider": "composio",
            "connected": self._connected_entity is not None,
            "fallback_to_praw": self._fallback_to_praw
        }


# Singleton
_composio_client: Optional[ComposioRedditClient] = None


def get_composio_client() -> ComposioRedditClient:
    global _composio_client
    if _composio_client is None:
        _composio_client = ComposioRedditClient()
    return _composio_client
