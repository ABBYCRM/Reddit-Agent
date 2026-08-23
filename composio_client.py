"""Composio Reddit transport.

The application talks to Composio over its server-side API. MCP tools are not
available to application processes, so this boundary deliberately has no PRAW
fallback and never silently turns an integration failure into an empty result.
"""
import logging
from typing import Any, Dict, List, Optional

import httpx

from config import get_settings

logger = logging.getLogger(__name__)


class ComposioError(RuntimeError):
    """A failed or unavailable Composio operation."""


class ComposioRedditClient:
    def __init__(self, http_client: Optional[httpx.AsyncClient] = None):
        self.settings = get_settings()
        self.api_key = self.settings.composio_api_key
        self.base_url = self.settings.composio_base_url.rstrip("/")
        self._connected_entity: Optional[str] = self.settings.composio_reddit_connected_account_id
        self._connected_user_id: Optional[str] = self.settings.composio_reddit_user_id
        self._client = http_client
        self._owns_client = http_client is None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    async def _request(self, method: str, endpoint: str, data: Optional[dict] = None) -> Dict[str, Any]:
        if not self.api_key:
            raise ComposioError("COMPOSIO_API_KEY is not configured")
        client = self._client or httpx.AsyncClient(timeout=self.settings.composio_request_timeout_seconds)
        headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
        try:
            response = await client.request(method, f"{self.base_url}{endpoint}", headers=headers, json=data)
            try:
                payload = response.json()
            except ValueError as exc:
                raise ComposioError(f"Composio returned non-JSON response ({response.status_code})") from exc
            if not 200 <= response.status_code < 300:
                detail = str(payload.get("message") or payload.get("error") or "request failed")[:300]
                raise ComposioError(f"Composio {response.status_code}: {detail}")
            if payload.get("success") is False or payload.get("error"):
                raise ComposioError(str(payload.get("error") or payload.get("message") or "Composio action failed")[:300])
            return payload
        except httpx.HTTPError as exc:
            raise ComposioError(f"Composio request failed: {exc}") from exc
        finally:
            if self._owns_client:
                await client.aclose()

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()

    async def get_connected_account(self) -> str:
        if self._connected_entity and self._connected_user_id:
            return self._connected_entity
        result = await self._request("GET", "/connected_accounts?toolkit_slugs=reddit&statuses=ACTIVE&limit=100")
        account_data = result.get("data", result)
        accounts = account_data.get("items", account_data.get("results", [])) if isinstance(account_data, dict) else []
        for account in accounts or []:
            account_id = account.get("id") or account.get("connectedAccountId")
            if not self._connected_entity or account_id == self._connected_entity:
                self._connected_entity = account_id
                self._connected_user_id = account.get("user_id") or account.get("userId") or self._connected_user_id
                break
        if not self._connected_entity or not self._connected_user_id:
            raise ComposioError("No connected Reddit account was found in Composio")
        return self._connected_entity

    async def _execute(self, action: str, data: dict) -> Dict[str, Any]:
        entity = await self.get_connected_account()
        result = await self._request("POST", f"/tools/execute/{action}", {
            "connected_account_id": entity,
            "user_id": self._connected_user_id,
            "arguments": data,
        })
        return result.get("data", result)

    @staticmethod
    def _items(payload: Dict[str, Any], *keys: str) -> List[Dict[str, Any]]:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                return value
        if isinstance(payload.get("data"), list):
            return payload["data"]
        return []

    async def search_subreddit(self, subreddit: str, query: str, sort: str = "new",
                               time_filter: str = "week", limit: int = 25) -> List[Dict[str, Any]]:
        payload = await self._execute("REDDIT_SEARCH_ACROSS_SUBREDDITS", {
            "search_query": f"subreddit:{subreddit} {query}", "sort": sort, "limit": limit, "restrict_sr": False
        })
        search_results = payload.get("search_results", payload) if isinstance(payload, dict) else {}
        children = search_results.get("data", {}).get("children", []) if isinstance(search_results, dict) else []
        return [item.get("data", item) for item in children if isinstance(item, dict)]

    async def get_hot_posts(self, subreddit: str, limit: int = 25) -> List[Dict[str, Any]]:
        payload = await self._execute("REDDIT_RETRIEVE_REDDIT_POST", {"subreddit": subreddit, "size": limit})
        return self._items(payload, "posts", "results", "items")

    async def get_submission_comments(self, submission_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        payload = await self._execute("REDDIT_RETRIEVE_POST_COMMENTS", {"article": submission_id})
        return self._items(payload, "comments", "replies", "items")

    async def post_comment(self, submission_id: str, text: str) -> Optional[Dict[str, Any]]:
        payload = await self._execute("REDDIT_POST_REDDIT_COMMENT", {"thing_id": submission_id, "text": text})
        return payload if payload else None

    async def send_dm(self, username: str, subject: str, message: str) -> bool:
        raise ComposioError("Direct-message sending is unavailable in the configured Composio Reddit toolkit")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "provider": "composio",
            "configured": self.configured,
            "connected": self._connected_entity is not None,
            "transport": "composio",
        }


_composio_client: Optional[ComposioRedditClient] = None


def get_composio_client() -> ComposioRedditClient:
    global _composio_client
    if _composio_client is None:
        _composio_client = ComposioRedditClient()
    return _composio_client