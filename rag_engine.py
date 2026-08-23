"""Small persistent RAG index backed by the application database."""
import hashlib
import json
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from database import RedditBundle, db_session


def _fallback_embedding(text: str, dimensions: int = 128) -> List[float]:
    vector = [0.0] * dimensions
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode()).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        vector[index] += 1.0 if digest[4] % 2 else -1.0
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b)) / (
        (math.sqrt(sum(x * x for x in a)) or 1.0) * (math.sqrt(sum(y * y for y in b)) or 1.0)
    )


class RAGEngine:
    def __init__(self):
        self.embedding_model = "sha256-token-v1"

    async def add_reddit_post(self, reddit_id: str, subreddit: str, content_type: str,
                              title: Optional[str], body: str, author: Optional[str],
                              score: int = 0, num_comments: int = 0,
                              intent_tags: Optional[List[str]] = None,
                              location_tags: Optional[List[str]] = None,
                              accident_type_tags: Optional[List[str]] = None,
                              reddit_created_utc: Optional[datetime] = None):
        doc = f"{title or ''}\n{body[:4000]}"
        embedding = _fallback_embedding(doc)
        with db_session() as db:
            bundle = db.query(RedditBundle).filter_by(reddit_id=reddit_id).first()
            values = dict(subreddit=subreddit, content_type=content_type, title=title or "",
                          body=body, author=author, embedding_model=self.embedding_model,
                          embedding_json=json.dumps(embedding), score=score, num_comments=num_comments,
                          intent_tags=intent_tags or [], location_tags=location_tags or [],
                          accident_type_tags=accident_type_tags or [], reddit_created_utc=reddit_created_utc,
                          indexed_at=datetime.utcnow())
            if bundle:
                for key, value in values.items():
                    setattr(bundle, key, value)
            else:
                db.add(RedditBundle(reddit_id=reddit_id, **values))

    async def search_similar(self, query: str, n_results: int = 5,
                             subreddit_filter: Optional[str] = None, min_score: int = 0) -> List[Dict[str, Any]]:
        query_vector = _fallback_embedding(query)
        with db_session() as db:
            query_obj = db.query(RedditBundle).filter(RedditBundle.score >= min_score)
            if subreddit_filter:
                query_obj = query_obj.filter(RedditBundle.subreddit == subreddit_filter)
            bundles = query_obj.all()
            results = []
            for bundle in bundles:
                try:
                    vector = json.loads(bundle.embedding_json or "[]")
                except (TypeError, ValueError):
                    vector = []
                relevance = cosine_similarity(query_vector, vector) if len(vector) == len(query_vector) else 0.0
                if not relevance:
                    relevance = self._text_similarity(query, f"{bundle.title} {bundle.body}")
                results.append({
                    "reddit_id": bundle.reddit_id,
                    "document": f"{bundle.title}\n{bundle.body}",
                    "metadata": {"subreddit": bundle.subreddit, "score": bundle.score, "author": bundle.author},
                    "distance": 1 - relevance, "relevance_score": relevance,
                })
            results.sort(key=lambda result: result["relevance_score"], reverse=True)
            return results[:n_results]

    @staticmethod
    def _text_similarity(query: str, text: str) -> float:
        query_words = set(query.lower().split())
        return len(query_words & set(text.lower().split())) / len(query_words) if query_words else 0.0

    async def get_context_for_response(self, post_title: str, post_body: str, n_results: int = 3) -> List[str]:
        return [item["document"] for item in await self.search_similar(f"{post_title} {post_body[:500]}", n_results)]

    async def initialize_kb(self):
        docs = [
            ("kb_about", "About CaseClosedFL", "CaseClosedFL is a Florida accident intake and eligibility screening service. We are not a law firm and do not provide legal advice. No attorney-client relationship is created."),
            ("kb_process", "How It Works", "Users can answer eligibility questions and may request intake review for referral to a participating attorney. There is no guarantee of representation or results."),
            ("kb_disclaimer", "Disclaimer", "This is general information, not legal advice. Only a licensed Florida attorney can give legal advice about a specific situation."),
            ("kb_florida", "Florida Accident Facts", "Florida car accident information includes PIP and fault considerations. Refer specific legal questions to a licensed Florida attorney."),
        ]
        for reddit_id, title, body in docs:
            await self.add_reddit_post(reddit_id, "kb", "kb", title, body, "caseclosedfl", score=100)

    async def get_kb_context(self, query: str, n_results: int = 3) -> List[str]:
        return [item["document"] for item in await self.search_similar(query, n_results, "kb")]

    def get_stats(self) -> Dict[str, Any]:
        with db_session() as db:
            total = db.query(RedditBundle).count()
            kb = db.query(RedditBundle).filter_by(subreddit="kb").count()
        return {"reddit_bundles_count": total - kb, "kb_documents_count": kb, "embedding_provider": self.embedding_model}


_rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine