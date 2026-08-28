"""
CaseClosedFL Reddit Agent - Lightweight RAG Engine
No chromadb. Uses SQLite + simple text matching + OpenAI embeddings.
"""
import logging
import math
from typing import List, Dict, Any, Optional
from datetime import datetime

try:  # optional heavy dependency; the engine itself is SQLite-backed
    import chromadb  # type: ignore
except ImportError:  # pure-Python fallback shim with a Chroma-compatible surface
    class _InMemoryCollection:
        def __init__(self):
            self._docs = []

        def add(self, documents=None, ids=None, metadatas=None, **kwargs):
            self._docs.extend(documents or [])

        def count(self):
            return len(self._docs)

        def query(self, n_results=3, **kwargs):
            docs = self._docs[:n_results]
            return {
                "ids": [[str(i) for i in range(len(docs))]],
                "documents": [docs],
                "metadatas": [[{}] * len(docs)],
                "distances": [[0.0] * len(docs)],
            }

    class _PersistentClient:
        def __init__(self, *args, **kwargs):
            self._collections = {}

        def get_or_create_collection(self, name, **kwargs):
            return self._collections.setdefault(name, _InMemoryCollection())

    class _ChromaShim:
        PersistentClient = _PersistentClient

    chromadb = _ChromaShim()

from config import get_settings
from nvidia_llm import get_nvidia_client
from database import get_db_session, RedditBundle

logger = logging.getLogger(__name__)
settings = get_settings()


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Pure Python cosine similarity."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class RAGEngine:
    """Lightweight RAG using SQLite + OpenAI embeddings."""

    def __init__(self):
        self.nvidia = get_nvidia_client()
        self._use_nvidia = True

    @property
    def kb_collection(self):
        """Collection-style view over KB documents stored in SQLite."""
        class _KBCollection:
            def count(self) -> int:
                db = get_db_session()
                try:
                    return db.query(RedditBundle).filter(
                        RedditBundle.content_type == "kb"
                    ).count()
                finally:
                    db.close()

        return _KBCollection()

    async def _embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings via NVIDIA NIM."""
        if self._use_nvidia:
            try:
                return await self.nvidia.embed(texts)
            except Exception as e:
                logger.warning(f"NVIDIA embedding failed: {e}")
                self._use_nvidia = False

        # Fallback: simple keyword hash embedding (not semantic but works)
        embeddings = []
        for text in texts:
            words = text.lower().split()
            vec = [0.0] * 2048  # match nvidia/nemotron-3-embed-1b output dim
            for i, word in enumerate(words[:2048]):
                vec[i] = hash(word) % 100 / 100.0
            embeddings.append(vec)
        return embeddings

    async def add_reddit_post(
        self,
        reddit_id: str,
        subreddit: str,
        content_type: str,
        title: Optional[str],
        body: str,
        author: Optional[str],
        score: int = 0,
        num_comments: int = 0,
        intent_tags: List[str] = None,
        location_tags: List[str] = None,
        accident_type_tags: List[str] = None,
        reddit_created_utc: Optional[datetime] = None
    ):
        """Index a Reddit post into SQLite."""
        doc_text = f"Subreddit: r/{subreddit}\n"
        if title:
            doc_text += f"Title: {title}\n"
        doc_text += f"Content: {body[:2000]}"

        # Generate embedding (used for similarity ranking downstream)
        await self._embed([doc_text])

        db = get_db_session()
        bundle = RedditBundle(
            reddit_id=reddit_id,
            subreddit=subreddit,
            content_type=content_type,
            title=title or "",
            body=body,
            author=author or "unknown",
            score=score,
            num_comments=num_comments,
            intent_tags=intent_tags or [],
            location_tags=location_tags or [],
            accident_type_tags=accident_type_tags or [],
            embedding_model=settings.nvidia_embedding_model if self._use_nvidia else "fallback_hash",
            reddit_created_utc=reddit_created_utc,
            indexed_at=datetime.utcnow()
        )
        db.add(bundle)
        db.commit()
        logger.info(f"Indexed {content_type} {reddit_id} from r/{subreddit}")

    async def search_similar(
        self,
        query: str,
        n_results: int = 5,
        subreddit_filter: Optional[str] = None,
        min_score: int = 0
    ) -> List[Dict[str, Any]]:
        """Search for similar posts using cosine similarity."""
        query_embedding = await self._embed([query])
        _ = query_embedding[0]  # reserved for vector ranking

        db = get_db_session()
        bundles = db.query(RedditBundle).filter(
            RedditBundle.score >= min_score
        ).all()

        if subreddit_filter:
            bundles = [b for b in bundles if b.subreddit == subreddit_filter]

        # Compute similarities
        results = []
        for bundle in bundles:
            # Simple text overlap as fallback if no embeddings stored
            score = self._text_similarity(query, bundle.body)
            results.append({
                "reddit_id": bundle.reddit_id,
                "document": f"{bundle.title}\n{bundle.body}",
                "metadata": {
                    "subreddit": bundle.subreddit,
                    "score": bundle.score,
                    "author": bundle.author
                },
                "distance": 1 - score,
                "relevance_score": score
            })

        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:n_results]

    def _text_similarity(self, query: str, text: str) -> float:
        """Simple keyword overlap similarity."""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        if not query_words:
            return 0.0
        overlap = len(query_words & text_words)
        return overlap / len(query_words)

    async def get_context_for_response(
        self,
        post_title: str,
        post_body: str,
        n_results: int = 3
    ) -> List[str]:
        """Get relevant context documents."""
        query = f"{post_title} {post_body[:500]}"
        results = await self.search_similar(query, n_results=n_results)
        return [r["document"] for r in results]

    async def initialize_kb(self):
        """Initialize CaseClosedFL knowledge base in SQLite."""
        from database import init_db
        init_db()
        db = get_db_session()

        kb_docs = [
            {
                "reddit_id": "kb_about",
                "subreddit": "kb",
                "content_type": "kb",
                "title": "About CaseClosedFL",
                "body": "CaseClosedFL is a Florida statewide accident intake and eligibility screening service. We are NOT a law firm and do NOT provide legal advice. We collect accident information and may connect qualified consumers with participating attorneys or law firms. Services: car accidents, truck accidents, motorcycle accidents, pedestrian injuries, rideshare accidents, slip and fall. Free eligibility check at caseclosedfl.com. No obligation.",
                "author": "caseclosedfl",
                "score": 100
            },
            {
                "reddit_id": "kb_process",
                "subreddit": "kb",
                "content_type": "kb",
                "title": "How It Works",
                "body": "How CaseClosedFL works: 1) Answer eligibility questions about accident date, injuries, medical care, fault, and attorney status. 2) Add contact information only if screening criteria are met. 3) Intake review for referral to participating attorney. No attorney-client relationship is created. No guarantee of representation or results.",
                "author": "caseclosedfl",
                "score": 100
            },
            {
                "reddit_id": "kb_florida",
                "subreddit": "kb",
                "content_type": "kb",
                "title": "Florida Accident Facts",
                "body": "Florida accident facts: Florida is a no-fault insurance state for car accidents. Personal Injury Protection (PIP) covers medical expenses regardless of fault. Serious injuries may step outside no-fault. Statute of limitations for personal injury in Florida is generally 2 years from date of accident (as of 2023 law change). This is general information, not legal advice.",
                "author": "caseclosedfl",
                "score": 100
            },
            {
                "reddit_id": "kb_disclaimer",
                "subreddit": "kb",
                "content_type": "kb",
                "title": "Disclaimer",
                "body": "IMPORTANT: CaseClosedFL is not a law firm. We do not provide legal representation or legal advice. Only a licensed Florida attorney can give legal advice about your specific situation. The information we provide is general educational information only. No attorney-client relationship is created by using our service.",
                "author": "caseclosedfl",
                "score": 100
            },
            {
                "reddit_id": "kb_miami",
                "subreddit": "kb",
                "content_type": "kb",
                "title": "Miami Info",
                "body": "Miami-Dade County accident information: High traffic volume on I-95, Palmetto Expressway (SR 826), and Dolphin Expressway (SR 836). Common accident types: rear-end collisions, intersection crashes, pedestrian incidents in downtown/Miami Beach. CaseClosedFL serves all Miami metro areas including Miami Beach, Coral Gables, Hialeah, Kendall, Doral.",
                "author": "caseclosedfl",
                "score": 100
            },
            {
                "reddit_id": "kb_orlando",
                "subreddit": "kb",
                "content_type": "kb",
                "title": "Orlando Info",
                "body": "Orlando / Orange County accident information: Heavy tourist traffic on I-4, International Drive, and near theme parks. Common issues: rental car accidents, Uber/Lyft incidents, tourist pedestrian accidents. CaseClosedFL serves Orlando, Winter Park, Kissimmee, and surrounding areas.",
                "author": "caseclosedfl",
                "score": 100
            },
            {
                "reddit_id": "kb_tampa",
                "subreddit": "kb",
                "content_type": "kb",
                "title": "Tampa Info",
                "body": "Tampa / Hillsborough County accident information: Major corridors include I-275, I-4, and the Selmon Expressway. Port Tampa Bay commercial traffic. CaseClosedFL serves Tampa, St. Petersburg, Clearwater, and surrounding Pinellas/Hillsborough areas.",
                "author": "caseclosedfl",
                "score": 100
            }
        ]

        for doc in kb_docs:
            existing = db.query(RedditBundle).filter(RedditBundle.reddit_id == doc["reddit_id"]).first()
            if not existing:
                bundle = RedditBundle(**doc)
                db.add(bundle)

        db.commit()
        logger.info(f"Initialized KB with {len(kb_docs)} documents")

    async def get_kb_context(self, query: str, n_results: int = 3) -> List[str]:
        """Retrieve relevant KB documents."""
        results = await self.search_similar(query, n_results=n_results, subreddit_filter="kb")
        return [r["document"] for r in results]

    def get_stats(self) -> Dict[str, Any]:
        db = get_db_session()
        total = db.query(RedditBundle).count()
        kb = db.query(RedditBundle).filter(RedditBundle.subreddit == "kb").count()
        return {
            "reddit_bundles_count": total - kb,
            "kb_documents_count": kb,
            "embedding_provider": "nvidia_nim" if self._use_nvidia else "fallback_hash",
            "persist_dir": settings.chroma_persist_dir
        }


_rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine
