"""
CaseClosedFL Reddit Agent - RAG Engine
ChromaDB vector store with NVIDIA embeddings for Reddit post retrieval.
Supports semantic search, hybrid retrieval, and contextual compression.
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from config import get_settings
from nvidia_llm import get_nvidia_client

logger = logging.getLogger(__name__)
settings = get_settings()


class RAGEngine:
    """
    Retrieval-Augmented Generation engine for Reddit content.
    Uses ChromaDB for vector storage with fallback to local embeddings.
    """

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Collection for Reddit posts
        self.collection = self.client.get_or_create_collection(
            name="reddit_bundles",
            metadata={"hnsw:space": "cosine"}
        )

        # Collection for CaseClosedFL knowledge base
        self.kb_collection = self.client.get_or_create_collection(
            name="caseclosed_kb",
            metadata={"hnsw:space": "cosine"}
        )

        # Local embedding fallback (if NVIDIA API unavailable)
        self._local_embedder: Optional[SentenceTransformer] = None
        self.nvidia = get_nvidia_client()
        self._use_nvidia = True

    @property
    def local_embedder(self) -> SentenceTransformer:
        """Lazy-load local embedding model."""
        if self._local_embedder is None:
            logger.info("Loading local embedding model (fallback)...")
            self._local_embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._local_embedder

    async def _embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using NVIDIA NIM or local fallback."""
        if self._use_nvidia:
            try:
                return await self.nvidia.embed(texts)
            except Exception as e:
                logger.warning(f"NVIDIA embedding failed, falling back to local: {e}")
                self._use_nvidia = False

        # Local fallback
        embeddings = self.local_embedder.encode(texts, convert_to_list=True)
        return embeddings

    async def add_reddit_post(
        self,
        reddit_id: str,
        subreddit: str,
        content_type: str,  # "post" or "comment"
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
        """Index a Reddit post/comment into the vector store."""
        # Prepare document
        doc_text = f"Subreddit: r/{subreddit}\n"
        if title:
            doc_text += f"Title: {title}\n"
        doc_text += f"Content: {body[:2000]}"

        metadata = {
            "reddit_id": reddit_id,
            "subreddit": subreddit,
            "content_type": content_type,
            "title": title or "",
            "author": author or "unknown",
            "score": score,
            "num_comments": num_comments,
            "intent_tags": ",".join(intent_tags or []),
            "location_tags": ",".join(location_tags or []),
            "accident_type_tags": ",".join(accident_type_tags or []),
            "indexed_at": datetime.utcnow().isoformat(),
            "reddit_created_utc": reddit_created_utc.isoformat() if reddit_created_utc else ""
        }

        # Generate embedding
        embeddings = await self._embed([doc_text])

        # Add to collection
        self.collection.add(
            ids=[reddit_id],
            embeddings=embeddings,
            documents=[doc_text],
            metadatas=[metadata]
        )

        logger.info(f"Indexed {content_type} {reddit_id} from r/{subreddit}")

    async def search_similar(
        self,
        query: str,
        n_results: int = 5,
        subreddit_filter: Optional[str] = None,
        min_score: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Search for similar Reddit posts.

        Args:
            query: Search query text
            n_results: Number of results to return
            subreddit_filter: Optional subreddit to filter by
            min_score: Minimum Reddit score filter
        """
        query_embedding = await self._embed([query])

        where_filter = {}
        if subreddit_filter:
            where_filter["subreddit"] = subreddit_filter
        if min_score > 0:
            where_filter["score"] = {"$gte": min_score}

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where_filter if where_filter else None,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "reddit_id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "relevance_score": 1 - results["distances"][0][i]  # Convert distance to similarity
            })

        return formatted_results

    async def search_by_intent(
        self,
        intent: str,
        location: Optional[str] = None,
        n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for posts matching a specific intent pattern."""
        query = f"Personal injury accident in {location or 'Florida'}: {intent}"
        return await self.search_similar(query, n_results=n_results)

    async def get_context_for_response(
        self,
        post_title: str,
        post_body: str,
        n_results: int = 3
    ) -> List[str]:
        """Get relevant context documents for crafting a response."""
        query = f"{post_title} {post_body[:500]}"
        results = await self.search_similar(query, n_results=n_results)
        return [r["document"] for r in results]

    async def initialize_kb(self):
        """Initialize CaseClosedFL knowledge base."""
        kb_docs = [
            {
                "id": "kb_about",
                "text": """CaseClosedFL is a Florida statewide accident intake and eligibility screening service. 
We are NOT a law firm and do NOT provide legal advice. We collect accident information and may 
connect qualified consumers with participating attorneys or law firms. Services: car accidents, 
truck accidents, motorcycle accidents, pedestrian injuries, rideshare accidents, slip and fall. 
Free eligibility check at caseclosedfl.com. No obligation."""
            },
            {
                "id": "kb_process",
                "text": """How CaseClosedFL works: 1) Answer eligibility questions about accident date, 
injuries, medical care, fault, and attorney status. 2) Add contact information only if screening 
criteria are met. 3) Intake review for referral to participating attorney. No attorney-client 
relationship is created. No guarantee of representation or results."""
            },
            {
                "id": "kb_florida_accidents",
                "text": """Florida accident facts: Florida is a no-fault insurance state for car accidents. 
Personal Injury Protection (PIP) covers medical expenses regardless of fault. Serious injuries may 
step outside no-fault. Statute of limitations for personal injury in Florida is generally 2 years 
from date of accident (as of 2023 law change). This is general information, not legal advice."""
            },
            {
                "id": "kb_disclaimer",
                "text": """IMPORTANT: CaseClosedFL is not a law firm. We do not provide legal representation 
or legal advice. Only a licensed Florida attorney can give legal advice about your specific situation. 
The information we provide is general educational information only. No attorney-client relationship 
is created by using our service."""
            },
            {
                "id": "kb_miami",
                "text": """Miami-Dade County accident information: High traffic volume on I-95, Palmetto 
Expressway (SR 826), and Dolphin Expressway (SR 836). Common accident types: rear-end collisions, 
intersection crashes, pedestrian incidents in downtown/Miami Beach. CaseClosedFL serves all Miami 
metro areas including Miami Beach, Coral Gables, Hialeah, Kendall, Doral."""
            },
            {
                "id": "kb_orlando",
                "text": """Orlando / Orange County accident information: Heavy tourist traffic on I-4, 
International Drive, and near theme parks. Common issues: rental car accidents, Uber/Lyft 
incidents, tourist pedestrian accidents. CaseClosedFL serves Orlando, Winter Park, Kissimmee, 
and surrounding areas."""
            },
            {
                "id": "kb_tampa",
                "text": """Tampa / Hillsborough County accident information: Major corridors include I-275, 
I-4, and the Selmon Expressway. Port Tampa Bay commercial traffic. CaseClosedFL serves Tampa, 
St. Petersburg, Clearwater, and surrounding Pinellas/Hillsborough areas."""
            }
        ]

        embeddings = await self._embed([doc["text"] for doc in kb_docs])

        self.kb_collection.add(
            ids=[doc["id"] for doc in kb_docs],
            embeddings=embeddings,
            documents=[doc["text"] for doc in kb_docs],
            metadatas=[{"source": "caseclosed_kb"} for _ in kb_docs]
        )

        logger.info(f"Initialized knowledge base with {len(kb_docs)} documents")

    async def get_kb_context(self, query: str, n_results: int = 3) -> List[str]:
        """Retrieve relevant knowledge base documents."""
        query_embedding = await self._embed([query])

        results = self.kb_collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents"]
        )

        return results["documents"][0] if results["documents"] else []

    def get_stats(self) -> Dict[str, Any]:
        """Get RAG engine statistics."""
        return {
            "reddit_bundles_count": self.collection.count(),
            "kb_documents_count": self.kb_collection.count(),
            "embedding_provider": "nvidia_nim" if self._use_nvidia else "local_sentence_transformers",
            "persist_dir": settings.chroma_persist_dir
        }


# Singleton
_rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine
