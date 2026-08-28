"""
CaseClosedFL Reddit Agent - Discovery Agent
Autonomously searches Reddit for high-intent personal injury posts.
Uses search queries, subreddit monitoring, and LLM scoring.
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from config import get_settings
from reddit_client import get_reddit_client
from nvidia_llm import get_nvidia_client
from rag_engine import get_rag_engine
from safety_guardrails import get_guardrails
from database import get_db_session, AgentRun

logger = logging.getLogger(__name__)
settings = get_settings()


async def _maybe_await(value):
    """Await coroutines; pass through plain values (e.g. sync mocks/fakes)."""
    import inspect
    if inspect.isawaitable(value):
        return await value
    return value


class DiscoveryAgent:
    """
    Agent responsible for discovering relevant Reddit posts.
    Runs on schedule, searches target subreddits, scores posts,
    and indexes them into the RAG system.
    """

    # Core search queries for personal injury leads in Florida
    BASE_QUERIES = [
        "car accident Florida",
        "rear ended Miami",
        "injured in crash Orlando",
        "hit by car Tampa",
        "truck accident Jacksonville",
        "motorcycle crash Florida",
        "Uber accident Miami",
        "slip and fall Florida",
        "insurance denying claim Florida",
        "not at fault accident Florida",
        "whiplash after crash",
        "pedestrian hit Miami",
        "Lyft driver accident",
        "pain after car accident",
        "medical bills after crash Florida"
    ]

    def __init__(self):
        self.reddit = get_reddit_client()
        self.llm = get_nvidia_client()
        self.rag = get_rag_engine()
        self.guardrails = get_guardrails()
        self.subreddits = settings.target_subreddits_list

    async def run(self) -> Dict[str, Any]:
        """
        Main discovery cycle.
        1. Generate fresh search queries
        2. Search target subreddits
        3. Score each post with LLM
        4. Run safety checks
        5. Index high-value posts into RAG
        6. Log results
        """
        run_id = None
        try:
            db = get_db_session()
            run_record = AgentRun(
                agent_name="discovery_agent",
                run_type="discovery",
                status="running",
                started_at=datetime.utcnow()
            )
            db.add(run_record)
            db.commit()
            run_id = run_record.id

            results = {
                "posts_found": 0,
                "posts_scored": 0,
                "posts_indexed": 0,
                "posts_blocked": 0,
                "high_intent_posts": [],
                "errors": []
            }

            # Step 1: Generate dynamic queries
            queries = await _maybe_await(self.llm.generate_search_queries([
                "Florida car accident",
                "personal injury help",
                "insurance problems",
                "accident lawyer questions"
            ]))
            queries = list(set(queries + self.BASE_QUERIES))[:20]
            logger.info(f"Using {len(queries)} search queries")

            # Step 2: Search each subreddit with each query
            all_posts = []
            for subreddit in self.subreddits:
                for query in queries:
                    try:
                        posts = self.reddit.search_subreddit(
                            subreddit_name=subreddit,
                            query=query,
                            sort="new",
                            time_filter="week",
                            limit=10
                        )
                        all_posts.extend(posts)
                    except Exception as e:
                        logger.warning(f"Search failed for r/{subreddit} query '{query}': {e}")
                        results["errors"].append(f"search_{subreddit}_{query}: {str(e)}")

            # Deduplicate by post ID
            seen_ids = set()
            unique_posts = []
            for post in all_posts:
                if post.id not in seen_ids:
                    seen_ids.add(post.id)
                    unique_posts.append(post)

            results["posts_found"] = len(unique_posts)
            logger.info(f"Found {len(unique_posts)} unique posts")

            # Step 3: Score and filter each post
            high_intent_posts = []
            for post in unique_posts:
                try:
                    # Skip if too old
                    post_age = datetime.utcnow() - datetime.utcfromtimestamp(post.created_utc)
                    if post_age > timedelta(days=7):
                        continue

                    # Run safety checks
                    safety_checks = self.guardrails.check_post_eligibility(
                        post_title=post.title,
                        post_body=post.selftext or "",
                        subreddit=str(post.subreddit)
                    )
                    can_proceed, block_reasons = self.guardrails.can_proceed(safety_checks)

                    if not can_proceed:
                        results["posts_blocked"] += 1
                        logger.debug(f"Post {post.id} blocked: {block_reasons}")
                        continue

                    # LLM intent analysis
                    analysis = await _maybe_await(self.llm.analyze_post_intent(
                        post_title=post.title,
                        post_body=post.selftext or "",
                        subreddit=str(post.subreddit)
                    ))

                    results["posts_scored"] += 1

                    # Index into RAG regardless of score
                    await _maybe_await(self.rag.add_reddit_post(
                        reddit_id=post.id,
                        subreddit=str(post.subreddit),
                        content_type="post",
                        title=post.title,
                        body=post.selftext or "",
                        author=str(post.author) if post.author else "deleted",
                        score=post.score,
                        num_comments=post.num_comments,
                        intent_tags=analysis.get("key_phrases", []),
                        location_tags=[analysis.get("location_hint")] if analysis.get("location_hint") else [],
                        accident_type_tags=[analysis.get("accident_type")] if analysis.get("accident_type") else [],
                        reddit_created_utc=datetime.utcfromtimestamp(post.created_utc)
                    ))
                    results["posts_indexed"] += 1

                    # Track high-intent posts
                    intent_score = analysis.get("intent_score", 0)
                    if intent_score >= settings.lead_score_threshold:
                        high_intent_posts.append({
                            "reddit_id": post.id,
                            "subreddit": str(post.subreddit),
                            "title": post.title,
                            "intent_score": intent_score,
                            "qualification_score": analysis.get("qualification_score", 0),
                            "lead_temperature": analysis.get("lead_temperature", "cold"),
                            "recommended_action": analysis.get("recommended_action", "ignore"),
                            "url": f"https://reddit.com{post.permalink}",
                            "analysis": analysis
                        })

                except Exception as e:
                    logger.error(f"Error processing post {post.id}: {e}")
                    results["errors"].append(f"process_{post.id}: {str(e)}")

            # Sort by intent score descending
            high_intent_posts.sort(key=lambda x: x["intent_score"], reverse=True)
            results["high_intent_posts"] = high_intent_posts[:50]  # Top 50

            # Update run record
            run_record.status = "success"
            run_record.completed_at = datetime.utcnow()
            run_record.items_processed = results["posts_found"]
            run_record.items_created = len(high_intent_posts)
            run_record.errors_count = len(results["errors"])
            run_record.log_output = str(results)
            db.commit()

            logger.info(
                f"Discovery complete: {results['posts_found']} found, "
                f"{results['posts_scored']} scored, {results['posts_indexed']} indexed, "
                f"{len(high_intent_posts)} high-intent"
            )

            return results

        except Exception as e:
            logger.error(f"Discovery agent failed: {e}")
            if run_id:
                db = get_db_session()
                run_record = db.query(AgentRun).filter(AgentRun.id == run_id).first()
                if run_record:
                    run_record.status = "failed"
                    run_record.completed_at = datetime.utcnow()
                    run_record.error_details = str(e)
                    db.commit()
            raise


# Singleton
discovery_agent: Optional[DiscoveryAgent] = None


def get_discovery_agent() -> DiscoveryAgent:
    global discovery_agent
    if discovery_agent is None:
        discovery_agent = DiscoveryAgent()
    return discovery_agent
