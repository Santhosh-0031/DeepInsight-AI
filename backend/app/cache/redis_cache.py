import os
import json
import hashlib
from typing import Optional, Dict, Any

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None

from ..embeddings import embed_text, cosine_similarity


CACHE_SIMILARITY_THRESHOLD = 0.90
CACHE_TTL_SECONDS = None


class SemanticCache:
    def __init__(self):
        self._redis = None
        self._cache_key_prefix = "dra:cache:"
        self._index_key = "dra:cache:index"

    async def _get_redis(self):
        """Get Redis connection with Upstash-safe settings."""
        if aioredis is None:
            print("[SemanticCache] redis package not installed.")
            return None

        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            print("[SemanticCache] REDIS_URL not set.")
            return None

        if redis_url and redis_url.startswith("redis://") and ".upstash.io" in redis_url:
            print("[SemanticCache] Upgrading Upstash Redis URL to use SSL (rediss://)")
            redis_url = redis_url.replace("redis://", "rediss://", 1)

        try:
            # Reconnect if needed
            if self._redis is None:
                self._redis = aioredis.from_url(
                    redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_timeout=5,
                    socket_connect_timeout=5,
                    retry_on_timeout=True,
                )

            await self._redis.ping()
            return self._redis

        except Exception as e:
            print(f"[SemanticCache] Redis reconnecting due to error: {e}")
            self._redis = None
            return None

    async def check_cache(self, query: str) -> Optional[Dict[str, Any]]:
        r = await self._get_redis()
        if r is None:
            return None

        try:
            query_embedding = await embed_text(query)
            if query_embedding is None:
                return None

            cached_entries = await r.hgetall(self._index_key)

            best_match_key = None
            best_similarity = 0.0

            for cache_key, cached_data_str in cached_entries.items():
                cached_data = json.loads(cached_data_str)
                cached_embedding = cached_data.get("embedding")

                if not cached_embedding:
                    continue

                similarity = cosine_similarity(query_embedding, cached_embedding)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_key = cache_key

            if best_similarity >= CACHE_SIMILARITY_THRESHOLD and best_match_key:
                report_key = f"{self._cache_key_prefix}{best_match_key}"
                report_data = await r.get(report_key)

                if report_data:
                    print(
                        f"[SemanticCache] Cache HIT (similarity: {best_similarity:.4f})"
                    )
                    return json.loads(report_data)
                else:
                    # Remove stale index entry
                    await r.hdel(self._index_key, best_match_key)

            print(
                f"[SemanticCache] Cache MISS (best similarity: {best_similarity:.4f})"
            )
            return None

        except Exception as e:
            print(f"[SemanticCache] Error checking cache: {e}")
            self._redis = None
            return None

    async def store_result(self, query: str, report_data: Dict[str, Any]) -> None:
        r = await self._get_redis()
        if r is None:
            return

        try:
            query_embedding = await embed_text(query)
            if query_embedding is None:
                return

            cache_key = hashlib.md5(query.encode()).hexdigest()

            index_entry = json.dumps(
                {
                    "query": query,
                    "embedding": query_embedding,
                }
            )

            await r.hset(self._index_key, cache_key, index_entry)

            await r.set(
                f"{self._cache_key_prefix}{cache_key}",
                json.dumps(report_data),
            )

            print(f"[SemanticCache] Stored result for query: {query[:60]}...")

        except Exception as e:
            print(f"[SemanticCache] Error storing result: {e}")
            self._redis = None

    async def clear_cache(self) -> bool:
        """Clear all cache entries from Redis."""
        r = await self._get_redis()
        if r is None:
            return False

        try:
            # Plan: Find keys with prefix and delete them + delete the index
            # Strategy: use scan_iter to find all keys starting with prefix
            count = 0
            async for key in r.scan_iter(f"{self._cache_key_prefix}*"):
                await r.delete(key)
                count += 1
            
            # Delete the index key
            await r.delete(self._index_key)
            
            print(f"[SemanticCache] Successfully cleared {count} cache entries and the index.")
            return True
        except Exception as e:
            print(f"[SemanticCache] Error clearing cache: {e}")
            return False

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
