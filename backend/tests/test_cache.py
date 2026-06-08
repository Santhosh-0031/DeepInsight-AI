"""
Test: Redis Semantic Cache (cache/redis_cache.py)

Robust tests for SemanticCache using mocked Redis and embeddings.
Designed to match redis.asyncio + decode_responses=True behavior.
"""

import json
import pytest
from unittest.mock import patch, AsyncMock

from app.cache.redis_cache import SemanticCache


# =========================================================
# Init & Redis Connection
# =========================================================

class TestSemanticCacheInit:

    def test_init_defaults(self):
        cache = SemanticCache()
        assert cache._redis is None
        assert cache._cache_key_prefix == "dra:cache:"
        assert cache._index_key == "dra:cache:index"

    @pytest.mark.asyncio
    async def test_get_redis_returns_none_when_ping_fails(self):
        cache = SemanticCache()

        with patch("app.cache.redis_cache.aioredis.from_url") as mock_from_url:
            mock_client = AsyncMock()
            mock_client.ping.side_effect = ConnectionError("refused")
            mock_from_url.return_value = mock_client

            with patch.dict("os.environ", {"REDIS_URL": "redis://test"}):
                result = await cache._get_redis()

        assert result is None


# =========================================================
# check_cache()
# =========================================================

class TestSemanticCacheCheckCache:

    @pytest.mark.asyncio
    async def test_cache_miss_when_index_empty(self):
        cache = SemanticCache()
        mock_redis = AsyncMock()
        mock_redis.hgetall.return_value = {}

        with patch.object(cache, "_get_redis", return_value=mock_redis), \
             patch("app.cache.redis_cache.embed_text", new_callable=AsyncMock) as mock_embed:

            mock_embed.return_value = [0.1] * 128

            result = await cache.check_cache("new query")

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_hit_returns_report(self):
        cache = SemanticCache()
        mock_redis = AsyncMock()

        embedding = [0.1] * 128
        report = {"content": "Cached report"}

        mock_redis.hgetall.return_value = {
            "abc123": json.dumps({"query": "AI research", "embedding": embedding})
        }
        mock_redis.get.return_value = json.dumps(report)

        with patch.object(cache, "_get_redis", return_value=mock_redis), \
             patch("app.cache.redis_cache.embed_text", new_callable=AsyncMock) as mock_embed, \
             patch("app.cache.redis_cache.cosine_similarity", return_value=0.95):

            mock_embed.return_value = embedding
            result = await cache.check_cache("AI research overview")

        assert result == report

    @pytest.mark.asyncio
    async def test_cache_low_similarity_returns_none(self):
        cache = SemanticCache()
        mock_redis = AsyncMock()

        mock_redis.hgetall.return_value = {
            "abc123": json.dumps(
                {"query": "cooking", "embedding": [0.9] * 128}
            )
        }

        with patch.object(cache, "_get_redis", return_value=mock_redis), \
             patch("app.cache.redis_cache.embed_text", new_callable=AsyncMock) as mock_embed, \
             patch("app.cache.redis_cache.cosine_similarity", return_value=0.2):

            mock_embed.return_value = [0.1] * 128
            result = await cache.check_cache("quantum physics")

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_embed_failure_returns_none(self):
        cache = SemanticCache()
        mock_redis = AsyncMock()

        with patch.object(cache, "_get_redis", return_value=mock_redis), \
             patch("app.cache.redis_cache.embed_text", new_callable=AsyncMock) as mock_embed:

            mock_embed.return_value = None
            result = await cache.check_cache("test query")

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_removes_stale_index_entry(self):
        """If index exists but report is missing, entry should be removed."""
        cache = SemanticCache()
        mock_redis = AsyncMock()

        embedding = [0.1] * 128

        mock_redis.hgetall.return_value = {
            "abc123": json.dumps({"query": "AI", "embedding": embedding})
        }
        mock_redis.get.return_value = None  # Missing report

        with patch.object(cache, "_get_redis", return_value=mock_redis), \
             patch("app.cache.redis_cache.embed_text", new_callable=AsyncMock) as mock_embed, \
             patch("app.cache.redis_cache.cosine_similarity", return_value=0.95):

            mock_embed.return_value = embedding
            result = await cache.check_cache("AI research")

        assert result is None
        mock_redis.hdel.assert_called_once()


# =========================================================
# store_result()
# =========================================================

class TestSemanticCacheStoreResult:

    @pytest.mark.asyncio
    async def test_store_result_writes_to_redis(self):
        cache = SemanticCache()
        mock_redis = AsyncMock()

        with patch.object(cache, "_get_redis", return_value=mock_redis), \
             patch("app.cache.redis_cache.embed_text", new_callable=AsyncMock) as mock_embed:

            mock_embed.return_value = [0.1] * 128

            await cache.store_result(
                "test query",
                {"content": "test report"}
            )

        mock_redis.hset.assert_called_once()
        mock_redis.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_result_skips_when_no_redis(self):
        cache = SemanticCache()

        with patch.object(cache, "_get_redis", return_value=None):
            await cache.store_result("query", {"content": "report"})
        # No exception = success

    @pytest.mark.asyncio
    async def test_store_result_skips_when_embed_fails(self):
        cache = SemanticCache()
        mock_redis = AsyncMock()

        with patch.object(cache, "_get_redis", return_value=mock_redis), \
             patch("app.cache.redis_cache.embed_text", new_callable=AsyncMock) as mock_embed:

            mock_embed.return_value = None
            await cache.store_result("query", {"content": "report"})

        mock_redis.hset.assert_not_called()


# =========================================================
# close()
# =========================================================

class TestSemanticCacheClose:

    @pytest.mark.asyncio
    async def test_close_resets_connection(self):
        cache = SemanticCache()
        mock_redis = AsyncMock()
        cache._redis = mock_redis

        await cache.close()

        mock_redis.aclose.assert_called_once()
        assert cache._redis is None

    @pytest.mark.asyncio
    async def test_close_noop_when_not_connected(self):
        cache = SemanticCache()
        cache._redis = None

        await cache.close()  # Should not raise
        assert cache._redis is None