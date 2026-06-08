import asyncio
import os
import sys
from dotenv import load_dotenv

# Add the project root to sys.path
sys.path.append(os.getcwd())

from backend.app.cache.redis_cache import SemanticCache

async def clear_redis():
    # Load environment variables
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)
        print(f"Loaded .env from {env_path}")
    else:
        print(f"Warning: .env not found at {env_path}")

    if not os.getenv("REDIS_URL"):
        print("Error: REDIS_URL not set in environment.")
        return

    print("--- Clearing Redis Semantic Cache ---")
    cache = SemanticCache()
    success = await cache.clear_cache()
    
    if success:
        print("Done! Redis cache is now fresh.")
    else:
        print("Failed to clear Redis cache. Check your connection settings.")
    
    await cache.close()

if __name__ == "__main__":
    asyncio.run(clear_redis())
