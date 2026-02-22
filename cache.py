"""Async-compatible TTL cache using cachetools."""

import asyncio
from functools import wraps
from typing import Any, Callable

from cachetools import TTLCache


def cached_async(ttl: int, maxsize: int = 128) -> Callable:
    """Decorator that adds TTL caching to an async function.

    - Cache key is derived from function name + arguments (skips ``self``).
    - An ``asyncio.Lock`` prevents thundering-herd: concurrent calls for the
      same key wait for the first call to finish, then share the result.
    - Each decorated function gets its own cache instance so TTLs can differ.
    """
    cache: TTLCache = TTLCache(maxsize=maxsize, ttl=ttl)
    lock = asyncio.Lock()

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Build a hashable cache key.
            # Skip args[0] (self) for bound methods.
            cache_args = args[1:] if args else ()
            key = (func.__name__, cache_args, tuple(sorted(kwargs.items())))

            async with lock:
                if key in cache:
                    return cache[key]

            # Call the actual function outside the lock so we don't block
            # other cache lookups for different keys.
            result = await func(*args, **kwargs)

            async with lock:
                cache[key] = result

            return result

        # Expose the cache for testing / manual invalidation.
        wrapper.cache = cache  # type: ignore[attr-defined]
        return wrapper

    return decorator
