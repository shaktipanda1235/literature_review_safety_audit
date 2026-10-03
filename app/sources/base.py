from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential


CACHE_DIR = os.path.join(".cache", "http")
DEFAULT_TTL = 300
DEFAULT_MIN_INTERVAL = 0.2
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


class HttpClient:
    """Async HTTP client with retries, per-host rate limiting and disk cache.

    - Uses `httpx.AsyncClient` for requests.
    - Retries on 429/5xx status codes using `tenacity.AsyncRetrying`.
    - Simple per-host rate limiter using an asyncio.Lock and last-request timestamps.
    - Simple file-based cache under `.cache/http/` with TTL.
    """

    def __init__(
        self,
        timeout: int = 20,
        retries: int = 3,
        min_interval: float = DEFAULT_MIN_INTERVAL,
        cache_dir: str = CACHE_DIR,
        cache_ttl: int = DEFAULT_TTL,
    ):
        self.timeout = timeout
        self.retries = retries
        self.min_interval = min_interval
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl

        self.client = httpx.AsyncClient(timeout=self.timeout)
        self._host_locks: Dict[str, asyncio.Lock] = {}
        self._last_request: Dict[str, float] = {}

    def _cache_key(self, method: str, url: str, params: Optional[Dict[str, Any]] = None, data: Optional[Any] = None) -> str:
        key_raw = f"{method}|{url}|{params}|{data}"
        return hashlib.sha256(key_raw.encode()).hexdigest()

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, key + ".json")

    def _read_cache(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._cache_path(key)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            ts = payload.get("_ts", 0)
            if time.time() - ts > self.cache_ttl:
                return None
            return payload.get("response")
        except Exception:
            return None

    def _write_cache(self, key: str, response: httpx.Response) -> None:
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            path = self._cache_path(key)
            payload = {
                "_ts": time.time(),
                "response": {
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "content": response.text,
                },
            }
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
        except Exception:
            # Do not fail on cache errors
            return

    async def _ensure_rate_limit(self, url: str) -> None:
        parsed = urlparse(url)
        host = parsed.netloc
        lock = self._host_locks.get(host)
        if lock is None:
            lock = asyncio.Lock()
            self._host_locks[host] = lock

        async with lock:
            last = self._last_request.get(host, 0)
            elapsed = time.time() - last
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self._last_request[host] = time.time()

    async def request(self, method: str, url: str, use_cache: bool = True, **kwargs) -> httpx.Response:
        """Perform an HTTP request with caching, rate-limiting and retry on retryable status codes.

        Returns an `httpx.Response`.
        """
        key = self._cache_key(method, url, params=kwargs.get("params"), data=kwargs.get("data"))
        if use_cache:
            cached = self._read_cache(key)
            if cached is not None:
                # Reconstruct a simple httpx.Response from cached payload
                req = httpx.Request(method, url)
                resp = httpx.Response(cached.get("status_code", 200), content=cached.get("content", ""), headers=cached.get("headers", {}), request=req)
                resp.extensions["from_cache"] = True
                return resp

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=10),
            retry=retry_if_exception(lambda e: isinstance(e, httpx.HTTPStatusError) and e.response is not None and e.response.status_code in RETRY_STATUS_CODES),
            reraise=True,
        ):
            with attempt:
                await self._ensure_rate_limit(url)

                resp = await self.client.request(method, url, **kwargs)

                if resp.status_code in RETRY_STATUS_CODES:
                    # Make httpx raise so tenacity can catch and retry
                    raise httpx.HTTPStatusError("Retryable status", request=resp.request, response=resp)

                if use_cache and resp.status_code == 200:
                    # best-effort cache write
                    self._write_cache(key, resp)

                return resp

        # If retries exhausted, last exception will be raised by AsyncRetrying

    async def close(self) -> None:
        await self.client.aclose()
