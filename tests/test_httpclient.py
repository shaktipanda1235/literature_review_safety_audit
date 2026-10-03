import asyncio
import time

import httpx
import pytest
import respx

from app.sources.base import HttpClient


@pytest.mark.asyncio
@respx.mock
async def test_retry_on_429():
    url = "https://example.com/retry"
    calls = []

    def handler(request):
        calls.append(time.time())
        if len(calls) == 1:
            return httpx.Response(429, text="retry")
        return httpx.Response(200, json={"ok": True})

    respx.get(url).mock(side_effect=handler)

    client = HttpClient(retries=3, min_interval=0.0)
    resp = await client.request("GET", url)
    await client.close()

    assert resp.status_code == 200
    assert len(calls) >= 2


@pytest.mark.asyncio
@respx.mock
async def test_cache_hit(tmp_path):
    url = "https://example.com/cache"
    # single server response
    respx.get(url).mock(return_value=httpx.Response(200, text="hello"))

    cache_dir = str(tmp_path / "cache")
    client = HttpClient(retries=1, min_interval=0.0, cache_dir=cache_dir)

    r1 = await client.request("GET", url, use_cache=True)
    # second request should be served from cache and not hit server
    r2 = await client.request("GET", url, use_cache=True)
    await client.close()

    assert r1.status_code == 200
    assert r2.status_code == 200
    # second response should be flagged from cache
    assert r2.extensions.get("from_cache") is True


@pytest.mark.asyncio
@respx.mock
async def test_rate_limiting():
    url = "https://example.com/slow"
    timestamps = []

    def handler(request):
        timestamps.append(time.time())
        return httpx.Response(200, text="ok")

    respx.get(url).mock(side_effect=handler)

    client = HttpClient(retries=1, min_interval=0.25)

    async def do():
        return await client.request("GET", url, use_cache=False)

    # fire two concurrent requests
    r1, r2 = await asyncio.gather(do(), do())
    await client.close()

    assert r1.status_code == 200 and r2.status_code == 200
    # ensure server saw calls spaced by at least min_interval (allow small epsilon)
    assert len(timestamps) == 2
    assert timestamps[1] - timestamps[0] >= 0.22
