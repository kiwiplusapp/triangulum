"""
Shared HTTP/WebSocket plumbing for live venue adapters.

Uses ``aiohttp`` when available and degrades to a documented failure otherwise
rather than silently importing something else. Live trading without a proper
async HTTP client is not a mode this engine offers.

Signing helpers live here because every venue reinvents the same HMAC dance with
one gratuitous difference, and having them side by side makes those differences
auditable:

    Binance   HMAC-SHA256 over the urlencoded query string, hex
    Bybit     HMAC-SHA256 over timestamp+key+recv_window+body, hex
    OKX       HMAC-SHA256 over timestamp+method+path+body, base64
    KuCoin    HMAC-SHA256 over timestamp+method+path+body, base64, plus a
              separately-signed passphrase
    Kraken    HMAC-SHA512 over path + SHA256(nonce + body), base64
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
import urllib.parse
from typing import Any, AsyncIterator, Mapping

from triangulum.core.constants import DEFAULT_HTTP_TIMEOUT_SEC, USER_AGENT
from triangulum.core.errors import (
    AuthenticationError,
    BadResponse,
    ConnectionLost,
    RateLimited,
    VenueMaintenance,
)

logger = logging.getLogger(__name__)

__all__ = ["HttpClient", "WebSocketClient", "sign_hmac", "AIOHTTP_AVAILABLE"]

try:  # pragma: no cover - environment dependent
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False


def sign_hmac(
    secret: str,
    payload: str,
    *,
    algorithm: str = "sha256",
    encoding: str = "hex",
) -> str:
    digestmod = getattr(hashlib, algorithm)
    mac = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), digestmod)
    if encoding == "hex":
        return mac.hexdigest()
    if encoding == "base64":
        return base64.b64encode(mac.digest()).decode("ascii")
    raise ValueError(f"unknown encoding {encoding!r}")


def sign_kraken(secret_b64: str, path: str, nonce: str, body: str) -> str:
    """Kraken's two-stage scheme: SHA256 the payload, then HMAC-SHA512 it."""
    sha = hashlib.sha256((nonce + body).encode("utf-8")).digest()
    mac = hmac.new(base64.b64decode(secret_b64), path.encode("utf-8") + sha, hashlib.sha512)
    return base64.b64encode(mac.digest()).decode("ascii")


class HttpClient:
    """
    Thin async REST client with venue-aware error translation.

    Error translation is the point: every venue signals "you are rate limited"
    differently (HTTP 429, HTTP 418, a 200 with an error code in the body, an
    HTML error page from a CDN), and the layers above must not have to care.
    """

    def __init__(
        self,
        base_url: str,
        *,
        venue: str = "",
        timeout_sec: float = DEFAULT_HTTP_TIMEOUT_SEC,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError(
                "aiohttp is required for live venue adapters. "
                "Install it with: pip install aiohttp"
            )
        self.base_url = base_url.rstrip("/")
        self.venue = venue
        self.timeout_sec = timeout_sec
        self._headers = {"User-Agent": USER_AGENT, **(headers or {})}
        self._session: Any = None
        self.requests = 0
        self.errors = 0

    async def __aenter__(self) -> "HttpClient":
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def start(self) -> None:
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout_sec)
            self._session = aiohttp.ClientSession(timeout=timeout, headers=self._headers)

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        data: Any = None,
        headers: Mapping[str, str] | None = None,
        retries: int = 2,
    ) -> Any:
        await self.start()
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        last_error: Exception | None = None

        for attempt in range(retries + 1):
            self.requests += 1
            try:
                async with self._session.request(
                    method, url, params=params, data=data, headers=headers
                ) as response:
                    text = await response.text()
                    self._raise_for_status(response.status, text, response.headers)
                    if not text:
                        return None
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError as exc:
                        raise BadResponse(
                            f"non-JSON response: {text[:200]}", venue=self.venue
                        ) from exc
            except RateLimited:
                raise
            except (AuthenticationError, BadResponse):
                raise
            except asyncio.TimeoutError as exc:
                last_error = ConnectionLost(f"timeout on {method} {path}", venue=self.venue)
                self.errors += 1
            except Exception as exc:  # aiohttp client errors
                last_error = ConnectionLost(str(exc), venue=self.venue)
                self.errors += 1
            if attempt < retries:
                await asyncio.sleep(0.25 * (2 ** attempt))

        raise last_error or ConnectionLost(f"{method} {path} failed", venue=self.venue)

    def _raise_for_status(self, status: int, text: str, headers: Mapping[str, str]) -> None:
        if status < 400:
            return
        if status in (401, 403):
            raise AuthenticationError(f"auth failed ({status}): {text[:200]}", venue=self.venue)
        if status in (418, 429):
            retry_after = float(headers.get("Retry-After", "1") or 1)
            raise RateLimited(
                f"rate limited ({status})", venue=self.venue, retry_after=retry_after
            )
        if status in (503, 502, 504):
            raise VenueMaintenance(f"venue unavailable ({status})", venue=self.venue)
        raise BadResponse(f"HTTP {status}: {text[:300]}", venue=self.venue)

    async def get(self, path: str, **kwargs: Any) -> Any:
        return await self.request("GET", path, **kwargs)

    async def post(self, path: str, **kwargs: Any) -> Any:
        return await self.request("POST", path, **kwargs)

    async def delete(self, path: str, **kwargs: Any) -> Any:
        return await self.request("DELETE", path, **kwargs)

    @staticmethod
    def urlencode(params: Mapping[str, Any]) -> str:
        return urllib.parse.urlencode(
            {k: v for k, v in params.items() if v is not None}, safe=","
        )


class WebSocketClient:
    """
    Reconnecting WebSocket wrapper.

    Deliberately does not resubscribe automatically: the adapter must decide
    what to do after a reconnect, and for order books the answer is always
    "discard everything and resnapshot", never "resume". Auto-resubscribe
    without resnapshot is how you end up trading a book with a hole in it.
    """

    def __init__(
        self,
        url: str,
        *,
        venue: str = "",
        ping_interval: float = 20.0,
        ping_timeout: float = 10.0,
        max_message_size: int = 8 * 1024 * 1024,
    ) -> None:
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp is required for live venue adapters")
        self.url = url
        self.venue = venue
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.max_message_size = max_message_size
        self._session: Any = None
        self._ws: Any = None
        self.messages_received = 0
        self.connects = 0

    async def connect(self) -> None:
        await self.close()
        self._session = aiohttp.ClientSession(headers={"User-Agent": USER_AGENT})
        self._ws = await self._session.ws_connect(
            self.url,
            heartbeat=self.ping_interval,
            receive_timeout=self.ping_interval + self.ping_timeout,
            max_msg_size=self.max_message_size,
            autoping=True,
        )
        self.connects += 1
        logger.info("ws connected: %s", self.url)

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def send_json(self, payload: Any) -> None:
        if self._ws is None:
            raise ConnectionLost("websocket not connected", venue=self.venue)
        await self._ws.send_str(json.dumps(payload))

    async def messages(self) -> AsyncIterator[Any]:
        if self._ws is None:
            raise ConnectionLost("websocket not connected", venue=self.venue)
        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                self.messages_received += 1
                try:
                    yield json.loads(msg.data)
                except json.JSONDecodeError:
                    logger.debug("ws: dropping non-JSON frame from %s", self.venue)
            elif msg.type == aiohttp.WSMsgType.BINARY:
                self.messages_received += 1
                try:
                    yield json.loads(msg.data.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                raise ConnectionLost(f"websocket closed: {msg.type}", venue=self.venue)

    @property
    def connected(self) -> bool:
        return self._ws is not None and not self._ws.closed
