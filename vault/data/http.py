"""
HTTP transport with automatic fallback.

Why this exists rather than a bare ``urllib.request.urlopen``:

Corporate and sandboxed networks frequently sit behind a CONNECT proxy that
re-terminates TLS. Some clients negotiate that tunnel correctly and some do
not, and the ones that do not fail in the least helpful way possible -- a read
timeout or a bare ``RemoteDisconnected``, with no error body, several seconds
after the request. Observed directly while building this: ``curl`` fetched a
FRED series in 1.4 seconds while ``urllib`` and ``requests`` both timed out
against the same URL through the same proxy in the same shell.

A data layer that dies on that is a data layer that works on the author's
laptop and nowhere else. So the transport is pluggable, urllib is tried first
(no subprocess, correct on an unproxied machine), and the system ``curl``
binary is the fallback. The choice is probed once and remembered.

This is not a workaround for a bug in urllib. It is the recognition that
"which HTTP client can reach the internet from here" is a property of the
environment, not of the code, and the code should discover it rather than
assume.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from typing import Protocol

logger = logging.getLogger(__name__)

__all__ = ["HttpTransport", "UrllibTransport", "CurlTransport", "get_transport", "http_get"]

DEFAULT_UA = "Vault/1.0 (macro research)"


class TransportError(Exception):
    """Any failure to retrieve a URL, from any transport."""


class HttpTransport(Protocol):
    name: str

    def get(self, url: str, *, timeout: float, user_agent: str) -> bytes: ...
    def available(self) -> bool: ...


class UrllibTransport:
    """Standard library. Correct on a normal network; preferred when it works."""

    name = "urllib"

    def available(self) -> bool:
        return True

    def get(self, url: str, *, timeout: float = 25.0, user_agent: str = DEFAULT_UA) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": user_agent})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            # A real HTTP status is a real answer -- do not fall back for it.
            raise TransportError(f"HTTP {exc.code} for {url}") from exc
        except Exception as exc:
            raise TransportError(f"{type(exc).__name__}: {exc}") from exc


class CurlTransport:
    """
    The system ``curl`` binary.

    Used when urllib cannot reach the network. ``--fail-with-body`` makes curl
    exit non-zero on a 4xx/5xx while still returning the body, so a policy
    denial is distinguishable from a transport failure.
    """

    name = "curl"

    def __init__(self, binary: str = "") -> None:
        self.binary = binary or shutil.which("curl") or ""

    def available(self) -> bool:
        return bool(self.binary)

    # curl exit codes that are worth retrying: transient transport faults
    # rather than a refusal. 92 is "HTTP/2 stream not closed cleanly", which is
    # the signature failure of a CONNECT proxy that mishandles HTTP/2
    # multiplexing -- observed here against a proxy that served the same URL
    # perfectly over HTTP/1.1. 28 is a timeout, 52 an empty reply, 56 a receive
    # error, 35 a TLS handshake fault.
    RETRYABLE = frozenset({18, 28, 35, 52, 55, 56, 92})

    def get(self, url: str, *, timeout: float = 25.0, user_agent: str = DEFAULT_UA,
            attempts: int = 3) -> bytes:
        if not self.binary:
            raise TransportError("curl is not installed")

        last = ""
        for attempt in range(attempts):
            try:
                result = subprocess.run(
                    [
                        self.binary, "-sS", "--fail-with-body",
                        # Pin HTTP/1.1. Some egress proxies renegotiate TLS and
                        # then fail to close HTTP/2 streams cleanly; the request
                        # succeeds, the body arrives, and curl still exits 92.
                        # HTTP/1.1 costs nothing here -- these are small, cold,
                        # one-off CSV fetches with no multiplexing to gain from.
                        "--http1.1",
                        "--retry", "0",
                        "--max-time", str(int(timeout)),
                        "-A", user_agent, url,
                    ],
                    capture_output=True, timeout=timeout + 8,
                )
            except subprocess.TimeoutExpired:
                last = f"timed out after {timeout}s"
                continue

            if result.returncode == 0:
                return result.stdout

            last = result.stderr.decode("utf-8", "replace").strip()[:200]
            if result.returncode not in self.RETRYABLE:
                raise TransportError(f"curl exit {result.returncode}: {last}")
            logger.debug(
                "curl attempt %d/%d failed (%d): %s",
                attempt + 1, attempts, result.returncode, last,
            )
            time.sleep(0.6 * (2 ** attempt))

        raise TransportError(f"curl failed after {attempts} attempts: {last}")


_PROBE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10&cosd=2025-01-01"
_selected: HttpTransport | None = None


def get_transport(*, force: str = "", probe_timeout: float = 12.0) -> HttpTransport:
    """
    Pick a working transport, probing once and caching the result.

    ``VAULT_HTTP_TRANSPORT=curl`` (or ``urllib``) skips the probe entirely,
    which matters in CI where the probe URL may be unreachable for reasons that
    have nothing to do with which client works.
    """
    global _selected

    forced = force or os.environ.get("VAULT_HTTP_TRANSPORT", "")
    if forced:
        transport = CurlTransport() if forced == "curl" else UrllibTransport()
        _selected = transport
        return transport

    if _selected is not None:
        return _selected

    for candidate in (UrllibTransport(), CurlTransport()):
        if not candidate.available():
            continue
        try:
            payload = candidate.get(_PROBE_URL, timeout=probe_timeout)
        except TransportError as exc:
            logger.info("transport %s failed the probe (%s); trying the next", candidate.name, exc)
            continue
        if not payload or len(payload) < 32:
            logger.info("transport %s returned an empty probe body; trying the next",
                        candidate.name)
            continue
        if candidate.name != "urllib":
            logger.warning(
                "using the %s transport: urllib could not reach the network from "
                "here, which usually means a CONNECT proxy it does not negotiate",
                candidate.name,
            )
        _selected = candidate
        return candidate

    # Nothing probed clean. Return urllib so callers get a real error with a
    # real traceback rather than a silent no-op.
    logger.error("no HTTP transport reached the network; falling back to urllib")
    _selected = UrllibTransport()
    return _selected


def http_get(url: str, *, timeout: float = 25.0, user_agent: str = DEFAULT_UA) -> bytes:
    return get_transport().get(url, timeout=timeout, user_agent=user_agent)


def reset_transport() -> None:
    """Clear the cached choice. For tests."""
    global _selected
    _selected = None
