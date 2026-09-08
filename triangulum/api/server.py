"""
HTTP + WebSocket server for the dashboard and the Claude Code bridge.

Built on ``http.server`` and a hand-rolled WebSocket implementation rather than
FastAPI. That choice is deliberate and worth defending, because it looks like
the wrong one:

**Zero dependencies for the control plane.** The dashboard is how you find out
that the engine is in trouble. It must come up when the engine is in trouble --
including when a dependency failed to install, when a virtualenv is half-built,
or when you are SSH'd into a box at 3am. A control plane that depends on the
same fragile stack as the thing it monitors is not a control plane.

**The WebSocket protocol is 80 lines.** RFC 6455's server side, for text frames
under 64KB with no extensions, is a SHA-1, a base64, and a frame parser. That is
less code than the FastAPI dependency graph's import statements.

**Binding.** The server binds to 127.0.0.1 by default and refuses to bind to
0.0.0.0 without an auth token. A trading engine's control plane on a public
interface with no auth is an incident waiting to happen, and the code should not
let you do it by accident.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import mimetypes
import os
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)

__all__ = ["DashboardServer", "WebSocketConnection"]

_WS_MAGIC = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_STATIC_ROOT = Path(__file__).parent.parent / "dashboard"


def _root_for(dashboard: "DashboardServer") -> Path:
    """
    Static root for this server instance.

    A subclass can serve a different UI from the same server by setting
    ``static_root``; without this the Vault HUD would be served Triangulum's
    HTML, which fails silently and confusingly (a 200 with the wrong page).
    """
    return getattr(dashboard, "static_root", None) or _STATIC_ROOT


class WebSocketConnection:
    """One upgraded connection. Text frames only; that is all we send."""

    def __init__(self, sock: socket.socket, address: tuple) -> None:
        self.sock = sock
        self.address = address
        self.open = True
        self.sent = 0
        self._lock = threading.Lock()

    def send(self, payload: str) -> bool:
        """Send a text frame. Returns False when the peer is gone."""
        if not self.open:
            return False
        data = payload.encode("utf-8")
        header = bytearray([0x81])          # FIN + text opcode
        length = len(data)
        if length < 126:
            header.append(length)
        elif length < 65536:
            header.append(126)
            header.extend(struct.pack(">H", length))
        else:
            header.append(127)
            header.extend(struct.pack(">Q", length))
        try:
            with self._lock:
                self.sock.sendall(bytes(header) + data)
            self.sent += 1
            return True
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.close()
            return False

    def send_json(self, payload: Any) -> bool:
        return self.send(json.dumps(payload, default=str))

    def receive(self, timeout: float = 0.1) -> str | None:
        """Read one text frame, or None on timeout/close."""
        if not self.open:
            return None
        self.sock.settimeout(timeout)
        try:
            header = self._read_exactly(2)
            if header is None:
                return None
            opcode = header[0] & 0x0F
            if opcode == 0x8:               # close
                self.close()
                return None
            masked = bool(header[1] & 0x80)
            length = header[1] & 0x7F
            if length == 126:
                extra = self._read_exactly(2)
                length = struct.unpack(">H", extra)[0] if extra else 0
            elif length == 127:
                extra = self._read_exactly(8)
                length = struct.unpack(">Q", extra)[0] if extra else 0
            if length > 1_048_576:          # 1MB cap on inbound control messages
                self.close()
                return None
            mask = self._read_exactly(4) if masked else b""
            body = self._read_exactly(length) if length else b""
            if body is None:
                return None
            if masked and mask:
                body = bytes(b ^ mask[i % 4] for i, b in enumerate(body))
            return body.decode("utf-8", errors="replace")
        except (socket.timeout, TimeoutError):
            return None
        except (OSError, struct.error):
            self.close()
            return None

    def _read_exactly(self, count: int) -> bytes | None:
        chunks = b""
        while len(chunks) < count:
            chunk = self.sock.recv(count - len(chunks))
            if not chunk:
                self.close()
                return None
            chunks += chunk
        return chunks

    def close(self) -> None:
        if not self.open:
            return
        self.open = False
        try:
            self.sock.close()
        except OSError:
            pass


class _Handler(BaseHTTPRequestHandler):
    server_version = "Triangulum/1.0"
    protocol_version = "HTTP/1.1"

    # Injected by DashboardServer.
    dashboard: "DashboardServer" = None  # type: ignore[assignment]

    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug("http: " + fmt, *args)

    # -- auth --------------------------------------------------------------

    def _authorized(self) -> bool:
        token = self.dashboard.auth_token
        if not token:
            return True
        header = self.headers.get("Authorization", "")
        if header.startswith("Bearer ") and header[7:] == token:
            return True
        # Allow the token in the query string so the dashboard can open a
        # WebSocket, which cannot set headers from the browser.
        return f"token={token}" in (self.path or "")

    def _deny(self) -> None:
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "34")
        self.end_headers()
        self.wfile.write(b'{"error":"authorization required"}')

    # -- routing -----------------------------------------------------------

    def do_GET(self) -> None:
        path = (self.path or "/").split("?")[0]

        if self.headers.get("Upgrade", "").lower() == "websocket":
            if not self._authorized():
                return self._deny()
            return self._upgrade()

        if not self._authorized() and path.startswith("/api/"):
            return self._deny()

        if path == "/" or path == "/index.html":
            return self._serve_file(_root_for(self.dashboard) / "templates" / "index.html", "text/html")
        if path.startswith("/static/"):
            return self._serve_static(path)
        if path.startswith("/api/"):
            return self._serve_api(path)
        self._send_json({"error": "not found", "path": path}, status=404)

    def do_POST(self) -> None:
        if not self._authorized():
            return self._deny()
        path = (self.path or "/").split("?")[0]
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return self._send_json({"error": "invalid JSON"}, status=400)

        handler = self.dashboard.post_routes.get(path)
        if handler is None:
            return self._send_json({"error": "not found", "path": path}, status=404)
        try:
            self._send_json(handler(body))
        except Exception as exc:
            logger.exception("POST %s failed", path)
            self._send_json({"error": str(exc)}, status=500)

    def _serve_api(self, path: str) -> None:
        handler = self.dashboard.get_routes.get(path)
        if handler is None:
            return self._send_json({"error": "not found", "path": path}, status=404)
        try:
            self._send_json(handler())
        except Exception as exc:
            logger.exception("GET %s failed", path)
            self._send_json({"error": str(exc)}, status=500)

    def _serve_static(self, path: str) -> None:
        relative = path[len("/static/"):]
        # Path traversal guard: resolve and confirm containment.
        base = _root_for(self.dashboard)
        target = (base / "static" / relative).resolve()
        root = (base / "static").resolve()
        if not str(target).startswith(str(root)) or not target.is_file():
            return self._send_json({"error": "not found"}, status=404)
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self._serve_file(target, content_type)

    def _serve_file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            return self._send_json({"error": f"missing {path.name}"}, status=404)
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: Any, *, status: int = 200) -> None:
        data = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    # -- websocket ---------------------------------------------------------

    def _upgrade(self) -> None:
        key = self.headers.get("Sec-WebSocket-Key", "")
        if not key:
            return self._send_json({"error": "missing Sec-WebSocket-Key"}, status=400)
        accept = base64.b64encode(
            hashlib.sha1((key + _WS_MAGIC).encode("ascii")).digest()
        ).decode("ascii")

        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept)
        self.end_headers()

        connection = WebSocketConnection(self.connection, self.client_address)
        self.dashboard.register(connection)
        # Send the current state immediately so the page renders without
        # waiting for the next broadcast tick.
        connection.send_json({"type": "snapshot", "data": self.dashboard.snapshot()})

        try:
            while connection.open and not self.dashboard.stopping:
                message = connection.receive(timeout=0.5)
                if message:
                    self.dashboard.on_message(connection, message)
        finally:
            self.dashboard.unregister(connection)

    # Keep the connection alive so the browser reuses it.
    def handle_one_request(self) -> None:  # pragma: no cover - stdlib override
        try:
            super().handle_one_request()
        except (ConnectionResetError, BrokenPipeError):
            self.close_connection = True


class DashboardServer:
    """Serves the HUD, the JSON API, and the live WebSocket feed."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8787,
        auth_token: str = "",
        snapshot_provider: Callable[[], dict] | None = None,
        broadcast_interval_ms: float = 250.0,
        allow_kill_switch: bool = True,
    ) -> None:
        if host not in ("127.0.0.1", "localhost", "::1") and not auth_token:
            raise ValueError(
                f"refusing to bind the control plane to {host} without an auth "
                f"token. Set TRIANGULUM_DASHBOARD_TOKEN, or bind to 127.0.0.1."
            )
        self.host = host
        self.port = port
        self.auth_token = auth_token
        self.snapshot_provider = snapshot_provider or (lambda: {})
        self.broadcast_interval = broadcast_interval_ms / 1000.0
        self.allow_kill_switch = allow_kill_switch

        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._broadcaster: threading.Thread | None = None
        self._connections: list[WebSocketConnection] = []
        self._lock = threading.Lock()
        self.stopping = False
        self._log_lines: list[dict] = []

        self.get_routes: dict[str, Callable[[], Any]] = {
            "/api/state": self.snapshot,
            "/api/health": self.health,
            "/api/logs": lambda: {"lines": self._log_lines[-500:]},
        }
        self.post_routes: dict[str, Callable[[dict], Any]] = {}
        # Subclasses override this to serve their own UI.
        self.static_root: Path | None = None
        self.command_handlers: dict[str, Callable[[dict], Any]] = {}

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        _Handler.dashboard = self
        self._server = ThreadingHTTPServer((self.host, self.port), _Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="dashboard-http", daemon=True
        )
        self._thread.start()
        self._broadcaster = threading.Thread(
            target=self._broadcast_loop, name="dashboard-broadcast", daemon=True
        )
        self._broadcaster.start()
        logger.info(
            "dashboard at http://%s:%d%s",
            self.host, self.port,
            f"?token={self.auth_token}" if self.auth_token else "",
        )

    def stop(self) -> None:
        self.stopping = True
        with self._lock:
            for connection in self._connections:
                connection.close()
            self._connections.clear()
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    @property
    def url(self) -> str:
        suffix = f"?token={self.auth_token}" if self.auth_token else ""
        return f"http://{self.host}:{self.port}/{suffix}"

    # -- connections -------------------------------------------------------

    def register(self, connection: WebSocketConnection) -> None:
        with self._lock:
            self._connections.append(connection)
        logger.info("dashboard client connected (%d total)", len(self._connections))

    def unregister(self, connection: WebSocketConnection) -> None:
        with self._lock:
            if connection in self._connections:
                self._connections.remove(connection)
        connection.close()

    def broadcast(self, message_type: str, data: Any) -> int:
        payload = json.dumps({"type": message_type, "data": data}, default=str)
        delivered = 0
        with self._lock:
            dead: list[WebSocketConnection] = []
            for connection in self._connections:
                if connection.send(payload):
                    delivered += 1
                else:
                    dead.append(connection)
            for connection in dead:
                self._connections.remove(connection)
        return delivered

    def _broadcast_loop(self) -> None:
        while not self.stopping:
            time.sleep(self.broadcast_interval)
            if not self._connections:
                continue
            try:
                self.broadcast("snapshot", self.snapshot())
            except Exception:
                logger.exception("dashboard broadcast failed")

    # -- messages ----------------------------------------------------------

    def on_message(self, connection: WebSocketConnection, raw: str) -> None:
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            return
        command = message.get("command", "")
        handler = self.command_handlers.get(command)
        if handler is None:
            connection.send_json({"type": "error", "data": f"unknown command {command}"})
            return
        try:
            connection.send_json({"type": "result", "data": handler(message)})
        except Exception as exc:
            logger.exception("dashboard command %s failed", command)
            connection.send_json({"type": "error", "data": str(exc)})

    def register_command(self, name: str, handler: Callable[[dict], Any]) -> None:
        self.command_handlers[name] = handler

    def push_log(self, level: str, message: str, ts_ns: int = 0) -> None:
        self._log_lines.append({
            "level": level, "message": message, "ts": ts_ns or time.time_ns(),
        })
        if len(self._log_lines) > 2000:
            self._log_lines = self._log_lines[-1000:]
        self.broadcast("log", self._log_lines[-1])

    # -- data --------------------------------------------------------------

    def snapshot(self) -> dict:
        return self.snapshot_provider()

    def health(self) -> dict:
        return {
            "ok": True,
            "connections": len(self._connections),
            "stopping": self.stopping,
            "uptime_hint": time.time(),
        }
