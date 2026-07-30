#!/usr/bin/env python3
"""Minimal same-origin intake API for the TTinker Grid early-access form."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import sqlite3
import threading
import time
import uuid
from collections import deque
from contextlib import closing
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("ttinker-grid")

MAX_BODY_BYTES = 16 * 1024
MAX_REQUESTS_PER_WINDOW = 5
RATE_LIMIT_WINDOW_SECONDS = 10 * 60
RATE_LIMIT_CLEANUP_INTERVAL_SECONDS = 60
MAX_RATE_LIMIT_CLIENTS = 10_000
MAX_MODELS_PER_NODE = 64
NODE_ONLINE_WINDOW_SECONDS = 120
NODE_TOKEN_PREFIX = "tg_node_"

ROLE_BUYER = "buyer"
ROLE_SUPPLIER = "supplier"

BUYER_INTENTS = frozenset(("少于 1M", "1M – 10M", "10M – 100M", "100M 以上"))
SUPPLIER_INTENTS = frozenset(
    ("单张消费级 GPU", "多张消费级 GPU", "数据中心 GPU", "已有推理集群")
)
INTENTS_BY_ROLE = {
    ROLE_BUYER: BUYER_INTENTS,
    ROLE_SUPPLIER: SUPPLIER_INTENTS,
}

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
EXPECTED_FIELDS = frozenset(("role", "name", "email", "intent", "consent"))
EXPECTED_HEARTBEAT_FIELDS = frozenset(("agent_version", "provider", "models"))


class RequestError(Exception):
    """A safe client-facing request error."""

    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _required_text(payload: dict[str, Any], field: str, max_length: int) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise RequestError(HTTPStatus.BAD_REQUEST, f"{field} 必须是文本")

    value = value.strip()
    if not value:
        raise RequestError(HTTPStatus.BAD_REQUEST, f"{field} 不能为空")
    if len(value) > max_length:
        raise RequestError(HTTPStatus.BAD_REQUEST, f"{field} 超过长度限制")
    return value


def validate_application(payload: Any) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise RequestError(HTTPStatus.BAD_REQUEST, "请求内容必须是 JSON 对象")

    unknown_fields = payload.keys() - EXPECTED_FIELDS
    if unknown_fields:
        raise RequestError(HTTPStatus.BAD_REQUEST, "请求包含不支持的字段")

    role = _required_text(payload, "role", 16)
    if role not in INTENTS_BY_ROLE:
        raise RequestError(HTTPStatus.BAD_REQUEST, "申请角色无效")

    name = _required_text(payload, "name", 80)
    email = _required_text(payload, "email", 254).casefold()
    if not EMAIL_PATTERN.fullmatch(email):
        raise RequestError(HTTPStatus.BAD_REQUEST, "邮箱格式无效")

    intent = _required_text(payload, "intent", 80)
    if intent not in INTENTS_BY_ROLE[role]:
        raise RequestError(HTTPStatus.BAD_REQUEST, "申请意向与角色不匹配")

    if payload.get("consent") is not True:
        raise RequestError(HTTPStatus.BAD_REQUEST, "提交前需要同意申请信息处理说明")

    return {
        "role": role,
        "name": name,
        "email": email,
        "intent": intent,
    }


def validate_heartbeat(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RequestError(HTTPStatus.BAD_REQUEST, "心跳内容必须是 JSON 对象")

    unknown_fields = payload.keys() - EXPECTED_HEARTBEAT_FIELDS
    if unknown_fields:
        raise RequestError(HTTPStatus.BAD_REQUEST, "心跳包含不支持的字段")

    agent_version = _required_text(payload, "agent_version", 32)
    provider = _required_text(payload, "provider", 64)
    models = payload.get("models")
    if not isinstance(models, list):
        raise RequestError(HTTPStatus.BAD_REQUEST, "models 必须是数组")
    if len(models) > MAX_MODELS_PER_NODE:
        raise RequestError(HTTPStatus.BAD_REQUEST, "模型数量超过单节点限制")

    normalized_models: list[str] = []
    seen_models: set[str] = set()
    for model in models:
        if not isinstance(model, str):
            raise RequestError(HTTPStatus.BAD_REQUEST, "模型 ID 必须是文本")
        model_id = model.strip()
        if not model_id or len(model_id) > 128:
            raise RequestError(HTTPStatus.BAD_REQUEST, "模型 ID 无效")
        if model_id not in seen_models:
            seen_models.add(model_id)
            normalized_models.append(model_id)

    return {
        "agent_version": agent_version,
        "provider": provider,
        "models": normalized_models,
    }


def hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


class ApplicationStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS early_access_applications (
                        id INTEGER PRIMARY KEY,
                        role TEXT NOT NULL CHECK (role IN ('buyer', 'supplier')),
                        name TEXT NOT NULL,
                        email TEXT NOT NULL,
                        intent TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE (role, email)
                    )
                    """
                )
        os.chmod(self.database_path, 0o600)

    def save(self, application: dict[str, str]) -> bool:
        """Persist an application and return True only when a new row was created."""

        timestamp = datetime.now(timezone.utc).isoformat()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT id
                FROM early_access_applications
                WHERE role = ? AND email = ?
                """,
                (application["role"], application["email"]),
            ).fetchone()

            if existing is None:
                connection.execute(
                    """
                    INSERT INTO early_access_applications (
                        role, name, email, intent, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        application["role"],
                        application["name"],
                        application["email"],
                        application["intent"],
                        timestamp,
                        timestamp,
                    ),
                )
                created = True
            else:
                connection.execute(
                    """
                    UPDATE early_access_applications
                    SET name = ?, intent = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        application["name"],
                        application["intent"],
                        timestamp,
                        existing["id"],
                    ),
                )
                created = False

            connection.commit()
            return created
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


class NodeRegistry:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS inference_nodes (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        token_hash TEXT NOT NULL UNIQUE,
                    token_prefix TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    revoked_at TEXT,
                    last_seen_at TEXT,
                        agent_version TEXT,
                        provider TEXT,
                        models_json TEXT NOT NULL DEFAULT '[]'
                    )
                    """
                )
        os.chmod(self.database_path, 0o600)

    def create_node(self, name: str) -> tuple[str, str]:
        normalized_name = name.strip()
        if not normalized_name or len(normalized_name) > 80:
            raise ValueError("Node name must contain 1 to 80 characters")

        node_id = str(uuid.uuid4())
        token = f"{NODE_TOKEN_PREFIX}{secrets.token_urlsafe(32)}"
        timestamp = datetime.now(timezone.utc).isoformat()
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO inference_nodes (
                        id, name, token_hash, token_prefix, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        node_id,
                        normalized_name,
                        hash_secret(token),
                        token[:16],
                        timestamp,
                    ),
                )
        return node_id, token

    def record_heartbeat(self, token: str, heartbeat: dict[str, Any]) -> dict[str, str]:
        token_hash = hash_secret(token)
        timestamp = datetime.now(timezone.utc).isoformat()
        with closing(self._connect()) as connection:
            with connection:
                node = connection.execute(
                    """
                SELECT id, name, token_hash
                FROM inference_nodes
                WHERE token_hash = ? AND revoked_at IS NULL
                    """,
                    (token_hash,),
                ).fetchone()
                if node is None or not hmac.compare_digest(node["token_hash"], token_hash):
                    raise RequestError(HTTPStatus.UNAUTHORIZED, "节点凭证无效")

                connection.execute(
                    """
                    UPDATE inference_nodes
                    SET last_seen_at = ?, agent_version = ?, provider = ?, models_json = ?
                    WHERE id = ?
                    """,
                    (
                        timestamp,
                        heartbeat["agent_version"],
                        heartbeat["provider"],
                        json.dumps(heartbeat["models"], ensure_ascii=False),
                        node["id"],
                    ),
                )
        return {"id": node["id"], "name": node["name"]}

    def snapshot(self, now: datetime | None = None) -> dict[str, int]:
        current_time = now or datetime.now(timezone.utc)
        cutoff = (current_time - timedelta(seconds=NODE_ONLINE_WINDOW_SECONDS)).isoformat()
        with closing(self._connect()) as connection:
            nodes = connection.execute(
                """
                SELECT models_json
                FROM inference_nodes
                WHERE revoked_at IS NULL
                  AND last_seen_at IS NOT NULL
                  AND last_seen_at >= ?
                """,
                (cutoff,),
            ).fetchall()

        model_ids: set[str] = set()
        for node in nodes:
            model_ids.update(json.loads(node["models_json"]))
        return {
            "connected_nodes": len(nodes),
            "advertised_models": len(model_ids),
        }

    def list_nodes(self) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            nodes = connection.execute(
                """
                SELECT id, name, token_prefix, created_at, last_seen_at,
                       revoked_at, agent_version, provider, models_json
                FROM inference_nodes
                ORDER BY created_at
                """
            ).fetchall()
        return [
            {
                "id": node["id"],
                "name": node["name"],
                "token_prefix": node["token_prefix"],
                "created_at": node["created_at"],
                "revoked_at": node["revoked_at"],
                "last_seen_at": node["last_seen_at"],
                "agent_version": node["agent_version"],
                "provider": node["provider"],
                "models": json.loads(node["models_json"]),
            }
            for node in nodes
        ]

    def revoke_node(self, node_id: str) -> bool:
        timestamp = datetime.now(timezone.utc).isoformat()
        with closing(self._connect()) as connection:
            with connection:
                result = connection.execute(
                    """
                    UPDATE inference_nodes
                    SET revoked_at = ?
                    WHERE id = ? AND revoked_at IS NULL
                    """,
                    (timestamp, node_id),
                )
        return result.rowcount == 1


class RateLimiter:
    """Small in-memory limiter; persistence is unnecessary for this low-risk endpoint."""

    def __init__(
        self,
        max_requests: int = MAX_REQUESTS_PER_WINDOW,
        window_seconds: int = RATE_LIMIT_WINDOW_SECONDS,
    ) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._next_cleanup_at = 0.0

    def _cleanup(self, current_time: float) -> None:
        if (
            current_time < self._next_cleanup_at
            and len(self._requests) < MAX_RATE_LIMIT_CLIENTS
        ):
            return

        cutoff = current_time - self.window_seconds
        for key, requests in tuple(self._requests.items()):
            while requests and requests[0] <= cutoff:
                requests.popleft()
            if not requests:
                del self._requests[key]

        while len(self._requests) >= MAX_RATE_LIMIT_CLIENTS:
            self._requests.pop(next(iter(self._requests)))
        self._next_cleanup_at = current_time + RATE_LIMIT_CLEANUP_INTERVAL_SECONDS

    def allow(self, client_key: str, now: float | None = None) -> bool:
        current_time = time.monotonic() if now is None else now
        cutoff = current_time - self.window_seconds

        with self._lock:
            self._cleanup(current_time)
            requests = self._requests.setdefault(client_key, deque())
            while requests and requests[0] <= cutoff:
                requests.popleft()

            if len(requests) >= self.max_requests:
                return False

            requests.append(current_time)
            return True


class GridRequestHandler(BaseHTTPRequestHandler):
    server_version = "TTinkerGrid/1"
    store: ApplicationStore
    node_registry: NodeRegistry
    rate_limiter: RateLimiter

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/healthz":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        if self.path == "/v1/status":
            node_status = self.node_registry.snapshot()
            self._send_json(
                HTTPStatus.OK,
                {
                    "service": "ttinker-grid-intake",
                    "status": "operational",
                    "accepting_applications": True,
                    "inference_available": False,
                    **node_status,
                },
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "接口不存在"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path == "/agent/v1/heartbeat":
            self._handle_node_heartbeat()
            return
        if self.path != "/v1/early-access":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "接口不存在"})
            return

        client_key = self.headers.get("X-Real-IP") or self.client_address[0]
        if not self.rate_limiter.allow(client_key):
            self._send_json(
                HTTPStatus.TOO_MANY_REQUESTS,
                {"error": "提交过于频繁，请稍后再试"},
            )
            return

        try:
            payload = self._read_json_body()
            application = validate_application(payload)
            created = self.store.save(application)
        except RequestError as error:
            self._send_json(error.status, {"error": error.message})
            return
        except sqlite3.Error:
            LOGGER.exception("Failed to persist an early-access application")
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "申请服务暂时不可用，请稍后再试"},
            )
            return
        except Exception:
            LOGGER.exception("Unexpected intake API failure")
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "服务器暂时无法处理请求"},
            )
            return

        status = HTTPStatus.CREATED if created else HTTPStatus.OK
        self._send_json(
            status,
            {
                "status": "accepted",
                "created": created,
            },
        )

    def _handle_node_heartbeat(self) -> None:
        authorization = self.headers.get("Authorization", "")
        scheme, separator, token = authorization.partition(" ")
        if (
            separator != " "
            or scheme.lower() != "bearer"
            or not token.startswith(NODE_TOKEN_PREFIX)
            or len(token) > 256
        ):
            self._send_json(
                HTTPStatus.UNAUTHORIZED,
                {"error": "缺少有效的节点凭证"},
                {"WWW-Authenticate": "Bearer"},
            )
            return

        try:
            heartbeat = validate_heartbeat(self._read_json_body())
            node = self.node_registry.record_heartbeat(token, heartbeat)
        except RequestError as error:
            extra_headers = (
                {"WWW-Authenticate": "Bearer"}
                if error.status == HTTPStatus.UNAUTHORIZED
                else None
            )
            self._send_json(error.status, {"error": error.message}, extra_headers)
            return
        except sqlite3.Error:
            LOGGER.exception("Failed to persist a node heartbeat")
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": "节点控制面暂时不可用"},
            )
            return

        self._send_json(
            HTTPStatus.OK,
            {
                "status": "accepted",
                "node": node,
                "next_heartbeat_seconds": 60,
            },
        )

    def _read_json_body(self) -> Any:
        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("application/json"):
            raise RequestError(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "Content-Type 必须是 application/json",
            )

        content_length_value = self.headers.get("Content-Length")
        if content_length_value is None:
            raise RequestError(HTTPStatus.LENGTH_REQUIRED, "缺少 Content-Length")
        try:
            content_length = int(content_length_value)
        except ValueError as error:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Content-Length 无效") from error

        if content_length <= 0:
            raise RequestError(HTTPStatus.BAD_REQUEST, "请求内容不能为空")
        if content_length > MAX_BODY_BYTES:
            raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "请求内容过大")

        body = self.rfile.read(content_length)
        try:
            return json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise RequestError(HTTPStatus.BAD_REQUEST, "JSON 格式无效") from error

    def _send_json(
        self,
        status: HTTPStatus,
        payload: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for header, value in (extra_headers or {}).items():
            self.send_header(header, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string: str, *args: Any) -> None:
        # Request bodies and client addresses are intentionally excluded from logs.
        LOGGER.debug("Handled %s request", self.command)


def create_server(
    host: str,
    port: int,
    database_path: Path,
    rate_limiter: RateLimiter | None = None,
    node_registry: NodeRegistry | None = None,
) -> ThreadingHTTPServer:
    store = ApplicationStore(database_path)
    limiter = rate_limiter or RateLimiter()
    registry = node_registry or NodeRegistry(database_path)

    class ConfiguredGridRequestHandler(GridRequestHandler):
        pass

    ConfiguredGridRequestHandler.store = store
    ConfiguredGridRequestHandler.rate_limiter = limiter
    ConfiguredGridRequestHandler.node_registry = registry

    server = ThreadingHTTPServer((host, port), ConfiguredGridRequestHandler)
    server.daemon_threads = True
    return server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TTinker Grid intake API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("/var/lib/ttinker-grid/applications.sqlite3"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    server = create_server(args.host, args.port, args.database)
    LOGGER.info("Intake API listening on %s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
