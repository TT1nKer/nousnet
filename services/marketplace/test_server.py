import json
import sqlite3
import tempfile
import threading
import unittest
from contextlib import closing
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from server import (
    ApplicationStore,
    NODE_TOKEN_PREFIX,
    NodeRegistry,
    RateLimiter,
    RequestError,
    create_server,
    hash_secret,
    validate_application,
    validate_heartbeat,
)


VALID_APPLICATION = {
    "role": "buyer",
    "name": "Example Lab",
    "email": "Team@Example.com",
    "intent": "1M – 10M",
    "consent": True,
}

VALID_HEARTBEAT = {
    "agent_version": "0.1.0",
    "provider": "openai-compatible",
    "models": ["qwen/qwen3-32b", "qwen/qwen3-32b", "deepseek/r1"],
}


class ValidationTests(unittest.TestCase):
    def test_normalizes_valid_application(self) -> None:
        application = validate_application(VALID_APPLICATION)

        self.assertEqual(application["email"], "team@example.com")
        self.assertEqual(application["name"], "Example Lab")

    def test_rejects_invalid_inputs(self) -> None:
        invalid_payloads = (
            None,
            {},
            {**VALID_APPLICATION, "role": "unknown"},
            {**VALID_APPLICATION, "email": "not-an-email"},
            {**VALID_APPLICATION, "intent": "数据中心 GPU"},
            {**VALID_APPLICATION, "consent": False},
            {**VALID_APPLICATION, "unexpected": "value"},
        )

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(RequestError):
                    validate_application(payload)

    def test_normalizes_heartbeat_models(self) -> None:
        heartbeat = validate_heartbeat(VALID_HEARTBEAT)

        self.assertEqual(heartbeat["models"], ["qwen/qwen3-32b", "deepseek/r1"])

    def test_rejects_invalid_heartbeats(self) -> None:
        invalid_payloads = (
            None,
            {},
            {**VALID_HEARTBEAT, "models": "qwen"},
            {**VALID_HEARTBEAT, "models": [""]},
            {**VALID_HEARTBEAT, "models": [1]},
            {**VALID_HEARTBEAT, "unexpected": True},
        )

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(RequestError):
                    validate_heartbeat(payload)


class ApplicationStoreTests(unittest.TestCase):
    def test_repeated_email_updates_instead_of_duplicating(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "applications.sqlite3"
            store = ApplicationStore(database_path)
            application = validate_application(VALID_APPLICATION)

            self.assertTrue(store.save(application))
            self.assertFalse(store.save({**application, "name": "Updated Lab"}))

            with closing(sqlite3.connect(database_path)) as connection:
                rows = connection.execute(
                    "SELECT name, email FROM early_access_applications"
                ).fetchall()

            self.assertEqual(rows, [("Updated Lab", "team@example.com")])
            self.assertEqual(database_path.stat().st_mode & 0o777, 0o600)


class NodeRegistryTests(unittest.TestCase):
    def test_creates_hashed_identity_and_records_heartbeat(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "applications.sqlite3"
            registry = NodeRegistry(database_path)

            node_id, token = registry.create_node("Test GPU")
            node = registry.record_heartbeat(token, validate_heartbeat(VALID_HEARTBEAT))
            snapshot = registry.snapshot()

            self.assertTrue(token.startswith(NODE_TOKEN_PREFIX))
            self.assertEqual(node["id"], node_id)
            self.assertEqual(snapshot, {"connected_nodes": 1, "advertised_models": 2})

            with closing(sqlite3.connect(database_path)) as connection:
                stored_hash = connection.execute(
                    "SELECT token_hash FROM inference_nodes WHERE id = ?",
                    (node_id,),
                ).fetchone()[0]

            self.assertEqual(stored_hash, hash_secret(token))
            self.assertNotEqual(stored_hash, token)

    def test_rejects_unknown_node_token(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "applications.sqlite3"
            registry = NodeRegistry(database_path)

            with self.assertRaises(RequestError):
                registry.record_heartbeat(
                    f"{NODE_TOKEN_PREFIX}unknown",
                    validate_heartbeat(VALID_HEARTBEAT),
                )

    def test_revoked_node_can_no_longer_send_heartbeats(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "applications.sqlite3"
            registry = NodeRegistry(database_path)
            node_id, token = registry.create_node("Revoked GPU")

            self.assertTrue(registry.revoke_node(node_id))
            self.assertFalse(registry.revoke_node(node_id))
            with self.assertRaises(RequestError):
                registry.record_heartbeat(token, validate_heartbeat(VALID_HEARTBEAT))


class RateLimiterTests(unittest.TestCase):
    def test_rejects_requests_after_limit_until_window_expires(self) -> None:
        limiter = RateLimiter(max_requests=2, window_seconds=10)

        self.assertTrue(limiter.allow("client", now=0))
        self.assertTrue(limiter.allow("client", now=1))
        self.assertFalse(limiter.allow("client", now=2))
        self.assertTrue(limiter.allow("client", now=11))

    def test_removes_expired_client_entries(self) -> None:
        limiter = RateLimiter(max_requests=2, window_seconds=10)

        self.assertTrue(limiter.allow("old-client", now=0))
        self.assertTrue(limiter.allow("new-client", now=61))

        self.assertNotIn("old-client", limiter._requests)


class HttpApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        database_path = Path(self.temporary_directory.name) / "applications.sqlite3"
        self.node_registry = NodeRegistry(database_path)
        _, self.node_token = self.node_registry.create_node("HTTP Test GPU")
        self.server = create_server(
            "127.0.0.1",
            0,
            database_path,
            RateLimiter(max_requests=20, window_seconds=60),
            self.node_registry,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temporary_directory.cleanup()

    def request_json(
        self,
        path: str,
        method: str = "GET",
        payload: object | None = None,
        token: str | None = None,
    ) -> tuple[int, dict[str, object]]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers=headers,
        )
        try:
            response = urlopen(request, timeout=2)
        except HTTPError as error:
            response = error

        with response:
            return response.status, json.loads(response.read())

    def test_status_reports_inference_as_unavailable(self) -> None:
        status, payload = self.request_json("/v1/status")

        self.assertEqual(status, 200)
        self.assertTrue(payload["accepting_applications"])
        self.assertFalse(payload["inference_available"])
        self.assertEqual(payload["connected_nodes"], 0)
        self.assertEqual(payload["advertised_models"], 0)

    def test_authenticated_heartbeat_updates_public_status(self) -> None:
        heartbeat_status, heartbeat = self.request_json(
            "/agent/v1/heartbeat",
            method="POST",
            payload=VALID_HEARTBEAT,
            token=self.node_token,
        )
        status_code, status = self.request_json("/v1/status")

        self.assertEqual(heartbeat_status, 200)
        self.assertEqual(heartbeat["status"], "accepted")
        self.assertEqual(status_code, 200)
        self.assertEqual(status["connected_nodes"], 1)
        self.assertEqual(status["advertised_models"], 2)

    def test_rejects_heartbeat_without_node_token(self) -> None:
        status, payload = self.request_json(
            "/agent/v1/heartbeat",
            method="POST",
            payload=VALID_HEARTBEAT,
        )

        self.assertEqual(status, 401)
        self.assertIn("error", payload)

    def test_application_create_and_idempotent_update(self) -> None:
        created_status, created = self.request_json(
            "/v1/early-access",
            method="POST",
            payload=VALID_APPLICATION,
        )
        updated_status, updated = self.request_json(
            "/v1/early-access",
            method="POST",
            payload={**VALID_APPLICATION, "name": "Updated Lab"},
        )

        self.assertEqual(created_status, 201)
        self.assertEqual(created, {"status": "accepted", "created": True})
        self.assertEqual(updated_status, 200)
        self.assertEqual(updated, {"status": "accepted", "created": False})

    def test_rejects_invalid_json_shape(self) -> None:
        status, payload = self.request_json(
            "/v1/early-access",
            method="POST",
            payload=[],
        )

        self.assertEqual(status, 400)
        self.assertIn("error", payload)


if __name__ == "__main__":
    unittest.main()
