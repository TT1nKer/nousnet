import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agent import AgentError, discover_models, send_heartbeat
from server import NodeRegistry, RateLimiter, create_server


class ProviderHandler(BaseHTTPRequestHandler):
    response_payload = {
        "object": "list",
        "data": [
            {"id": "qwen/qwen3-32b", "object": "model"},
            {"id": "deepseek/r1", "object": "model"},
            {"id": "qwen/qwen3-32b", "object": "model"},
        ],
    }

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/v1/models":
            self.send_error(404)
            return
        body = json.dumps(self.response_payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string: str, *args: object) -> None:
        pass


class AgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        database_path = Path(self.temporary_directory.name) / "applications.sqlite3"
        self.registry = NodeRegistry(database_path)
        _, self.token = self.registry.create_node("Agent Test GPU")

        self.control_server = create_server(
            "127.0.0.1",
            0,
            database_path,
            RateLimiter(max_requests=20, window_seconds=60),
            self.registry,
        )
        self.control_thread = threading.Thread(
            target=self.control_server.serve_forever,
            daemon=True,
        )
        self.control_thread.start()

        self.provider_server = ThreadingHTTPServer(("127.0.0.1", 0), ProviderHandler)
        self.provider_thread = threading.Thread(
            target=self.provider_server.serve_forever,
            daemon=True,
        )
        self.provider_thread.start()

    def tearDown(self) -> None:
        for server, thread in (
            (self.provider_server, self.provider_thread),
            (self.control_server, self.control_thread),
        ):
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.temporary_directory.cleanup()

    def test_discovers_models_and_sends_authenticated_heartbeat(self) -> None:
        provider_host, provider_port = self.provider_server.server_address
        control_host, control_port = self.control_server.server_address

        models = discover_models(f"http://{provider_host}:{provider_port}/v1")
        response = send_heartbeat(
            f"http://{control_host}:{control_port}",
            self.token,
            models,
            "test-provider",
        )

        self.assertEqual(models, ["qwen/qwen3-32b", "deepseek/r1"])
        self.assertEqual(response["status"], "accepted")
        self.assertEqual(
            self.registry.snapshot(),
            {"connected_nodes": 1, "advertised_models": 2},
        )

    def test_rejects_invalid_node_token(self) -> None:
        control_host, control_port = self.control_server.server_address

        with self.assertRaises(AgentError):
            send_heartbeat(
                f"http://{control_host}:{control_port}",
                "tg_node_invalid",
                ["qwen/qwen3-32b"],
                "test-provider",
            )


if __name__ == "__main__":
    unittest.main()
