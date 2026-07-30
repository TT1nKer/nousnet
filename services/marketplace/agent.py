#!/usr/bin/env python3
"""TTinker Grid node agent for OpenAI-compatible local inference servers."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


LOGGER = logging.getLogger("ttinker-grid-agent")

AGENT_VERSION = "0.1.0"
DEFAULT_CONTROL_URL = "https://ttinker.net/grid/api"
DEFAULT_PROVIDER_URL = "http://127.0.0.1:8000/v1"
DEFAULT_HEARTBEAT_SECONDS = 60
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_MODELS_PER_NODE = 64
NODE_TOKEN_ENVIRONMENT_VARIABLE = "TTINKER_NODE_TOKEN"


class AgentError(Exception):
    pass


def request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    token: str | None = None,
    timeout: float = 10,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(url, data=body, method=method, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            response_body = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as error:
        with error:
            response_body = error.read(MAX_RESPONSE_BYTES + 1)
        try:
            message = json.loads(response_body).get("error")
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
            message = None
        raise AgentError(message or f"HTTP {error.code} from {url}") from error
    except URLError as error:
        raise AgentError(f"Unable to reach {url}: {error.reason}") from error

    if len(response_body) > MAX_RESPONSE_BYTES:
        raise AgentError(f"Response from {url} exceeds size limit")
    try:
        payload_data = json.loads(response_body)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise AgentError(f"Invalid JSON response from {url}") from error
    if not isinstance(payload_data, dict):
        raise AgentError(f"JSON response from {url} must be an object")
    return payload_data


def discover_models(provider_url: str, timeout: float = 10) -> list[str]:
    response = request_json(f"{provider_url.rstrip('/')}/models", timeout=timeout)
    models = response.get("data")
    if not isinstance(models, list):
        raise AgentError("Provider /models response does not contain a data array")

    model_ids: list[str] = []
    seen_models: set[str] = set()
    for model in models:
        if not isinstance(model, dict):
            continue
        model_id = model.get("id")
        if isinstance(model_id, str) and model_id and model_id not in seen_models:
            if len(model_ids) >= MAX_MODELS_PER_NODE:
                raise AgentError(
                    f"Provider advertises more than {MAX_MODELS_PER_NODE} models"
                )
            seen_models.add(model_id)
            model_ids.append(model_id)
    return model_ids


def send_heartbeat(
    control_url: str,
    token: str,
    models: list[str],
    provider: str,
    timeout: float = 10,
) -> dict[str, Any]:
    return request_json(
        f"{control_url.rstrip('/')}/agent/v1/heartbeat",
        method="POST",
        token=token,
        timeout=timeout,
        payload={
            "agent_version": AGENT_VERSION,
            "provider": provider,
            "models": models,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a TTinker Grid node heartbeat agent")
    parser.add_argument("--control-url", default=DEFAULT_CONTROL_URL)
    parser.add_argument("--provider-url", default=DEFAULT_PROVIDER_URL)
    parser.add_argument("--provider-name", default="openai-compatible")
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_HEARTBEAT_SECONDS,
        help="Heartbeat interval in seconds (minimum 30)",
    )
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def run_once(args: argparse.Namespace, token: str) -> None:
    models = discover_models(args.provider_url, timeout=args.timeout)
    result = send_heartbeat(
        args.control_url,
        token,
        models,
        args.provider_name,
        timeout=args.timeout,
    )
    node = result.get("node", {})
    LOGGER.info(
        "Heartbeat accepted for node %s; advertised models: %d",
        node.get("id", "unknown"),
        len(models),
    )


def main() -> int:
    args = parse_args()
    if args.interval < 30:
        raise SystemExit("--interval must be at least 30 seconds")

    token = os.environ.get(NODE_TOKEN_ENVIRONMENT_VARIABLE)
    if not token:
        raise SystemExit(
            f"Set {NODE_TOKEN_ENVIRONMENT_VARIABLE}; do not pass node tokens on the command line"
        )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.once:
        try:
            run_once(args, token)
        except AgentError as error:
            LOGGER.error("%s", error)
            return 1
        return 0

    while True:
        try:
            run_once(args, token)
        except AgentError as error:
            LOGGER.warning("%s", error)
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
