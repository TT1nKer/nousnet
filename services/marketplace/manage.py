#!/usr/bin/env python3
"""Operator-only management commands for TTinker Grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from server import NodeRegistry


DEFAULT_DATABASE = Path("/var/lib/ttinker-grid/applications.sqlite3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage TTinker Grid")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_node = subparsers.add_parser(
        "create-node",
        help="Create a node identity and print its token once",
    )
    create_node.add_argument("--name", required=True)
    subparsers.add_parser("list-nodes", help="List nodes without secret tokens")
    revoke_node = subparsers.add_parser("revoke-node", help="Revoke a node credential")
    revoke_node.add_argument("--id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = NodeRegistry(args.database)

    if args.command == "create-node":
        node_id, token = registry.create_node(args.name)
        print(f"node_id={node_id}")
        print(f"token={token}")
        print("Save this token now; only its SHA-256 hash is stored.")
        return

    if args.command == "revoke-node":
        if not registry.revoke_node(args.id):
            raise SystemExit("Node does not exist or is already revoked")
        print(f"revoked_node_id={args.id}")
        return

    print(json.dumps(registry.list_nodes(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
