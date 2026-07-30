# TTinker Grid marketplace service

This directory owns the public TTinker Grid product surface and its
compatibility control plane. Grid and Koinon are layers of the same distributed
compute system: Koinon executes work, while Grid coordinates supply, demand,
measurement, reputation, pricing, and settlement.

Static prototype for the open inference market concept.

- No build step or external runtime dependencies.
- All market quotes and routing metrics are explicitly simulated.
- The early-access form submits to a same-origin Python API and persists validated,
  idempotent applications in a private SQLite database.
- The status endpoint explicitly reports that intake is operational while inference
  remains unavailable.
- `manage.py create-node` issues one-time node credentials whose SHA-256 hashes are
  stored in SQLite.
- `agent.py` discovers models from a local OpenAI-compatible `/models` endpoint and
  submits authenticated heartbeats without exposing the provider to the public internet.
- Production deployment uses timestamped release directories with atomic
  `/opt/ttinker-grid/current` (API) and `/var/www/ttinker/grid/current`
  (static site) symlink switches.

## Boundary with the Koinon runtime

The marketplace service owns application intake, operator-issued node
credentials, public service status, and the static website. It does not execute
or route inference requests.

Koinon's inference runtime remains in:

- `architectures/inference-only/inference-node` for node discovery and the
  OpenAI-compatible gateway;
- `shared/inference` for inference protocol types and execution.

`agent.py` and the HTTP node registry are a compatibility path for the currently
deployed prototype. New execution features should be added to the Koinon
inference runtime instead of creating a second router here. The eventual
integration boundary is a read-only marketplace status adapter backed by the
Koinon gateway.

See the project [vision](../../docs/vision.md) and
[roadmap](../../docs/roadmap.md) for the approved system boundaries and current
delivery order.
