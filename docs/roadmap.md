# Koinon Grid Roadmap

## Current goal

Deliver M2: make the Koinon gateway's executable node state the authoritative
source for Grid registration and availability, then retire the marketplace's
compatibility heartbeat view without turning Grid into a second inference
router.

## Completed

- Imported the TTinker Grid marketplace prototype into
  `services/marketplace`.
- Preserved the boundary that Grid does not execute or route inference.
- Validated `qwen3:8b` locally on an RTX 3080 without exposing Ollama publicly.
- Disabled the experimental Windows background tasks after validation.
- Approved the Koinon/Grid/Psyche system vision and layered architecture.
- Completed M0 documentation: vision, current-state claims, boundaries, and
  milestone definitions.
- Added a provider-neutral asynchronous inference runtime and a localhost-only
  Ollama backend while preserving the existing vLLM path.
- Added persistent node identity, mandatory gateway and node allowlists,
  gateway Bearer authentication, exact model routing, bounded requests, and
  explicit HTTP failure contracts.
- Added a GPU-free local P2P acceptance test and Windows CI coverage for both
  inference crates and release binaries.
- Completed M1 acceptance on Windows with `qwen3:8b` and an RTX 3080: one
  authenticated request traversed the gateway and iroh P2P path and returned
  HTTP 200 in 3.441 seconds including a 2.27-second cold load.
- Verified that Ollama remained bound to `127.0.0.1`, operational logs did not
  contain the acceptance prompt or response, the stale node became unavailable
  with HTTP 503, GPU inference processes exited, no firewall rule was added,
  and both experimental scheduled tasks remained disabled.

## In progress

- Mapping the Grid registration and heartbeat call path to the gateway's
  signed, model-specific node state.
- Defining the smallest read-only status adapter and compatibility migration
  that preserve the execution/economic-layer boundary.

## Next steps

1. Locate the Grid registration, credential, and compatibility heartbeat
   boundaries and the gateway's authoritative node-state interface.
2. Define a read-only adapter that reports only signed, allowlisted, non-stale
   capacity and never accepts execution commands.
3. Bind Grid registrations to persistent Koinon Endpoint IDs.
4. Add behavior-focused tests for unknown, unauthorized, stale, and
   model-mismatched nodes.
5. Migrate marketplace availability reads to the adapter while keeping the
   legacy heartbeat explicitly non-authoritative during a bounded transition.
6. Add revocable user API-key lifecycle without exposing credentials to Grid.

## Milestones

### M0 — Shared vision and truthful claims

Status: completed.

The root documentation defines Koinon, Grid, and Psyche; distinguishes current
capabilities from simulations; and records the execution and economic-layer
boundary.

### M1 — Safe single-node inference

Status: completed.

The gateway authenticates clients, selects an explicitly allowed node with the
requested model, sends a bounded P2P request, and returns an OpenAI-compatible
response. The node is manually enabled, uses an Ollama or vLLM backend, admits
one request at a time by default, and releases resources according to the
configured policy. GPU-free integration tests and a Windows RTX 3080 acceptance
run verified the production-shaped Ollama path and its shutdown behavior.

### M2 — Authoritative Grid control plane

Status: in progress.

Grid registration binds a node identity to its persistent Koinon endpoint.
Marketplace status is backed by gateway state, user API keys can be revoked,
and the compatibility heartbeat registry begins a documented retirement
period.

### M3 — Trusted node product

Status: planned.

A signed Windows-first client provides system credential storage, visible
state, resource controls, and one-click pause without hidden PowerShell,
silent elevation, scheduled tasks, or antivirus exclusions. macOS and Linux
follow the same lifecycle contract.

### M4 — Measured Grid market

Status: planned.

Reproducible benchmarks, request metering, quotes, routing policy, reputation,
and a test-balance ledger replace simulated market data. Real payment is
excluded until measurements and dispute handling are validated.

### M5 — Scale and additional workloads

Status: planned.

The runtime adds streaming, end-to-end cancellation, retries with idempotency,
multi-node scheduling, privacy tiers, and model caching. Psyche training jobs
can then enter Grid as another measured workload.

## Blockers

There is no blocker for local M2 development. Public node enrollment remains
blocked until registration is bound to persistent signed node identity and a
visible node client exists. Production economic claims remain blocked until
the Grid status adapter, authoritative metering, revocation, and dispute
handling are complete.

## Related files

- [`../README.md`](../README.md)
- [`vision.md`](vision.md)
- [`superpowers/specs/2026-07-30-koinon-grid-vision-design.md`](superpowers/specs/2026-07-30-koinon-grid-vision-design.md)
- [`../architectures/inference-only/inference-node`](../architectures/inference-only/inference-node)
- [`../shared/inference`](../shared/inference)
- [`../services/marketplace`](../services/marketplace)
