# Koinon Grid Roadmap

## Current goal

Deliver M1: a consent-first, authenticated, single-node inference path from an
OpenAI-compatible client through the Koinon gateway and outbound iroh P2P to a
manually enabled Ollama model.

## Completed

- Imported the TTinker Grid marketplace prototype into
  `services/marketplace`.
- Preserved the boundary that Grid does not execute or route inference.
- Validated `qwen3:8b` locally on an RTX 3080 without exposing Ollama publicly.
- Disabled the experimental Windows background tasks after validation.
- Approved the Koinon/Grid/Psyche system vision and layered architecture.
- Completed M0 documentation: vision, current-state claims, boundaries, and
  milestone definitions.

## In progress

- Converting the approved M1 design into a test-driven implementation plan.
- Defining persistent node identity, gateway allowlisting, API authentication,
  and provider-neutral execution boundaries.

## Next steps

1. Add a tested asynchronous inference backend boundary.
2. Add an Ollama backend with bounded input, output, and timeouts.
3. Persist the node's iroh identity with user-only storage permissions.
4. Require an explicit gateway node allowlist.
5. Add gateway API authentication and exact model selection.
6. Run an automated local P2P integration test without a GPU.
7. Run one manual Windows RTX 3080 acceptance request, then return the node to
   `Disabled`.
8. Replace Grid's compatibility heartbeat status with a read-only gateway
   status adapter.

## Milestones

### M0 — Shared vision and truthful claims

Status: completed.

The root documentation defines Koinon, Grid, and Psyche; distinguishes current
capabilities from simulations; and records the execution and economic-layer
boundary.

### M1 — Safe single-node inference

Status: next implementation milestone.

The gateway authenticates clients, selects an explicitly allowed node with the
requested model, sends a bounded P2P request, and returns an OpenAI-compatible
response. The node is manually enabled, uses an Ollama or vLLM backend, admits
one request at a time by default, and releases resources according to the
user-selected policy.

### M2 — Authoritative Grid control plane

Status: planned.

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

There is no blocker for local M1 development. Public node enrollment and
production economic claims remain blocked until persistent signed node
identity, a visible node client, authoritative metering, and the Grid status
adapter are complete.

## Related files

- [`../README.md`](../README.md)
- [`vision.md`](vision.md)
- [`superpowers/specs/2026-07-30-koinon-grid-vision-design.md`](superpowers/specs/2026-07-30-koinon-grid-vision-design.md)
- [`../architectures/inference-only/inference-node`](../architectures/inference-only/inference-node)
- [`../shared/inference`](../shared/inference)
- [`../services/marketplace`](../services/marketplace)
