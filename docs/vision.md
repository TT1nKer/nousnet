# Koinon Grid Vision

Koinon Grid is a coordination system for distributed compute. It turns
heterogeneous hardware operated by mutually untrusted parties into resources
that can be discovered, invoked, measured, and traded while preserving the
operator's control over the machine.

## System thesis

A compute market without reliable execution is a simulation. A distributed
runtime without a way to coordinate supply and demand remains an infrastructure
experiment. Koinon and Grid therefore form one system with separate layers:

- **Koinon execution plane:** node identity, discovery, routing, P2P transport,
  workload protocols, deadlines, cancellation, and resource admission.
- **Koinon control plane:** model catalog, health, authorization, policy, and
  the authoritative view of executable capacity.
- **Grid economic plane:** measurement, quotes, reputation, metering, disputes,
  and settlement.
- **Product experience:** explicit node controls, API access, usage, earnings,
  and understandable trust and privacy information.
- **Workload backends:** Ollama and vLLM for inference, with Psyche training
  workloads joining only after inference execution and measurement are proven.

Grid may depend on Koinon's execution facts. Koinon must not depend on Grid's
market rules, and the marketplace service must never become a second inference
router.

## Product principles

### User sovereignty

Installing software does not grant permission to use a GPU. Nodes are disabled
by default, run visibly, establish outbound connections only, and can stop
accepting work immediately. Background startup, warm models, concurrency, and
resource retention always require explicit opt-in.

### Execution before economics

Real routing, bounded failures, cancellation, and reproducible measurement come
before prices or payments. Quotes remain labeled as simulations until the
underlying execution and metering paths are authoritative.

### One execution truth

The Koinon gateway owns executable node state. Grid reads that state through a
defined adapter instead of maintaining an independent routing registry.
Compatibility heartbeat data may coexist during migration but cannot be used to
claim that inference is available.

### Explicit trust

Network reachability is not authorization. Gateways authenticate API clients,
and enrolled node identities are bound to persistent P2P endpoint identities.
Public gossip alone never makes a node eligible for requests.

### Minimal data

Prompts and model responses are not persisted by default. Operational telemetry
is limited to request identifiers, model identifiers, token counts, timings,
state transitions, and normalized error categories. Credentials are never
accepted on command lines or written to logs.

### Stable layers, replaceable implementations

Wire contracts and lifecycle states remain small and stable. Ollama, vLLM, and
future backends implement a shared execution boundary without leaking
provider-specific behavior into the gateway or Grid.

## Current reality

The repository already contains:

- an iroh-based inference network and direct request protocol;
- an OpenAI-compatible gateway endpoint;
- a vLLM-backed inference node;
- stale-node cleanup and bounded P2P reads;
- a Grid marketplace prototype with applications, node credentials, heartbeat
  discovery, and an explicitly unavailable inference status;
- a locally validated Ollama `qwen3:8b` model on an RTX 3080.

The repository does not yet contain:

- authenticated public gateway access;
- persistent, allowlisted P2P node identities;
- an Ollama execution backend in the Rust node;
- exact model routing and production error contracts;
- an end-to-end Grid-to-Koinon source of truth;
- authoritative metering, pricing, reputation, or settlement.

The [roadmap](roadmap.md) orders work so that documentation and product claims
never run ahead of these capabilities.
