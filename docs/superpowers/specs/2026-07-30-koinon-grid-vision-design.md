# Koinon Grid Vision and M1 Architecture Design

Status: approved on 2026-07-30.

## Purpose

Unify Koinon, TTinker Grid, and Psyche as layers of one distributed compute
system, then deliver the first product proof: a safe inference request from an
OpenAI-compatible gateway through outbound P2P to a user-controlled Ollama
node.

## Problem

The repository contains useful but disconnected pieces:

- Psyche is documented primarily as a distributed training system.
- The inference architecture already contains an OpenAI-compatible gateway,
  iroh discovery, direct P2P requests, and a vLLM node.
- Grid presents a compute market but currently has only application intake,
  one-time node credentials, heartbeat discovery, and simulated market data.
- The experimental Windows setup proved local Ollama inference but relied on
  SSH, PowerShell, and scheduled tasks that are unsuitable for a user product.

Treating Grid as an unrelated website would duplicate control state. Treating
the marketplace as the inference router would duplicate execution. Treating
installation as permission to use hardware would violate the product's central
trust boundary.

## Product model

Koinon and Grid form one system with separate responsibilities:

1. Koinon execution handles P2P identity, routing, inference, deadlines,
   cancellation, and resource admission.
2. Koinon control state describes which enrolled nodes and models are actually
   executable.
3. Grid uses that execution truth to coordinate supply, demand, measurement,
   reputation, pricing, and settlement.
4. Psyche remains the distributed training workload family and networking
   foundation. Existing crate and protocol names remain compatible during M1.

The marketplace may read Koinon state but may not execute requests. Koinon may
operate without Grid's economic rules.

## Alternatives considered

### Shared backend boundary

Retain the existing gateway, iroh network, and inference protocol. Replace the
node's hard dependency on vLLM with a tested asynchronous backend boundary and
implement vLLM and Ollama backends.

This is the selected approach because it produces one node lifecycle and one
request path.

### Separate Ollama P2P bridge

Leave the vLLM node unchanged and create another P2P binary for Ollama. This
reduces the first edit but duplicates lifecycle, configuration, protocol
handling, and future security fixes.

### Marketplace relay

Extend the Python heartbeat Agent to accept and forward inference. This is
rejected because it creates a second router, exposes a compatibility component
to untrusted request data, and perpetuates the background-script installation
model.

## Architecture

```text
OpenAI-compatible client
        |
        | HTTPS + Bearer API key
        v
Koinon Gateway
  - authenticate and validate
  - exact model selection
  - allowed node identity
  - deadline and cancellation
        |
        | iroh direct P2P
        v
Koinon Node Runtime
  - explicit enabled state
  - concurrency admission
  - lifecycle and resource policy
        |
        +-- VllmBackend
        |
        +-- OllamaBackend --> localhost-only Ollama

Grid reads gateway status; it is not in the request path.
```

The dependency direction follows the repository's local-context contract:

- **L0:** inference wire types, lifecycle states, normalized failures, and
  validation constants;
- **L1:** `InferenceBackend`, Ollama transport, vLLM adapter, API-key validator,
  and exact-model node selector;
- **L2:** node request admission and gateway request flow;
- **L3:** CLI configuration, Axum HTTP conversion, iroh wiring, and the future
  Grid status adapter.

M1 extracts only the authentication and node-selection responsibilities needed
to test the current 643-line gateway. It does not rewrite the full event loop.

## Node lifecycle

The externally visible states are:

```text
Disabled -> Starting -> Ready -> Busy -> Ready
                |         |
                v         v
              Error     Draining -> Disabled
```

- Installation leaves the node `Disabled`.
- M1 starts the node only through an explicit foreground command.
- The node advertises a model only after the configured backend passes a health
  check.
- Consumer nodes admit one request at a time by default.
- Pause removes availability before rejecting new work.
- Graceful pause drains the current request up to a fixed deadline; immediate
  pause cancels it.
- Node shutdown broadcasts unavailability and releases backend resources.
- Hidden windows, scheduled tasks, startup registration, silent elevation, and
  antivirus exclusions are outside M1 and prohibited by the product contract.

The existing inference binaries generate a new iroh key on every run. M1 must
instead load a persistent identity created by an explicit initialization
command. The private key is stored in a user-only file, never passed on the
command line, and the node fails closed if required file permissions cannot be
verified. Grid enrollment binds the resulting Endpoint ID in M2.

## Resource policy

M1 defines a conservative default:

- concurrency: one;
- provider endpoint: localhost only;
- no provider startup by the Koinon node;
- no warm-model promise;
- bounded request and response sizes;
- explicit request and shutdown deadlines.

For Ollama, safe mode releases the loaded model after a request. Retaining a
warm model is a later explicit user option. Provider-specific resource controls
remain inside `OllamaBackend`; they do not enter the wire protocol.

## Request flow

1. The gateway authenticates the client API key.
2. HTTP input is validated before conversion to the internal request type.
3. The selector finds a non-stale, allowed, ready node that advertises the exact
   requested model.
4. The gateway creates a request identifier and deadline.
5. The P2P layer sends the bounded request to the selected Endpoint ID.
6. The node acquires its single concurrency permit and invokes the backend.
7. The backend returns generated text or a normalized failure.
8. The gateway removes pending state on success, error, timeout, or client
   cancellation and converts the result to an OpenAI-compatible response.

Streaming is rejected during M1 instead of being silently treated as a
non-streaming request.

## Authentication and trust

- Gateway HTTP requires a Bearer API key loaded from environment or a
  permission-restricted file. M1 implements the environment source.
  Comparison is constant-time and neither the key nor its prefix is logged.
- M1 gateway configuration contains an explicit allowlist of persistent iroh
  Endpoint IDs. An empty allowlist cannot serve inference.
- M1 nodes accept inference connections only from the configured gateway
  Endpoint IDs. An empty gateway peer list cannot start an inference node.
- Gossip discovery supplies reachability and health only. It never grants
  authorization.
- Grid's existing `tg_node_` heartbeat credential remains a compatibility
  credential and does not authorize P2P inference.
- M2 replaces the static allowlist with signed Grid enrollment bound to the
  persistent Endpoint ID.
- Prompts and generated content are not stored or emitted to operational logs.

## Error contract

The external gateway maps normalized failures as follows:

- `400 Bad Request`: malformed input, a missing or invalid model identifier,
  unsupported streaming, or invalid bounds;
- `401 Unauthorized`: missing or invalid API key;
- `503 Service Unavailable`: no allowed ready node for the requested model;
- `502 Bad Gateway`: backend rejection, malformed backend response, or P2P
  execution failure;
- `504 Gateway Timeout`: deadline exceeded.

Every terminal path removes pending request state and releases the node's
concurrency permit. Client disconnect propagates cancellation when the P2P
protocol can observe it; otherwise the server deadline bounds the orphaned
work. Full cancellation messages and streaming are completed in M5.

## M1 scope

M1 includes:

- asynchronous backend contract;
- vLLM adaptation without removing current behavior;
- Ollama health, model discovery, chat, timeout, malformed-response, and
  resource-release behavior;
- persistent node identity;
- gateway API-key authentication;
- explicit Endpoint ID allowlist;
- exact model routing;
- bounded non-streaming requests and explicit errors;
- automated local tests and one manual RTX 3080 acceptance test.

M1 excludes:

- marketplace payments or real quotes;
- automated provider installation or startup;
- background node execution;
- GUI or signed installers;
- streaming;
- retries across nodes;
- public enrollment;
- training jobs.

## Verification

Automated tests require no GPU:

- backend contract success, timeout, unreachable provider, oversized body, and
  malformed response;
- Ollama request and response mapping against a local fake HTTP server;
- lifecycle transitions and concurrency admission;
- persistent identity creation, reload, and permission failure;
- API-key rejection without leaking secrets;
- exact model selection, stale nodes, unauthorized nodes, and no capacity;
- pending-state cleanup for success, error, timeout, and disconnect;
- local gateway-to-node P2P completion.

Manual M1 acceptance uses the existing Windows RTX 3080 and `qwen3:8b`:

1. verify both experimental scheduled tasks remain disabled;
2. start Ollama and the Koinon node visibly;
3. send one authenticated OpenAI-compatible request through the gateway;
4. confirm the provider still listens only on localhost;
5. stop the node;
6. confirm availability disappears and Ollama/model GPU processes are gone.

## Delivery order

1. Commit the shared vision, truthful current state, and roadmap.
2. Write a task-level M1 implementation plan with red-green TDD steps.
3. Implement backend and identity foundations.
4. Harden gateway authentication and routing.
5. Add automated P2P integration coverage.
6. Perform the manual Windows acceptance test.
7. Begin M2 only after the M1 acceptance criteria pass.

This design supersedes the earlier assumption that Grid is merely a Koinon
website. Grid is the economic layer of the same system, while Koinon remains
the authoritative execution core.
