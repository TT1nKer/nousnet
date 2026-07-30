# Safe Ollama Inference M1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver an authenticated OpenAI-compatible request through the Koinon gateway and outbound iroh P2P to a manually enabled, localhost-only Ollama node.

**Architecture:** `psyche-inference` gains a provider-neutral asynchronous backend contract and a lifecycle runtime. The existing vLLM engine and a new native Ollama HTTP adapter implement that contract. The gateway authenticates API clients, enforces a non-empty Endpoint ID allowlist, selects an exact model match, and maps bounded P2P outcomes to explicit HTTP errors.

**Tech Stack:** Rust 1.97, Tokio, async-trait, reqwest 0.12, Axum 0.7, iroh 0.97, postcard, Clap, Windows DPAPI, existing Python/vLLM bridge.

## Global Constraints

- Installation leaves a node disabled; M1 runs only through an explicit foreground command.
- Do not create scheduled tasks, hidden windows, startup registration, silent elevation, inbound provider ports, or antivirus exclusions.
- The Ollama provider URL must resolve to a loopback host.
- Consumer-node concurrency defaults to exactly one.
- Gateway HTTP requires a Bearer API key sourced from an environment variable or permission-restricted file.
- Gateway inference requires a non-empty allowlist of persistent iroh Endpoint IDs.
- Nodes accept inference only from the non-empty configured gateway Endpoint ID list.
- Gossip provides reachability only; it never authorizes a node.
- Model routing is an exact string match.
- M1 rejects `stream: true` with `400 Bad Request`.
- Prompts, generated text, API keys, node private keys, and token prefixes must not be logged.
- Automated tests require no Python package, model download, or GPU.
- Existing default vLLM builds remain supported; Ollama-only builds must not link Python.
- Grid remains outside the execution path.
- Do not modify or stage the existing untracked `koinon-dashboard/` or `research/` directories.

## Baseline and workspace setup

The current Cargo baseline cannot fetch the workspace's pinned `hf-hub` Git
dependency through the user's stale global Git proxy, and a direct HTTP/2 fetch
also failed. Do not change global Git configuration. Before the first Cargo
test, fetch dependencies with process-local configuration:

```bash
GIT_CONFIG_GLOBAL=/dev/null \
GIT_CONFIG_COUNT=1 \
GIT_CONFIG_KEY_0=http.version \
GIT_CONFIG_VALUE_0=HTTP/1.1 \
CARGO_NET_GIT_FETCH_WITH_CLI=true \
cargo fetch --locked
```

If that command still fails, record the exact fetch error and continue only
with checks whose dependencies are already available. Never report Rust tests
as passing until the full commands below complete successfully.

Implement in an isolated worktree created with
`superpowers:using-git-worktrees`. Keep each task's commit separate.

## File responsibility map

- `shared/inference/src/backend.rs`: provider-neutral backend contract and normalized backend failures.
- `shared/inference/src/runtime.rs`: node lifecycle, backend replacement, concurrency admission, and draining.
- `shared/inference/src/backends/ollama.rs`: localhost Ollama native API adapter and bounded response reader.
- `shared/inference/src/backends/vllm.rs`: compatibility adapter around the existing Python-backed `InferenceNode`.
- `shared/inference/src/protocol_handler.rs`: P2P conversion into the runtime; no provider logic.
- `architectures/inference-only/inference-node/src/identity.rs`: persistent iroh identity storage and platform protection.
- `architectures/inference-only/inference-node/src/node_cli.rs`: testable CLI and backend configuration conversion.
- `architectures/inference-only/inference-node/src/gateway/auth.rs`: Bearer authentication without logging secrets.
- `architectures/inference-only/inference-node/src/gateway/routing.rs`: allowed-node catalog and exact-model selection.
- `architectures/inference-only/inference-node/src/gateway/http.rs`: OpenAI HTTP validation, response conversion, and error mapping.
- `architectures/inference-only/inference-node/src/bin/gateway-node.rs`: iroh event loop and process wiring only.
- `architectures/inference-only/inference-node/src/main.rs`: visible node process wiring only.

---

### Task 1: Make the inference core build without Python

**Files:**
- Modify: `shared/inference/Cargo.toml`
- Modify: `shared/inference/src/lib.rs`
- Modify: `shared/inference/tests/test_inference_node.rs`
- Modify: `shared/inference/tests/test_vllm_integration.rs`

**Interfaces:**
- Produces: Cargo features `vllm`, `ollama`, and `vllm-tests`.
- Preserves: default builds export `InferenceNode` and `vllm`.
- Guarantees: `--no-default-features` does not include `pyo3`.

- [ ] **Step 1: Run the dependency-isolation check and verify it fails**

Run:

```bash
if cargo tree -p psyche-inference --no-default-features -e normal | rg -q 'pyo3 v'; then
  exit 1
fi
```

Expected: exit 1 because `pyo3` is currently unconditional.

- [ ] **Step 2: Feature-gate the Python implementation**

Change `shared/inference/Cargo.toml` to contain:

```toml
[features]
default = ["vllm"]
vllm = ["dep:pyo3"]
ollama = []
vllm-tests = ["vllm"]

[dependencies]
pyo3 = { workspace = true, optional = true }
```

Keep the existing non-Python dependencies unchanged.

Gate the modules and exports in `shared/inference/src/lib.rs`:

```rust
#[cfg(feature = "vllm")]
pub mod node;
pub mod protocol;
pub mod protocol_handler;
#[cfg(feature = "vllm")]
pub mod vllm;

#[cfg(feature = "vllm")]
pub use node::InferenceNode;
```

- [ ] **Step 3: Repair the feature-gated integration fixtures**

In `test_inference_node.rs`, replace every stale `prompt` request with:

```rust
messages: vec![psyche_inference::ChatMessage {
    role: "user".to_string(),
    content: "Once upon a time".to_string(),
}],
```

Update `test_vllm_integration.rs` calls to `run_inference` to pass a
`Vec<ChatMessage>` instead of a string. These tests remain feature-gated but
must type-check when enabled.

- [ ] **Step 4: Verify feature isolation and default compatibility**

Run:

```bash
if cargo tree -p psyche-inference --no-default-features -e normal | rg -q 'pyo3 v'; then
  exit 1
fi
cargo check -p psyche-inference --no-default-features
cargo test -p psyche-inference --lib
cargo check -p psyche-inference --features vllm-tests --tests
```

Expected: all commands exit 0. The last command may link Python but must not
load a model or require a GPU.

- [ ] **Step 5: Commit**

```bash
git add shared/inference
git commit -m "Decouple inference protocol from Python runtime"
```

---

### Task 2: Add the backend contract and lifecycle runtime

**Files:**
- Create: `shared/inference/src/backend.rs`
- Create: `shared/inference/src/runtime.rs`
- Modify: `shared/inference/src/lib.rs`

**Interfaces:**
- Produces: `InferenceBackend`, `BackendError`, `BackendErrorKind`.
- Produces: `InferenceRuntime`, `NodeLifecycleState`, `RuntimeError`.
- Consumes: existing `InferenceRequest` and `InferenceResponse`.
- Invariant: a runtime admits at most one request by default and never queues a second request silently.

- [ ] **Step 1: Write failing runtime tests**

Add tests at the bottom of `runtime.rs` using a fake backend:

```rust
#[derive(Debug)]
struct BlockingBackend {
    release: Arc<Notify>,
}

#[async_trait]
impl InferenceBackend for BlockingBackend {
    fn model_name(&self) -> &str {
        "qwen3:8b"
    }

    async fn health(&self) -> Result<(), BackendError> {
        Ok(())
    }

    async fn infer(
        &self,
        request: &InferenceRequest,
    ) -> Result<InferenceResponse, BackendError> {
        self.release.notified().await;
        Ok(InferenceResponse {
            request_id: request.request_id.clone(),
            generated_text: "ready".to_string(),
            full_text: "ready".to_string(),
            finish_reason: Some("stop".to_string()),
        })
    }
}

#[tokio::test]
async fn disabled_runtime_rejects_requests() {
    let runtime = InferenceRuntime::new(1);
    let error = runtime.execute(test_request("disabled")).await.unwrap_err();
    assert_eq!(error, RuntimeError::NotReady);
}

#[tokio::test]
async fn second_request_is_rejected_while_busy() {
    let release = Arc::new(Notify::new());
    let runtime = Arc::new(InferenceRuntime::new(1));
    runtime
        .start(Arc::new(BlockingBackend {
            release: release.clone(),
        }))
        .await
        .unwrap();

    let first_runtime = runtime.clone();
    let first = tokio::spawn(async move {
        first_runtime.execute(test_request("first")).await
    });
    tokio::task::yield_now().await;

    assert_eq!(
        runtime.execute(test_request("second")).await.unwrap_err(),
        RuntimeError::Busy
    );
    release.notify_one();
    first.await.unwrap().unwrap();
}
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
cargo test -p psyche-inference runtime::tests --no-default-features
```

Expected: compile failure because `InferenceRuntime` and the backend types do
not exist.

- [ ] **Step 3: Add the backend contract**

Create `backend.rs` with this public contract:

```rust
use async_trait::async_trait;
use crate::{InferenceRequest, InferenceResponse};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendErrorKind {
    Unavailable,
    Timeout,
    InvalidResponse,
    Execution,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{kind:?}: {message}")]
pub struct BackendError {
    pub kind: BackendErrorKind,
    pub message: String,
}

#[async_trait]
pub trait InferenceBackend: std::fmt::Debug + Send + Sync {
    fn model_name(&self) -> &str;
    async fn health(&self) -> Result<(), BackendError>;
    async fn infer(
        &self,
        request: &InferenceRequest,
    ) -> Result<InferenceResponse, BackendError>;
    async fn shutdown(&self) -> Result<(), BackendError> {
        Ok(())
    }
}
```

Add `async-trait.workspace = true` and `thiserror.workspace = true` to the
crate dependencies.

- [ ] **Step 4: Implement lifecycle and admission**

Create `runtime.rs` with:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeLifecycleState {
    Disabled,
    Starting,
    Ready,
    Busy,
    Draining,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RuntimeError {
    #[error("node is not ready")]
    NotReady,
    #[error("node is busy")]
    Busy,
    #[error(transparent)]
    Backend(#[from] BackendError),
}

pub struct InferenceRuntime {
    backend: RwLock<Option<Arc<dyn InferenceBackend>>>,
    state: RwLock<NodeLifecycleState>,
    permits: Semaphore,
}
```

`start` must set `Starting`, call `health`, then set `Ready`; failures set
`Error`. `execute` must use `try_acquire`, set `Busy`, invoke the backend, and
restore `Ready` on every result. `drain` must set `Draining`, acquire all
permits within its caller-provided deadline, call `shutdown`, clear the
backend, and set `Disabled`.

- [ ] **Step 5: Export and verify**

Export the new public types from `lib.rs`, then run:

```bash
cargo test -p psyche-inference runtime::tests --no-default-features
cargo test -p psyche-inference --lib
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add shared/inference
git commit -m "Add inference backend lifecycle"
```

---

### Task 3: Adapt vLLM and the P2P protocol to the runtime

**Files:**
- Create: `shared/inference/src/backends/mod.rs`
- Create: `shared/inference/src/backends/vllm.rs`
- Modify: `shared/inference/src/protocol_handler.rs`
- Modify: `shared/inference/src/lib.rs`
- Modify: `architectures/inference-only/inference-node/src/main.rs`
- Modify: `architectures/inference-only/inference-node/Cargo.toml`

**Interfaces:**
- Produces: `VllmBackend::initialize(model, tensor_parallel, memory)`.
- Changes: `InferenceProtocol::new` consumes `Arc<InferenceRuntime>`.
- Preserves: default CLI still starts vLLM when invoked with the existing vLLM arguments.

- [ ] **Step 1: Write the failing adapter test**

Add this feature-gated test to `backends/vllm.rs`:

```rust
#[test]
fn adapter_exposes_configured_model_without_loading_python() {
    let backend = VllmBackend::uninitialized(
        "gpt2".to_string(),
        Some(1),
        Some(0.3),
    );
    assert_eq!(backend.model_name(), "gpt2");
}
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
cargo test -p psyche-inference backends::vllm::tests --features vllm
```

Expected: compile failure because `VllmBackend` does not exist.

- [ ] **Step 3: Add the vLLM adapter**

`VllmBackend` owns:

```rust
pub struct VllmBackend {
    model_name: String,
    node: Arc<std::sync::Mutex<InferenceNode>>,
}
```

Its constructor initializes the existing `InferenceNode`. `infer` clones an
`Arc<Mutex<InferenceNode>>` into `tokio::task::spawn_blocking`, calls the
existing synchronous `inference`, and maps join or engine errors to
`BackendErrorKind::Execution`. `shutdown` uses the same blocking boundary and
calls `InferenceNode::shutdown`. Do not block a Tokio worker thread or change
Python bridge semantics in this task.

- [ ] **Step 4: Move protocol execution to `InferenceRuntime`**

Change `InferenceProtocol` to contain:

```rust
runtime: Arc<InferenceRuntime>
```

`process_request` calls `runtime.execute(request).await`. A runtime failure
returns an error from the P2P handler; do not encode failures into
`finish_reason`.

- [ ] **Step 5: Wire the current node process**

In `main.rs`, create one `Arc<InferenceRuntime>`, initialize a `VllmBackend`
when `--model-name` is present, and pass the runtime to
`InferenceProtocol::new`. Existing gossip `LoadModel` handling replaces the
runtime backend rather than replacing `Option<InferenceNode>` directly.

Add crate features:

```toml
[features]
default = ["vllm"]
vllm = ["psyche-inference/vllm", "dep:pyo3"]

[dependencies]
psyche-inference = { workspace = true, default-features = false }
pyo3 = { workspace = true, optional = true }
```

Guard Python initialization with `#[cfg(feature = "vllm")]`.

- [ ] **Step 6: Verify default and protocol-only builds**

Run:

```bash
cargo test -p psyche-inference --lib
cargo check -p psyche-inference-node
cargo check -p psyche-inference-node --no-default-features
```

Expected: all commands pass; the no-default build contains protocol and CLI
infrastructure but no usable backend yet.

- [ ] **Step 7: Commit**

```bash
git add shared/inference architectures/inference-only/inference-node
git commit -m "Route P2P inference through backend runtime"
```

---

### Task 4: Implement the localhost-only Ollama backend

**Files:**
- Create: `shared/inference/src/backends/ollama.rs`
- Create: `shared/inference/tests/test_ollama_backend.rs`
- Modify: `shared/inference/src/backends/mod.rs`
- Modify: `shared/inference/Cargo.toml`

**Interfaces:**
- Produces: `OllamaBackend::new(base_url, model, timeout)`.
- Consumes: Ollama native `GET /api/tags` and `POST /api/chat`.
- Invariant: request JSON contains `stream: false`, `think: false`, and `keep_alive: 0`.
- Invariant: only `localhost`, `127.0.0.0/8`, or `[::1]` URLs are accepted.

- [ ] **Step 1: Write failing URL and mapping tests**

Use an Axum test server bound to `127.0.0.1:0`. Add tests:

```rust
#[test]
fn rejects_non_loopback_provider() {
    let error = OllamaBackend::new(
        "http://192.0.2.10:11434",
        "qwen3:8b",
        Duration::from_secs(10),
    )
    .unwrap_err();
    assert_eq!(error.kind, BackendErrorKind::Unavailable);
}

#[tokio::test]
async fn maps_chat_and_releases_model() {
    let captured = Arc::new(Mutex::new(None));
    let provider_url = spawn_fake_ollama(captured.clone()).await;
    let backend = OllamaBackend::new(
        &provider_url,
        "qwen3:8b",
        Duration::from_secs(2),
    )
    .unwrap();

    let response = backend.infer(&test_request()).await.unwrap();
    assert_eq!(response.generated_text, "READY");

    let body = captured.lock().await.clone().unwrap();
    assert_eq!(body["model"], "qwen3:8b");
    assert_eq!(body["stream"], false);
    assert_eq!(body["think"], false);
    assert_eq!(body["keep_alive"], 0);
}
```

Also add tests for timeout, missing model in `/api/tags`, malformed JSON, and a
response larger than `1 MiB`.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
cargo test -p psyche-inference --no-default-features --features ollama --test test_ollama_backend
```

Expected: compile failure because `OllamaBackend` does not exist.

- [ ] **Step 3: Add dependencies and request types**

Add:

```toml
ollama = ["dep:reqwest", "dep:futures-util"]
reqwest = { version = "0.12", features = ["json", "stream"], optional = true }
futures-util = { workspace = true, optional = true }

[dev-dependencies]
axum.workspace = true
```

Use private `Serialize` request structs and `Deserialize` response structs.
Never serialize internal credentials or log request bodies.

- [ ] **Step 4: Implement bounded health and inference**

`new` parses `reqwest::Url` and verifies `host()` is loopback. `health`
requests `/api/tags`, applies the configured timeout, reads at most `1 MiB`,
and requires an exact model name. `infer` posts `/api/chat` with native Ollama
options:

```rust
OllamaChatRequest {
    model: self.model_name.clone(),
    messages: request.messages.clone(),
    stream: false,
    think: false,
    keep_alive: 0,
    options: OllamaOptions {
        temperature: request.temperature,
        top_p: request.top_p,
        num_predict: request.max_tokens,
    },
}
```

Read the body incrementally from `bytes_stream`; return
`BackendErrorKind::InvalidResponse` immediately after the accumulated size
exceeds `1 MiB`.

- [ ] **Step 5: Verify backend behavior**

Run:

```bash
cargo test -p psyche-inference --no-default-features --features ollama --test test_ollama_backend
cargo test -p psyche-inference --no-default-features --features ollama --lib
```

Expected: all Ollama tests pass without Ollama or a GPU.

- [ ] **Step 6: Commit**

```bash
git add shared/inference
git commit -m "Add safe localhost Ollama backend"
```

---

### Task 5: Add persistent node identity and explicit Ollama CLI

**Files:**
- Create: `architectures/inference-only/inference-node/src/identity.rs`
- Create: `architectures/inference-only/inference-node/src/node_cli.rs`
- Modify: `architectures/inference-only/inference-node/src/lib.rs`
- Modify: `architectures/inference-only/inference-node/src/main.rs`
- Modify: `architectures/inference-only/inference-node/Cargo.toml`
- Modify: `Cargo.lock`

**Interfaces:**
- Produces: `create_identity(path) -> Result<EndpointId>`.
- Produces: `load_identity(path) -> Result<SecretKey>`.
- Produces: `NodeBackendConfig::{Vllm, Ollama}`.
- CLI: `init-identity --identity-file PATH`.
- CLI: `run --identity-file PATH --backend ollama --model-name MODEL --provider-url URL`.

- [ ] **Step 1: Write failing identity tests**

In `identity.rs` add:

```rust
#[test]
fn create_then_load_preserves_endpoint_id() {
    let directory = tempfile::tempdir().unwrap();
    let identity_file = directory.path().join("node.identity");
    let endpoint_id = create_identity(&identity_file).unwrap();
    assert_eq!(load_identity(&identity_file).unwrap().public(), endpoint_id);
}

#[test]
fn create_refuses_to_overwrite_existing_identity() {
    let directory = tempfile::tempdir().unwrap();
    let identity_file = directory.path().join("node.identity");
    create_identity(&identity_file).unwrap();
    assert!(create_identity(&identity_file).is_err());
}
```

On Unix, assert mode `0o600`. On Windows, decrypt the saved DPAPI blob under
the same user and assert the plaintext key bytes never appear in the file.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
cargo test -p psyche-inference-node --lib identity::tests
```

Expected: compile failure because `identity` does not exist.

- [ ] **Step 3: Implement platform-protected storage**

The common module generates an iroh `SecretKey` and delegates byte protection:

```rust
#[cfg(unix)]
fn protect_and_write(path: &Path, key_bytes: &[u8; 32]) -> Result<()>;

#[cfg(windows)]
fn protect_and_write(path: &Path, key_bytes: &[u8; 32]) -> Result<()>;
```

Unix creates a new file with `OpenOptionsExt::mode(0o600)` and rejects any
existing file. Windows uses `CryptProtectData` with
`CRYPTPROTECT_UI_FORBIDDEN`, writes only the DPAPI ciphertext, zeroizes the
plaintext buffer, and frees the returned allocation with `LocalFree`.
`load_identity` performs the inverse and requires exactly 32 plaintext bytes.

Add target-specific `windows-sys` features
`Win32_Security_Cryptography`, `Win32_System_Memory`, and
`Win32_Foundation`, plus:

```toml
zeroize = "1.8.2"

[target.'cfg(windows)'.dependencies]
windows-sys = { version = "0.61.2", features = [
  "Win32_Foundation",
  "Win32_Security_Cryptography",
  "Win32_System_Memory",
] }

[dev-dependencies]
tempfile = "3.25.0"
```

- [ ] **Step 4: Write failing CLI parsing tests**

In `node_cli.rs`:

```rust
#[test]
fn ollama_run_requires_identity_model_and_loopback_provider() {
    let cli = Cli::try_parse_from([
        "psyche-inference-node",
        "run",
        "--identity-file", "node.identity",
        "--backend", "ollama",
        "--model-name", "qwen3:8b",
        "--provider-url", "http://127.0.0.1:11434",
    ])
    .unwrap();
    assert!(matches!(cli.backend_config().unwrap(), NodeBackendConfig::Ollama { .. }));
}
```

Add rejection tests for a missing identity, missing model, and a non-loopback
provider.

- [ ] **Step 5: Implement CLI and node wiring**

Move Clap definitions from `main.rs` into `node_cli.rs`. Do not derive `Debug`
for any struct that can contain secrets. `init-identity` prints only the
Endpoint ID. The run path loads the key and passes `Some(secret_key)` to
`NetworkConnection::init_with_custom_protocol`.

Add the feature forwarding now that the shared Ollama backend exists:

```toml
ollama = ["psyche-inference/ollama"]
```

Create the node-side network allowlist from the configured bootstrap gateway
Endpoint IDs:

```rust
let allowed_gateways = bootstrap_peers.iter().map(|peer| peer.id);
let gateway_allowlist = allowlist::AllowDynamic::with_nodes(allowed_gateways);
```

Fail startup when the bootstrap peer list is empty. Pass `gateway_allowlist`
instead of `AllowAll`, so another discovered peer cannot invoke the node
directly.

For `BackendKind::Ollama`, construct `OllamaBackend` and never initialize
Python. For `BackendKind::Vllm`, preserve the current defaults and Python
initialization.

- [ ] **Step 6: Verify both feature builds**

Run:

```bash
cargo test -p psyche-inference-node --lib
cargo check -p psyche-inference-node
cargo check -p psyche-inference-node --no-default-features --features ollama
```

Expected: all commands pass. The Ollama-only dependency tree contains no
`pyo3`.

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml Cargo.lock architectures/inference-only/inference-node
git commit -m "Add persistent identity and Ollama node CLI"
```

---

### Task 6: Add gateway authentication and exact-model routing

**Files:**
- Create: `architectures/inference-only/inference-node/src/gateway/mod.rs`
- Create: `architectures/inference-only/inference-node/src/gateway/auth.rs`
- Create: `architectures/inference-only/inference-node/src/gateway/routing.rs`
- Modify: `architectures/inference-only/inference-node/src/lib.rs`
- Modify: `architectures/inference-only/inference-node/Cargo.toml`
- Modify: `Cargo.lock`

**Interfaces:**
- Produces: `ApiKeyAuthenticator::from_secret`.
- Produces: `ApiKeyAuthenticator::authorize(&HeaderValue)`.
- Produces: `NodeCatalog::{upsert, remove, remove_stale, select}`.
- Consumes: a non-empty `HashSet<EndpointId>` loaded from the gateway allowlist.

- [ ] **Step 1: Write failing authentication tests**

```rust
#[test]
fn accepts_only_the_exact_bearer_key() {
    let auth = ApiKeyAuthenticator::from_secret("gateway-secret").unwrap();
    assert!(auth.authorize(&HeaderValue::from_static("Bearer gateway-secret")));
    assert!(!auth.authorize(&HeaderValue::from_static("Bearer gateway-secreu")));
    assert!(!auth.authorize(&HeaderValue::from_static("gateway-secret")));
}

#[test]
fn rejects_empty_gateway_key() {
    assert!(ApiKeyAuthenticator::from_secret("  ").is_err());
}
```

- [ ] **Step 2: Write failing routing tests**

```rust
#[test]
fn selects_only_allowed_exact_model_matches() {
    let allowed = endpoint_id(1);
    let denied = endpoint_id(2);
    let mut catalog = NodeCatalog::new(HashSet::from([allowed]));
    catalog.upsert(node(denied, "qwen3:8b"));
    catalog.upsert(node(allowed, "llama3:8b"));
    assert_eq!(catalog.select("qwen3:8b", Instant::now()), None);

    catalog.upsert(node(allowed, "qwen3:8b"));
    assert_eq!(catalog.select("qwen3:8b", Instant::now()), Some(allowed));
}
```

Also test stale removal at 90 seconds and empty allowlist rejection.

- [ ] **Step 3: Run and verify RED**

Run:

```bash
cargo test -p psyche-inference-node --lib gateway::
```

Expected: compile failure because `gateway` does not exist.

- [ ] **Step 4: Implement constant-time authentication**

Store the API key in `zeroize::Zeroizing<Vec<u8>>`. Parse exactly one
case-insensitive `Bearer` scheme and compare equal-length key bytes using
`subtle::ConstantTimeEq`. Error messages must not include the key or prefix.
Add:

```toml
subtle = "2.6.1"
zeroize = "1.8.2"
```

- [ ] **Step 5: Implement the catalog**

`NodeCatalog` owns the allowed Endpoint IDs and node records. `select` filters
by allowlist, `last_seen <= 90s`, and exact model equality, then chooses the
least recently selected node for deterministic round-robin behavior. Do not
accept a requested model merely because some other model is loaded.

- [ ] **Step 6: Verify**

Run:

```bash
cargo test -p psyche-inference-node --lib gateway::
```

Expected: all gateway policy tests pass.

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml Cargo.lock architectures/inference-only/inference-node
git commit -m "Add gateway authentication and routing policy"
```

---

### Task 7: Harden the OpenAI HTTP request flow and cleanup

**Files:**
- Create: `architectures/inference-only/inference-node/src/gateway/http.rs`
- Modify: `architectures/inference-only/inference-node/src/gateway/mod.rs`
- Modify: `architectures/inference-only/inference-node/src/bin/gateway-node.rs`
- Modify: `architectures/inference-only/inference-node/Cargo.toml`

**Interfaces:**
- Produces: `build_gateway_router(state) -> axum::Router`.
- Produces: OpenAI-compatible success response.
- Produces: normalized `400`, `401`, `502`, `503`, and `504` JSON errors.
- Changes: pending channels carry `Result<InferenceResponse, GatewayFailure>`.

- [ ] **Step 1: Write failing HTTP tests**

Use `tower::ServiceExt::oneshot` against the router:

```rust
#[tokio::test]
async fn rejects_missing_api_key() {
    let response = test_router()
        .oneshot(chat_request(None, "qwen3:8b", false))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn rejects_streaming_explicitly() {
    let response = test_router()
        .oneshot(chat_request(Some("gateway-secret"), "qwen3:8b", true))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn returns_service_unavailable_without_exact_capacity() {
    let response = test_router()
        .oneshot(chat_request(Some("gateway-secret"), "missing", false))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
}
```

Add tests that inject success, backend failure, and timeout into the pending
channel and assert pending length returns to zero.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
cargo test -p psyche-inference-node --lib gateway::http::tests
```

Expected: compile failure because `gateway::http` does not exist.

- [ ] **Step 3: Move HTTP contracts and validation**

Move `ChatMessage`, `ChatCompletionRequest`, response types, `AppError`, and
the handler from `gateway-node.rs` into `gateway/http.rs`. Add explicit bounds:

```rust
const MAX_MESSAGES: usize = 128;
const MAX_MESSAGE_BYTES: usize = 256 * 1024;
const MAX_TOTAL_MESSAGE_BYTES: usize = 1024 * 1024;
const MAX_TOKENS: usize = 4096;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
```

Reject empty messages, unsupported roles, empty content, non-finite
temperature/top-p, out-of-range values, oversized content, and `stream: true`.

- [ ] **Step 4: Guarantee pending cleanup**

Use a request guard containing the request ID and state. Its `Drop`
implementation sends the request ID to a cleanup channel; the gateway event
loop owns that channel and removes pending entries asynchronously. Success,
explicit failure, timeout, and handler cancellation therefore converge on the
same cleanup path.

Change pending senders to:

```rust
oneshot::Sender<Result<InferenceResponse, GatewayFailure>>
```

P2P connection and backend failures send `GatewayFailure::NodeExecution`;
timeouts send `GatewayFailure::Timeout` instead of silently dropping the
sender.

- [ ] **Step 5: Wire gateway configuration**

`gateway-node` must require:

- `KOINON_GATEWAY_API_KEY`;
- `--allowed-peer-file PATH`, containing a JSON array of Endpoint ID strings.

Load the file before binding HTTP. Fail startup on an empty or malformed
allowlist. Initialize the network with `AllowDynamic::with_nodes` instead of
`AllowAll`.

Enable Tower's router test utility without changing its major version:

```toml
tower = { version = "0.4", features = ["util"] }
```

- [ ] **Step 6: Verify HTTP and default builds**

Run:

```bash
cargo test -p psyche-inference-node --lib gateway::
cargo check -p psyche-inference-node --bin gateway-node
```

Expected: tests and check pass.

- [ ] **Step 7: Commit**

```bash
git add architectures/inference-only/inference-node
git commit -m "Harden gateway request lifecycle"
```

---

### Task 8: Add a GPU-free local P2P integration test

**Files:**
- Create: `architectures/inference-only/inference-node/tests/p2p_ollama.rs`
- Create: `architectures/inference-only/inference-node/src/p2p_client.rs`
- Modify: `architectures/inference-only/inference-node/src/lib.rs`
- Modify: `architectures/inference-only/inference-node/src/bin/gateway-node.rs`

**Interfaces:**
- Produces: `send_inference_request(endpoint, peer_id, request, deadline)`.
- Verifies: fake Ollama HTTP → `OllamaBackend` → runtime → iroh P2P → response.

- [ ] **Step 1: Move the P2P client into the library**

Move the existing `send_inference_request` function from the gateway binary to
`p2p_client.rs`. Add an explicit deadline parameter and retain the `10 MiB`
response bound.

- [ ] **Step 2: Write the failing integration test**

The test must:

1. bind a fake Ollama Axum server to `127.0.0.1:0`;
2. create `OllamaBackend` and `InferenceRuntime`;
3. start an iroh endpoint with `InferenceProtocol` under local discovery and
   relays disabled, allowing only the client Endpoint ID;
4. start a client endpoint allowing only the node Endpoint ID;
5. call `send_inference_request`;
6. assert request ID and `"READY"` generated text;
7. cancel both endpoints and join their tasks.

Core assertion:

```rust
let response = send_inference_request(
    client_endpoint,
    node_endpoint_id,
    test_request("p2p-ollama"),
    Duration::from_secs(5),
)
.await
.unwrap();

assert_eq!(response.request_id, "p2p-ollama");
assert_eq!(response.generated_text, "READY");
```

- [ ] **Step 3: Run and verify RED**

Run:

```bash
cargo test -p psyche-inference-node --no-default-features --features ollama --test p2p_ollama
```

Expected: failure before wiring is complete.

- [ ] **Step 4: Complete minimal P2P wiring and cleanup**

Use ephemeral keys only inside this test. The production CLI continues to
require persistent identity. Ensure every spawned task is cancelled and joined
even when an assertion fails by storing cancellation tokens and using a test
cleanup guard.

- [ ] **Step 5: Verify M1 automated acceptance**

Run:

```bash
cargo test -p psyche-inference --no-default-features --features ollama
cargo test -p psyche-inference-node --no-default-features --features ollama
cargo test -p psyche-inference --lib
cargo test -p psyche-inference-node --lib
cargo check -p psyche-inference-node
cargo fmt --all -- --check
cargo clippy -p psyche-inference -p psyche-inference-node --all-targets --all-features -- -D warnings
```

Expected: all commands exit 0. GPU-marked tests may compile but must not skip a
failure by returning early; tests requiring real vLLM remain explicitly
feature-gated and are reported separately.

- [ ] **Step 6: Commit**

```bash
git add architectures/inference-only/inference-node
git commit -m "Test Ollama inference over local P2P"
```

---

### Task 9: Perform Windows acceptance and close M1

**Files:**
- Modify: `docs/roadmap.md`
- Modify: `docs/superpowers/specs/2026-07-30-koinon-grid-vision-design.md` only if actual behavior requires a documented correction

**Interfaces:**
- Validates: the production-shaped request path on the existing RTX 3080.
- Preserves: both experimental Windows scheduled tasks remain disabled.

- [ ] **Step 1: Build the Ollama-only node**

From a visible Windows developer shell with an already installed Rust
toolchain, run:

```bash
cargo build -p psyche-inference-node \
  --no-default-features \
  --features ollama \
  --release
```

Produce and record the SHA-256 of the binary. Do not copy a binary whose local
tests failed. If the Windows toolchain is absent, stop and record that as the
manual acceptance blocker; do not install a toolchain or execute an unsigned
download through remote automation.

- [ ] **Step 2: Verify Windows preconditions without changing state**

Read-only checks must confirm:

- `TTinker Grid Agent` is disabled;
- `TTinker Grid Ollama` is disabled;
- no Ollama or Koinon process is running;
- port `11434` is not publicly bound;
- the expected NVIDIA RTX 3080 is present.

- [ ] **Step 3: Initialize identity and configure allowlist**

Run the native `init-identity` command visibly on Windows. Capture only its
public Endpoint ID. Add that ID to the gateway allowlist file. Store the
gateway API key in `KOINON_GATEWAY_API_KEY`; do not put it on either command
line.

- [ ] **Step 4: Start the provider and node visibly**

Start Ollama and the node in foreground sessions. Do not enable or alter either
scheduled task. Verify the node advertises exactly `qwen3:8b`.

- [ ] **Step 5: Send one acceptance request**

Send an authenticated non-streaming `/v1/chat/completions` request through the
gateway. Record:

- HTTP status;
- model;
- cold-load seconds;
- total seconds;
- generated token count and tokens/second when available;
- gateway request ID;
- absence of prompt/response text from operational logs.

- [ ] **Step 6: Verify shutdown and failure behavior**

Stop the node and Ollama visibly. Confirm:

- no Ollama, llama-server, or Koinon node process remains;
- GPU inference processes are absent;
- both scheduled tasks remain disabled;
- a new gateway request returns `503` after the stale window;
- no inbound firewall rule or public listener was created.

- [ ] **Step 7: Update roadmap truthfully**

Mark M1 complete only if every automated and manual acceptance check passed.
Otherwise keep M1 in progress and record the concrete blocker under
`## Blockers`; do not soften failed criteria.

- [ ] **Step 8: Final verification**

Run:

```bash
git diff --check
git status --short
```

Re-run the full automated acceptance commands from Task 8 after any code
correction made during manual acceptance.

- [ ] **Step 9: Commit**

```bash
git add docs/roadmap.md docs/superpowers/specs/2026-07-30-koinon-grid-vision-design.md
git commit -m "Record M1 inference acceptance"
```

Do not add the spec file if it did not change.

## Completion criteria

M1 is complete only when:

- the Ollama-only node builds without Python;
- automated tests pass without a GPU;
- gateway API authentication and node allowlisting are mandatory;
- exact model routing and documented error statuses are tested;
- the Windows request succeeds over P2P with no public Ollama listener;
- stopping the node removes capacity and releases GPU processes;
- background tasks remain disabled;
- roadmap claims match observed behavior.
