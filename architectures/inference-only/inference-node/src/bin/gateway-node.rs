//! Gateway node for inference requests
//!
//! Usage:
//! 1. Exposes HTTP API on localhost:8000
//! 2. Discovers inference nodes via gossip
//! 3. Routes requests to available inference nodes via gossip
//! 4. Returns responses to HTTP clients
//!
//!   cargo run --bin gateway-node -- --discovery-mode local

use anyhow::{Context, Result};
use axum::{
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use iroh::EndpointAddr;
use psyche_inference::{
    InferenceGossipMessage, InferenceMessage, InferenceRequest, InferenceResponse, INFERENCE_ALPN,
};
use psyche_inference_node::gateway::{
    auth::ApiKeyAuthenticator,
    http::{
        build_gateway_router, GatewayFailure, GatewayState, RoutedInferenceRequest, REQUEST_TIMEOUT,
    },
    routing::{load_endpoint_allowlist, NodeRecord},
};
use psyche_metrics::ClientMetrics;
use psyche_network::{
    allowlist, DiscoveryMode, EndpointId, NetworkConnection, NetworkEvent, RelayKind,
};
use std::{fs, path::PathBuf, sync::Arc, time::Duration};
use tokio::{sync::mpsc, time::sleep};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};
use zeroize::Zeroizing;

const P2P_REQUEST_TIMEOUT: Duration = Duration::from_secs(29);

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "127.0.0.1:8000")]
    listen_addr: String,

    /// what discovery to use - public n0 or local
    #[arg(long, env = "IROH_DISCOVERY", default_value = "n0")]
    discovery_mode: DiscoveryMode,

    /// what relays to use - public n0 or the private Psyche ones
    #[arg(long, env = "IROH_RELAY", default_value = "psyche")]
    relay_kind: RelayKind,

    #[arg(long)]
    bootstrap_peer_file: Option<PathBuf>,

    /// JSON array of inference node Endpoint IDs authorized by this gateway.
    #[arg(long)]
    allowed_peer_file: PathBuf,

    #[arg(long)]
    write_endpoint_file: Option<PathBuf>,
}

#[derive(Clone)]
struct GatewayAdminState {
    gossip_tx: mpsc::Sender<InferenceGossipMessage>,
    endpoint_addr: EndpointAddr,
}

#[derive(serde::Deserialize, Debug, Clone)]
#[serde(tag = "source_type", rename_all = "lowercase")]
enum LoadModelSource {
    #[serde(rename = "huggingface")]
    HuggingFace {
        source_path: Option<String>,
    },
    Local {
        source_path: String,
    },
}

#[derive(serde::Deserialize)]
struct LoadModelRequest {
    model_name: String,
    #[serde(flatten)]
    source: LoadModelSource,
}

#[axum::debug_handler]
async fn handle_load_model(
    State(state): State<Arc<GatewayAdminState>>,
    Json(req): Json<LoadModelRequest>,
) -> Result<String, AdminError> {
    use psyche_inference::ModelSource;

    info!(
        "Admin API: Received LoadModel request for model: {} (source: {:?})",
        req.model_name, req.source
    );

    let model_source = match req.source {
        LoadModelSource::HuggingFace { source_path } => {
            let path = source_path.unwrap_or_else(|| req.model_name.clone());
            ModelSource::HuggingFace(path)
        }
        LoadModelSource::Local { source_path } => ModelSource::Local(source_path),
    };

    let load_msg = InferenceGossipMessage::LoadModel {
        model_name: req.model_name.clone(),
        model_source,
    };

    state.gossip_tx.send(load_msg).await.map_err(|e| {
        error!("Failed to broadcast LoadModel message: {:#}", e);
        AdminError::DispatchUnavailable
    })?;

    info!(
        "Successfully broadcasted LoadModel message for: {}",
        req.model_name
    );
    Ok(format!(
        "LoadModel broadcast sent for model: {}",
        req.model_name
    ))
}

#[axum::debug_handler]
async fn handle_bootstrap(State(state): State<Arc<GatewayAdminState>>) -> Json<EndpointAddr> {
    info!(
        "Bootstrap request: returning endpoint addr {}",
        state.endpoint_addr.id.fmt_short()
    );
    Json(state.endpoint_addr.clone())
}

#[derive(Debug)]
enum AdminError {
    Unauthorized,
    DispatchUnavailable,
}

impl IntoResponse for AdminError {
    fn into_response(self) -> Response {
        let (status, code, message) = match self {
            Self::Unauthorized => (
                StatusCode::UNAUTHORIZED,
                "invalid_api_key",
                "A valid Bearer API key is required.",
            ),
            Self::DispatchUnavailable => (
                StatusCode::BAD_GATEWAY,
                "dispatch_unavailable",
                "The gateway could not dispatch the administrative request.",
            ),
        };
        (
            status,
            Json(serde_json::json!({
                "error": {
                    "message": message,
                    "type": "gateway_error",
                    "code": code,
                }
            })),
        )
            .into_response()
    }
}

async fn require_admin_authentication(
    State(authenticator): State<ApiKeyAuthenticator>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Response {
    match headers
        .get("authorization")
        .filter(|header| authenticator.authorize(header))
    {
        Some(_) => next.run(request).await,
        None => AdminError::Unauthorized.into_response(),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    run_gateway().await
}

async fn send_inference_request(
    endpoint: iroh::Endpoint,
    peer_id: EndpointId,
    request: InferenceRequest,
) -> Result<InferenceResponse> {
    info!(
        "Connecting to peer {} with ALPN {:?}",
        peer_id.fmt_short(),
        std::str::from_utf8(INFERENCE_ALPN)
    );

    // connect to peer and open bidirectional stream
    let connection = endpoint
        .connect(peer_id, INFERENCE_ALPN)
        .await
        .context("Failed to connect to peer")?;

    info!("Connected, opening bidirectional stream");
    let (mut send, mut recv) = connection
        .open_bi()
        .await
        .context("Failed to open bidirectional stream")?;

    let message = InferenceMessage::Request(request);
    let request_bytes =
        postcard::to_allocvec(&message).context("Failed to serialize inference request")?;

    info!("Sending {} bytes", request_bytes.len());
    send.write_all(&request_bytes)
        .await
        .context("Failed to write request")?;

    info!("Finishing send stream");
    send.finish()?;

    info!("Reading response...");
    let response_bytes = recv
        .read_to_end(10 * 1024 * 1024)
        .await
        .context("Failed to read response")?; // 10MB max

    info!("Received {} bytes, deserializing", response_bytes.len());
    let response_message: InferenceMessage = postcard::from_bytes(&response_bytes)
        .context("Failed to deserialize inference response")?;

    match response_message {
        InferenceMessage::Response(response) => {
            info!("Successfully received inference response");
            Ok(response)
        }
        _ => anyhow::bail!("Unexpected message type from inference node"),
    }
}

async fn run_gateway() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let api_key = Zeroizing::new(
        std::env::var("KOINON_GATEWAY_API_KEY").context("KOINON_GATEWAY_API_KEY must be set")?,
    );
    let authenticator = ApiKeyAuthenticator::from_secret(&api_key)?;
    let allowed_endpoint_ids = load_endpoint_allowlist(&args.allowed_peer_file)?;

    info!("Starting gateway node");
    info!("  HTTP API: http://{}", args.listen_addr);
    info!("  Discovery mode: {:?}", args.discovery_mode);
    info!("  Relay kind: {:?}", args.relay_kind);
    info!(
        "  Authorized inference nodes: {}",
        allowed_endpoint_ids.len()
    );

    let bootstrap_peers = psyche_inference_node::load_bootstrap_peers(
        args.bootstrap_peer_file.as_ref(),
        "No bootstrap peers configured (gateway will be a bootstrap node)",
    )?;
    let mut transport_endpoint_ids = allowed_endpoint_ids.clone();
    transport_endpoint_ids.extend(bootstrap_peers.iter().map(|peer| peer.id));
    let transport_allowlist = allowlist::AllowDynamic::with_nodes(transport_endpoint_ids);

    let cancel = CancellationToken::new();

    info!("Initializing P2P network...");
    let metrics = Arc::new(ClientMetrics::default());
    let run_id = "inference";

    type P2PNetwork = NetworkConnection<InferenceGossipMessage, ()>;

    let mut network = P2PNetwork::init(
        run_id,
        None,
        None,
        args.discovery_mode,
        args.relay_kind,
        bootstrap_peers,
        None,
        transport_allowlist,
        metrics.clone(),
        Some(cancel.clone()),
    )
    .await
    .context("Failed to initialize P2P network")?;

    info!("P2P network initialized");
    info!("  Endpoint ID: {}", network.endpoint_id());

    let endpoint_file = if let Ok(file_path) = std::env::var("PSYCHE_GATEWAY_ENDPOINT_FILE") {
        info!("Found PSYCHE_GATEWAY_ENDPOINT_FILE env var: {}", file_path);
        Some(PathBuf::from(file_path))
    } else {
        info!("No PSYCHE_GATEWAY_ENDPOINT_FILE env var, checking CLI args");
        args.write_endpoint_file.clone()
    };

    let endpoint_addr = network.router().endpoint().addr();
    let endpoints = vec![endpoint_addr.clone()];

    if let Some(ref endpoint_file) = endpoint_file {
        let content =
            serde_json::to_string(&endpoints).context("Failed to serialize endpoint address")?;
        fs::write(endpoint_file, content).context("Failed to write endpoint file")?;
        info!("Wrote gateway endpoint to {:?}", endpoint_file);
        info!("Other nodes can bootstrap using this file");
    } else {
        let endpoint_json = serde_json::to_string_pretty(&endpoints)
            .context("Failed to serialize endpoint address")?;
        info!("Gateway endpoint address (use for bootstrapping inference nodes):");
        println!("\n{}\n", endpoint_json);
    }

    info!("Waiting for gossip mesh to stabilize...");
    sleep(Duration::from_secs(5)).await;

    info!("Gossip mesh should be ready");

    let (request_tx, mut request_rx) = mpsc::channel::<RoutedInferenceRequest>(100);
    let (cleanup_tx, mut cleanup_rx) = mpsc::unbounded_channel::<String>();
    let (gossip_tx, mut gossip_rx) = mpsc::channel::<InferenceGossipMessage>(100);

    let state = Arc::new(GatewayState::new(
        authenticator.clone(),
        allowed_endpoint_ids,
        request_tx,
        cleanup_tx,
        REQUEST_TIMEOUT,
    )?);
    let admin_state = Arc::new(GatewayAdminState {
        gossip_tx,
        endpoint_addr,
    });

    info!("Gateway ready! Listening on http://{}", args.listen_addr);
    info!("Discovering inference nodes...");

    let network_handle = {
        let state = state.clone();
        let cancel = cancel.clone();
        tokio::spawn(async move {
            let mut task_set = tokio::task::JoinSet::new();

            let mut cleanup_interval = tokio::time::interval(Duration::from_secs(15));
            cleanup_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                tokio::select! {
                    _ = cancel.cancelled() => {
                        info!("Network task shutting down");
                        info!("Aborting {} active P2P request tasks", task_set.len());
                        task_set.shutdown().await;
                        break;
                    }

                    _ = cleanup_interval.tick() => {
                        let now = std::time::Instant::now();
                        for node_id in state.remove_stale_nodes(now) {
                            warn!("Removing stale node {}", node_id.fmt_short());
                        }
                    }

                    Some(request_id) = cleanup_rx.recv() => {
                        state.remove_pending_request(&request_id).await;
                    }

                    Some(routed_request) = request_rx.recv() => {
                        let request_id = routed_request.request.request_id.clone();
                        let target_peer_id = routed_request.endpoint_id;
                        info!("Sending inference request {} to {} via direct P2P",
                              request_id, target_peer_id.fmt_short());

                        let endpoint = network.router().endpoint().clone();
                        let state_clone = state.clone();
                        task_set.spawn(async move {
                            let result = tokio::time::timeout(
                                P2P_REQUEST_TIMEOUT,
                                send_inference_request(
                                    endpoint,
                                    target_peer_id,
                                    routed_request.request,
                                ),
                            )
                            .await;

                            match result {
                                Ok(Ok(response)) => {
                                    info!("Received inference response for {}", request_id);
                                    state_clone
                                        .complete_request(&request_id, Ok(response))
                                        .await;
                                }
                                Ok(Err(error)) => {
                                    error!("Inference P2P request failed: {error:#}");
                                    state_clone
                                        .complete_request(
                                            &request_id,
                                            Err(GatewayFailure::NodeExecution),
                                        )
                                        .await;
                                }
                                Err(_) => {
                                    error!("Inference request {} timed out", request_id);
                                    state_clone
                                        .complete_request(&request_id, Err(GatewayFailure::Timeout))
                                        .await;
                                }
                            }
                        });
                    }

                    Some(result) = task_set.join_next(), if !task_set.is_empty() => {
                        if let Err(error) = result {
                            error!("P2P request task failed: {error}");
                        }
                    }

                    Some(gossip_msg) = gossip_rx.recv() => {
                        if let Err(e) = network.broadcast(&gossip_msg) {
                            error!("Failed to broadcast gossip message: {:#}", e);
                        }
                    }

                    event = network.poll_next() => {
                        match event {
                            Ok(Some(NetworkEvent::MessageReceived((peer_id, msg)))) => {
                                info!("Received gossip message from {}", peer_id.fmt_short());
                                match msg {
                                    InferenceGossipMessage::NodeAvailable {
                                        model_name,
                                        checkpoint_id: _,
                                        capabilities: _,
                                        timestamp_ms: _,
                                    } => {
                                        let accepted = state.upsert_node(NodeRecord {
                                            endpoint_id: peer_id,
                                            model_name: model_name.clone(),
                                            last_seen: std::time::Instant::now(),
                                        });
                                        if accepted {
                                            info!("Heartbeat from {} (model: {})",
                                                peer_id.fmt_short(),
                                                model_name.as_deref().unwrap_or("<idle>"));
                                        } else {
                                            warn!("Ignoring unauthorized node {}", peer_id.fmt_short());
                                        }
                                    }
                                    InferenceGossipMessage::NodeUnavailable => {
                                        info!("Inference node {} went offline", peer_id.fmt_short());
                                        state.remove_node(&peer_id);
                                    }
                                    InferenceGossipMessage::LoadModel { .. } => {
                                        debug!("Ignoring LoadModel message (gateways don't load models)");
                                    }
                                    InferenceGossipMessage::ReloadCheckpoint { checkpoint_id, checkpoint_source } => {
                                        debug!("Checkpoint reload notification: {} from {}", checkpoint_id, checkpoint_source);
                                    }
                                }
                            }
                            Ok(Some(_)) => {
                                debug!("Other network event (ignored)");
                            }
                            Ok(None) => {}
                            Err(e) => {
                                error!("Network error: {:#}", e);
                            }
                        }
                    }
                }
            }
        })
    };

    let admin_router = Router::new()
        .route("/admin/load-model", post(handle_load_model))
        .route("/bootstrap", get(handle_bootstrap))
        .with_state(admin_state)
        .layer(axum::middleware::from_fn_with_state(
            authenticator,
            require_admin_authentication,
        ));
    let app = build_gateway_router(state).merge(admin_router);

    let listener = tokio::net::TcpListener::bind(&args.listen_addr)
        .await
        .context("Failed to bind HTTP server")?;

    info!("HTTP server listening on {}", args.listen_addr);

    let server_cancel = cancel.clone();
    let server_handle = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(server_cancel.cancelled_owned())
            .await
            .context("HTTP server error")
    });

    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            info!("Received shutdown signal");
        }
        _ = cancel.cancelled() => {
            info!("Cancellation requested");
        }
    }

    info!("Shutting down...");
    cancel.cancel();

    let (network_result, server_result) = tokio::join!(network_handle, server_handle);
    network_result.context("network task failed")?;
    server_result.context("HTTP server task failed")??;

    info!("Shutdown complete");
    Ok(())
}
