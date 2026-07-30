//! Psyche Inference Node
//!
//! A standalone node for serving LLM inference over the Psyche P2P network.
//!
//! Architecture:
//! - Joins P2P network via iroh (gossip + direct connections)
//! - Announces availability via gossip
//! - Handles inference requests via direct P2P connections
//! - Supports dynamic checkpoint reloading

use anyhow::{ensure, Context, Result};
use clap::Parser;
#[cfg(feature = "ollama")]
use psyche_inference::backends::ollama::OllamaBackend;
#[cfg(feature = "vllm")]
use psyche_inference::{backends::vllm::VllmBackend, ModelSource};
use psyche_inference::{
    InferenceGossipMessage, InferenceProtocol, InferenceRuntime, INFERENCE_ALPN,
};
use psyche_inference_node::{
    identity::{create_identity, load_identity},
    node_cli::{Cli, NodeBackendConfig, NodeCommand},
    p2p::{InferenceNetwork, PeerAllowlist},
};
use std::sync::Arc;
use std::{fs, time::Duration};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[derive(Debug, Clone)]
enum ModelLoadState {
    Idle,
    #[cfg(feature = "vllm")]
    Loading(String),
    Loaded(String),
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let run_args = match Cli::parse().into_command()? {
        NodeCommand::InitIdentity { identity_file } => {
            let endpoint_id = create_identity(&identity_file)?;
            println!("{endpoint_id}");
            return Ok(());
        }
        NodeCommand::PrintAllHelp => {
            clap_markdown::print_help_markdown::<Cli>();
            return Ok(());
        }
        NodeCommand::Run(run_args) => run_args,
    };
    let backend_config = run_args.backend_config()?;
    let identity_secret_key = load_identity(run_args.identity_file()?)?;

    info!("Starting Psyche Inference Node");
    match &backend_config {
        NodeBackendConfig::Vllm {
            model_name,
            tensor_parallel_size,
            gpu_memory_utilization,
        } => {
            info!("Backend: vLLM");
            info!("Model: {}", model_name.as_deref().unwrap_or("<idle>"));
            info!("Tensor Parallel Size: {}", tensor_parallel_size);
            info!("GPU Memory Utilization: {}", gpu_memory_utilization);
        }
        NodeBackendConfig::Ollama { model_name, .. } => {
            info!("Backend: Ollama");
            info!("Model: {}", model_name);
        }
    }
    info!("Endpoint ID: {}", identity_secret_key.public());

    let capabilities: Vec<String> = if run_args.capabilities.is_empty() {
        vec![]
    } else {
        run_args
            .capabilities
            .split(',')
            .map(|s| s.trim().to_string())
            .collect()
    };

    info!("Discovery mode: {:?}", run_args.discovery_mode);
    info!("Relay kind: {:?}", run_args.relay_kind);
    info!("Capabilities: {:?}", capabilities);

    let mut bootstrap_peers = psyche_inference_node::load_bootstrap_peers(
        run_args.bootstrap_peer_file.as_ref(),
        "No bootstrap peers configured (no env vars or CLI args)",
    )?;

    if bootstrap_peers.is_empty() {
        if let Some(ref url) = run_args.bootstrap_url {
            match psyche_inference_node::fetch_bootstrap_peer(url).await {
                Ok(peer) => {
                    info!("Fetched bootstrap peer from {}", url);
                    bootstrap_peers.push(peer);
                }
                Err(e) => {
                    warn!("Failed to fetch bootstrap peer from {}: {:#}", url, e);
                }
            }
        }
    }
    ensure!(
        !bootstrap_peers.is_empty(),
        "at least one bootstrap gateway is required"
    );
    let gateway_allowlist = PeerAllowlist::with_nodes(bootstrap_peers.iter().map(|peer| peer.id));

    let cancel = CancellationToken::new();

    let inference_runtime = Arc::new(InferenceRuntime::new(1));
    match &backend_config {
        NodeBackendConfig::Vllm {
            model_name: Some(model_name),
            tensor_parallel_size,
            gpu_memory_utilization,
        } => {
            #[cfg(feature = "vllm")]
            {
                info!("Initializing Python interpreter...");
                pyo3::prepare_freethreaded_python();
                info!("Initializing vLLM engine with model: {}...", model_name);
                let backend = VllmBackend::initialize(
                    model_name.clone(),
                    Some(*tensor_parallel_size),
                    Some(*gpu_memory_utilization),
                )
                .await
                .context("Failed to initialize vLLM engine")?;
                inference_runtime
                    .start(Arc::new(backend))
                    .await
                    .context("Failed to start inference runtime")?;
                info!("vLLM engine initialized successfully");
            }

            #[cfg(not(feature = "vllm"))]
            {
                let _ = (tensor_parallel_size, gpu_memory_utilization);
                anyhow::bail!(
                    "model {} requires vLLM, but this binary was built without it",
                    model_name
                );
            }
        }
        NodeBackendConfig::Vllm {
            model_name: None, ..
        } => info!("No initial vLLM model - starting in idle mode"),
        NodeBackendConfig::Ollama {
            model_name,
            provider_url,
            timeout,
        } => {
            #[cfg(feature = "ollama")]
            {
                let backend = OllamaBackend::new(provider_url, model_name.clone(), *timeout)
                    .context("Invalid Ollama backend configuration")?;
                inference_runtime
                    .start(Arc::new(backend))
                    .await
                    .context("Failed to start Ollama backend")?;
                info!("Ollama backend is healthy");
            }

            #[cfg(not(feature = "ollama"))]
            {
                let _ = (provider_url, timeout);
                anyhow::bail!(
                    "model {} requires Ollama, but this binary was built without it",
                    model_name
                );
            }
        }
    }

    let initial_model_name = match &backend_config {
        NodeBackendConfig::Vllm { model_name, .. } => model_name.clone(),
        NodeBackendConfig::Ollama { model_name, .. } => Some(model_name.clone()),
    };
    let model_state = Arc::new(RwLock::new(if let Some(model) = initial_model_name {
        ModelLoadState::Loaded(model.clone())
    } else {
        ModelLoadState::Idle
    }));
    #[cfg(feature = "vllm")]
    let vllm_reload_config = match &backend_config {
        NodeBackendConfig::Vllm {
            tensor_parallel_size,
            gpu_memory_utilization,
            ..
        } => Some((*tensor_parallel_size, *gpu_memory_utilization)),
        NodeBackendConfig::Ollama { .. } => None,
    };

    info!("Initializing P2P network...");

    let run_id = "inference";

    type P2PNetwork = InferenceNetwork<InferenceGossipMessage>;

    info!("Registering inference protocol handler...");
    let inference_protocol = InferenceProtocol::new(inference_runtime.clone());

    let mut network = P2PNetwork::init_with_protocol(
        run_id,
        run_args.discovery_mode,
        run_args.relay_kind,
        bootstrap_peers,
        Some(identity_secret_key),
        gateway_allowlist.clone(),
        Some(cancel.clone()),
        (INFERENCE_ALPN, inference_protocol),
    )
    .await
    .context("Failed to initialize P2P network")?;

    info!("P2P network initialized");
    info!("  Endpoint ID: {}", network.endpoint_id());
    info!("Protocol handler registered");

    if let Some(ref endpoint_file) = run_args.write_endpoint_file {
        let endpoint_addr = network.endpoint_addr();
        let content = serde_json::to_string(&endpoint_addr)
            .context("Failed to serialize endpoint address")?;
        fs::write(endpoint_file, content).context("Failed to write endpoint file")?;
        info!("Wrote endpoint to {:?}", endpoint_file);
    }

    tokio::time::sleep(Duration::from_secs(2)).await;

    // announce availability via gossip
    let model_name_for_broadcast = match &*model_state.read().await {
        ModelLoadState::Loaded(name) => Some(name.clone()),
        _ => None,
    };
    let availability_msg = InferenceGossipMessage::NodeAvailable {
        model_name: model_name_for_broadcast.clone(),
        checkpoint_id: None,
        capabilities: capabilities.clone(),
        timestamp_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64,
    };

    network
        .broadcast(&availability_msg)
        .context("Failed to broadcast availability")?;

    info!(
        "Broadcasted availability to network (model: {})",
        model_name_for_broadcast.as_deref().unwrap_or("<idle>")
    );
    info!("Inference node ready! Listening for requests...");

    // heartbeat for re-announcing availability
    let mut heartbeat_interval = tokio::time::interval(std::time::Duration::from_secs(30));
    heartbeat_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    // re-bootstrap every 20 heartbeats (10 min)
    let mut rebootstrap_interval = tokio::time::interval(std::time::Duration::from_secs(600));
    rebootstrap_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    rebootstrap_interval.tick().await;

    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                info!("Received shutdown signal");
                break;
            }

            _ = cancel.cancelled() => {
                info!("Cancellation requested");
                break;
            }

            _ = heartbeat_interval.tick() => {
                let model_name_for_broadcast = match &*model_state.read().await {
                    ModelLoadState::Loaded(name) => Some(name.clone()),
                    _ => None,
                };
                let availability_msg = InferenceGossipMessage::NodeAvailable {
                    model_name: model_name_for_broadcast.clone(),
                    checkpoint_id: None,
                    capabilities: capabilities.clone(),
                    timestamp_ms: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                };
                if let Err(e) = network.broadcast(&availability_msg) {
                    warn!("Failed to broadcast: {:#}", e);
                } else if let Some(ref model) = model_name_for_broadcast {
                    debug!("Re-broadcast successful (model: {})", model);
                } else {
                    debug!("Re-broadcast successful (idle)");
                }
            }

            _ = rebootstrap_interval.tick() => {
                if let Some(ref url) = run_args.bootstrap_url {
                    match psyche_inference_node::fetch_bootstrap_peer(url).await {
                        Ok(peer) => {
                            gateway_allowlist.add(peer.id);
                            network.add_peer(peer.clone());
                            debug!("Re-bootstrapped from {}: peer {}", url, peer.id.fmt_short());
                        }
                        Err(e) => {
                            warn!("Re-bootstrap from {} failed: {:#}", url, e);
                        }
                    }
                }
            }

            event = network.poll_next() => {
                match event {
                    Ok(Some((peer_id, msg))) => {
                        debug!("Received gossip message from {}: {:?}", peer_id.fmt_short(), msg);

                        match msg {
                            InferenceGossipMessage::NodeAvailable { model_name, checkpoint_id, capabilities, timestamp_ms: _ } => {
                                info!("Peer {} is available: model={:?}, checkpoint={:?}, caps={:?}",
                                      peer_id.fmt_short(), model_name, checkpoint_id, capabilities);
                            }
                            InferenceGossipMessage::NodeUnavailable => {
                                info!("Peer {} is no longer available", peer_id.fmt_short());
                            }
                            InferenceGossipMessage::LoadModel { model_name: requested_model, model_source } => {
                                info!("Received LoadModel request from {}: model={}, source={:?}",
                                      peer_id.fmt_short(), requested_model, model_source);

                                let should_load = match &*model_state.read().await {
                                    ModelLoadState::Loaded(name) if name == &requested_model => {
                                        info!("Model {} already loaded, skipping", requested_model);
                                        false
                                    }
                                    #[cfg(feature = "vllm")]
                                    ModelLoadState::Loading(name) => {
                                        info!("Model load already in progress ({}), skipping concurrent load request for {}",
                                              name, requested_model);
                                        false
                                    }
                                    _ => true,
                                };

                                if should_load {
                                    #[cfg(feature = "vllm")]
                                    {
                                        let Some((tensor_parallel_size, gpu_memory_utilization)) =
                                            vllm_reload_config
                                        else {
                                            let _ = model_source;
                                            error!(
                                                "Cannot dynamically load model {}: node is not configured for vLLM",
                                                requested_model
                                            );
                                            continue;
                                        };
                                        let model_path = match model_source {
                                            ModelSource::HuggingFace(name) | ModelSource::Local(name) => name,
                                        };
                                        *model_state.write().await = ModelLoadState::Loading(requested_model.clone());
                                        info!("Loading new model: {} (background task)", requested_model);

                                        // Loading runs behind the backend's blocking boundary so heartbeats continue.
                                        let inference_runtime_clone = inference_runtime.clone();
                                        let model_state_clone = model_state.clone();
                                        let requested_model_clone = requested_model.clone();

                                        tokio::spawn(async move {
                                            info!("Draining existing inference backend");
                                            if let Err(e) = inference_runtime_clone
                                                .drain(Duration::from_secs(60))
                                                .await
                                            {
                                                error!("Failed to drain existing backend: {:#}", e);
                                                *model_state_clone.write().await = ModelLoadState::Idle;
                                                return;
                                            }

                                            // Give vLLM time to release GPU memory before loading a new model.
                                            info!("Waiting 5s for GPU memory to be released...");
                                            tokio::time::sleep(Duration::from_secs(5)).await;

                                            let load_result = VllmBackend::initialize(
                                                model_path,
                                                Some(tensor_parallel_size),
                                                Some(gpu_memory_utilization),
                                            )
                                            .await;

                                            match load_result {
                                                Ok(backend) => {
                                                    if let Err(e) = inference_runtime_clone
                                                        .start(Arc::new(backend))
                                                        .await
                                                    {
                                                        error!("Failed to start model {}: {:#}", requested_model_clone, e);
                                                        *model_state_clone.write().await = ModelLoadState::Idle;
                                                        return;
                                                    }
                                                    *model_state_clone.write().await =
                                                        ModelLoadState::Loaded(requested_model_clone.clone());
                                                    info!("Successfully loaded model: {}", requested_model_clone);
                                                }
                                                Err(e) => {
                                                    error!("Failed to load model {}: {:#}", requested_model_clone, e);
                                                    *model_state_clone.write().await = ModelLoadState::Idle;
                                                }
                                            }
                                        });
                                    }

                                    #[cfg(not(feature = "vllm"))]
                                    {
                                        let _ = model_source;
                                        error!(
                                            "Cannot load model {}: binary has no vLLM backend",
                                            requested_model
                                        );
                                    }
                                }
                            }
                            InferenceGossipMessage::ReloadCheckpoint { checkpoint_id, checkpoint_source } => {
                                info!("Received checkpoint reload request: {} from {}",
                                      checkpoint_id, checkpoint_source);
                                // TODO: implement checkpoint reloading for RL training
                                warn!("Checkpoint reloading not yet implemented");
                            }
                        }
                    }
                    Ok(None) => {
                    }
                    Err(e) => {
                        error!("Network error: {:#}", e);
                    }
                }
            }
        }
    }

    info!("Shutting down inference node...");
    if let Err(error) = network.broadcast(&InferenceGossipMessage::NodeUnavailable) {
        warn!("Failed to broadcast node unavailability: {error:#}");
    }
    inference_runtime
        .drain(Duration::from_secs(60))
        .await
        .context("Failed to drain inference runtime")?;
    info!("Shutdown complete");

    Ok(())
}
