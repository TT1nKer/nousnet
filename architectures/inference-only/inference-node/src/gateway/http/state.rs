use crate::gateway::{
    auth::ApiKeyAuthenticator,
    routing::{NodeCatalog, NodeRecord},
};
use anyhow::{ensure, Result};
use iroh::EndpointId;
use psyche_inference::{InferenceRequest, InferenceResponse};
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Mutex, MutexGuard},
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, oneshot, RwLock};

type GatewayResult = std::result::Result<InferenceResponse, GatewayFailure>;
pub(super) type PendingSender = oneshot::Sender<GatewayResult>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GatewayFailure {
    NodeExecution,
    Timeout,
}

#[derive(Debug)]
pub struct RoutedInferenceRequest {
    pub endpoint_id: EndpointId,
    pub model_name: String,
    pub request: InferenceRequest,
}

pub struct GatewayState {
    pub(super) authenticator: ApiKeyAuthenticator,
    catalog: Mutex<NodeCatalog>,
    pending_requests: RwLock<HashMap<String, PendingSender>>,
    pub(super) request_tx: mpsc::Sender<RoutedInferenceRequest>,
    pub(super) cleanup_tx: mpsc::UnboundedSender<String>,
    pub(super) request_timeout: Duration,
}

impl GatewayState {
    pub fn new(
        authenticator: ApiKeyAuthenticator,
        allowed_endpoint_ids: HashSet<EndpointId>,
        request_tx: mpsc::Sender<RoutedInferenceRequest>,
        cleanup_tx: mpsc::UnboundedSender<String>,
        request_timeout: Duration,
    ) -> Result<Self> {
        ensure!(
            !request_timeout.is_zero(),
            "gateway request timeout must be positive"
        );
        Ok(Self {
            authenticator,
            catalog: Mutex::new(NodeCatalog::new(allowed_endpoint_ids)?),
            pending_requests: RwLock::new(HashMap::new()),
            request_tx,
            cleanup_tx,
            request_timeout,
        })
    }

    pub fn upsert_node(&self, record: NodeRecord) -> bool {
        self.catalog().upsert(record)
    }

    pub fn remove_node(&self, endpoint_id: &EndpointId) -> Option<NodeRecord> {
        self.catalog().remove(endpoint_id)
    }

    pub fn remove_stale_nodes(&self, now: Instant) -> Vec<EndpointId> {
        self.catalog().remove_stale(now)
    }

    pub(super) fn select_node(&self, model_name: &str, now: Instant) -> Option<EndpointId> {
        self.catalog().select(model_name, now)
    }

    pub(super) async fn insert_pending_request(&self, request_id: String, sender: PendingSender) {
        self.pending_requests
            .write()
            .await
            .insert(request_id, sender);
    }

    pub async fn complete_request(
        &self,
        request_id: &str,
        result: std::result::Result<InferenceResponse, GatewayFailure>,
    ) -> bool {
        let sender = self.pending_requests.write().await.remove(request_id);
        sender.is_some_and(|sender| sender.send(result).is_ok())
    }

    pub async fn remove_pending_request(&self, request_id: &str) -> bool {
        self.pending_requests
            .write()
            .await
            .remove(request_id)
            .is_some()
    }

    pub async fn pending_request_count(&self) -> usize {
        self.pending_requests.read().await.len()
    }

    fn catalog(&self) -> MutexGuard<'_, NodeCatalog> {
        self.catalog
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }
}

pub async fn run_pending_cleanup(
    state: Arc<GatewayState>,
    mut cleanup_rx: mpsc::UnboundedReceiver<String>,
) {
    while let Some(request_id) = cleanup_rx.recv().await {
        state.remove_pending_request(&request_id).await;
    }
}
