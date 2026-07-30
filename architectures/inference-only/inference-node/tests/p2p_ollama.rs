#![cfg(feature = "ollama")]

use axum::{routing::get, routing::post, Json, Router};
use iroh::SecretKey;
use psyche_inference::{
    backends::ollama::OllamaBackend, ChatMessage, InferenceGossipMessage, InferenceProtocol,
    InferenceRequest, InferenceRuntime, INFERENCE_ALPN,
};
use psyche_inference_node::{
    p2p::{DiscoveryMode, InferenceNetwork, PeerAllowlist, RelayKind},
    p2p_client::send_inference_request,
};
use serde_json::json;
use std::{sync::Arc, time::Duration};
use tokio::{net::TcpListener, task::JoinHandle};
use tokio_util::sync::CancellationToken;

struct FakeOllama {
    base_url: String,
    cancel: CancellationToken,
    task: Option<JoinHandle<()>>,
}

impl FakeOllama {
    async fn start() -> Self {
        let app = Router::new()
            .route(
                "/api/tags",
                get(|| async {
                    Json(json!({
                        "models": [{
                            "name": "qwen3:8b",
                            "model": "qwen3:8b"
                        }]
                    }))
                }),
            )
            .route(
                "/api/chat",
                post(|| async {
                    Json(json!({
                        "model": "qwen3:8b",
                        "message": {
                            "role": "assistant",
                            "content": "READY"
                        },
                        "done": true,
                        "done_reason": "stop"
                    }))
                }),
            );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let cancel = CancellationToken::new();
        let server_cancel = cancel.clone();
        let task = tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(server_cancel.cancelled_owned())
                .await
                .unwrap();
        });

        Self {
            base_url,
            cancel,
            task: Some(task),
        }
    }

    async fn shutdown(mut self) {
        self.cancel.cancel();
        if let Some(task) = self.task.take() {
            task.await.unwrap();
        }
    }
}

impl Drop for FakeOllama {
    fn drop(&mut self) {
        self.cancel.cancel();
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

struct CancelOnDrop(Vec<CancellationToken>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        for token in &self.0 {
            token.cancel();
        }
    }
}

fn test_request() -> InferenceRequest {
    InferenceRequest {
        request_id: "p2p-ollama".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "Reply READY".to_string(),
        }],
        max_tokens: 16,
        temperature: 0.2,
        top_p: 0.8,
        stream: false,
    }
}

#[tokio::test]
async fn returns_real_backend_response_over_allowlisted_p2p() {
    let fake_ollama = FakeOllama::start().await;
    let backend =
        OllamaBackend::new(&fake_ollama.base_url, "qwen3:8b", Duration::from_secs(2)).unwrap();
    let runtime = Arc::new(InferenceRuntime::new(1));
    runtime.start(Arc::new(backend)).await.unwrap();

    let node_secret = SecretKey::generate(&mut rand::rng());
    let client_secret = SecretKey::generate(&mut rand::rng());
    let node_id = node_secret.public();
    let client_id = client_secret.public();
    let node_cancel = CancellationToken::new();
    let client_cancel = CancellationToken::new();
    let _cancel_on_drop = CancelOnDrop(vec![node_cancel.clone(), client_cancel.clone()]);

    type TestNetwork = InferenceNetwork<InferenceGossipMessage>;
    let node = TestNetwork::init_with_protocol(
        "p2p-ollama-test",
        DiscoveryMode::Local,
        RelayKind::Disabled,
        vec![],
        Some(node_secret),
        PeerAllowlist::with_nodes([client_id]),
        Some(node_cancel),
        (INFERENCE_ALPN, InferenceProtocol::new(runtime)),
    )
    .await
    .unwrap();
    let client = TestNetwork::init(
        "p2p-ollama-test",
        DiscoveryMode::Local,
        RelayKind::Disabled,
        vec![node.endpoint_addr()],
        Some(client_secret),
        PeerAllowlist::with_nodes([node_id]),
        Some(client_cancel),
    )
    .await
    .unwrap();

    let response = send_inference_request(
        client.endpoint(),
        node.endpoint_id(),
        test_request(),
        Duration::from_secs(5),
    )
    .await;

    client.shutdown().await.unwrap();
    node.shutdown().await.unwrap();
    fake_ollama.shutdown().await;

    let response = response.unwrap();
    assert_eq!(response.request_id, "p2p-ollama");
    assert_eq!(response.generated_text, "READY");
}
