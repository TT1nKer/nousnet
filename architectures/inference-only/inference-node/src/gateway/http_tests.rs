use super::{
    auth::ApiKeyAuthenticator,
    http::{
        build_gateway_router, run_pending_cleanup, GatewayFailure, GatewayState,
        RoutedInferenceRequest,
    },
    routing::NodeRecord,
};
use axum::{
    body::{to_bytes, Body},
    http::{Request, StatusCode},
    Router,
};
use iroh::{EndpointId, SecretKey};
use psyche_inference::InferenceResponse;
use serde_json::{json, Value};
use std::{collections::HashSet, sync::Arc, time::Duration};
use tokio::{
    sync::{mpsc, Mutex},
    time::sleep,
};
use tower::ServiceExt;

struct TestGateway {
    router: Router,
    state: Arc<GatewayState>,
    requests: Arc<Mutex<mpsc::Receiver<RoutedInferenceRequest>>>,
    endpoint_id: EndpointId,
}

fn endpoint_id(seed: u8) -> EndpointId {
    SecretKey::from_bytes(&[seed; 32]).public()
}

fn test_gateway(model_name: Option<&str>, timeout: Duration) -> TestGateway {
    let endpoint_id = endpoint_id(1);
    let authenticator = ApiKeyAuthenticator::from_secret("gateway-secret").unwrap();
    let (request_tx, request_rx) = mpsc::channel(8);
    let (cleanup_tx, cleanup_rx) = mpsc::unbounded_channel();
    let state = Arc::new(
        GatewayState::new(
            authenticator,
            HashSet::from([endpoint_id]),
            request_tx,
            cleanup_tx,
            timeout,
        )
        .unwrap(),
    );
    if let Some(model_name) = model_name {
        state.upsert_node(NodeRecord {
            endpoint_id,
            model_name: Some(model_name.to_string()),
            last_seen: std::time::Instant::now(),
        });
    }
    tokio::spawn(run_pending_cleanup(state.clone(), cleanup_rx));

    TestGateway {
        router: build_gateway_router(state.clone()),
        state,
        requests: Arc::new(Mutex::new(request_rx)),
        endpoint_id,
    }
}

fn chat_request(api_key: Option<&str>, body: Value) -> Request<Body> {
    let mut builder = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json");
    if let Some(api_key) = api_key {
        builder = builder.header("authorization", format!("Bearer {api_key}"));
    }
    builder.body(Body::from(body.to_string())).unwrap()
}

fn valid_request(model: &str) -> Value {
    json!({
        "model": model,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 64,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": false
    })
}

async fn error_code(response: axum::response::Response) -> String {
    let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    let body: Value = serde_json::from_slice(&body).unwrap();
    body["error"]["code"].as_str().unwrap().to_string()
}

async fn assert_pending_empty(state: &GatewayState) {
    for _ in 0..100 {
        if state.pending_request_count().await == 0 {
            return;
        }
        sleep(Duration::from_millis(1)).await;
    }
    panic!("pending request was not cleaned up");
}

#[tokio::test]
async fn rejects_missing_api_key() {
    let gateway = test_gateway(Some("qwen3:8b"), Duration::from_secs(1));
    let response = gateway
        .router
        .oneshot(chat_request(None, valid_request("qwen3:8b")))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(error_code(response).await, "invalid_api_key");
}

#[tokio::test]
async fn rejects_streaming_explicitly() {
    let gateway = test_gateway(Some("qwen3:8b"), Duration::from_secs(1));
    let mut body = valid_request("qwen3:8b");
    body["stream"] = true.into();
    let response = gateway
        .router
        .oneshot(chat_request(Some("gateway-secret"), body))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(error_code(response).await, "streaming_unsupported");
}

#[tokio::test]
async fn returns_service_unavailable_without_exact_capacity() {
    let gateway = test_gateway(Some("qwen3:8b"), Duration::from_secs(1));
    let response = gateway
        .router
        .oneshot(chat_request(
            Some("gateway-secret"),
            valid_request("missing"),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(error_code(response).await, "model_unavailable");
}

#[tokio::test]
async fn returns_success_and_cleans_pending_request() {
    let gateway = test_gateway(Some("qwen3:8b"), Duration::from_secs(1));
    let state = gateway.state.clone();
    let requests = gateway.requests.clone();
    let responder = tokio::spawn(async move {
        let dispatched = requests.lock().await.recv().await.unwrap();
        let request_id = dispatched.request.request_id.clone();
        state
            .complete_request(
                &request_id,
                Ok(InferenceResponse {
                    request_id: request_id.clone(),
                    generated_text: "world".to_string(),
                    full_text: "hello world".to_string(),
                    finish_reason: Some("stop".to_string()),
                }),
            )
            .await;
    });

    let response = gateway
        .router
        .oneshot(chat_request(
            Some("gateway-secret"),
            valid_request("qwen3:8b"),
        ))
        .await
        .unwrap();
    responder.await.unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_pending_empty(&gateway.state).await;
}

#[tokio::test]
async fn maps_node_failure_and_cleans_pending_request() {
    let gateway = test_gateway(Some("qwen3:8b"), Duration::from_secs(1));
    let state = gateway.state.clone();
    let requests = gateway.requests.clone();
    let responder = tokio::spawn(async move {
        let dispatched = requests.lock().await.recv().await.unwrap();
        state
            .complete_request(
                &dispatched.request.request_id,
                Err(GatewayFailure::NodeExecution),
            )
            .await;
    });

    let response = gateway
        .router
        .oneshot(chat_request(
            Some("gateway-secret"),
            valid_request("qwen3:8b"),
        ))
        .await
        .unwrap();
    responder.await.unwrap();

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(error_code(response).await, "node_execution_failed");
    assert_pending_empty(&gateway.state).await;
}

#[tokio::test]
async fn maps_timeout_and_cleans_pending_request() {
    let gateway = test_gateway(Some("qwen3:8b"), Duration::from_millis(10));
    let _requests = gateway.requests.clone();
    let response = gateway
        .router
        .oneshot(chat_request(
            Some("gateway-secret"),
            valid_request("qwen3:8b"),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
    assert_eq!(error_code(response).await, "inference_timeout");
    assert_pending_empty(&gateway.state).await;
}

#[tokio::test]
async fn cleans_pending_request_when_handler_is_cancelled() {
    let gateway = test_gateway(Some("qwen3:8b"), Duration::from_secs(1));
    let requests = gateway.requests.clone();
    let response_task = tokio::spawn(gateway.router.oneshot(chat_request(
        Some("gateway-secret"),
        valid_request("qwen3:8b"),
    )));
    requests.lock().await.recv().await.unwrap();
    response_task.abort();
    let _ = response_task.await;

    assert_pending_empty(&gateway.state).await;
}

#[tokio::test]
async fn maps_closed_dispatch_queue_and_cleans_pending_request() {
    let gateway = test_gateway(Some("qwen3:8b"), Duration::from_secs(1));
    gateway.requests.lock().await.close();
    let response = gateway
        .router
        .oneshot(chat_request(
            Some("gateway-secret"),
            valid_request("qwen3:8b"),
        ))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert_eq!(error_code(response).await, "dispatch_unavailable");
    assert_pending_empty(&gateway.state).await;
}

#[tokio::test]
async fn rejects_malformed_and_out_of_bounds_requests() {
    let gateway = test_gateway(Some("qwen3:8b"), Duration::from_secs(1));
    let invalid_requests = [
        json!({"model": "qwen3:8b", "messages": []}),
        json!({"model": "qwen3:8b", "messages": [{"role": "owner", "content": "hi"}]}),
        json!({"model": "qwen3:8b", "messages": [{"role": "user", "content": "  "}]}),
        json!({"model": "qwen3:8b", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4097}),
        json!({"model": "qwen3:8b", "messages": [{"role": "user", "content": "hi"}], "temperature": 2.1}),
        json!({"model": "qwen3:8b", "messages": [{"role": "user", "content": "hi"}], "top_p": 1.1}),
    ];

    for body in invalid_requests {
        let response = gateway
            .router
            .clone()
            .oneshot(chat_request(Some("gateway-secret"), body))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(error_code(response).await, "invalid_request");
    }

    let oversized = json!({
        "model": "qwen3:8b",
        "messages": [{"role": "user", "content": "x".repeat(256 * 1024 + 1)}]
    });
    let response = gateway
        .router
        .oneshot(chat_request(Some("gateway-secret"), oversized))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert_eq!(error_code(response).await, "invalid_request");
}

#[tokio::test]
async fn dispatches_only_to_the_selected_exact_model_node() {
    let gateway = test_gateway(Some("qwen3:8b"), Duration::from_millis(10));
    let requests = gateway.requests.clone();
    let expected_endpoint_id = gateway.endpoint_id;
    let response_task = tokio::spawn(gateway.router.oneshot(chat_request(
        Some("gateway-secret"),
        valid_request("qwen3:8b"),
    )));

    let dispatched = requests.lock().await.recv().await.unwrap();
    assert_eq!(dispatched.endpoint_id, expected_endpoint_id);
    assert_eq!(dispatched.model_name, "qwen3:8b");
    let _ = response_task.await.unwrap().unwrap();
}
