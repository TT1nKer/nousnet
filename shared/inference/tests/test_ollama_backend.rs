#![cfg(feature = "ollama")]

use axum::{
    body::{Body, Bytes},
    extract::State,
    http::{header::CONTENT_TYPE, Response, StatusCode},
    routing::{get, post},
    Router,
};
use psyche_inference::{
    backends::ollama::OllamaBackend, BackendErrorKind, ChatMessage, InferenceBackend,
    InferenceRequest,
};
use serde_json::{json, Value};
use std::{sync::Arc, time::Duration};
use tokio::{net::TcpListener, sync::Mutex};

#[derive(Clone)]
enum ChatBehavior {
    Json(Value),
    Raw(Vec<u8>),
    Delayed(Duration, Value),
}

#[derive(Clone)]
struct FakeOllamaState {
    captured_chat: Arc<Mutex<Option<Value>>>,
    tags_body: Arc<Vec<u8>>,
    chat_behavior: ChatBehavior,
}

fn test_request() -> InferenceRequest {
    InferenceRequest {
        request_id: "ollama-test".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "reply READY".to_string(),
        }],
        max_tokens: 16,
        temperature: 0.2,
        top_p: 0.8,
        stream: false,
    }
}

fn chat_response(content: &str) -> Value {
    json!({
        "model": "qwen3:8b",
        "message": {
            "role": "assistant",
            "content": content
        },
        "done": true,
        "done_reason": "stop"
    })
}

async fn spawn_fake_ollama(
    tags_body: Value,
    chat_behavior: ChatBehavior,
    captured_chat: Arc<Mutex<Option<Value>>>,
) -> String {
    let state = FakeOllamaState {
        captured_chat,
        tags_body: Arc::new(serde_json::to_vec(&tags_body).unwrap()),
        chat_behavior,
    };
    let app = Router::new()
        .route("/api/tags", get(fake_tags))
        .route("/api/chat", post(fake_chat))
        .with_state(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    format!("http://{address}")
}

async fn fake_tags(State(state): State<FakeOllamaState>) -> Response<Body> {
    json_response((*state.tags_body).clone())
}

async fn fake_chat(State(state): State<FakeOllamaState>, body: Bytes) -> Response<Body> {
    let request: Value = serde_json::from_slice(&body).unwrap();
    *state.captured_chat.lock().await = Some(request);

    match state.chat_behavior {
        ChatBehavior::Json(value) => json_response(serde_json::to_vec(&value).unwrap()),
        ChatBehavior::Raw(body) => json_response(body),
        ChatBehavior::Delayed(delay, value) => {
            tokio::time::sleep(delay).await;
            json_response(serde_json::to_vec(&value).unwrap())
        }
    }
}

fn json_response(body: Vec<u8>) -> Response<Body> {
    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .unwrap()
}

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

#[test]
fn rejects_empty_model_name() {
    let error =
        OllamaBackend::new("http://127.0.0.1:11434", "  ", Duration::from_secs(10)).unwrap_err();
    assert_eq!(error.kind, BackendErrorKind::Unavailable);
}

#[test]
fn rejects_non_http_or_credentialed_provider() {
    for provider in [
        "https://127.0.0.1:11434",
        "http://user:secret@127.0.0.1:11434",
    ] {
        let error = OllamaBackend::new(provider, "qwen3:8b", Duration::from_secs(10)).unwrap_err();
        assert_eq!(error.kind, BackendErrorKind::Unavailable);
    }
}

#[tokio::test]
async fn maps_chat_and_releases_model() {
    let captured = Arc::new(Mutex::new(None));
    let provider_url = spawn_fake_ollama(
        json!({"models": [{"name": "qwen3:8b", "model": "qwen3:8b"}]}),
        ChatBehavior::Json(chat_response("READY")),
        captured.clone(),
    )
    .await;
    let backend = OllamaBackend::new(&provider_url, "qwen3:8b", Duration::from_secs(2)).unwrap();

    let response = backend.infer(&test_request()).await.unwrap();
    assert_eq!(response.generated_text, "READY");
    assert_eq!(response.finish_reason.as_deref(), Some("stop"));

    let body = captured.lock().await.clone().unwrap();
    assert_eq!(body["model"], "qwen3:8b");
    assert_eq!(body["messages"][0]["role"], "user");
    assert_eq!(body["stream"], false);
    assert_eq!(body["think"], false);
    assert_eq!(body["keep_alive"], 0);
    assert_eq!(body["options"]["temperature"], 0.2);
    assert_eq!(body["options"]["top_p"], 0.8);
    assert_eq!(body["options"]["num_predict"], 16);
}

#[tokio::test]
async fn health_requires_exact_model_name() {
    let provider_url = spawn_fake_ollama(
        json!({"models": [{"name": "qwen3:8b-latest", "model": "qwen3:8b-latest"}]}),
        ChatBehavior::Json(chat_response("unused")),
        Arc::new(Mutex::new(None)),
    )
    .await;
    let backend = OllamaBackend::new(&provider_url, "qwen3:8b", Duration::from_secs(2)).unwrap();

    let error = backend.health().await.unwrap_err();
    assert_eq!(error.kind, BackendErrorKind::Unavailable);
}

#[tokio::test]
async fn maps_provider_timeout() {
    let provider_url = spawn_fake_ollama(
        json!({"models": [{"name": "qwen3:8b"}]}),
        ChatBehavior::Delayed(Duration::from_millis(100), chat_response("late")),
        Arc::new(Mutex::new(None)),
    )
    .await;
    let backend = OllamaBackend::new(&provider_url, "qwen3:8b", Duration::from_millis(10)).unwrap();

    let error = backend.infer(&test_request()).await.unwrap_err();
    assert_eq!(error.kind, BackendErrorKind::Timeout);
}

#[tokio::test]
async fn rejects_malformed_chat_response() {
    let provider_url = spawn_fake_ollama(
        json!({"models": [{"name": "qwen3:8b"}]}),
        ChatBehavior::Raw(b"{not-json".to_vec()),
        Arc::new(Mutex::new(None)),
    )
    .await;
    let backend = OllamaBackend::new(&provider_url, "qwen3:8b", Duration::from_secs(2)).unwrap();

    let error = backend.infer(&test_request()).await.unwrap_err();
    assert_eq!(error.kind, BackendErrorKind::InvalidResponse);
}

#[tokio::test]
async fn rejects_chat_response_larger_than_one_mebibyte() {
    let oversized_content = "x".repeat(1024 * 1024);
    let provider_url = spawn_fake_ollama(
        json!({"models": [{"name": "qwen3:8b"}]}),
        ChatBehavior::Json(chat_response(&oversized_content)),
        Arc::new(Mutex::new(None)),
    )
    .await;
    let backend = OllamaBackend::new(&provider_url, "qwen3:8b", Duration::from_secs(2)).unwrap();

    let error = backend.infer(&test_request()).await.unwrap_err();
    assert_eq!(error.kind, BackendErrorKind::InvalidResponse);
}
