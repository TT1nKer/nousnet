//! Authenticated OpenAI-compatible HTTP request flow.

mod contracts;
mod state;

pub use state::{run_pending_cleanup, GatewayFailure, GatewayState, RoutedInferenceRequest};

use axum::{
    body::{to_bytes, Body},
    extract::State,
    http::HeaderMap,
    routing::post,
    Json, Router,
};
use contracts::{
    ChatCompletionChoice, ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse,
    HttpError, ValidatedRequest, MAX_HTTP_BODY_BYTES,
};
use psyche_inference::InferenceRequest;
use std::{
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::oneshot;

pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

pub fn build_gateway_router(state: Arc<GatewayState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(handle_inference))
        .with_state(state)
}

async fn handle_inference(
    State(state): State<Arc<GatewayState>>,
    headers: HeaderMap,
    body: Body,
) -> Result<Json<ChatCompletionResponse>, HttpError> {
    let authorization = headers
        .get("authorization")
        .ok_or(HttpError::Unauthorized)?;
    if !state.authenticator.authorize(authorization) {
        return Err(HttpError::Unauthorized);
    }

    let body = to_bytes(body, MAX_HTTP_BODY_BYTES)
        .await
        .map_err(|_| HttpError::InvalidRequest("request body is too large"))?;
    let request: ChatCompletionRequest = serde_json::from_slice(&body)
        .map_err(|_| HttpError::InvalidRequest("request body is not valid JSON"))?;
    let request = ValidatedRequest::try_from(request)?;

    let endpoint_id = state
        .select_node(&request.model, Instant::now())
        .ok_or(HttpError::ModelUnavailable)?;
    let request_id = uuid::Uuid::new_v4().to_string();
    let inference_request = InferenceRequest {
        request_id: request_id.clone(),
        messages: request.messages,
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        stream: false,
    };
    let (response_tx, response_rx) = oneshot::channel();
    state
        .insert_pending_request(request_id.clone(), response_tx)
        .await;
    let _pending_guard = PendingRequestGuard {
        request_id,
        cleanup_tx: state.cleanup_tx.clone(),
    };

    state
        .request_tx
        .send(RoutedInferenceRequest {
            endpoint_id,
            model_name: request.model.clone(),
            request: inference_request,
        })
        .await
        .map_err(|_| HttpError::DispatchUnavailable)?;

    let response = tokio::time::timeout(state.request_timeout, response_rx)
        .await
        .map_err(|_| HttpError::Timeout)?
        .map_err(|_| HttpError::NodeExecution)?
        .map_err(HttpError::from)?;

    Ok(Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}", response.request_id),
        object: "chat.completion",
        created: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        model: request.model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatCompletionMessage {
                role: "assistant".to_string(),
                content: response.generated_text,
            },
            finish_reason: response.finish_reason,
        }],
    }))
}

struct PendingRequestGuard {
    request_id: String,
    cleanup_tx: tokio::sync::mpsc::UnboundedSender<String>,
}

impl Drop for PendingRequestGuard {
    fn drop(&mut self) {
        let _ = self.cleanup_tx.send(self.request_id.clone());
    }
}
