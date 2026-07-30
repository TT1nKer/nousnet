use super::state::GatewayFailure;
use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use psyche_inference::ChatMessage;
use serde::{Deserialize, Serialize};

const MAX_MESSAGES: usize = 128;
const MAX_MESSAGE_BYTES: usize = 256 * 1024;
const MAX_TOTAL_MESSAGE_BYTES: usize = 1024 * 1024;
const MAX_MODEL_BYTES: usize = 256;
const MAX_TOKENS: usize = 4096;

pub(super) const MAX_HTTP_BODY_BYTES: usize = MAX_TOTAL_MESSAGE_BYTES + 64 * 1024;

#[derive(Deserialize)]
pub(super) struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatCompletionMessage>,
    max_tokens: Option<usize>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    #[serde(default)]
    stream: bool,
}

#[derive(Deserialize, Serialize)]
pub(super) struct ChatCompletionMessage {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
pub(super) struct ChatCompletionChoice {
    pub index: usize,
    pub message: ChatCompletionMessage,
    pub finish_reason: Option<String>,
}

#[derive(Serialize)]
pub(super) struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
}

pub(super) struct ValidatedRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
}

impl TryFrom<ChatCompletionRequest> for ValidatedRequest {
    type Error = HttpError;

    fn try_from(request: ChatCompletionRequest) -> Result<Self, Self::Error> {
        if request.stream {
            return Err(HttpError::StreamingUnsupported);
        }
        if request.model.trim().is_empty() || request.model.len() > MAX_MODEL_BYTES {
            return Err(HttpError::InvalidRequest(
                "model must be a non-empty bounded string",
            ));
        }
        if request.messages.is_empty() || request.messages.len() > MAX_MESSAGES {
            return Err(HttpError::InvalidRequest(
                "messages must contain between 1 and 128 entries",
            ));
        }

        let mut total_message_bytes = 0usize;
        let mut messages = Vec::with_capacity(request.messages.len());
        for message in request.messages {
            validate_message(&message, &mut total_message_bytes)?;
            messages.push(ChatMessage {
                role: message.role,
                content: message.content,
            });
        }

        let max_tokens = request.max_tokens.unwrap_or(100);
        if !(1..=MAX_TOKENS).contains(&max_tokens) {
            return Err(HttpError::InvalidRequest(
                "max_tokens must be between 1 and 4096",
            ));
        }
        let temperature = request.temperature.unwrap_or(1.0);
        if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
            return Err(HttpError::InvalidRequest(
                "temperature must be finite and between 0 and 2",
            ));
        }
        let top_p = request.top_p.unwrap_or(1.0);
        if !top_p.is_finite() || !(0.0..=1.0).contains(&top_p) {
            return Err(HttpError::InvalidRequest(
                "top_p must be finite and between 0 and 1",
            ));
        }

        Ok(Self {
            model: request.model,
            messages,
            max_tokens,
            temperature,
            top_p,
        })
    }
}

fn validate_message(
    message: &ChatCompletionMessage,
    total_message_bytes: &mut usize,
) -> Result<(), HttpError> {
    if !matches!(
        message.role.as_str(),
        "system" | "user" | "assistant" | "tool"
    ) {
        return Err(HttpError::InvalidRequest("message role is unsupported"));
    }
    if message.content.trim().is_empty() {
        return Err(HttpError::InvalidRequest(
            "message content must not be empty",
        ));
    }
    let message_bytes = message.content.len();
    if message_bytes > MAX_MESSAGE_BYTES {
        return Err(HttpError::InvalidRequest("message content is too large"));
    }
    *total_message_bytes =
        total_message_bytes
            .checked_add(message_bytes)
            .ok_or(HttpError::InvalidRequest(
                "total message content is too large",
            ))?;
    if *total_message_bytes > MAX_TOTAL_MESSAGE_BYTES {
        return Err(HttpError::InvalidRequest(
            "total message content is too large",
        ));
    }
    Ok(())
}

#[derive(Debug)]
pub(super) enum HttpError {
    Unauthorized,
    InvalidRequest(&'static str),
    StreamingUnsupported,
    ModelUnavailable,
    DispatchUnavailable,
    NodeExecution,
    Timeout,
}

impl From<GatewayFailure> for HttpError {
    fn from(failure: GatewayFailure) -> Self {
        match failure {
            GatewayFailure::NodeExecution => Self::NodeExecution,
            GatewayFailure::Timeout => Self::Timeout,
        }
    }
}

#[derive(Serialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Serialize)]
struct ErrorBody {
    message: &'static str,
    r#type: &'static str,
    code: &'static str,
}

impl IntoResponse for HttpError {
    fn into_response(self) -> Response {
        let (status, error_type, code, message) = match self {
            Self::Unauthorized => (
                StatusCode::UNAUTHORIZED,
                "authentication_error",
                "invalid_api_key",
                "A valid Bearer API key is required.",
            ),
            Self::InvalidRequest(message) => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "invalid_request",
                message,
            ),
            Self::StreamingUnsupported => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "streaming_unsupported",
                "Streaming is not supported.",
            ),
            Self::ModelUnavailable => (
                StatusCode::SERVICE_UNAVAILABLE,
                "service_unavailable_error",
                "model_unavailable",
                "No authorized inference node currently serves the requested model.",
            ),
            Self::DispatchUnavailable => (
                StatusCode::BAD_GATEWAY,
                "gateway_error",
                "dispatch_unavailable",
                "The gateway could not dispatch the inference request.",
            ),
            Self::NodeExecution => (
                StatusCode::BAD_GATEWAY,
                "gateway_error",
                "node_execution_failed",
                "The inference node failed to execute the request.",
            ),
            Self::Timeout => (
                StatusCode::GATEWAY_TIMEOUT,
                "timeout_error",
                "inference_timeout",
                "The inference request timed out.",
            ),
        };
        (
            status,
            Json(ErrorEnvelope {
                error: ErrorBody {
                    message,
                    r#type: error_type,
                    code,
                },
            }),
        )
            .into_response()
    }
}
