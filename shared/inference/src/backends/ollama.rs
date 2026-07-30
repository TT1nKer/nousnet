//! Local Ollama inference backend.

use crate::{
    BackendError, BackendErrorKind, ChatMessage, InferenceBackend, InferenceRequest,
    InferenceResponse,
};
use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::{Client, Response, Url};
use serde::{Deserialize, Serialize};
use std::{net::IpAddr, time::Duration};

const MAX_RESPONSE_BYTES: usize = 1024 * 1024;

#[derive(Debug)]
pub struct OllamaBackend {
    base_url: Url,
    model_name: String,
    client: Client,
}

impl OllamaBackend {
    pub fn new(
        base_url: &str,
        model_name: impl Into<String>,
        timeout: Duration,
    ) -> Result<Self, BackendError> {
        let base_url = Url::parse(base_url)
            .map_err(|error| unavailable_error(format!("invalid Ollama URL: {error}")))?;
        validate_provider_url(&base_url)?;
        let model_name = model_name.into();
        if model_name.trim().is_empty() {
            return Err(unavailable_error("Ollama model name must not be empty"));
        }

        let client = Client::builder()
            .no_proxy()
            .timeout(timeout)
            .build()
            .map_err(|error| unavailable_error(format!("failed to create HTTP client: {error}")))?;

        Ok(Self {
            base_url,
            model_name,
            client,
        })
    }

    fn endpoint(&self, path: &str) -> Url {
        let mut endpoint = self.base_url.clone();
        endpoint.set_path(path);
        endpoint.set_query(None);
        endpoint.set_fragment(None);
        endpoint
    }
}

#[async_trait]
impl InferenceBackend for OllamaBackend {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    async fn health(&self) -> Result<(), BackendError> {
        let response = self
            .client
            .get(self.endpoint("/api/tags"))
            .send()
            .await
            .map_err(|error| request_error(error, BackendErrorKind::Unavailable))?;
        let body = read_response(response, BackendErrorKind::Unavailable).await?;
        let tags: OllamaTagsResponse = serde_json::from_slice(&body)
            .map_err(|error| invalid_response(format!("invalid tags response: {error}")))?;

        let has_model = tags.models.iter().any(|model| {
            model.name == self.model_name
                || model.model.as_deref() == Some(self.model_name.as_str())
        });
        if !has_model {
            return Err(unavailable_error(format!(
                "Ollama model {} is not installed",
                self.model_name
            )));
        }
        Ok(())
    }

    async fn infer(&self, request: &InferenceRequest) -> Result<InferenceResponse, BackendError> {
        let payload = OllamaChatRequest {
            model: &self.model_name,
            messages: &request.messages,
            stream: false,
            think: false,
            keep_alive: 0,
            options: OllamaOptions {
                temperature: request.temperature,
                top_p: request.top_p,
                num_predict: request.max_tokens,
            },
        };
        let response = self
            .client
            .post(self.endpoint("/api/chat"))
            .json(&payload)
            .send()
            .await
            .map_err(|error| request_error(error, BackendErrorKind::Execution))?;
        let body = read_response(response, BackendErrorKind::Execution).await?;
        let response: OllamaChatResponse = serde_json::from_slice(&body)
            .map_err(|error| invalid_response(format!("invalid chat response: {error}")))?;

        if response.model != self.model_name {
            return Err(invalid_response(format!(
                "Ollama responded with unexpected model {}",
                response.model
            )));
        }
        if !response.done {
            return Err(invalid_response(
                "Ollama returned an incomplete non-streaming response",
            ));
        }

        let generated_text = response.message.content;
        Ok(InferenceResponse {
            request_id: request.request_id.clone(),
            full_text: generated_text.clone(),
            generated_text,
            finish_reason: response.done_reason.or_else(|| Some("stop".to_string())),
        })
    }
}

fn validate_provider_url(url: &Url) -> Result<(), BackendError> {
    if url.scheme() != "http" {
        return Err(unavailable_error(
            "Ollama provider must use HTTP on the local machine",
        ));
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(unavailable_error(
            "Ollama provider URL must not contain credentials",
        ));
    }

    let is_loopback = match url.host_str() {
        Some("localhost") => true,
        Some(host) => host
            .parse::<IpAddr>()
            .is_ok_and(|address| address.is_loopback()),
        None => false,
    };
    if !is_loopback {
        return Err(unavailable_error(
            "Ollama provider must resolve to a loopback address",
        ));
    }
    Ok(())
}

async fn read_response(
    response: Response,
    status_error_kind: BackendErrorKind,
) -> Result<Vec<u8>, BackendError> {
    if !response.status().is_success() {
        return Err(BackendError {
            kind: status_error_kind,
            message: format!("Ollama returned HTTP {}", response.status()),
        });
    }

    let mut body = Vec::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| request_error(error, status_error_kind))?;
        if body.len().saturating_add(chunk.len()) > MAX_RESPONSE_BYTES {
            return Err(invalid_response("Ollama response exceeded the 1 MiB limit"));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn request_error(error: reqwest::Error, default_kind: BackendErrorKind) -> BackendError {
    BackendError {
        kind: if error.is_timeout() {
            BackendErrorKind::Timeout
        } else {
            default_kind
        },
        message: format!("Ollama request failed: {error}"),
    }
}

fn unavailable_error(message: impl Into<String>) -> BackendError {
    BackendError {
        kind: BackendErrorKind::Unavailable,
        message: message.into(),
    }
}

fn invalid_response(message: impl Into<String>) -> BackendError {
    BackendError {
        kind: BackendErrorKind::InvalidResponse,
        message: message.into(),
    }
}

#[derive(Debug, Serialize)]
struct OllamaChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    stream: bool,
    think: bool,
    keep_alive: u8,
    options: OllamaOptions,
}

#[derive(Debug, Serialize)]
struct OllamaOptions {
    temperature: f64,
    top_p: f64,
    num_predict: usize,
}

#[derive(Debug, Deserialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaModel>,
}

#[derive(Debug, Deserialize)]
struct OllamaModel {
    name: String,
    model: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OllamaChatResponse {
    model: String,
    message: OllamaChatMessage,
    done: bool,
    done_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OllamaChatMessage {
    content: String,
}
