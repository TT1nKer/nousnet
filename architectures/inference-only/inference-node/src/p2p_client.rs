use anyhow::{Context, Result};
use iroh::{Endpoint, EndpointId};
use psyche_inference::{InferenceMessage, InferenceRequest, InferenceResponse, INFERENCE_ALPN};
use std::time::Duration;
use thiserror::Error;
use tracing::info;

const MAX_RESPONSE_BYTES: usize = 10 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum P2PRequestError {
    #[error("inference P2P request exceeded its deadline")]
    DeadlineElapsed,
    #[error(transparent)]
    Request(#[from] anyhow::Error),
}

pub async fn send_inference_request(
    endpoint: Endpoint,
    peer_id: EndpointId,
    request: InferenceRequest,
    deadline: Duration,
) -> Result<InferenceResponse, P2PRequestError> {
    tokio::time::timeout(
        deadline,
        send_inference_request_without_deadline(endpoint, peer_id, request),
    )
    .await
    .map_err(|_| P2PRequestError::DeadlineElapsed)?
    .map_err(P2PRequestError::Request)
}

async fn send_inference_request_without_deadline(
    endpoint: Endpoint,
    peer_id: EndpointId,
    request: InferenceRequest,
) -> Result<InferenceResponse> {
    info!(
        "Connecting to peer {} with ALPN {:?}",
        peer_id.fmt_short(),
        std::str::from_utf8(INFERENCE_ALPN)
    );
    let connection = endpoint
        .connect(peer_id, INFERENCE_ALPN)
        .await
        .context("Failed to connect to peer")?;

    info!("Connected, opening bidirectional stream");
    let (mut send, mut recv) = connection
        .open_bi()
        .await
        .context("Failed to open bidirectional stream")?;

    let request_bytes = postcard::to_allocvec(&InferenceMessage::Request(request))
        .context("Failed to serialize inference request")?;
    info!("Sending {} bytes", request_bytes.len());
    send.write_all(&request_bytes)
        .await
        .context("Failed to write request")?;
    send.finish().context("Failed to finish request stream")?;

    info!("Reading response");
    let response_bytes = recv
        .read_to_end(MAX_RESPONSE_BYTES)
        .await
        .context("Failed to read response")?;
    info!("Received {} response bytes", response_bytes.len());
    let response_message: InferenceMessage = postcard::from_bytes(&response_bytes)
        .context("Failed to deserialize inference response")?;

    match response_message {
        InferenceMessage::Response(response) => Ok(response),
        _ => anyhow::bail!("Unexpected message type from inference node"),
    }
}
