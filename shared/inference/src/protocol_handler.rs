//! Direct P2P protocol handler for inference requests
//!
//! This implements iroh's ProtocolHandler trait to accept incoming
//! inference requests over direct P2P connections.

use crate::{InferenceMessage, InferenceRequest, InferenceResponse, InferenceRuntime};
use anyhow::{Context, Result};
use iroh::endpoint::Connection;
use iroh::protocol::{AcceptError, ProtocolHandler};
use std::sync::Arc;
use tracing::{debug, error, info};

pub const INFERENCE_ALPN: &[u8] = b"/psyche/inference/1";

#[derive(Clone, Debug)]
pub struct InferenceProtocol {
    runtime: Arc<InferenceRuntime>,
}

impl InferenceProtocol {
    pub fn new(runtime: Arc<InferenceRuntime>) -> Self {
        Self { runtime }
    }

    async fn handle_connection(&self, connection: Connection) -> Result<()> {
        let peer_id = connection.remote_id();
        debug!(
            "Accepting inference connection from {}",
            peer_id.fmt_short()
        );

        // bidirectional stream
        let (mut send, mut recv) = connection.accept_bi().await?;

        let request_bytes = recv.read_to_end(1024 * 1024).await?;
        let message: InferenceMessage = postcard::from_bytes(&request_bytes)
            .context("Failed to deserialize inference message")?;

        match message {
            InferenceMessage::Request(request) => {
                info!(
                    "Received inference request {} from {}",
                    request.request_id,
                    peer_id.fmt_short()
                );

                let response = self.process_request(request).await?;

                info!("Serializing response for {}", peer_id.fmt_short());
                let response_msg = InferenceMessage::Response(response);
                let response_bytes =
                    postcard::to_allocvec(&response_msg).context("Failed to serialize response")?;

                info!(
                    "Writing {} bytes to {}",
                    response_bytes.len(),
                    peer_id.fmt_short()
                );
                send.write_all(&response_bytes).await?;

                info!("Finishing send stream to {}", peer_id.fmt_short());
                send.finish()?;

                // adaptive delay to ensure data is flushed before connection is dropped
                // without this, the connection might close before the peer reads all bytes
                // base 50ms + 10ms per MB of data
                let size_mb = response_bytes.len() as f64 / (1024.0 * 1024.0);
                let delay_ms = 50 + (size_mb * 10.0) as u64;
                debug!(
                    "Waiting {}ms for {} bytes to flush",
                    delay_ms,
                    response_bytes.len()
                );
                tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;

                info!(
                    "Successfully sent inference response to {}",
                    peer_id.fmt_short()
                );
            }
            _ => {
                error!("Unexpected message type from {}", peer_id.fmt_short());
            }
        }

        Ok(())
    }

    async fn process_request(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        info!("Processing inference request: {}", request.request_id);
        self.runtime
            .execute(request)
            .await
            .context("Failed to run inference")
    }
}

impl ProtocolHandler for InferenceProtocol {
    async fn accept(&self, connection: Connection) -> Result<(), AcceptError> {
        self.handle_connection(connection).await.map_err(|e| {
            error!("Error handling inference connection: {:#}", e);
            let io_error = std::io::Error::other(e.to_string());
            AcceptError::from_err(io_error)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChatMessage;

    #[tokio::test]
    async fn disabled_runtime_returns_an_error_instead_of_a_fake_response() {
        let protocol = InferenceProtocol::new(Arc::new(InferenceRuntime::new(1)));
        let error = protocol
            .process_request(InferenceRequest {
                request_id: "disabled".to_string(),
                messages: vec![ChatMessage {
                    role: "user".to_string(),
                    content: "hello".to_string(),
                }],
                max_tokens: 8,
                temperature: 0.7,
                top_p: 0.9,
                stream: false,
            })
            .await
            .unwrap_err();

        assert!(format!("{error:#}").contains("node is not ready"));
    }
}
