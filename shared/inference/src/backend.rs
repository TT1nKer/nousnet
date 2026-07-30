use crate::{InferenceRequest, InferenceResponse};
use async_trait::async_trait;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendErrorKind {
    Unavailable,
    Timeout,
    InvalidResponse,
    Execution,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{kind:?}: {message}")]
pub struct BackendError {
    pub kind: BackendErrorKind,
    pub message: String,
}

#[async_trait]
pub trait InferenceBackend: std::fmt::Debug + Send + Sync {
    fn model_name(&self) -> &str;

    async fn health(&self) -> Result<(), BackendError>;

    async fn infer(&self, request: &InferenceRequest) -> Result<InferenceResponse, BackendError>;

    async fn shutdown(&self) -> Result<(), BackendError> {
        Ok(())
    }
}
