//! Psyche Inference

pub mod backend;
pub mod protocol;
#[cfg(feature = "vllm")]
pub mod protocol_handler;
pub mod runtime;

#[cfg(feature = "vllm")]
pub mod node;
#[cfg(feature = "vllm")]
pub mod vllm;

pub use backend::{BackendError, BackendErrorKind, InferenceBackend};
#[cfg(feature = "vllm")]
pub use node::InferenceNode;
pub use protocol::{
    ChatMessage, InferenceGossipMessage, InferenceMessage, InferenceRequest, InferenceResponse,
    ModelSource,
};
#[cfg(feature = "vllm")]
pub use protocol_handler::{InferenceProtocol, INFERENCE_ALPN};
pub use runtime::{InferenceRuntime, NodeLifecycleState, RuntimeError};
