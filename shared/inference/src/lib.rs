//! Psyche Inference

pub mod protocol;
#[cfg(feature = "vllm")]
pub mod protocol_handler;

#[cfg(feature = "vllm")]
pub mod node;
#[cfg(feature = "vllm")]
pub mod vllm;

#[cfg(feature = "vllm")]
pub use node::InferenceNode;
pub use protocol::{
    ChatMessage, InferenceGossipMessage, InferenceMessage, InferenceRequest, InferenceResponse,
    ModelSource,
};
#[cfg(feature = "vllm")]
pub use protocol_handler::{InferenceProtocol, INFERENCE_ALPN};
