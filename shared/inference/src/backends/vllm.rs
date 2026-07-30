use crate::{
    BackendError, BackendErrorKind, InferenceBackend, InferenceNode, InferenceRequest,
    InferenceResponse,
};
use async_trait::async_trait;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

#[derive(Debug)]
pub struct VllmBackend {
    model_name: String,
    node: Arc<Mutex<InferenceNode>>,
    initialized: AtomicBool,
}

impl VllmBackend {
    pub async fn initialize(
        model_name: String,
        tensor_parallel_size: Option<usize>,
        gpu_memory_utilization: Option<f64>,
    ) -> Result<Self, BackendError> {
        let backend =
            Self::new_uninitialized(model_name, tensor_parallel_size, gpu_memory_utilization);
        let node = backend.node.clone();

        tokio::task::spawn_blocking(move || {
            node.lock()
                .map_err(|_| execution_error("vLLM node lock is poisoned"))?
                .initialize(tensor_parallel_size, gpu_memory_utilization)
                .map_err(|error| execution_error(format!("vLLM initialization failed: {error}")))
        })
        .await
        .map_err(|error| execution_error(format!("vLLM initialization task failed: {error}")))??;

        backend.initialized.store(true, Ordering::Release);
        Ok(backend)
    }

    fn new_uninitialized(
        model_name: String,
        tensor_parallel_size: Option<usize>,
        gpu_memory_utilization: Option<f64>,
    ) -> Self {
        let node = InferenceNode::new(
            model_name.clone(),
            tensor_parallel_size,
            gpu_memory_utilization,
        );
        Self {
            model_name,
            node: Arc::new(Mutex::new(node)),
            initialized: AtomicBool::new(false),
        }
    }

    #[cfg(test)]
    fn uninitialized(
        model_name: String,
        tensor_parallel_size: Option<usize>,
        gpu_memory_utilization: Option<f64>,
    ) -> Self {
        Self::new_uninitialized(model_name, tensor_parallel_size, gpu_memory_utilization)
    }
}

#[async_trait]
impl InferenceBackend for VllmBackend {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    async fn health(&self) -> Result<(), BackendError> {
        if self.initialized.load(Ordering::Acquire) {
            Ok(())
        } else {
            Err(BackendError {
                kind: BackendErrorKind::Unavailable,
                message: "vLLM backend is not initialized".to_string(),
            })
        }
    }

    async fn infer(&self, request: &InferenceRequest) -> Result<InferenceResponse, BackendError> {
        self.health().await?;

        let node = self.node.clone();
        let request = request.clone();
        tokio::task::spawn_blocking(move || {
            node.lock()
                .map_err(|_| execution_error("vLLM node lock is poisoned"))?
                .inference(&request)
                .map_err(|error| execution_error(format!("vLLM inference failed: {error}")))
        })
        .await
        .map_err(|error| execution_error(format!("vLLM inference task failed: {error}")))?
    }

    async fn shutdown(&self) -> Result<(), BackendError> {
        if !self.initialized.load(Ordering::Acquire) {
            return Ok(());
        }

        let node = self.node.clone();
        tokio::task::spawn_blocking(move || {
            node.lock()
                .map_err(|_| execution_error("vLLM node lock is poisoned"))?
                .shutdown()
                .map_err(|error| execution_error(format!("vLLM shutdown failed: {error}")))
        })
        .await
        .map_err(|error| execution_error(format!("vLLM shutdown task failed: {error}")))??;

        self.initialized.store(false, Ordering::Release);
        Ok(())
    }
}

fn execution_error(message: impl Into<String>) -> BackendError {
    BackendError {
        kind: BackendErrorKind::Execution,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InferenceBackend;

    #[test]
    fn adapter_exposes_configured_model_without_loading_python() {
        let backend = VllmBackend::uninitialized("gpt2".to_string(), Some(1), Some(0.3));
        assert_eq!(backend.model_name(), "gpt2");
    }
}
