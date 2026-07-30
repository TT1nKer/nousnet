use crate::{BackendError, InferenceBackend, InferenceRequest, InferenceResponse};
use std::{sync::Arc, time::Duration};
use tokio::sync::{RwLock, Semaphore};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeLifecycleState {
    Disabled,
    Starting,
    Ready,
    Busy,
    Draining,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RuntimeError {
    #[error("node is not ready")]
    NotReady,
    #[error("node is busy")]
    Busy,
    #[error("node did not drain before the deadline")]
    DrainTimeout,
    #[error(transparent)]
    Backend(#[from] BackendError),
}

#[derive(Debug)]
pub struct InferenceRuntime {
    backend: RwLock<Option<Arc<dyn InferenceBackend>>>,
    state: RwLock<NodeLifecycleState>,
    permits: Semaphore,
    max_concurrent_requests: u32,
}

impl InferenceRuntime {
    pub fn new(max_concurrent_requests: usize) -> Self {
        assert!(
            max_concurrent_requests > 0,
            "max_concurrent_requests must be greater than zero"
        );
        let max_concurrent_requests = u32::try_from(max_concurrent_requests)
            .expect("max_concurrent_requests must fit in u32");

        Self {
            backend: RwLock::new(None),
            state: RwLock::new(NodeLifecycleState::Disabled),
            permits: Semaphore::new(max_concurrent_requests as usize),
            max_concurrent_requests,
        }
    }

    pub async fn state(&self) -> NodeLifecycleState {
        *self.state.read().await
    }

    pub async fn start(&self, backend: Arc<dyn InferenceBackend>) -> Result<(), RuntimeError> {
        {
            let mut state = self.state.write().await;
            if !matches!(
                *state,
                NodeLifecycleState::Disabled | NodeLifecycleState::Error
            ) {
                return Err(RuntimeError::NotReady);
            }
            *state = NodeLifecycleState::Starting;
        }

        if let Err(error) = backend.health().await {
            *self.state.write().await = NodeLifecycleState::Error;
            return Err(error.into());
        }

        *self.backend.write().await = Some(backend);
        *self.state.write().await = NodeLifecycleState::Ready;
        Ok(())
    }

    pub async fn execute(
        &self,
        request: InferenceRequest,
    ) -> Result<InferenceResponse, RuntimeError> {
        if !matches!(
            self.state().await,
            NodeLifecycleState::Ready | NodeLifecycleState::Busy
        ) {
            return Err(RuntimeError::NotReady);
        }

        let permit = self.permits.try_acquire().map_err(|_| RuntimeError::Busy)?;
        let backend = self
            .backend
            .read()
            .await
            .clone()
            .ok_or(RuntimeError::NotReady)?;

        {
            let mut state = self.state.write().await;
            if !matches!(*state, NodeLifecycleState::Ready | NodeLifecycleState::Busy) {
                return Err(RuntimeError::NotReady);
            }
            *state = NodeLifecycleState::Busy;
        }

        let result = backend.infer(&request).await.map_err(RuntimeError::from);
        {
            let mut state = self.state.write().await;
            if matches!(*state, NodeLifecycleState::Ready | NodeLifecycleState::Busy) {
                let is_last_request =
                    self.permits.available_permits() + 1 == self.max_concurrent_requests as usize;
                *state = if is_last_request {
                    NodeLifecycleState::Ready
                } else {
                    NodeLifecycleState::Busy
                };
            }
        }
        drop(permit);

        result
    }

    pub async fn drain(&self, deadline: Duration) -> Result<(), RuntimeError> {
        {
            let mut state = self.state.write().await;
            if *state == NodeLifecycleState::Disabled {
                return Ok(());
            }
            if *state == NodeLifecycleState::Starting {
                return Err(RuntimeError::NotReady);
            }
            *state = NodeLifecycleState::Draining;
        }

        let all_permits = tokio::time::timeout(
            deadline,
            self.permits.acquire_many(self.max_concurrent_requests),
        )
        .await
        .map_err(|_| RuntimeError::DrainTimeout)?
        .map_err(|_| RuntimeError::NotReady)?;

        let backend = self.backend.read().await.clone();
        if let Some(backend) = backend {
            if let Err(error) = backend.shutdown().await {
                *self.state.write().await = NodeLifecycleState::Error;
                return Err(error.into());
            }
        }

        self.backend.write().await.take();
        *self.state.write().await = NodeLifecycleState::Disabled;
        drop(all_permits);
        Ok(())
    }
}
