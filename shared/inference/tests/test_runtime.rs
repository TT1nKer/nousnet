use async_trait::async_trait;
use psyche_inference::{
    BackendError, BackendErrorKind, ChatMessage, InferenceBackend, InferenceRequest,
    InferenceResponse, InferenceRuntime, NodeLifecycleState, RuntimeError,
};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio::sync::Notify;

#[derive(Debug)]
struct BlockingBackend {
    entered: Arc<Notify>,
    release: Arc<Notify>,
    shutdown_called: Arc<AtomicBool>,
}

#[async_trait]
impl InferenceBackend for BlockingBackend {
    fn model_name(&self) -> &str {
        "qwen3:8b"
    }

    async fn health(&self) -> Result<(), BackendError> {
        Ok(())
    }

    async fn infer(&self, request: &InferenceRequest) -> Result<InferenceResponse, BackendError> {
        self.entered.notify_one();
        self.release.notified().await;
        Ok(success_response(request))
    }

    async fn shutdown(&self) -> Result<(), BackendError> {
        self.shutdown_called.store(true, Ordering::SeqCst);
        Ok(())
    }
}

#[derive(Debug)]
struct FailingBackend {
    fail_health: bool,
}

#[async_trait]
impl InferenceBackend for FailingBackend {
    fn model_name(&self) -> &str {
        "qwen3:8b"
    }

    async fn health(&self) -> Result<(), BackendError> {
        if self.fail_health {
            Err(test_backend_error())
        } else {
            Ok(())
        }
    }

    async fn infer(&self, _request: &InferenceRequest) -> Result<InferenceResponse, BackendError> {
        Err(test_backend_error())
    }
}

fn blocking_backend() -> (BlockingBackend, Arc<Notify>, Arc<Notify>, Arc<AtomicBool>) {
    let entered = Arc::new(Notify::new());
    let release = Arc::new(Notify::new());
    let shutdown_called = Arc::new(AtomicBool::new(false));
    (
        BlockingBackend {
            entered: entered.clone(),
            release: release.clone(),
            shutdown_called: shutdown_called.clone(),
        },
        entered,
        release,
        shutdown_called,
    )
}

fn test_request(request_id: &str) -> InferenceRequest {
    InferenceRequest {
        request_id: request_id.to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "hello".to_string(),
        }],
        max_tokens: 8,
        temperature: 0.7,
        top_p: 0.9,
        stream: false,
    }
}

fn success_response(request: &InferenceRequest) -> InferenceResponse {
    InferenceResponse {
        request_id: request.request_id.clone(),
        generated_text: "ready".to_string(),
        full_text: "ready".to_string(),
        finish_reason: Some("stop".to_string()),
    }
}

fn test_backend_error() -> BackendError {
    BackendError {
        kind: BackendErrorKind::Execution,
        message: "test failure".to_string(),
    }
}

#[tokio::test]
async fn disabled_runtime_rejects_requests() {
    let runtime = InferenceRuntime::new(1);
    let error = runtime.execute(test_request("disabled")).await.unwrap_err();
    assert_eq!(error, RuntimeError::NotReady);
}

#[tokio::test]
async fn second_request_is_rejected_while_busy() {
    let (backend, entered, release, _) = blocking_backend();
    let runtime = Arc::new(InferenceRuntime::new(1));
    runtime.start(Arc::new(backend)).await.unwrap();

    let first_runtime = runtime.clone();
    let first = tokio::spawn(async move { first_runtime.execute(test_request("first")).await });
    entered.notified().await;

    assert_eq!(
        runtime.execute(test_request("second")).await.unwrap_err(),
        RuntimeError::Busy
    );
    release.notify_one();
    first.await.unwrap().unwrap();
    assert_eq!(runtime.state().await, NodeLifecycleState::Ready);
}

#[tokio::test]
async fn failed_health_check_leaves_runtime_in_error() {
    let runtime = InferenceRuntime::new(1);
    let error = runtime
        .start(Arc::new(FailingBackend { fail_health: true }))
        .await
        .unwrap_err();

    assert_eq!(error, RuntimeError::Backend(test_backend_error()));
    assert_eq!(runtime.state().await, NodeLifecycleState::Error);
}

#[tokio::test]
async fn inference_failure_releases_capacity_and_restores_ready_state() {
    let runtime = InferenceRuntime::new(1);
    runtime
        .start(Arc::new(FailingBackend { fail_health: false }))
        .await
        .unwrap();

    assert_eq!(
        runtime.execute(test_request("failed")).await.unwrap_err(),
        RuntimeError::Backend(test_backend_error())
    );
    assert_eq!(runtime.state().await, NodeLifecycleState::Ready);
}

#[tokio::test]
async fn drain_waits_for_active_request_then_shuts_down() {
    let (backend, entered, release, shutdown_called) = blocking_backend();
    let runtime = Arc::new(InferenceRuntime::new(1));
    runtime.start(Arc::new(backend)).await.unwrap();

    let request_runtime = runtime.clone();
    let request =
        tokio::spawn(async move { request_runtime.execute(test_request("active")).await });
    entered.notified().await;

    let drain_runtime = runtime.clone();
    let drain = tokio::spawn(async move { drain_runtime.drain(Duration::from_secs(1)).await });
    tokio::task::yield_now().await;
    assert_eq!(runtime.state().await, NodeLifecycleState::Draining);
    assert_eq!(
        runtime.execute(test_request("late")).await.unwrap_err(),
        RuntimeError::NotReady
    );

    release.notify_one();
    request.await.unwrap().unwrap();
    drain.await.unwrap().unwrap();
    assert!(shutdown_called.load(Ordering::SeqCst));
    assert_eq!(runtime.state().await, NodeLifecycleState::Disabled);
}

#[tokio::test]
async fn drain_timeout_keeps_admission_closed_and_allows_retry() {
    let (backend, entered, release, _) = blocking_backend();
    let runtime = Arc::new(InferenceRuntime::new(1));
    runtime.start(Arc::new(backend)).await.unwrap();

    let request_runtime = runtime.clone();
    let request =
        tokio::spawn(async move { request_runtime.execute(test_request("active")).await });
    entered.notified().await;

    assert_eq!(
        runtime.drain(Duration::from_millis(1)).await.unwrap_err(),
        RuntimeError::DrainTimeout
    );
    assert_eq!(runtime.state().await, NodeLifecycleState::Draining);
    assert_eq!(
        runtime.execute(test_request("late")).await.unwrap_err(),
        RuntimeError::NotReady
    );

    release.notify_one();
    request.await.unwrap().unwrap();
    runtime.drain(Duration::from_secs(1)).await.unwrap();
    assert_eq!(runtime.state().await, NodeLifecycleState::Disabled);
}
