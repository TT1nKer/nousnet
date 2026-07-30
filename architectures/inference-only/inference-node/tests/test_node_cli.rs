use clap::Parser;
use psyche_inference_node::node_cli::{Cli, NodeBackendConfig, NodeCommand};

#[test]
fn init_identity_does_not_require_run_arguments() {
    let cli = Cli::try_parse_from([
        "psyche-inference-node",
        "init-identity",
        "--identity-file",
        "node.identity",
    ])
    .unwrap();
    assert!(matches!(
        cli.into_command().unwrap(),
        NodeCommand::InitIdentity { .. }
    ));
}

#[test]
fn legacy_top_level_run_defaults_to_vllm() {
    let cli =
        Cli::try_parse_from(["psyche-inference-node", "--identity-file", "node.identity"]).unwrap();
    assert!(matches!(
        cli.backend_config().unwrap(),
        NodeBackendConfig::Vllm {
            model_name: None,
            ..
        }
    ));
}

#[test]
fn ollama_run_requires_identity_model_and_loopback_provider() {
    let cli = Cli::try_parse_from([
        "psyche-inference-node",
        "run",
        "--identity-file",
        "node.identity",
        "--backend",
        "ollama",
        "--model-name",
        "qwen3:8b",
        "--provider-url",
        "http://127.0.0.1:11434",
    ])
    .unwrap();
    assert!(matches!(
        cli.backend_config().unwrap(),
        NodeBackendConfig::Ollama { .. }
    ));
}

#[test]
fn run_rejects_missing_identity() {
    let cli = Cli::try_parse_from(["psyche-inference-node", "run"]).unwrap();
    assert!(cli.run_args().is_err());
}

#[test]
fn ollama_rejects_missing_model_or_provider() {
    for arguments in [
        vec![
            "psyche-inference-node",
            "run",
            "--identity-file",
            "node.identity",
            "--backend",
            "ollama",
            "--provider-url",
            "http://127.0.0.1:11434",
        ],
        vec![
            "psyche-inference-node",
            "run",
            "--identity-file",
            "node.identity",
            "--backend",
            "ollama",
            "--model-name",
            "qwen3:8b",
        ],
    ] {
        let cli = Cli::try_parse_from(arguments).unwrap();
        assert!(cli.backend_config().is_err());
    }
}

#[test]
fn ollama_rejects_non_loopback_provider() {
    let cli = Cli::try_parse_from([
        "psyche-inference-node",
        "run",
        "--identity-file",
        "node.identity",
        "--backend",
        "ollama",
        "--model-name",
        "qwen3:8b",
        "--provider-url",
        "http://192.0.2.10:11434",
    ])
    .unwrap();
    assert!(cli.backend_config().is_err());
}
