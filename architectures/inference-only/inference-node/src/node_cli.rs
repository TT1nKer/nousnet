use anyhow::{bail, ensure, Context, Result};
use clap::{Args as ClapArgs, Parser, Subcommand, ValueEnum};
use psyche_network::{DiscoveryMode, RelayKind};
use std::{net::IpAddr, path::PathBuf, time::Duration};

#[derive(Parser)]
#[command(name = "psyche-inference-node")]
pub struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    #[command(flatten)]
    run_args: RunArgs,
}

#[derive(Subcommand)]
enum Commands {
    /// Create a persistent node identity without starting the node.
    InitIdentity {
        #[arg(long)]
        identity_file: PathBuf,
    },

    /// Run the inference node.
    Run(Box<RunArgs>),

    // Prints the help, optionally as markdown. Used for docs generation.
    #[clap(hide = true)]
    PrintAllHelp {
        #[arg(long, required = true)]
        markdown: bool,
    },
}

pub enum NodeCommand {
    InitIdentity { identity_file: PathBuf },
    Run(RunArgs),
    PrintAllHelp,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum BackendKind {
    Vllm,
    Ollama,
}

#[derive(ClapArgs, Clone)]
pub struct RunArgs {
    #[arg(long)]
    pub identity_file: Option<PathBuf>,

    #[arg(long, value_enum, default_value_t = BackendKind::Vllm)]
    pub backend: BackendKind,

    #[arg(long)]
    pub model_name: Option<String>,

    #[arg(long)]
    pub provider_url: Option<String>,

    #[arg(long, default_value = "120")]
    pub provider_timeout_secs: u64,

    #[arg(long, default_value = "1")]
    pub tensor_parallel_size: usize,

    #[arg(long, default_value = "0.9")]
    pub gpu_memory_utilization: f64,

    #[arg(long)]
    pub checkpoint_path: Option<PathBuf>,

    /// what discovery to use - public n0 or local
    #[arg(long, env = "IROH_DISCOVERY", default_value = "n0")]
    pub discovery_mode: DiscoveryMode,

    /// what relays to use - public n0 or the private Psyche ones
    #[arg(long, env = "IROH_RELAY", default_value = "psyche")]
    pub relay_kind: RelayKind,

    #[arg(long)]
    pub relay_url: Option<String>,

    /// node capabilities (comma-separated, e.g. "streaming,tool_use")
    #[arg(long, default_value = "")]
    pub capabilities: String,

    /// gateway HTTP URL to fetch bootstrap peer from
    #[arg(long, env = "PSYCHE_GATEWAY_URL")]
    pub bootstrap_url: Option<String>,

    /// bootstrap peer file (JSON file with gateway endpoint address)
    #[arg(long)]
    pub bootstrap_peer_file: Option<PathBuf>,

    /// write endpoint address to file for other nodes to bootstrap from
    #[arg(long)]
    pub write_endpoint_file: Option<PathBuf>,
}

pub enum NodeBackendConfig {
    Vllm {
        model_name: Option<String>,
        tensor_parallel_size: usize,
        gpu_memory_utilization: f64,
    },
    Ollama {
        model_name: String,
        provider_url: String,
        timeout: Duration,
    },
}

impl Cli {
    pub fn into_command(self) -> Result<NodeCommand> {
        match self.command {
            Some(Commands::InitIdentity { identity_file }) => {
                Ok(NodeCommand::InitIdentity { identity_file })
            }
            Some(Commands::Run(run_args)) => {
                run_args.validate()?;
                Ok(NodeCommand::Run(*run_args))
            }
            Some(Commands::PrintAllHelp { markdown }) => {
                ensure!(markdown, "--markdown is required");
                Ok(NodeCommand::PrintAllHelp)
            }
            None => {
                self.run_args.validate()?;
                Ok(NodeCommand::Run(self.run_args))
            }
        }
    }

    pub fn run_args(&self) -> Result<&RunArgs> {
        let run_args = match &self.command {
            Some(Commands::Run(run_args)) => run_args,
            None => &self.run_args,
            _ => bail!("command does not run an inference node"),
        };
        run_args.validate()?;
        Ok(run_args)
    }

    pub fn backend_config(&self) -> Result<NodeBackendConfig> {
        self.run_args()?.backend_config()
    }
}

impl RunArgs {
    pub fn identity_file(&self) -> Result<&PathBuf> {
        self.identity_file
            .as_ref()
            .context("--identity-file is required")
    }

    pub fn backend_config(&self) -> Result<NodeBackendConfig> {
        match self.backend {
            BackendKind::Vllm => Ok(NodeBackendConfig::Vllm {
                model_name: self.model_name.clone(),
                tensor_parallel_size: self.tensor_parallel_size,
                gpu_memory_utilization: self.gpu_memory_utilization,
            }),
            BackendKind::Ollama => {
                let model_name = self
                    .model_name
                    .as_ref()
                    .filter(|model| !model.trim().is_empty())
                    .context("--model-name is required for the Ollama backend")?
                    .clone();
                let provider_url = self
                    .provider_url
                    .as_ref()
                    .context("--provider-url is required for the Ollama backend")?
                    .clone();
                validate_loopback_provider(&provider_url)?;
                Ok(NodeBackendConfig::Ollama {
                    model_name,
                    provider_url,
                    timeout: Duration::from_secs(self.provider_timeout_secs),
                })
            }
        }
    }

    fn validate(&self) -> Result<()> {
        self.identity_file()?;
        self.backend_config()?;
        Ok(())
    }
}

fn validate_loopback_provider(provider_url: &str) -> Result<()> {
    let url = reqwest::Url::parse(provider_url).context("invalid Ollama provider URL")?;
    ensure!(
        url.scheme() == "http",
        "Ollama provider must use local HTTP"
    );
    ensure!(
        url.username().is_empty() && url.password().is_none(),
        "Ollama provider URL must not contain credentials"
    );
    let is_loopback = match url.host_str() {
        Some("localhost") => true,
        Some(host) => host
            .parse::<IpAddr>()
            .is_ok_and(|address| address.is_loopback()),
        None => false,
    };
    ensure!(is_loopback, "Ollama provider must use a loopback address");
    Ok(())
}
