use iroh::{RelayMode, RelayUrl};
use iroh_gossip::proto::TopicId;
use iroh_relay::{RelayConfig, RelayMap, RelayQuicConfig};
use sha2::{Digest, Sha256};
use std::str::FromStr;

const GOSSIP_TOPIC: &str = "psyche gossip";
const USE_RELAY_HOSTNAME: &str = "use1-1.relay.nousresearch.psyche.iroh.link";
const USW_RELAY_HOSTNAME: &str = "usw1-1.relay.nousresearch.psyche.iroh.link";

#[derive(Clone, Copy, Debug)]
pub enum DiscoveryMode {
    Local,
    N0,
}

impl FromStr for DiscoveryMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "local" => Ok(Self::Local),
            "n0" => Ok(Self::N0),
            _ => Err(format!(
                "invalid discovery mode {value:?}; expected local or n0"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum RelayKind {
    Disabled,
    Psyche,
    N0,
}

impl FromStr for RelayKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "disabled" => Ok(Self::Disabled),
            "psyche" => Ok(Self::Psyche),
            "n0" => Ok(Self::N0),
            _ => Err(format!(
                "invalid relay kind {value:?}; expected disabled, psyche, or n0"
            )),
        }
    }
}

pub(super) fn relay_mode(kind: RelayKind) -> RelayMode {
    match kind {
        RelayKind::Disabled => RelayMode::Disabled,
        RelayKind::N0 => RelayMode::Default,
        RelayKind::Psyche => RelayMode::Custom(RelayMap::from_iter([
            relay_config(USE_RELAY_HOSTNAME),
            relay_config(USW_RELAY_HOSTNAME),
        ])),
    }
}

fn relay_config(hostname: &str) -> RelayConfig {
    let url: RelayUrl = format!("https://{hostname}")
        .parse()
        .expect("static relay URL must be valid");
    RelayConfig {
        url,
        quic: Some(RelayQuicConfig::default()),
    }
}

pub(super) fn gossip_topic(run_id: &str) -> TopicId {
    let mut hasher = Sha256::new();
    hasher.update(GOSSIP_TOPIC);
    hasher.update(run_id);
    TopicId::from_bytes(hasher.finalize().into())
}
