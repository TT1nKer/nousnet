mod allowlist;
mod config;
mod signed_message;

pub use allowlist::PeerAllowlist;
pub use config::{DiscoveryMode, RelayKind};

use allowlist::AllowlistHook;
use anyhow::{Context, Result};
use config::{gossip_topic, relay_mode};
use futures_util::StreamExt;
use iroh::{
    address_lookup::{dns::DnsAddressLookup, memory::MemoryLookup, pkarr::PkarrPublisher},
    endpoint::QuicTransportConfig,
    protocol::{ProtocolHandler, Router},
    Endpoint, EndpointAddr, EndpointId, SecretKey,
};
use iroh_gossip::{
    api::{Event, GossipReceiver, GossipSender},
    net::Gossip,
    proto::{HyparviewConfig, PlumtreeConfig},
};
use serde::{de::DeserializeOwned, Serialize};
use signed_message::SignedMessage;
use std::{fmt::Debug, sync::Arc, time::Duration};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, warn};

pub trait NetworkMessage: Serialize + DeserializeOwned + Debug + Send + Sync + 'static {}

impl<T> NetworkMessage for T where T: Serialize + DeserializeOwned + Debug + Send + Sync + 'static {}

pub struct InferenceNetwork<M: NetworkMessage> {
    endpoint: Endpoint,
    router: Arc<Router>,
    gossip_sender: GossipSender,
    gossip_receiver: GossipReceiver,
    memory_lookup: MemoryLookup,
    allowlist: PeerAllowlist,
    message: std::marker::PhantomData<M>,
}

impl<M: NetworkMessage> InferenceNetwork<M> {
    pub async fn init(
        run_id: &str,
        discovery_mode: DiscoveryMode,
        relay_kind: RelayKind,
        bootstrap_peers: Vec<EndpointAddr>,
        secret_key: Option<SecretKey>,
        allowlist: PeerAllowlist,
        cancel: Option<CancellationToken>,
    ) -> Result<Self> {
        Self::init_internal::<Gossip>(
            run_id,
            discovery_mode,
            relay_kind,
            bootstrap_peers,
            secret_key,
            allowlist,
            cancel,
            None,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn init_with_protocol<P>(
        run_id: &str,
        discovery_mode: DiscoveryMode,
        relay_kind: RelayKind,
        bootstrap_peers: Vec<EndpointAddr>,
        secret_key: Option<SecretKey>,
        allowlist: PeerAllowlist,
        cancel: Option<CancellationToken>,
        protocol: (&'static [u8], P),
    ) -> Result<Self>
    where
        P: ProtocolHandler + Clone,
    {
        Self::init_internal(
            run_id,
            discovery_mode,
            relay_kind,
            bootstrap_peers,
            secret_key,
            allowlist,
            cancel,
            Some(protocol),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn init_internal<P>(
        run_id: &str,
        discovery_mode: DiscoveryMode,
        relay_kind: RelayKind,
        bootstrap_peers: Vec<EndpointAddr>,
        secret_key: Option<SecretKey>,
        allowlist: PeerAllowlist,
        cancel: Option<CancellationToken>,
        protocol: Option<(&'static [u8], P)>,
    ) -> Result<Self>
    where
        P: ProtocolHandler + Clone,
    {
        let secret_key = secret_key.unwrap_or_else(|| SecretKey::generate(&mut rand::rng()));
        let memory_lookup = MemoryLookup::from_endpoint_info(bootstrap_peers.iter().cloned());
        let transport_config = QuicTransportConfig::builder()
            .max_idle_timeout(Some(Duration::from_secs(120).try_into()?))
            .keep_alive_interval(Duration::from_secs(5))
            .set_max_remote_nat_traversal_addresses(12)
            .build();
        let mut endpoint_builder = Endpoint::builder(iroh::endpoint::presets::N0)
            .secret_key(secret_key)
            .relay_mode(relay_mode(relay_kind))
            .transport_config(transport_config)
            .clear_address_lookup()
            .address_lookup(memory_lookup.clone())
            .hooks(AllowlistHook::new(allowlist.clone()));

        if matches!(discovery_mode, DiscoveryMode::N0) {
            endpoint_builder = endpoint_builder
                .address_lookup(DnsAddressLookup::n0_dns().build())
                .address_lookup(PkarrPublisher::n0_dns());
        }
        let endpoint = endpoint_builder
            .bind()
            .await
            .context("failed to bind inference endpoint")?;

        if matches!(discovery_mode, DiscoveryMode::N0) {
            if let Some(cancel) = cancel {
                tokio::select! {
                    _ = endpoint.online() => {}
                    _ = cancel.cancelled() => anyhow::bail!("inference network startup cancelled"),
                }
            } else {
                endpoint.online().await;
            }
        }

        let gossip = Gossip::builder()
            .max_message_size(4096)
            .membership_config(HyparviewConfig {
                active_view_capacity: 8,
                shuffle_interval: Duration::from_secs(30),
                neighbor_request_timeout: Duration::from_secs(2),
                ..HyparviewConfig::default()
            })
            .broadcast_config(PlumtreeConfig {
                graft_timeout_2: Duration::from_millis(200),
                message_cache_retention: Duration::from_secs(60),
                message_id_retention: Duration::from_secs(120),
                ..PlumtreeConfig::default()
            })
            .spawn(endpoint.clone());
        let mut router_builder =
            Router::builder(endpoint.clone()).accept(iroh_gossip::ALPN, gossip.clone());
        if let Some((alpn, handler)) = protocol {
            router_builder = router_builder.accept(alpn, handler);
        }
        let router = Arc::new(router_builder.spawn());
        let bootstrap_ids = bootstrap_peers.iter().map(|peer| peer.id).collect();
        let (gossip_sender, gossip_receiver) = gossip
            .subscribe(gossip_topic(run_id), bootstrap_ids)
            .await
            .context("failed to subscribe to inference gossip")?
            .split();

        Ok(Self {
            endpoint,
            router,
            gossip_sender,
            gossip_receiver,
            memory_lookup,
            allowlist,
            message: std::marker::PhantomData,
        })
    }

    pub fn endpoint_id(&self) -> EndpointId {
        self.endpoint.id()
    }

    pub fn endpoint(&self) -> Endpoint {
        self.endpoint.clone()
    }

    pub fn endpoint_addr(&self) -> EndpointAddr {
        self.endpoint.addr()
    }

    pub fn broadcast(&self, message: &M) -> Result<()> {
        let encoded = SignedMessage::sign_and_encode(self.endpoint.secret_key(), message)?;
        let sender = self.gossip_sender.clone();
        tokio::spawn(async move {
            if let Err(error) = sender.broadcast(encoded).await {
                error!("failed to broadcast inference gossip: {error}");
            }
        });
        Ok(())
    }

    pub fn add_peer(&self, peer: EndpointAddr) {
        let peer_id = peer.id;
        self.memory_lookup.add_endpoint_info(peer);
        let sender = self.gossip_sender.clone();
        tokio::spawn(async move {
            if let Err(error) = sender.join_peers(vec![peer_id]).await {
                warn!(
                    "failed to join inference peer {}: {error}",
                    peer_id.fmt_short()
                );
            }
        });
    }

    pub async fn poll_next(&mut self) -> Result<Option<(EndpointId, M)>> {
        let Some(event) = self.gossip_receiver.next().await else {
            return Ok(None);
        };
        let event = event.context("inference gossip stream failed")?;
        match event {
            Event::Received(message) => {
                let (origin, decoded) = SignedMessage::verify_and_decode(&message.content)
                    .context("invalid signed inference gossip")?;
                if !self.allowlist.allowed(origin) {
                    warn!(
                        "ignoring gossip signed by unauthorized peer {}",
                        origin.fmt_short()
                    );
                    return Ok(None);
                }
                Ok(Some((origin, decoded)))
            }
            Event::NeighborUp(endpoint_id) => {
                debug!(
                    "inference gossip neighbor connected: {}",
                    endpoint_id.fmt_short()
                );
                Ok(None)
            }
            Event::NeighborDown(endpoint_id) => {
                debug!(
                    "inference gossip neighbor disconnected: {}",
                    endpoint_id.fmt_short()
                );
                Ok(None)
            }
            Event::Lagged => {
                warn!("inference gossip lagged and dropped messages");
                Ok(None)
            }
        }
    }

    pub async fn shutdown(&self) -> Result<()> {
        self.router
            .shutdown()
            .await
            .context("failed to shut down inference network")
    }
}
