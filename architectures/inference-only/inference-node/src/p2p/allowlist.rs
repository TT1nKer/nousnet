use iroh::{
    endpoint::{AfterHandshakeOutcome, ConnectionInfo, EndpointHooks},
    EndpointId,
};
use std::{
    collections::HashSet,
    sync::{Arc, RwLock},
};

#[derive(Clone, Debug)]
pub struct PeerAllowlist {
    allow_all: bool,
    allowed_endpoint_ids: Arc<RwLock<HashSet<EndpointId>>>,
}

impl PeerAllowlist {
    pub fn allow_all() -> Self {
        Self {
            allow_all: true,
            allowed_endpoint_ids: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    pub fn with_nodes(endpoint_ids: impl IntoIterator<Item = EndpointId>) -> Self {
        Self {
            allow_all: false,
            allowed_endpoint_ids: Arc::new(RwLock::new(endpoint_ids.into_iter().collect())),
        }
    }

    pub fn add(&self, endpoint_id: EndpointId) {
        self.allowed_endpoint_ids
            .write()
            .unwrap_or_else(|error| error.into_inner())
            .insert(endpoint_id);
    }

    pub(crate) fn allowed(&self, endpoint_id: EndpointId) -> bool {
        self.allow_all
            || self
                .allowed_endpoint_ids
                .read()
                .unwrap_or_else(|error| error.into_inner())
                .contains(&endpoint_id)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct AllowlistHook {
    allowlist: PeerAllowlist,
}

impl AllowlistHook {
    pub fn new(allowlist: PeerAllowlist) -> Self {
        Self { allowlist }
    }
}

impl EndpointHooks for AllowlistHook {
    async fn after_handshake(&self, connection: &ConnectionInfo) -> AfterHandshakeOutcome {
        if self.allowlist.allowed(connection.remote_id()) {
            AfterHandshakeOutcome::Accept
        } else {
            AfterHandshakeOutcome::Reject {
                error_code: 1u32.into(),
                reason: b"not in inference allowlist".to_vec(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iroh::SecretKey;

    fn endpoint_id(seed: u8) -> EndpointId {
        SecretKey::from_bytes(&[seed; 32]).public()
    }

    #[test]
    fn dynamic_allowlist_rejects_unknown_peers() {
        let allowed = endpoint_id(1);
        let added_later = endpoint_id(2);
        let denied = endpoint_id(3);
        let allowlist = PeerAllowlist::with_nodes([allowed]);

        assert!(allowlist.allowed(allowed));
        assert!(!allowlist.allowed(denied));
        allowlist.add(added_later);
        assert!(allowlist.allowed(added_later));
    }
}
