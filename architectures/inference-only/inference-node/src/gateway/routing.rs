use anyhow::{ensure, Context, Result};
use iroh::EndpointId;
use std::{
    collections::{hash_map::Entry, HashMap, HashSet},
    fs::File,
    io::Read,
    path::Path,
    time::{Duration, Instant},
};

const STALE_AFTER: Duration = Duration::from_secs(90);
const MAX_ALLOWLIST_BYTES: u64 = 1024 * 1024;

pub fn load_endpoint_allowlist(path: &Path) -> Result<HashSet<EndpointId>> {
    let file = File::open(path)
        .with_context(|| format!("failed to open gateway allowlist {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("failed to inspect gateway allowlist {}", path.display()))?;
    ensure!(
        metadata.is_file(),
        "gateway allowlist must be a regular file"
    );
    ensure!(
        metadata.len() <= MAX_ALLOWLIST_BYTES,
        "gateway allowlist exceeds 1 MiB"
    );

    let mut contents = Vec::new();
    file.take(MAX_ALLOWLIST_BYTES + 1)
        .read_to_end(&mut contents)
        .with_context(|| format!("failed to read gateway allowlist {}", path.display()))?;
    ensure!(
        contents.len() as u64 <= MAX_ALLOWLIST_BYTES,
        "gateway allowlist exceeds 1 MiB"
    );
    let encoded_ids: Vec<String> = serde_json::from_slice(&contents)
        .context("gateway allowlist must be a JSON string array")?;
    ensure!(
        !encoded_ids.is_empty(),
        "gateway endpoint allowlist must not be empty"
    );

    encoded_ids
        .into_iter()
        .enumerate()
        .map(|(index, encoded_id)| {
            encoded_id
                .parse()
                .with_context(|| format!("invalid endpoint ID at allowlist index {index}"))
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct NodeRecord {
    pub endpoint_id: EndpointId,
    pub model_name: Option<String>,
    pub last_seen: Instant,
}

#[derive(Clone, Debug)]
struct CatalogEntry {
    record: NodeRecord,
    last_selected: Option<u128>,
}

#[derive(Debug)]
pub struct NodeCatalog {
    allowed_endpoint_ids: HashSet<EndpointId>,
    entries: HashMap<EndpointId, CatalogEntry>,
    selection_counter: u128,
}

impl NodeCatalog {
    pub fn new(allowed_endpoint_ids: HashSet<EndpointId>) -> Result<Self> {
        ensure!(
            !allowed_endpoint_ids.is_empty(),
            "gateway endpoint allowlist must not be empty"
        );
        Ok(Self {
            allowed_endpoint_ids,
            entries: HashMap::new(),
            selection_counter: 0,
        })
    }

    pub fn upsert(&mut self, record: NodeRecord) -> bool {
        if !self.allowed_endpoint_ids.contains(&record.endpoint_id) {
            return false;
        }

        match self.entries.entry(record.endpoint_id) {
            Entry::Occupied(mut entry) => entry.get_mut().record = record,
            Entry::Vacant(entry) => {
                entry.insert(CatalogEntry {
                    record,
                    last_selected: None,
                });
            }
        }
        true
    }

    pub fn remove(&mut self, endpoint_id: &EndpointId) -> Option<NodeRecord> {
        self.entries.remove(endpoint_id).map(|entry| entry.record)
    }

    pub fn remove_stale(&mut self, now: Instant) -> Vec<EndpointId> {
        let mut removed: Vec<_> = self
            .entries
            .iter()
            .filter_map(|(endpoint_id, entry)| {
                (now.saturating_duration_since(entry.record.last_seen) > STALE_AFTER)
                    .then_some(*endpoint_id)
            })
            .collect();
        removed.sort_by_key(|endpoint_id| *endpoint_id.as_bytes());
        for endpoint_id in &removed {
            self.entries.remove(endpoint_id);
        }
        removed
    }

    pub fn select(&mut self, model_name: &str, now: Instant) -> Option<EndpointId> {
        let selected = self
            .entries
            .iter()
            .filter(|(endpoint_id, entry)| {
                self.allowed_endpoint_ids.contains(*endpoint_id)
                    && now.saturating_duration_since(entry.record.last_seen) <= STALE_AFTER
                    && entry.record.model_name.as_deref() == Some(model_name)
            })
            .min_by_key(|(endpoint_id, entry)| {
                (entry.last_selected.unwrap_or(0), *endpoint_id.as_bytes())
            })
            .map(|(endpoint_id, _)| *endpoint_id)?;

        self.selection_counter += 1;
        self.entries
            .get_mut(&selected)
            .expect("selected entry must still exist")
            .last_selected = Some(self.selection_counter);
        Some(selected)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iroh::SecretKey;
    use std::{collections::HashSet, fs, time::Duration};

    fn endpoint_id(seed: u8) -> EndpointId {
        SecretKey::from_bytes(&[seed; 32]).public()
    }

    fn node(endpoint_id: EndpointId, model_name: &str, last_seen: Instant) -> NodeRecord {
        NodeRecord {
            endpoint_id,
            model_name: Some(model_name.to_string()),
            last_seen,
        }
    }

    #[test]
    fn selects_only_allowed_exact_model_matches() {
        let now = Instant::now();
        let allowed = endpoint_id(1);
        let denied = endpoint_id(2);
        let mut catalog = NodeCatalog::new(HashSet::from([allowed])).unwrap();
        catalog.upsert(node(denied, "qwen3:8b", now));
        catalog.upsert(node(allowed, "llama3:8b", now));
        assert_eq!(catalog.select("qwen3:8b", now), None);

        catalog.upsert(node(allowed, "qwen3:8b", now));
        assert_eq!(catalog.select("qwen3:8b", now), Some(allowed));
    }

    #[test]
    fn rejects_empty_allowlist() {
        assert!(NodeCatalog::new(HashSet::new()).is_err());
    }

    #[test]
    fn removes_nodes_older_than_ninety_seconds() {
        let first_seen = Instant::now();
        let allowed = endpoint_id(1);
        let mut catalog = NodeCatalog::new(HashSet::from([allowed])).unwrap();
        catalog.upsert(node(allowed, "qwen3:8b", first_seen));

        let exactly_fresh = first_seen + Duration::from_secs(90);
        assert_eq!(catalog.select("qwen3:8b", exactly_fresh), Some(allowed));

        let removed = catalog.remove_stale(first_seen + Duration::from_secs(91));
        assert_eq!(removed, vec![allowed]);
        assert_eq!(catalog.select("qwen3:8b", exactly_fresh), None);
    }

    #[test]
    fn round_robin_is_deterministic() {
        let now = Instant::now();
        let candidates = [endpoint_id(1), endpoint_id(2)];
        let (first, second) = if candidates[0].as_bytes() < candidates[1].as_bytes() {
            (candidates[0], candidates[1])
        } else {
            (candidates[1], candidates[0])
        };
        let mut catalog = NodeCatalog::new(HashSet::from([first, second])).unwrap();
        catalog.upsert(node(second, "qwen3:8b", now));
        catalog.upsert(node(first, "qwen3:8b", now));

        assert_eq!(catalog.select("qwen3:8b", now), Some(first));
        assert_eq!(catalog.select("qwen3:8b", now), Some(second));
        assert_eq!(catalog.select("qwen3:8b", now), Some(first));
    }

    #[test]
    fn loads_only_nonempty_valid_endpoint_id_arrays() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("allowed-peers.json");
        let allowed = endpoint_id(1);
        fs::write(
            &path,
            serde_json::to_vec(&vec![allowed.to_string()]).unwrap(),
        )
        .unwrap();
        assert_eq!(
            load_endpoint_allowlist(&path).unwrap(),
            HashSet::from([allowed])
        );

        fs::write(&path, "[]").unwrap();
        assert!(load_endpoint_allowlist(&path).is_err());
        fs::write(&path, r#"["not-an-endpoint-id"]"#).unwrap();
        assert!(load_endpoint_allowlist(&path).is_err());
    }
}
