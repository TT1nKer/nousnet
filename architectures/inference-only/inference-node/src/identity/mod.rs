use anyhow::{ensure, Result};
use iroh::{EndpointId, SecretKey};
use std::path::Path;
use zeroize::{Zeroize, Zeroizing};

#[cfg(unix)]
mod unix;
#[cfg(windows)]
mod windows;

#[cfg(unix)]
use unix::{protect_and_write, read_and_unprotect};
#[cfg(windows)]
use windows::{protect_and_write, read_and_unprotect};

pub fn create_identity(path: &Path) -> Result<EndpointId> {
    let secret_key = SecretKey::generate(&mut rand::rng());
    let endpoint_id = secret_key.public();
    let mut key_bytes = secret_key.to_bytes();
    let write_result = protect_and_write(path, &key_bytes);
    key_bytes.zeroize();
    write_result?;
    Ok(endpoint_id)
}

pub fn load_identity(path: &Path) -> Result<SecretKey> {
    let protected_bytes = read_and_unprotect(path)?;
    ensure!(
        protected_bytes.len() == 32,
        "identity must contain exactly 32 secret bytes"
    );
    let key_bytes = Zeroizing::new(
        <[u8; 32]>::try_from(protected_bytes.as_slice())
            .expect("identity length was checked above"),
    );
    Ok(SecretKey::from_bytes(&key_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_then_load_preserves_endpoint_id() {
        let directory = tempfile::tempdir().unwrap();
        let identity_file = directory.path().join("node.identity");
        let endpoint_id = create_identity(&identity_file).unwrap();
        assert_eq!(load_identity(&identity_file).unwrap().public(), endpoint_id);
    }

    #[test]
    fn create_refuses_to_overwrite_existing_identity() {
        let directory = tempfile::tempdir().unwrap();
        let identity_file = directory.path().join("node.identity");
        create_identity(&identity_file).unwrap();
        assert!(create_identity(&identity_file).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn unix_identity_has_private_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let identity_file = directory.path().join("node.identity");
        create_identity(&identity_file).unwrap();

        let mode = std::fs::metadata(identity_file)
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600);
    }

    #[cfg(unix)]
    #[test]
    fn unix_load_rejects_group_or_world_access() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let identity_file = directory.path().join("node.identity");
        create_identity(&identity_file).unwrap();
        std::fs::set_permissions(&identity_file, std::fs::Permissions::from_mode(0o644)).unwrap();

        assert!(load_identity(&identity_file).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn unix_load_rejects_symbolic_links() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let identity_file = directory.path().join("node.identity");
        let identity_link = directory.path().join("identity.link");
        create_identity(&identity_file).unwrap();
        symlink(&identity_file, &identity_link).unwrap();

        assert!(load_identity(&identity_link).is_err());
    }

    #[cfg(windows)]
    #[test]
    fn windows_identity_is_dpapi_encrypted() {
        let directory = tempfile::tempdir().unwrap();
        let identity_file = directory.path().join("node.identity");
        create_identity(&identity_file).unwrap();
        let key = load_identity(&identity_file).unwrap();
        let ciphertext = std::fs::read(identity_file).unwrap();

        assert!(!ciphertext
            .windows(key.to_bytes().len())
            .any(|window| window == key.to_bytes()));
    }
}
