use anyhow::{ensure, Context, Result};
use std::{
    fs::{self, OpenOptions, Permissions},
    io::{Read, Write},
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
    path::Path,
};
use zeroize::Zeroizing;

pub(super) fn protect_and_write(path: &Path, key_bytes: &[u8; 32]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
        .with_context(|| format!("failed to create identity file {}", path.display()))?;
    if let Err(error) = (|| -> Result<()> {
        file.set_permissions(Permissions::from_mode(0o600))
            .context("failed to restrict identity file permissions")?;
        file.write_all(key_bytes)
            .context("failed to write identity file")?;
        file.sync_all().context("failed to sync identity file")?;
        Ok(())
    })() {
        drop(file);
        let _ = fs::remove_file(path);
        return Err(error);
    }
    Ok(())
}

pub(super) fn read_and_unprotect(path: &Path) -> Result<Zeroizing<Vec<u8>>> {
    let path_metadata = fs::symlink_metadata(path)
        .with_context(|| format!("failed to inspect identity file {}", path.display()))?;
    ensure!(
        path_metadata.file_type().is_file(),
        "identity path must be a regular file"
    );

    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .with_context(|| format!("failed to open identity file {}", path.display()))?;
    let metadata = file
        .metadata()
        .context("failed to inspect opened identity file")?;
    ensure!(metadata.is_file(), "identity must be a regular file");
    ensure!(
        metadata.permissions().mode() & 0o077 == 0,
        "identity file must not be accessible by group or other users"
    );

    let mut bytes = Zeroizing::new(Vec::with_capacity(33));
    file.take(33)
        .read_to_end(&mut bytes)
        .context("failed to read identity file")?;
    Ok(bytes)
}
