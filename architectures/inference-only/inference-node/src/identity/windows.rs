use anyhow::{ensure, Context, Result};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::Path,
    ptr::{null, null_mut},
};
use windows_sys::Win32::{
    Foundation::LocalFree,
    Security::Cryptography::{
        CryptProtectData, CryptUnprotectData, CRYPTPROTECT_UI_FORBIDDEN, CRYPT_INTEGER_BLOB,
    },
};
use zeroize::Zeroizing;

pub(super) fn protect_and_write(path: &Path, key_bytes: &[u8; 32]) -> Result<()> {
    let input = CRYPT_INTEGER_BLOB {
        cbData: key_bytes.len() as u32,
        pbData: key_bytes.as_ptr().cast_mut(),
    };
    let mut output = CRYPT_INTEGER_BLOB::default();
    let protected = unsafe {
        CryptProtectData(
            &input,
            null(),
            null(),
            null(),
            null(),
            CRYPTPROTECT_UI_FORBIDDEN,
            &mut output,
        )
    };
    if protected == 0 {
        return Err(std::io::Error::last_os_error()).context("DPAPI encryption failed");
    }

    let ciphertext =
        unsafe { std::slice::from_raw_parts(output.pbData, output.cbData as usize) }.to_vec();
    let free_result = unsafe { LocalFree(output.pbData.cast()) };
    ensure!(free_result.is_null(), "failed to release DPAPI output");

    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("failed to create identity file {}", path.display()))?;
    if let Err(error) = (|| -> Result<()> {
        file.write_all(&ciphertext)
            .context("failed to write encrypted identity")?;
        file.sync_all()
            .context("failed to sync encrypted identity")?;
        Ok(())
    })() {
        drop(file);
        let _ = fs::remove_file(path);
        return Err(error);
    }
    Ok(())
}

pub(super) fn read_and_unprotect(path: &Path) -> Result<Zeroizing<Vec<u8>>> {
    const MAX_ENCRYPTED_IDENTITY_BYTES: u64 = 16 * 1024;

    let ciphertext = fs::read(path)
        .with_context(|| format!("failed to read identity file {}", path.display()))?;
    ensure!(
        !ciphertext.is_empty() && ciphertext.len() as u64 <= MAX_ENCRYPTED_IDENTITY_BYTES,
        "encrypted identity has an invalid size"
    );
    let input = CRYPT_INTEGER_BLOB {
        cbData: ciphertext.len() as u32,
        pbData: ciphertext.as_ptr().cast_mut(),
    };
    let mut output = CRYPT_INTEGER_BLOB::default();
    let unprotected = unsafe {
        CryptUnprotectData(
            &input,
            null_mut(),
            null(),
            null(),
            null(),
            CRYPTPROTECT_UI_FORBIDDEN,
            &mut output,
        )
    };
    if unprotected == 0 {
        return Err(std::io::Error::last_os_error()).context("DPAPI decryption failed");
    }

    let plaintext = Zeroizing::new(
        unsafe { std::slice::from_raw_parts(output.pbData, output.cbData as usize) }.to_vec(),
    );
    unsafe {
        std::ptr::write_bytes(output.pbData, 0, output.cbData as usize);
    }
    let free_result = unsafe { LocalFree(output.pbData.cast()) };
    ensure!(free_result.is_null(), "failed to release DPAPI plaintext");
    Ok(plaintext)
}
