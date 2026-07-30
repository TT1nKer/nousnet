use anyhow::{ensure, Result};
use axum::http::HeaderValue;
use subtle::ConstantTimeEq;
use zeroize::Zeroizing;

#[derive(Clone)]
pub struct ApiKeyAuthenticator {
    secret: Zeroizing<Vec<u8>>,
}

impl ApiKeyAuthenticator {
    pub fn from_secret(secret: &str) -> Result<Self> {
        ensure!(
            !secret.is_empty() && secret.as_bytes().iter().all(|byte| byte.is_ascii_graphic()),
            "gateway API key must contain only visible ASCII characters"
        );
        Ok(Self {
            secret: Zeroizing::new(secret.as_bytes().to_vec()),
        })
    }

    pub fn authorize(&self, authorization: &HeaderValue) -> bool {
        let Ok(authorization) = authorization.to_str() else {
            return false;
        };
        let Some((scheme, candidate)) = authorization.split_once(' ') else {
            return false;
        };
        if !scheme.eq_ignore_ascii_case("Bearer")
            || candidate.is_empty()
            || candidate.bytes().any(|byte| byte.is_ascii_whitespace())
            || candidate.len() != self.secret.len()
        {
            return false;
        }

        bool::from(self.secret.as_slice().ct_eq(candidate.as_bytes()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    #[test]
    fn accepts_only_the_exact_bearer_key() {
        let auth = ApiKeyAuthenticator::from_secret("gateway-secret").unwrap();
        assert!(auth.authorize(&HeaderValue::from_static("Bearer gateway-secret")));
        assert!(auth.authorize(&HeaderValue::from_static("bearer gateway-secret")));
        assert!(!auth.authorize(&HeaderValue::from_static("Bearer gateway-secreu")));
        assert!(!auth.authorize(&HeaderValue::from_static("gateway-secret")));
        assert!(!auth.authorize(&HeaderValue::from_static("Bearer  gateway-secret")));
    }

    #[test]
    fn rejects_empty_gateway_key() {
        assert!(ApiKeyAuthenticator::from_secret("  ").is_err());
        assert!(ApiKeyAuthenticator::from_secret("gateway secret").is_err());
        assert!(ApiKeyAuthenticator::from_secret("网关密钥").is_err());
    }
}
