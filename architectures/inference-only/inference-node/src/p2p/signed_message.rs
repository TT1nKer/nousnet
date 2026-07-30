use super::NetworkMessage;
use anyhow::Result;
use bytes::Bytes;
use iroh::{EndpointId, SecretKey, Signature};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

#[derive(Debug, Serialize, Deserialize)]
struct SignedMessage<M: NetworkMessage> {
    from: EndpointId,
    data: Bytes,
    signature: Signature,
    message: PhantomData<M>,
}

impl<M: NetworkMessage> SignedMessage<M> {
    pub fn sign_and_encode(secret_key: &SecretKey, message: &M) -> Result<Bytes> {
        let data: Bytes = postcard::to_stdvec(message)?.into();
        let encoded = Self {
            from: secret_key.public(),
            signature: secret_key.sign(&data),
            data,
            message: PhantomData,
        };
        Ok(postcard::to_stdvec(&encoded)?.into())
    }

    pub fn verify_and_decode(encoded: &[u8]) -> Result<(EndpointId, M)> {
        let signed: Self = postcard::from_bytes(encoded)?;
        signed.from.verify(&signed.data, &signed.signature)?;
        let message = postcard::from_bytes(&signed.data)?;
        Ok((signed.from, message))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Deserialize, PartialEq, Serialize)]
    struct TestMessage {
        value: u8,
    }

    #[test]
    fn verifies_origin_and_rejects_tampering() {
        let key = SecretKey::from_bytes(&[7; 32]);
        let encoded = SignedMessage::sign_and_encode(&key, &TestMessage { value: 42 }).unwrap();
        let (origin, decoded) = SignedMessage::<TestMessage>::verify_and_decode(&encoded).unwrap();
        assert_eq!(origin, key.public());
        assert_eq!(decoded, TestMessage { value: 42 });

        let mut tampered = encoded.to_vec();
        let last = tampered.last_mut().unwrap();
        *last ^= 1;
        assert!(SignedMessage::<TestMessage>::verify_and_decode(&tampered).is_err());
    }
}
