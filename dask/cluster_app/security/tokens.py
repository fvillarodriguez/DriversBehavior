from __future__ import annotations

import base64
import hmac
import secrets
from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True, slots=True)
class PairingToken:
    token: str
    digest: str


class PairingTokenService:
    """Creates and verifies node pairing tokens without storing plaintext tokens."""

    def __init__(self, secret: str):
        if not secret:
            raise ValueError("A non-empty cluster secret is required")
        self.secret = secret.encode("utf-8")

    def issue(self) -> PairingToken:
        raw = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii").rstrip("=")
        return PairingToken(token=raw, digest=self.digest(raw))

    def digest(self, token: str) -> str:
        return hmac.new(self.secret, token.encode("utf-8"), sha256).hexdigest()

    def verify(self, token: str, expected_digest: str) -> bool:
        return hmac.compare_digest(self.digest(token), expected_digest)

