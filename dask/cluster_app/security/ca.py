from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CertificateBundle:
    ca_cert: Path
    cert: Path
    key: Path
    fingerprint: str


def _crypto():
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "cryptography is required for TLS certificate generation. Install the project venv."
        ) from exc
    return x509, hashes, serialization, rsa, NameOID


class CertificateAuthority:
    def __init__(self, certs_dir: str | Path, common_name: str = "Dask Cluster App CA"):
        self.certs_dir = Path(certs_dir)
        self.certs_dir.mkdir(parents=True, exist_ok=True)
        self.common_name = common_name
        self.ca_cert_path = self.certs_dir / "ca.pem"
        self.ca_key_path = self.certs_dir / "ca-key.pem"

    def ensure(self, valid_days: int = 3650) -> tuple[Path, Path]:
        if self.ca_cert_path.exists() and self.ca_key_path.exists():
            return self.ca_cert_path, self.ca_key_path
        x509, hashes, serialization, rsa, NameOID = _crypto()
        key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
        subject = issuer = x509.Name(
            [
                x509.NameAttribute(NameOID.COMMON_NAME, self.common_name),
            ]
        )
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.now(UTC) - timedelta(minutes=1))
            .not_valid_after(datetime.now(UTC) + timedelta(days=valid_days))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_cert_sign=True,
                    crl_sign=True,
                    key_encipherment=False,
                    content_commitment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .sign(key, hashes.SHA256())
        )
        self.ca_key_path.write_bytes(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        self.ca_cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        return self.ca_cert_path, self.ca_key_path

    def issue_node_certificate(
        self,
        node_id: str,
        hosts: list[str],
        valid_days: int = 180,
    ) -> CertificateBundle:
        x509, hashes, serialization, rsa, NameOID = _crypto()
        self.ensure()
        ca_key = serialization.load_pem_private_key(self.ca_key_path.read_bytes(), password=None)
        ca_cert = x509.load_pem_x509_certificate(self.ca_cert_path.read_bytes())
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, node_id)])
        alt_names = [x509.DNSName(host) for host in hosts if host and not _is_ip(host)]
        alt_names.extend(x509.IPAddress(_to_ip(host)) for host in hosts if host and _is_ip(host))
        builder = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(ca_cert.subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.now(UTC) - timedelta(minutes=1))
            .not_valid_after(datetime.now(UTC) + timedelta(days=valid_days))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        )
        if alt_names:
            builder = builder.add_extension(x509.SubjectAlternativeName(alt_names), critical=False)
        cert = builder.sign(ca_key, hashes.SHA256())
        cert_path = self.certs_dir / f"{node_id}.pem"
        key_path = self.certs_dir / f"{node_id}-key.pem"
        key_path.write_bytes(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        fingerprint = cert.fingerprint(hashes.SHA256()).hex()
        return CertificateBundle(self.ca_cert_path, cert_path, key_path, fingerprint)


def _is_ip(value: str) -> bool:
    try:
        _to_ip(value)
    except ValueError:
        return False
    return True


def _to_ip(value: str):
    import ipaddress

    return ipaddress.ip_address(value)

