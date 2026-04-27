from __future__ import annotations

from pathlib import Path

from cluster_app.security.ca import CertificateAuthority, CertificateBundle


def ensure_node_cert(
    certs_dir: str | Path,
    node_id: str,
    hosts: list[str],
    valid_days: int,
) -> CertificateBundle:
    ca = CertificateAuthority(certs_dir)
    ca.ensure()
    cert_path = Path(certs_dir) / f"{node_id}.pem"
    key_path = Path(certs_dir) / f"{node_id}-key.pem"
    if cert_path.exists() and key_path.exists():
        # Re-issue is deliberately explicit so accidental startup does not rotate certs.
        return CertificateBundle(ca.ca_cert_path, cert_path, key_path, fingerprint_file(cert_path))
    return ca.issue_node_certificate(node_id, hosts, valid_days)


def fingerprint_file(cert_path: str | Path) -> str:
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes
    except ModuleNotFoundError as exc:
        raise RuntimeError("cryptography is required to read certificate fingerprints.") from exc
    cert = x509.load_pem_x509_certificate(Path(cert_path).read_bytes())
    return cert.fingerprint(hashes.SHA256()).hex()

