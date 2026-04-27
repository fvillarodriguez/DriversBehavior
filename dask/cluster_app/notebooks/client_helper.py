from __future__ import annotations

import os
from pathlib import Path


def connect(scheduler_address: str | None = None, certs_dir: str | Path | None = None):
    """Connect notebook code to the managed Dask scheduler using app TLS files."""
    try:
        from distributed import Client
    except ModuleNotFoundError as exc:
        raise RuntimeError("distributed is required for notebook connections.") from exc
    from cluster_app.security.ca import CertificateBundle
    from cluster_app.security.tls_config import dask_security

    address = scheduler_address or os.environ.get("DASK_SCHEDULER_ADDRESS")
    if not address:
        raise ValueError("scheduler_address or DASK_SCHEDULER_ADDRESS is required")
    cert_root = Path(certs_dir or os.environ.get("CLUSTER_APP_CERTS_DIR", "~/.dask-cluster-app/certs")).expanduser()
    node_id = os.environ.get("CLUSTER_APP_NODE_ID", "client")
    bundle = CertificateBundle(
        ca_cert=cert_root / "ca.pem",
        cert=cert_root / f"{node_id}.pem",
        key=cert_root / f"{node_id}-key.pem",
        fingerprint="",
    )
    return Client(address, security=dask_security(bundle))

