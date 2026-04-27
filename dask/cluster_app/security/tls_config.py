from __future__ import annotations

from pathlib import Path
from typing import Any

from cluster_app.security.ca import CertificateBundle


def dask_security(bundle: CertificateBundle) -> Any:
    try:
        from distributed.security import Security
    except ModuleNotFoundError as exc:
        raise RuntimeError("distributed is required to build Dask TLS Security objects.") from exc
    return Security(
        tls_ca_file=str(bundle.ca_cert),
        tls_client_cert=str(bundle.cert),
        tls_client_key=str(bundle.key),
        tls_scheduler_cert=str(bundle.cert),
        tls_scheduler_key=str(bundle.key),
        tls_worker_cert=str(bundle.cert),
        tls_worker_key=str(bundle.key),
        require_encryption=True,
    )


def dask_tls_env(bundle: CertificateBundle) -> dict[str, str]:
    return {
        "DASK_DISTRIBUTED__COMM__REQUIRE_ENCRYPTION": "true",
        "DASK_DISTRIBUTED__COMM__TLS__CA_FILE": str(Path(bundle.ca_cert)),
        "DASK_DISTRIBUTED__COMM__TLS__CLIENT__CERT": str(Path(bundle.cert)),
        "DASK_DISTRIBUTED__COMM__TLS__CLIENT__KEY": str(Path(bundle.key)),
        "DASK_DISTRIBUTED__COMM__TLS__SCHEDULER__CERT": str(Path(bundle.cert)),
        "DASK_DISTRIBUTED__COMM__TLS__SCHEDULER__KEY": str(Path(bundle.key)),
        "DASK_DISTRIBUTED__COMM__TLS__WORKER__CERT": str(Path(bundle.cert)),
        "DASK_DISTRIBUTED__COMM__TLS__WORKER__KEY": str(Path(bundle.key)),
    }

