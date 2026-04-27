from __future__ import annotations

from dataclasses import dataclass

from cluster_app.security.ca import CertificateBundle
from cluster_app.security.tls_config import dask_security


@dataclass(frozen=True, slots=True)
class DaskClientFactory:
    scheduler_address: str
    bundle: CertificateBundle

    def connect(self):
        try:
            from distributed import Client
        except ModuleNotFoundError as exc:
            raise RuntimeError("distributed is required to connect to Dask.") from exc
        return Client(self.scheduler_address, security=dask_security(self.bundle))

