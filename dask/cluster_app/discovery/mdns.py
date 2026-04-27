from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DiscoveredService:
    name: str
    host: str
    port: int
    properties: dict[str, str]


def _zeroconf():
    try:
        from zeroconf import ServiceBrowser, ServiceInfo, Zeroconf
    except ModuleNotFoundError as exc:
        raise RuntimeError("zeroconf is required for mDNS discovery. Install the project venv.") from exc
    return Zeroconf, ServiceInfo, ServiceBrowser


class MdnsPublisher:
    def __init__(
        self,
        service_type: str,
        name: str,
        host: str,
        port: int,
        properties: dict[str, str] | None = None,
    ):
        Zeroconf, ServiceInfo, _ = _zeroconf()
        self.zeroconf = Zeroconf()
        self.info = ServiceInfo(
            service_type,
            f"{name}.{service_type}",
            addresses=[socket.inet_aton(host)],
            port=port,
            properties={k: v.encode("utf-8") for k, v in (properties or {}).items()},
            server=f"{name}.local.",
        )

    def start(self) -> None:
        try:
            self.zeroconf.register_service(self.info)
        except Exception as e:
            # Handle service name conflicts by trying a modified name
            Zeroconf, ServiceInfo, _ = _zeroconf()
            from zeroconf import NonUniqueNameException
            if isinstance(e, NonUniqueNameException):
                # Try appending a counter to make the name unique
                base_name = self.info.name
                counter = 1
                while counter < 10:  # Try up to 10 times
                    try:
                        modified_info = ServiceInfo(
                            self.info.type,
                            f"{base_name[:-1]}{counter}.{self.info.type}",
                            addresses=self.info.addresses,
                            port=self.info.port,
                            properties=self.info.properties,
                            server=self.info.server,
                        )
                        self.zeroconf.register_service(modified_info)
                        self.info = modified_info  # Update our info to the registered one
                        return
                    except NonUniqueNameException:
                        counter += 1
                        continue
                    except Exception:
                        # If it's not a name conflict, re-raise
                        raise
            # If we get here, either it wasn't a name conflict or we exhausted retries
            raise

    def close(self) -> None:
        self.zeroconf.unregister_service(self.info)
        self.zeroconf.close()


def discover(service_type: str, timeout: float = 3.0) -> list[DiscoveredService]:
    Zeroconf, _, ServiceBrowser = _zeroconf()
    zc = Zeroconf()
    listener = _Listener(zc)
    browser = None
    try:
        browser = ServiceBrowser(zc, service_type, listener)
        time.sleep(timeout)
        return listener.services
    finally:
        try:
            if browser is not None:
                browser.cancel()
        finally:
            zc.close()


class _Listener:
    def __init__(self, zeroconf: Any):
        self.zeroconf = zeroconf
        self.services: list[DiscoveredService] = []

    def add_service(self, zc: Any, type_: str, name: str) -> None:
        info = zc.get_service_info(type_, name)
        if info is None or not info.addresses:
            return
        host = socket.inet_ntoa(info.addresses[0])
        props = {
            key.decode("utf-8"): value.decode("utf-8")
            for key, value in (info.properties or {}).items()
            if isinstance(key, bytes) and isinstance(value, bytes)
        }
        self.services.append(DiscoveredService(name=name, host=host, port=info.port, properties=props))

    def update_service(self, zc: Any, type_: str, name: str) -> None:
        self.add_service(zc, type_, name)

    def remove_service(self, zc: Any, type_: str, name: str) -> None:
        self.services = [service for service in self.services if service.name != name]
