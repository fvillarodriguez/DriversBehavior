from __future__ import annotations

import unittest
from unittest.mock import patch

from cluster_app.discovery import mdns
from cluster_app.discovery.election import Candidate, choose_scheduler
from cluster_app.security.tokens import PairingTokenService


class DiscoverySecurityTests(unittest.TestCase):
    def test_scheduler_prefers_preferred_node_then_uuid(self) -> None:
        candidates = [
            Candidate("b", "b", "127.0.0.1", 8080),
            Candidate("c", "c", "127.0.0.1", 8080, preferred=True),
            Candidate("a", "a", "127.0.0.1", 8080),
        ]
        self.assertEqual(choose_scheduler(candidates).uuid, "c")
        self.assertEqual(choose_scheduler([candidates[0], candidates[2]]).uuid, "a")

    def test_pairing_token_verification(self) -> None:
        service = PairingTokenService("cluster-secret")
        token = service.issue()
        self.assertTrue(service.verify(token.token, token.digest))
        self.assertFalse(service.verify(token.token + "x", token.digest))

    def test_mdns_discover_closes_browser_and_zeroconf(self) -> None:
        zeroconfs = []
        browsers = []

        class FakeZeroconf:
            def __init__(self) -> None:
                self.closed = False
                zeroconfs.append(self)

            def close(self) -> None:
                self.closed = True

        class FakeBrowser:
            def __init__(self, zc, service_type, listener) -> None:
                self.canceled = False
                browsers.append(self)

            def cancel(self) -> None:
                self.canceled = True

        with (
            patch.object(mdns, "_zeroconf", return_value=(FakeZeroconf, object, FakeBrowser)),
            patch("cluster_app.discovery.mdns.time.sleep"),
        ):
            self.assertEqual(mdns.discover("_test._tcp.local.", timeout=0.01), [])

        self.assertTrue(browsers[0].canceled)
        self.assertTrue(zeroconfs[0].closed)

    def test_mdns_discover_closes_zeroconf_when_browser_fails(self) -> None:
        zeroconfs = []

        class FakeZeroconf:
            def __init__(self) -> None:
                self.closed = False
                zeroconfs.append(self)

            def close(self) -> None:
                self.closed = True

        class RaisingBrowser:
            def __init__(self, zc, service_type, listener) -> None:
                raise RuntimeError("browser failed")

        with patch.object(mdns, "_zeroconf", return_value=(FakeZeroconf, object, RaisingBrowser)):
            with self.assertRaises(RuntimeError):
                mdns.discover("_test._tcp.local.", timeout=0.01)

        self.assertTrue(zeroconfs[0].closed)


if __name__ == "__main__":
    unittest.main()
