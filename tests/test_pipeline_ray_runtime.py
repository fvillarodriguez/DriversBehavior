from types import SimpleNamespace

import pytest

from src import pipeline_ray_runtime as runtime_module


class _FakeRayModule:
    def __init__(self):
        self._initialized = False
        self.init_calls = []

    def is_initialized(self):
        return self._initialized

    def init(self, **kwargs):
        self._initialized = True
        self.init_calls.append(dict(kwargs))

    def nodes(self):
        return [
            {"Alive": True, "Resources": {"CPU": 4.0}},
            {"Alive": False, "Resources": {"CPU": 8.0}},
            {"Alive": True, "Resources": {"CPU": 2.0}},
        ]

    def cluster_resources(self):
        return {"CPU": 6.0}


def test_connect_ray_cluster_uses_saved_address_and_alive_nodes(monkeypatch):
    fake_ray = _FakeRayModule()
    fake_config = SimpleNamespace(ray_address="ray://cluster")
    fake_status = SimpleNamespace(ok=True, combined_output="ok")

    monkeypatch.setattr(
        runtime_module.ray_cluster_manager,
        "load_config",
        lambda: fake_config,
    )
    monkeypatch.setattr(
        runtime_module.ray_cluster_manager,
        "automatic_bridge_config",
        lambda config: config,
    )
    monkeypatch.setattr(
        runtime_module.ray_cluster_manager,
        "ray_status",
        lambda config: fake_status,
    )
    monkeypatch.setattr(
        runtime_module.ray_cluster_manager,
        "runtime_connection_health_checks",
        lambda config: [],
    )

    runtime = runtime_module.connect_ray_cluster(ray_module=fake_ray)

    assert fake_ray.init_calls == [
        {"address": "ray://cluster", "ignore_reinit_error": True}
    ]
    assert runtime.config.ray_address == "ray://cluster"
    assert runtime.total_cpus == 6
    assert runtime.max_node_cpus == 4
    assert runtime.active_nodes == 2
    assert len(runtime.nodes) == 2


def test_connect_ray_cluster_raises_when_status_is_not_ok(monkeypatch):
    fake_config = SimpleNamespace(ray_address="ray://cluster")
    fake_status = SimpleNamespace(ok=False, combined_output="cluster down")

    monkeypatch.setattr(
        runtime_module.ray_cluster_manager,
        "load_config",
        lambda: fake_config,
    )
    monkeypatch.setattr(
        runtime_module.ray_cluster_manager,
        "automatic_bridge_config",
        lambda config: config,
    )
    monkeypatch.setattr(
        runtime_module.ray_cluster_manager,
        "ray_status",
        lambda config: fake_status,
    )
    monkeypatch.setattr(
        runtime_module.ray_cluster_manager,
        "runtime_connection_health_checks",
        lambda config: [],
    )

    with pytest.raises(RuntimeError, match="cluster down"):
        runtime_module.connect_ray_cluster(ray_module=_FakeRayModule())


def test_connect_ray_cluster_raises_when_health_checks_block(monkeypatch):
    fake_config = SimpleNamespace(ray_address="ray://cluster")

    monkeypatch.setattr(
        runtime_module.ray_cluster_manager,
        "load_config",
        lambda: fake_config,
    )
    monkeypatch.setattr(
        runtime_module.ray_cluster_manager,
        "automatic_bridge_config",
        lambda config: config,
    )
    monkeypatch.setattr(
        runtime_module.ray_cluster_manager,
        "runtime_connection_health_checks",
        lambda config: [
            runtime_module.ray_cluster_manager.CheckResult(
                name="Disco /tmp",
                ok=False,
                detail="sin margen para Ray",
                blocking=True,
            )
        ],
    )

    with pytest.raises(RuntimeError, match="sin margen para Ray"):
        runtime_module.connect_ray_cluster(ray_module=_FakeRayModule())


def test_normalize_execution_backend_defaults_to_local():
    assert runtime_module.normalize_execution_backend(None) == "local"
    assert runtime_module.normalize_execution_backend("") == "local"
    assert runtime_module.normalize_execution_backend("RAY_CLUSTER") == "ray_cluster"
    assert runtime_module.normalize_execution_backend("unknown") == "local"
