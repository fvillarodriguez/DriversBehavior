from types import SimpleNamespace

from src import pipeline_dask_runtime as runtime_module


class _FakeClient:
    def scheduler_info(self):
        return {
            "workers": {
                "worker-a": {"resources": {"CPU": 4.0}, "nthreads": 4},
                "worker-b": {"resources": {"CPU": 2.0}, "nthreads": 2},
            }
        }


class _FakeCluster:
    def __init__(self):
        self.config = SimpleNamespace()
        self.started = False
        self._client = _FakeClient()

    def start(self):
        self.started = True

    def client(self):
        return self._client

    def status(self):
        return {
            "scheduler": {"address": "tls://cluster:8786"},
            "dashboard": "http://cluster:8787/status",
        }


def test_connect_dask_cluster_uses_scheduler_info():
    cluster = _FakeCluster()

    runtime = runtime_module.connect_dask_cluster(cluster=cluster)

    assert cluster.started is True
    assert runtime.address == "tls://cluster:8786"
    assert runtime.dashboard_url == "http://cluster:8787/status"
    assert runtime.total_cpus == 6
    assert runtime.max_node_cpus == 4
    assert runtime.active_nodes == 2
    assert runtime.has_cpu_resource is True


def test_normalize_execution_backend_defaults_to_local():
    assert runtime_module.normalize_execution_backend(None) == "local"
    assert runtime_module.normalize_execution_backend("") == "local"
    assert runtime_module.normalize_execution_backend("DASK_CLUSTER") == "dask_cluster"
    assert runtime_module.normalize_execution_backend("unknown") == "local"


def test_dask_submit_resources_skips_missing_cpu_resource():
    runtime = runtime_module.DaskClusterRuntime(
        config=SimpleNamespace(),
        cluster=SimpleNamespace(),
        client=SimpleNamespace(),
        scheduler_info={},
        workers={},
        total_cpus=1,
        max_node_cpus=1,
        active_nodes=0,
        address="",
        dashboard_url=None,
        has_cpu_resource=False,
    )

    assert runtime_module.dask_submit_resources(runtime, 4) is None

