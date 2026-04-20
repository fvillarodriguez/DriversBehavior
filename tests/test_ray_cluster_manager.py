import json
from pathlib import Path

from src import ray_cluster_manager as manager


class FakeRunner:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def run(self, args, *, cwd=manager.ROOT_DIR, timeout=30, env=None):
        self.calls.append(list(args))
        if self.responses:
            return self.responses.pop(0)
        return manager.CommandResult(ok=True, returncode=0, command=" ".join(args))


def test_config_serialization_does_not_store_secret_material(tmp_path: Path):
    config = manager.RayClusterConfig(
        ssh_user="felipe",
        ssh_key_path="~/.ssh/id_ed25519",
        remote_repo_path="/Users/felipe/Desktop/SUMO",
    )
    path = tmp_path / "config.json"

    manager.save_config(config, path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["ssh_key_path"] == "~/.ssh/id_ed25519"
    assert "password" not in payload
    assert "private_key" not in payload
    assert manager.load_config(path) == config


def test_build_head_start_args_uses_fixed_ray_ports():
    config = manager.RayClusterConfig(head_ip="10.10.10.1", head_cpus=8)

    args = manager.build_head_start_args(config, root_dir=Path("/repo"))

    assert args[0] == "/repo/.venv/bin/ray"
    assert "--head" in args
    assert "--node-ip-address=10.10.10.1" in args
    assert "--port=6379" in args
    assert "--dashboard-port=8265" in args
    assert "--object-manager-port=8076" in args
    assert "--node-manager-port=8077" in args
    assert "--min-worker-port=10002" in args
    assert "--max-worker-port=10100" in args
    assert "--num-cpus=8" in args
    assert "--disable-usage-stats" in args


def test_build_worker_start_script_uses_repo_and_calculated_cpus():
    config = manager.RayClusterConfig(
        head_ip="10.10.10.1",
        worker_ip="10.10.10.2",
        remote_repo_path="/Users/felipe/Desktop/SUMO",
        worker_reserved_cpus=2,
    )

    script = manager.build_worker_start_script(config)

    assert "cd /Users/felipe/Desktop/SUMO" in script
    assert "CPUS=$(($(sysctl -n hw.ncpu)-2))" in script
    assert ".venv/bin/ray stop" in script
    assert "--address=10.10.10.1:6379" in script
    assert "--node-ip-address=10.10.10.2" in script
    assert '--num-cpus="$CPUS"' in script
    assert "--disable-usage-stats" in script


def test_command_runner_reports_missing_executable():
    result = manager.CommandRunner().run(["/definitely/not/a/command"], timeout=1)

    assert not result.ok
    assert result.returncode == 127
    assert "/definitely/not/a/command" in result.command


def test_run_remote_script_uses_ssh_key_and_batch_mode():
    config = manager.RayClusterConfig(
        ssh_user="felipe",
        worker_ip="10.10.10.2",
        ssh_key_path="~/.ssh/id_ed25519",
    )
    fake = FakeRunner([manager.CommandResult(ok=True, returncode=0, command="ssh")])

    result = manager.run_remote_script(config, "echo ok", runner=fake)

    assert result.ok
    call = fake.calls[0]
    assert call[:2] == ["ssh", "-i"]
    assert "BatchMode=yes" in call
    assert "felipe@10.10.10.2" in call
    assert call[-1].startswith("bash -lc ")


def test_check_ports_available_reports_busy_port():
    config = manager.RayClusterConfig()
    fake = FakeRunner(
        [
            manager.CommandResult(
                ok=False,
                returncode=1,
                stdout="Puertos ocupados: 6379",
                command="lsof",
            )
        ]
    )

    check = manager.check_ports_available(config, runner=fake)

    assert check.name == "Puertos head"
    assert not check.ok
    assert "6379" in check.detail


def test_parse_json_from_output_uses_last_json_line():
    payload = manager.parse_json_from_output(
        "ray log line\n"
        '{"ignored": true}\n'
        '{"tasks": 4, "tasks_by_host": {"mac-a": 2, "mac-b": 2}}\n'
    )

    assert payload == {"tasks": 4, "tasks_by_host": {"mac-a": 2, "mac-b": 2}}


def test_parse_ray_status_summary_counts_active_nodes_and_usage():
    summary = manager.parse_ray_status_summary(
        """
Node status
---------------------------------------------------------------
Active:
 1 node_abc
 1 node_def
Pending:
 (no pending nodes)

Resources
---------------------------------------------------------------
Usage:
 2.0/16.0 CPU
 0.0/32.0 memory
Demands:
 (no resource demands)
"""
    )

    assert summary["active_nodes"] == 2
    assert summary["usage"] == {"CPU": "2.0/16.0", "memory": "0.0/32.0"}


def test_run_preflight_skips_remote_checks_when_ssh_fails():
    config = manager.RayClusterConfig()
    fake = FakeRunner(
        [
            manager.CommandResult(ok=True, returncode=0, stdout="inet 10.10.10.1 netmask 0xfffffffc\nstatus: active\n", command="ifconfig"),
            manager.CommandResult(ok=False, returncode=255, stderr="Permission denied", command="ssh true"),
            manager.CommandResult(ok=False, returncode=1, stderr="timeout", command="ping"),
            manager.CommandResult(ok=True, returncode=0, stdout="Python 3.12.12\n", command="python"),
            manager.CommandResult(ok=True, returncode=0, stdout="ray, version 2.53.0\n", command="ray"),
            manager.CommandResult(ok=True, returncode=0, command="ray stop"),
            manager.CommandResult(ok=True, returncode=0, stdout="Puertos libres\n", command="ports"),
        ]
    )

    checks = manager.run_preflight(config, runner=fake)
    by_name = {check.name: check for check in checks}

    assert by_name["Thunderbolt local"].ok
    assert not by_name["SSH worker"].ok
    assert by_name["Thunderbolt worker"].detail == "Omitido porque SSH no conecta."
    assert by_name["Python worker"].detail == "Omitido porque SSH no conecta."
    assert by_name["Puertos head"].ok
    assert by_name["Puertos worker"].detail == "Omitido porque SSH no conecta."
