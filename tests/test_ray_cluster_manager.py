import json
from pathlib import Path

import pytest

from src import ray_cluster_manager as manager


class FakeRunner:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.envs = []

    def run(self, args, *, cwd=manager.ROOT_DIR, timeout=30, env=None):
        self.calls.append(list(args))
        self.envs.append(dict(env or {}))
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


def test_automatic_bridge_config_forces_default_bridge_profile():
    config = manager.RayClusterConfig(
        head_ip="192.168.1.10",
        worker_ip="192.168.1.11",
        netmask="255.255.255.0",
        ssh_user="felipe",
    )

    normalized = manager.automatic_bridge_config(config)

    assert normalized.head_ip == manager.DEFAULT_HEAD_IP
    assert normalized.worker_ip == manager.DEFAULT_WORKER_IP
    assert normalized.netmask == manager.DEFAULT_NETMASK
    assert normalized.ssh_user == "felipe"


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


def test_ssh_public_key_path_derives_pub_file():
    assert manager.ssh_public_key_path("~/.ssh/id_ed25519").name == "id_ed25519.pub"


def test_ssh_private_key_path_rejects_pub_file():
    with pytest.raises(ValueError):
        manager.ssh_private_key_path("~/.ssh/id_ed25519.pub")


def test_detect_private_keys_finds_standard_keys(tmp_path: Path):
    ssh_dir = tmp_path / ".ssh"
    ssh_dir.mkdir()
    (ssh_dir / "id_ed25519").write_text("PRIVATE", encoding="utf-8")
    (ssh_dir / "id_ed25519.pub").write_text("PUBLIC", encoding="utf-8")
    (ssh_dir / "id_rsa").write_text("PRIVATE", encoding="utf-8")

    found = manager.detect_private_keys(ssh_dir)

    assert found == [ssh_dir / "id_ed25519", ssh_dir / "id_rsa"]


def test_read_public_key_prefers_existing_pub_file(tmp_path: Path):
    private_key = tmp_path / "id_ed25519"
    public_key = tmp_path / "id_ed25519.pub"
    private_key.write_text("PRIVATE", encoding="utf-8")
    public_key.write_text("ssh-ed25519 AAAAB3NzaC1lZDI1NTE5AAAAIGZha2U= test@sumo\n", encoding="utf-8")
    config = manager.RayClusterConfig(ssh_key_path=str(private_key))

    exported = manager.read_public_key(config)

    assert exported == "ssh-ed25519 AAAAB3NzaC1lZDI1NTE5AAAAIGZha2U= test@sumo"


def test_read_public_key_falls_back_to_ssh_keygen(tmp_path: Path):
    private_key = tmp_path / "id_ed25519"
    private_key.write_text("PRIVATE", encoding="utf-8")
    config = manager.RayClusterConfig(ssh_key_path=str(private_key))
    fake = FakeRunner(
        [
            manager.CommandResult(
                ok=True,
                returncode=0,
                stdout="ssh-ed25519 AAAAB3NzaC1lZDI1NTE5AAAAIGZha2U= worker@mac\n",
                command="ssh-keygen",
            )
        ]
    )

    exported = manager.read_public_key(config, runner=fake)

    assert exported == "ssh-ed25519 AAAAB3NzaC1lZDI1NTE5AAAAIGZha2U= worker@mac"
    assert fake.calls[0][:3] == ["ssh-keygen", "-y", "-f"]


def test_import_public_key_appends_once(tmp_path: Path):
    target = tmp_path / ".ssh" / "authorized_keys"
    public_key = "ssh-ed25519 AAAAB3NzaC1lZDI1NTE5AAAAIGZha2U= test@sumo"

    first = manager.import_public_key(public_key, target_path=target)
    second = manager.import_public_key(public_key, target_path=target)

    assert "importada" in first
    assert "ya estaba presente" in second
    assert target.read_text(encoding="utf-8") == f"{public_key}\n"


def test_import_public_key_rejects_invalid_payload(tmp_path: Path):
    with pytest.raises(ValueError):
        manager.import_public_key("esto no es una llave valida", target_path=tmp_path / "authorized_keys")


def test_check_config_warnings_ignores_blank_private_key_path():
    warnings = manager.check_config_warnings(manager.RayClusterConfig(ssh_key_path=""))

    assert warnings == []


def test_check_config_warnings_ignores_pub_path_and_keeps_ssh_automatic():
    warnings = manager.check_config_warnings(manager.RayClusterConfig(ssh_key_path="~/.ssh/id_ed25519.pub"))

    assert warnings == []


def test_run_remote_script_falls_back_to_automatic_private_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    private_key = tmp_path / "id_ed25519"
    private_key.write_text("PRIVATE", encoding="utf-8")
    monkeypatch.setattr(manager, "default_ssh_private_key_path", lambda: private_key)
    config = manager.RayClusterConfig(ssh_key_path="~/.ssh/id_ed25519.pub")
    fake = FakeRunner([manager.CommandResult(ok=True, returncode=0, command="ssh")])

    result = manager.run_remote_script(config, "echo ok", runner=fake)

    assert result.ok
    assert str(private_key) in fake.calls[0]


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


def test_prepare_ssh_access_requests_password_when_ssh_is_not_ready(tmp_path: Path):
    private_key = tmp_path / "id_ed25519"
    private_key.write_text("PRIVATE", encoding="utf-8")
    config = manager.RayClusterConfig(ssh_key_path=str(private_key))
    fake = FakeRunner([manager.CommandResult(ok=False, returncode=255, stderr="Permission denied", command="ssh true")])

    results = manager.prepare_ssh_access(config, password="", runner=fake)

    assert len(results) == 2
    assert results[0].ok
    assert not results[1].ok
    assert "password del worker" in results[1].stderr.lower()


def test_bootstrap_ssh_access_uses_expect_and_ssh_copy_id(tmp_path: Path):
    private_key = tmp_path / "id_ed25519"
    public_key = tmp_path / "id_ed25519.pub"
    private_key.write_text("PRIVATE", encoding="utf-8")
    public_key.write_text("ssh-ed25519 AAAAB3NzaC1lZDI1NTE5AAAAIGZha2U= test@sumo\n", encoding="utf-8")
    config = manager.RayClusterConfig(ssh_user="felipe", ssh_key_path=str(private_key))
    fake = FakeRunner([manager.CommandResult(ok=True, returncode=0, command="expect")])

    result = manager.bootstrap_ssh_access(config, "secret", runner=fake)

    assert result.ok
    call = fake.calls[0]
    assert call[:2] == ["expect", "-c"]
    assert call[4:7] == ["ssh-copy-id", "-i", str(public_key)]
    assert call[-1] == "felipe@10.10.10.2"
    assert fake.envs[0][manager.SSH_BOOTSTRAP_PASSWORD_ENV] == "secret"


def test_configure_local_bridge_uses_macos_admin_prompt():
    config = manager.RayClusterConfig()
    fake = FakeRunner([manager.CommandResult(ok=True, returncode=0, command="osascript")])

    result = manager.configure_local_bridge(config, runner=fake)

    assert result.ok
    call = fake.calls[0]
    assert call[:2] == ["osascript", "-e"]
    assert "networksetup -setmanual 'Thunderbolt Bridge' 10.10.10.1 255.255.255.252 0.0.0.0" in call[2]
    assert "administrator privileges" in call[2]


def test_configure_remote_bridge_uses_ssh_to_apply_worker_profile():
    config = manager.RayClusterConfig(
        ssh_user="felipe",
        ssh_key_path="~/.ssh/id_ed25519",
        remote_repo_path="/Users/felipe/Desktop/SUMO",
    )
    fake = FakeRunner([manager.CommandResult(ok=True, returncode=0, command="ssh")])

    result = manager.configure_remote_bridge(config, runner=fake)

    assert result.ok
    call = fake.calls[0]
    assert call[:2] == ["ssh", "-i"]
    assert "felipe@10.10.10.2" in call
    assert "networksetup -setmanual" in call[-1]
    assert "Thunderbolt Bridge" in call[-1]
    assert "10.10.10.2" in call[-1]
    assert "255.255.255.252" in call[-1]


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
