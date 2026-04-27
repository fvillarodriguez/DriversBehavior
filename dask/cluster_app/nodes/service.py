from __future__ import annotations

import platform
import sys
from dataclasses import dataclass

from cluster_app.platform.macos import install_launch_daemon_path, launch_daemon_plist
from cluster_app.platform.windows import install_service_command


@dataclass(frozen=True, slots=True)
class ServicePlan:
    supported: bool
    description: str
    commands: list[list[str]]
    file_path: str | None = None
    file_content: str | None = None


def plan_service_install(config_path: str | None = None) -> ServicePlan:
    system = platform.system()
    python_exe = sys.executable
    if system == "Windows":
        cmd = install_service_command(python_exe)
        return ServicePlan(True, cmd.description, [cmd.command])
    if system == "Darwin":
        path = install_launch_daemon_path()
        return ServicePlan(
            True,
            "Install macOS LaunchDaemon before login",
            [["sudo", "launchctl", "load", "-w", str(path)]],
            str(path),
            launch_daemon_plist(python_exe, config_path),
        )
    return ServicePlan(False, "Service install is only implemented for Windows and macOS.", [])

