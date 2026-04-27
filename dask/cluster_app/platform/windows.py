from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WindowsServiceCommand:
    description: str
    command: list[str]


def firewall_rule_commands(app_path: str, ports: list[int]) -> list[WindowsServiceCommand]:
    commands: list[WindowsServiceCommand] = [
        WindowsServiceCommand(
            "Allow cluster app executable",
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                f"New-NetFirewallRule -DisplayName 'Dask Cluster App' -Direction Inbound -Program '{app_path}' -Action Allow",
            ],
        )
    ]
    for port in ports:
        commands.append(
            WindowsServiceCommand(
                f"Allow TCP {port}",
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    f"New-NetFirewallRule -DisplayName 'Dask Cluster App TCP {port}' -Direction Inbound -Protocol TCP -LocalPort {port} -Action Allow",
                ],
            )
        )
    return commands


def install_service_command(python_exe: str) -> WindowsServiceCommand:
    return WindowsServiceCommand(
        "Install Windows service",
        [
            "sc.exe",
            "create",
            "DaskClusterApp",
            "start=",
            "auto",
            "binPath=",
            f'"{python_exe}" -m cluster_app.main agent',
        ],
    )

