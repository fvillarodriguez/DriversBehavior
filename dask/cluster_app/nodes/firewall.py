from __future__ import annotations

import platform
from dataclasses import dataclass

from cluster_app.platform.macos import firewall_rule_commands as macos_firewall_rule_commands
from cluster_app.platform.windows import firewall_rule_commands as windows_firewall_rule_commands


@dataclass(frozen=True, slots=True)
class FirewallPlan:
    supported: bool
    commands: list[list[str]]
    note: str


def plan_firewall(app_path: str, ports: list[int]) -> FirewallPlan:
    system = platform.system()
    if system == "Windows":
        return FirewallPlan(
            True,
            [item.command for item in windows_firewall_rule_commands(app_path, ports)],
            "Run from elevated PowerShell or administrator context.",
        )
    if system == "Darwin":
        return FirewallPlan(
            True,
            [item.command for item in macos_firewall_rule_commands(app_path)],
            "May require sudo and macOS approval depending on signing policy.",
        )
    return FirewallPlan(False, [], "Firewall automation is only implemented for Windows and macOS.")

