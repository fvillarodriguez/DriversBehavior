from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MacOSCommand:
    description: str
    command: list[str]


def firewall_rule_commands(app_path: str) -> list[MacOSCommand]:
    return [
        MacOSCommand("Unblock app in Application Firewall", ["/usr/libexec/ApplicationFirewall/socketfilterfw", "--unblockapp", app_path]),
        MacOSCommand("Allow signed app automatically", ["/usr/libexec/ApplicationFirewall/socketfilterfw", "--add", app_path]),
    ]


def launch_daemon_plist(python_exe: str, config_path: str | None = None) -> str:
    args = f"<string>{python_exe}</string><string>-m</string><string>cluster_app.main</string><string>agent</string>"
    if config_path:
        args += f"<string>--config</string><string>{config_path}</string>"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>local.dask-cluster-app.agent</string>
  <key>ProgramArguments</key>
  <array>{args}</array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>/var/log/dask-cluster-app.log</string>
  <key>StandardErrorPath</key>
  <string>/var/log/dask-cluster-app.err</string>
</dict>
</plist>
"""


def install_launch_daemon_path() -> Path:
    return Path("/Library/LaunchDaemons/local.dask-cluster-app.agent.plist")

