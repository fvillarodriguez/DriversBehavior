from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

from cluster_app.nodes.hardware import detect_hardware


@dataclass(frozen=True, slots=True)
class NodeIdentity:
    uuid: str
    name: str


def load_node_identity(identity_path: str | Path) -> NodeIdentity:
    path = Path(identity_path)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return NodeIdentity(data["uuid"], data["name"])
    path.parent.mkdir(parents=True, exist_ok=True)
    hardware = detect_hardware()
    identity = NodeIdentity(uuid.uuid4().hex, hardware.hostname)
    path.write_text(
        json.dumps({"uuid": identity.uuid, "name": identity.name}, indent=2),
        encoding="utf-8",
    )
    return identity


def service_instance_name(identity: NodeIdentity) -> str:
    safe_name = "".join(char if char.isalnum() or char in "-_" else "-" for char in identity.name)
    return f"{safe_name}-{identity.uuid[:8]}"
