from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        fval = float(value)
        if not math.isfinite(fval):
            return None
        return fval
    if isinstance(value, (float,)):
        if not math.isfinite(value):
            return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated sampler memory probe worker.")
    parser.add_argument("--payload", required=True, help="Path to serialized payload (.pt)")
    parser.add_argument("--output", required=True, help="Path to output result (.json)")
    args = parser.parse_args()

    payload_path = Path(args.payload)
    output_path = Path(args.output)

    from src.graph_builder_app import _probe_sampler_memory_usage

    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    graph_data = payload.get("graph_data")
    sequence_index = payload.get("sequence_index")
    base_params = payload.get("base_params", {})
    sampler_config = payload.get("sampler_config", {})
    batch_size = int(payload.get("batch_size", 1))
    probe_steps = int(payload.get("probe_steps", 1))
    device = torch.device(str(payload.get("device", "cpu")))
    sampling_seed = int(payload.get("sampling_seed", 0))
    total_memory_bytes = int(payload.get("total_memory_bytes", 0))
    fidelity_full_epoch = bool(payload.get("fidelity_full_epoch", False))

    result = _probe_sampler_memory_usage(
        graph_data=graph_data,
        sequence_index=sequence_index,
        base_params=base_params,
        sampler_config=sampler_config,
        batch_size=batch_size,
        probe_steps=probe_steps,
        device=device,
        sampling_seed=sampling_seed,
        total_memory_bytes=total_memory_bytes,
        fidelity_full_epoch=fidelity_full_epoch,
        use_subprocess_isolation=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(result), fh, ensure_ascii=True)


if __name__ == "__main__":
    main()
