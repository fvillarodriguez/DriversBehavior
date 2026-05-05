from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from SUMO import (  # noqa: E402
    FlowColumns,
    build_irl_transition_dataset,
    build_sumo_fcd_transition_dataset,
    is_sumo_fcd_transition_dataset,
    run_sumo_pipeline,
)
from src.marl_core import build_expert_datasets, MAIRLManager  # noqa: E402


def _sample_porticos() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "portico": ["P1", "P2", "P3"],
            "km": [0.0, 5.0, 10.0],
            "calzada": ["Poniente", "Poniente", "Poniente"],
            "eje": ["RUTA 5 SUR", "RUTA 5 SUR", "RUTA 5 SUR"],
            "orden": [0, 1, 2],
            "edge_id_sumo": ["e1", "e2", "e3"],
            "lane_id_sumo": ["0", "0", "0"],
            "pos_m": [0.0, 5000.0, 10000.0],
        }
    )


def _sample_flujos() -> pd.DataFrame:
    base = pd.Timestamp("2024-01-01 08:00:00")
    return pd.DataFrame(
        {
            "FECHA": [
                base,
                base + pd.Timedelta(minutes=5),
                base + pd.Timedelta(minutes=10),
                base + pd.Timedelta(seconds=30),
                base + pd.Timedelta(minutes=5, seconds=30),
                base + pd.Timedelta(minutes=10, seconds=30),
            ],
            "VELOCIDAD": [80.0, 82.0, 84.0, 76.0, 78.0, 80.0],
            "CATEGORIA": [1, 1, 1, 3, 3, 3],
            "MATRICULA": ["AA11", "AA11", "AA11", "BB22", "BB22", "BB22"],
            "PORTICO": ["P1", "P2", "P3", "P1", "P2", "P3"],
            "CARRIL": ["1", "1", "1", "1", "1", "1"],
        }
    )


def test_sumo_pipeline_builds_segment_level_irl_transitions():
    result = run_sumo_pipeline(
        _sample_flujos(),
        _sample_porticos(),
        flow_cols=FlowColumns(),
    )

    transitions = result.irl_transitions

    assert not transitions.empty
    assert len(transitions) == 4
    assert {
        "trajectory_id",
        "plate",
        "agent_id",
        "vehicle_class",
        "state_speed_kmh",
        "next_speed_kmh",
        "state_lane",
        "next_lane",
        "next_headway_s",
        "next_relative_speed_kmh",
        "flow_context_vehicle_count",
        "action_delta_speed_kmh",
        "action_accel_m_s2",
        "action_lane_change",
        "vms_active",
        "temporal_split",
    }.issubset(transitions.columns)
    assert (transitions["delta_t_s"] > 0).all()
    assert set(transitions["vehicle_class"]) == {"car", "truck"}
    assert transitions["trajectory_id"].str.contains("_").all()


def test_irl_transition_dataset_assigns_temporal_splits_by_agent():
    base = pd.Timestamp("2024-01-01 08:00:00")
    rows = []
    for class_id, plate_prefix in [(1, "AA"), (3, "BB")]:
        for i in range(10):
            start_time = base + pd.Timedelta(minutes=i)
            rows.append(
                {
                    "trip_id": f"{plate_prefix}{i}_trip",
                    "plate": f"{plate_prefix}{i}",
                    "start_portico": "P1",
                    "end_portico": "P2",
                    "start_time": start_time,
                    "end_time": start_time + pd.Timedelta(minutes=5),
                    "start_speed_kmh": 70.0 + i,
                    "end_speed_kmh": 72.0 + i,
                    "start_lane": 1,
                    "end_lane": 1,
                    "delta_t_s": 300.0,
                    "avg_speed_kmh": 71.0 + i,
                    "speed_change_kmh": 2.0,
                    "accel_m_s2": 0.1,
                    "lane_change": 0,
                    "vehicle_type_id": class_id,
                }
            )

    transitions = build_irl_transition_dataset(
        pd.DataFrame(rows),
        pd.DataFrame(),
        pd.DataFrame(),
    )

    assert set(transitions["temporal_split"]) == {"train", "validation", "test"}
    for _, group in transitions.groupby("agent_id"):
        assert set(group["temporal_split"]) == {"train", "validation", "test"}


def test_sumo_fcd_transition_dataset_uses_consecutive_vehicle_samples(tmp_path):
    fcd_path = tmp_path / "sumo_fcd.xml"
    fcd_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<fcd-export>
  <timestep time="0.0">
    <vehicle id="veh_a" type="car" speed="10.0" acceleration="0.0" lane="edge0_0" edge="edge0" pos="10.0" x="0.0" y="0.0"/>
    <vehicle id="veh_b" type="truck" speed="8.0" acceleration="0.0" lane="edge0_0" edge="edge0" pos="25.0" x="15.0" y="0.0"/>
  </timestep>
  <timestep time="1.0">
    <vehicle id="veh_a" type="car" speed="12.0" acceleration="2.0" lane="edge0_1" edge="edge0" pos="22.0" x="12.0" y="3.5"/>
    <vehicle id="veh_b" type="truck" speed="8.5" acceleration="0.5" lane="edge0_0" edge="edge0" pos="33.0" x="23.0" y="0.0"/>
  </timestep>
  <timestep time="2.0">
    <vehicle id="veh_a" type="car" speed="11.0" acceleration="-1.0" lane="edge0_1" edge="edge0" pos="33.0" x="23.0" y="3.5"/>
    <vehicle id="veh_b" type="truck" speed="9.0" acceleration="0.5" lane="edge0_0" edge="edge0" pos="42.0" x="32.0" y="0.0"/>
  </timestep>
</fcd-export>
""",
        encoding="utf-8",
    )

    transitions = build_sumo_fcd_transition_dataset(fcd_path)

    assert len(transitions) == 4
    assert is_sumo_fcd_transition_dataset(transitions)
    assert set(transitions["agent_id"]) == {"car", "truck"}
    assert (transitions["source"] == "sumo_fcd").all()
    assert (transitions["delta_t_s"] == 1.0).all()

    veh_a_first = transitions[transitions["trajectory_id"] == "veh_a"].iloc[0]
    assert veh_a_first["state_speed_kmh"] == pytest.approx(36.0)
    assert veh_a_first["next_speed_kmh"] == pytest.approx(43.2)
    assert veh_a_first["action_delta_speed_kmh"] == pytest.approx(7.2)
    assert veh_a_first["action_accel_m_s2"] == pytest.approx(2.0)
    assert veh_a_first["action_lane_change"] == pytest.approx(1.0)
    assert veh_a_first["state_headway_s"] == pytest.approx(1.5)


def test_sumo_fcd_dataset_feeds_next_states_without_proxy(tmp_path):
    fcd_path = tmp_path / "sumo_fcd.xml"
    fcd_path.write_text(
        """<fcd-export>
  <timestep time="0.0">
    <vehicle id="veh_a" type="car" speed="10.0" lane="edge0_0" edge="edge0" pos="10.0"/>
  </timestep>
  <timestep time="1.0">
    <vehicle id="veh_a" type="car" speed="12.0" lane="edge0_1" edge="edge0" pos="22.0"/>
  </timestep>
  <timestep time="2.0">
    <vehicle id="veh_a" type="car" speed="11.0" lane="edge0_1" edge="edge0" pos="33.0"/>
  </timestep>
</fcd-export>
""",
        encoding="utf-8",
    )
    transitions = build_sumo_fcd_transition_dataset(fcd_path)

    datasets, _, feature_cols, _, _, _ = build_expert_datasets(
        transitions,
        feature_cols=["state_speed_kmh", "state_lane", "state_pos_m"],
        agent_ids=["car"],
    )

    assert "state_pos_m" in feature_cols
    assert datasets["car"].states.shape == datasets["car"].next_states.shape
    assert not (datasets["car"].states == datasets["car"].next_states).all()


def test_non_fcd_dataset_is_not_accepted_for_fcd_training():
    assert not is_sumo_fcd_transition_dataset(_transition_training_frame())


def _transition_training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "agent_id": ["car", "car", "truck", "truck"],
            "state_speed_kmh": [80.0, 82.0, 70.0, 72.0],
            "next_speed_kmh": [82.0, 84.0, 72.0, 71.0],
            "state_lane": [1.0, 1.0, 2.0, 2.0],
            "next_lane": [1.0, 2.0, 2.0, 1.0],
            "state_headway_s": [2.0, 2.5, 3.0, 3.5],
            "next_headway_s": [2.5, 3.0, 3.5, 4.0],
            "state_relative_speed_kmh": [1.0, 0.5, -1.0, -0.5],
            "next_relative_speed_kmh": [0.5, 0.0, -0.5, -1.5],
            "flow_context_vehicle_count": [20, 22, 15, 16],
            "flow_context_mean_speed_kmh": [78.0, 79.0, 68.0, 69.0],
            "flow_context_density_veh_per_km": [12.0, 13.0, 10.0, 11.0],
            "vms_active": [False, False, True, True],
            "action_delta_speed_kmh": [2.0, 2.0, 2.0, -1.0],
            "action_accel_m_s2": [0.2, 0.1, 0.15, -0.2],
            "action_lane_change": [0.0, 1.0, 0.0, 1.0],
        }
    )


def test_mairl_dataset_uses_state_action_next_state():
    datasets, _, feature_cols, action_cols, _, _ = build_expert_datasets(
        _transition_training_frame(),
        agent_ids=["car", "truck"],
    )

    assert set(datasets) == {"car", "truck"}
    assert "state_speed_kmh" in feature_cols
    assert action_cols == [
        "action_delta_speed_kmh",
        "action_accel_m_s2",
        "action_lane_change",
    ]
    assert datasets["car"].states.shape == datasets["car"].next_states.shape
    assert datasets["car"].actions.shape[1] == 3
    assert (datasets["car"].actions <= 1.0).all()
    assert (datasets["car"].actions >= -1.0).all()


def test_mairl_manager_runs_one_transition_training_step():
    pytest.importorskip("torch")

    manager = MAIRLManager(
        expert_df=_transition_training_frame(),
        agent_config=[
            {"id": "car", "name": "Policy_car"},
            {"id": "truck", "name": "Policy_truck"},
        ],
        hidden_sizes=(8,),
        device="cpu",
    )

    metrics = manager.train_step(batch_size=2)
    policy_actions = manager.get_policy_actions()

    assert metrics["agents"] == pytest.approx(2.0)
    assert metrics["transitions"] == pytest.approx(4.0)
    assert set(policy_actions) == {"car", "truck"}
    assert "action_accel_m_s2" in policy_actions["car"]
