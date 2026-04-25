from __future__ import annotations

from pathlib import Path

from src import crash_prediction_history_store as store


def test_insert_query_and_star_record(tmp_path):
    db_path = tmp_path / "history.sqlite"
    record_id = store.insert_record(
        db_path,
        stage="Optuna",
        context_key="ctx",
        feature_context_key="feature-ctx",
        batch_key="optuna-batch-1",
        features_path="/tmp/features.duckdb",
        model_name="XGBoost",
        optuna_objective="Balanced F1",
        threshold_objective="far",
        calibration_method="sigmoid",
        params={"n_trials": 20},
        metrics={"best_score": 0.42},
        artifacts=[
            {
                "path": "/tmp/optuna_trials.csv",
                "role": "trials",
                "generated": True,
                "delete_on_record_delete": True,
            }
        ],
    )

    records = store.query_previous_optuna(
        db_path,
        feature_context_key="feature-ctx",
        model_name="XGBoost",
        optuna_objective="Balanced F1",
        calibration_method="sigmoid",
        threshold_objective="far",
    )

    assert [record["id"] for record in records] == [record_id]
    assert records[0]["batch_key"] == "optuna-batch-1"
    assert records[0]["params"]["n_trials"] == 20
    assert records[0]["metrics"]["best_score"] == 0.42
    assert records[0]["artifacts"][0]["role"] == "trials"

    assert store.set_starred(db_path, record_id, True) is True
    starred = store.list_records(db_path, starred=True)
    assert len(starred) == 1
    assert starred[0]["starred"] is True


def test_list_record_summaries_skip_large_payloads_until_detail_load(tmp_path):
    db_path = tmp_path / "history.sqlite"
    record_id = store.insert_record(
        db_path,
        stage="Modelos",
        context_key="ctx",
        feature_context_key="feature-ctx",
        model_name="SVM",
        params={"large": "x" * 1000},
        metrics={"score": 0.7},
        metadata={"raw": "y" * 1000},
    )

    summaries = store.list_record_summaries(db_path, limit=10)

    assert [item["id"] for item in summaries] == [record_id]
    assert summaries[0]["model_name"] == "SVM"
    assert "params" not in summaries[0]
    assert "metrics" not in summaries[0]
    assert "metadata" not in summaries[0]

    detail = store.get_record(db_path, record_id)
    assert detail is not None
    assert detail["batch_key"] is None
    assert detail["params"]["large"] == "x" * 1000
    assert detail["metadata"]["raw"] == "y" * 1000


def test_context_key_changes_with_date_range_and_tramo():
    base = {
        "event_files": ["eventos.csv"],
        "features_path": "features.duckdb",
        "features_source": "duckdb",
        "features_rows": 10,
        "features_cols": 3,
        "dataset_fingerprint": "fp",
        "selected_features": ["a", "b"],
    }
    first = store.build_context_key(
        **base,
        features_date_min="2024-01-01",
        features_date_max="2024-01-31",
        tramo_label="N | A | 1 -> 2",
    )
    second = store.build_context_key(
        **base,
        features_date_min="2024-02-01",
        features_date_max="2024-02-28",
        tramo_label="N | A | 1 -> 2",
    )
    third = store.build_context_key(
        **base,
        features_date_min="2024-01-01",
        features_date_max="2024-01-31",
        tramo_label="N | A | 2 -> 3",
    )

    assert first != second
    assert first != third


def test_maybe_insert_generation_record_skips_loads(tmp_path):
    db_path = tmp_path / "history.sqlite"
    result = store.maybe_insert_generation_record(
        db_path,
        generated=False,
        stage="Feature engineering",
        context_key="load-context",
    )

    assert result is None
    assert store.list_records(db_path) == []


def test_delete_record_removes_only_generated_unreferenced_artifacts(tmp_path):
    db_path = tmp_path / "history.sqlite"
    generated_path = tmp_path / "generated.csv"
    input_path = tmp_path / "input.csv"
    shared_path = tmp_path / "shared.csv"
    generated_path.write_text("generated", encoding="utf-8")
    input_path.write_text("input", encoding="utf-8")
    shared_path.write_text("shared", encoding="utf-8")

    record_id = store.insert_record(
        db_path,
        stage="Balance",
        context_key="ctx",
        artifacts=[
            {
                "path": str(generated_path),
                "role": "snapshot",
                "generated": True,
                "delete_on_record_delete": True,
            },
            {
                "path": str(input_path),
                "role": "events",
                "generated": False,
                "delete_on_record_delete": False,
            },
            {
                "path": str(shared_path),
                "role": "shared",
                "generated": True,
                "delete_on_record_delete": True,
            },
        ],
    )
    store.insert_record(
        db_path,
        stage="Modelos",
        context_key="ctx-2",
        artifacts=[
            {
                "path": str(shared_path),
                "role": "shared",
                "generated": True,
                "delete_on_record_delete": True,
            }
        ],
    )

    result = store.delete_record(db_path, record_id)

    assert result["deleted"] is True
    assert not generated_path.exists()
    assert input_path.exists()
    assert shared_path.exists()
    assert result["skipped_paths"] == [
        {"path": str(shared_path), "reason": "referenced_by_other_record"}
    ]


def test_query_previous_models_filters_by_optuna_context_and_protocols(tmp_path):
    db_path = tmp_path / "history.sqlite"
    store.insert_record(
        db_path,
        stage="Modelos",
        feature_context_key="feature-ctx",
        optuna_context_key="optuna-a",
        model_name="XGBoost",
        threshold_objective="far",
        calibration_method="sigmoid",
        balance_strategy="smote",
        protocols=["robust"],
    )
    store.insert_record(
        db_path,
        stage="Modelos",
        feature_context_key="feature-ctx",
        optuna_context_key="optuna-b",
        model_name="XGBoost",
        threshold_objective="far",
        calibration_method="sigmoid",
        balance_strategy="smote",
        protocols=["conservative"],
    )

    records = store.query_previous_models(
        db_path,
        feature_context_key="feature-ctx",
        optuna_context_key="optuna-a",
        model_name="XGBoost",
        threshold_objective="far",
        calibration_method="sigmoid",
        balance_strategy="smote",
        protocols=["robust"],
    )

    assert len(records) == 1
    assert records[0]["optuna_context_key"] == "optuna-a"


def test_optuna_snapshots_are_independent_records(tmp_path):
    db_path = tmp_path / "history.sqlite"
    first = store.insert_record(
        db_path,
        stage="Optuna",
        record_uid="optuna-run-1",
        feature_context_key="feature-ctx",
        feature_signature_value=store.feature_signature(["a"]),
        model_name="XGBoost",
        metrics={"best_score": 0.1},
    )
    second = store.insert_record(
        db_path,
        stage="Optuna",
        record_uid="optuna-run-2",
        feature_context_key="feature-ctx",
        feature_signature_value=store.feature_signature(["a"]),
        model_name="XGBoost",
        metrics={"best_score": 0.2},
    )

    assert first != second
    records = store.query_previous_optuna(
        db_path,
        feature_context_key="feature-ctx",
        feature_signature_value=store.feature_signature(["a"]),
        model_name="XGBoost",
    )
    assert [record["metrics"]["best_score"] for record in records] == [0.2, 0.1]
