#!/usr/bin/env python3
#export DYLD_LIBRARY_PATH="/opt/homebrew/opt/gl2ps/lib:/opt/homebrew/opt/open-scene-graph/lib:/opt/homebrew/opt/gdal/lib:/opt/homebrew/opt/ffmpeg/lib:$HOME/Library/Python/3.12/lib/python/site-packages/pyarrow:${DYLD_LIBRARY_PATH}"

#source .venv/bin/activate
#export DYLD_LIBRARY_PATH="/opt/homebrew/opt/gl2ps/lib:/opt/homebrew/opt/open-scene-graph/lib:/opt/homebrew/opt/gdal/lib:/opt/homebrew/opt/ffmpeg/lib:$HOME/Library/Python/3.12/lib/python/site-packages/pyarrow:$DYLD_LIBRARY_PATH"

"""
CLI tool to explore vehicle flows, run SUMO pipelines, and clustering workflows.
"""
from __future__ import annotations

import os
import shutil
import sys
import subprocess
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from decimal import Decimal
from numbers import Integral, Number
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("PANDAS_NO_ARROW", "1")

try:
    import pyarrow 
    def _ensure_pyarrow_shims() -> None:
        lib = getattr(pyarrow, "lib", None)
        if lib is None:
            return
        shim_map = {
            "is_bool": lambda value: isinstance(value, bool),
            "is_integer": lambda value: isinstance(value, Integral) and not isinstance(value, bool),
            "is_float": lambda value: isinstance(value, float),
            "is_complex": lambda value: isinstance(value, complex),
            "is_scalar": lambda value: isinstance(value, (Number, str, bytes, bool, Decimal)),
            "is_decimal": lambda value: isinstance(value, Decimal),
            "is_interval": lambda value: False,
            "is_list_like": lambda value: isinstance(value, Iterable) and not isinstance(value, (str, bytes)),
            "is_iterator": lambda value: isinstance(value, Iterator),
            "no_default": object(),
        }
        for name, fn in shim_map.items():
            if not hasattr(lib, name):
                setattr(lib, name, fn)
    _ensure_pyarrow_shims()
except ImportError:
    pyarrow = None  

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from SUMO import SegmentFilter, SUMOResult, prompt_segment_filter, run_sumo_pipeline  # noqa: E402
from utils import (  # noqa: E402
    FlowColumns,
    clear_flow_table,
    get_flow_db_summary,
    get_flow_table_columns,
    get_flow_table_sample,
    import_flujos_to_duckdb,
    load_flujos,
    load_porticos,
)
from clustering import (  # noqa: E402
    handle_cluster_statistics,
    handle_cluster_features_visualization,
    handle_cluster_visualization,
    handle_clusterization,
    has_cluster_features_db,
    list_cluster_summary_files,
    list_cluster_label_files,
)

SIMULATION_DIR = ROOT_DIR / "simulación"
BREW_DYLD_DIRS = [
    Path("/opt/homebrew/opt/gl2ps/lib"),
    Path("/opt/homebrew/opt/open-scene-graph/lib"),
    Path("/opt/homebrew/opt/gdal/lib"),
    Path("/opt/homebrew/opt/ffmpeg/lib"),
]
_PYARROW_LIB_WARNING_SHOWN = False


def _normalize_dyld_chunks(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    normalized: List[str] = []
    for raw_value in values:
        if not raw_value:
            continue
        for chunk in raw_value.split(":"):
            path = chunk.strip()
            if not path or path in seen:
                continue
            if Path(path).expanduser().exists():
                seen.add(path)
                normalized.append(path)
    return normalized


def _pyarrow_lib_dir() -> Optional[Path]:
    override = os.environ.get("SUMO_PYARROW_LIB_DIR")
    if override:
        override_path = Path(override).expanduser()
        if override_path.exists():
            return override_path
    py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = []
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if virtual_env:
        candidates.append(Path(virtual_env) / "lib" / py_version / "site-packages" / "pyarrow")
    home_py = (
        Path.home()
        / "Library"
        / "Python"
        / f"{sys.version_info.major}.{sys.version_info.minor}"
        / "lib/python/site-packages/pyarrow"
    )
    candidates.append(home_py)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _warn_missing_pyarrow_lib() -> None:
    global _PYARROW_LIB_WARNING_SHOWN
    if _PYARROW_LIB_WARNING_SHOWN:
        return
    print(
        "⚠️ Could not find the pyarrow/libarrow folder. "
        "Install pyarrow>=18 with 'pip install --user pyarrow==18.1.0' "
        "or set SUMO_PYARROW_LIB_DIR to point to those binaries."
    )
    _PYARROW_LIB_WARNING_SHOWN = True


def _build_sumo_subprocess_env() -> Dict[str, str]:
    env = os.environ.copy()
    if sys.platform != "darwin":
        return env
    dyld_parts: List[str] = []
    dyld_parts.extend(str(path) for path in BREW_DYLD_DIRS if path.exists())
    pyarrow_dir = _pyarrow_lib_dir()
    if pyarrow_dir:
        dyld_parts.append(str(pyarrow_dir))
    else:
        if "pyarrow" not in env.get("DYLD_LIBRARY_PATH", ""):
            _warn_missing_pyarrow_lib()
    extra = os.environ.get("SUMO_EXTRA_DYLD_PATHS")
    if extra:
        dyld_parts.append(extra)
    existing = env.get("DYLD_LIBRARY_PATH")
    if existing:
        dyld_parts.append(existing)
    normalized = _normalize_dyld_chunks(dyld_parts)
    if normalized:
        env["DYLD_LIBRARY_PATH"] = ":".join(normalized)
    return env


@dataclass
class AnalysisSession:
    flujos_df: Optional[pd.DataFrame] = None
    porticos_df: Optional[pd.DataFrame] = None
    somu_result: Optional[SUMOResult] = None


def find_sumo_binary(executable: str) -> Optional[Path]:
    """
    Locate a SUMO executable by checking SUMO_HOME/bin and the system PATH.
    """
    candidates: List[Path] = []
    sumo_home = os.environ.get("SUMO_HOME")
    suffixes = [""]
    if os.name == "nt" and not executable.lower().endswith(".exe"):
        suffixes.append(".exe")
    if sumo_home:
        for suffix in suffixes:
            candidates.append(Path(sumo_home) / "bin" / f"{executable}{suffix}")
    for suffix in suffixes:
        which_path = shutil.which(f"{executable}{suffix}")
        if which_path:
            candidates.append(Path(which_path))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def prompt_path(
    prompt_message: str,
    default: Optional[Path] = None,
    must_exist: bool = True,
) -> Optional[Path]:
    """
    Prompts the user for a path, allowing Enter to accept the default value.
    """
    default_text = f" [{default}]" if default else ""
    while True:
        user_input = input(f"{prompt_message}{default_text}: ").strip()
        if user_input.lower() in {"q", "quit", "salir"}:
            return None
        if not user_input:
            if default is None:
                print("⚠️ You must enter a path or type 'q' to cancel.")
                continue
            path_value = default
        else:
            path_value = Path(user_input).expanduser()
        if must_exist and not path_value.exists():
            print(f"⚠️ Path '{path_value}' does not exist. Try again.")
            continue
        return path_value


def print_somu_summary(result: SUMOResult) -> None:
    clean_count = len(result.clean_events)
    trajectories_count = result.trajectories["trip_id"].nunique() if not result.trajectories.empty else 0
    segments_count = len(result.segments)
    macro_count = len(result.macro_metrics)
    headway_count = len(result.headways)

    print("\n📊 SUMO summary")
    if result.segment_filter is not None:
        print(f"Analyzed segment: {result.segment_filter.description()}")
    print(f"Clean detections: {clean_count:,}")
    print(f"Detected trips: {trajectories_count:,}")
    print(f"Reconstructed segments: {segments_count:,}")
    print(f"Macro windows (flow/speed): {macro_count:,}")
    print(f"Computed headways: {headway_count:,}")
    if result.sumo_trips:
        print(f"SUMO trips generated: {len(result.sumo_trips):,}")
        if result.sumo_trips_path:
            print(f"Trips XML file: {result.sumo_trips_path}")
        if result.depart_summary_path:
            print(f"Depart summary file: {result.depart_summary_path}")
    if result.sumo_warning:
        print(f"⚠️ SUMO warning: {result.sumo_warning}")



def handle_somu_pipeline(session: AnalysisSession) -> None:
    fc = FlowColumns()
    if session.flujos_df is None:
        print("⚠️ No flow data is loaded. Loading now...")
        flujos_df = load_flujos()
        if flujos_df is None:
            print("❌ Flow data was not loaded.")
            return
        session.flujos_df = flujos_df

    if session.porticos_df is None:
        print("⚠️ No gantry data is loaded. Loading now...")
        porticos_df = load_porticos()
        if porticos_df is None:
            print("❌ Gantry data was not loaded.")
            return
        session.porticos_df = porticos_df

    segment_filter = prompt_segment_filter(session.porticos_df)

    print("\n🚀 Running SUMO pipeline...")
    try:
        result = run_sumo_pipeline(
            session.flujos_df,
            session.porticos_df,
            flow_cols=fc,
            output_dir=SIMULATION_DIR,
            segment_filter=segment_filter,
        )
    except ValueError as exc:
        print(f"❌ SUMO error: {exc}")
        return

    session.somu_result = result
    print_somu_summary(result)


def _get_existing_sumo_trips_path(session: AnalysisSession) -> Optional[Path]:
    candidates: List[Path] = []
    if session.somu_result and session.somu_result.sumo_trips_path:
        candidates.append(Path(session.somu_result.sumo_trips_path))
    primary = SIMULATION_DIR / "sumo_trips.rou.xml"
    fallback = ROOT_DIR / "Resultados" / "sumo_trips.rou.xml"
    candidates.extend([primary, fallback])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def handle_generate_routes_file(session: AnalysisSession) -> None:
    trips_path = _get_existing_sumo_trips_path(session)
    if trips_path is None:
        print(
            "⚠️ sumo_trips.rou.xml was not found. Run the SUMO pipeline first to generate it."
        )
        return

    default_net = SIMULATION_DIR / "highway.net.xml"
    if not default_net.exists():
        default_net = None
    net_path = prompt_path(
        "Network (.net.xml) file to use",
        default=default_net,
        must_exist=True,
    )
    if net_path is None:
        print("Operation canceled.")
        return

    default_routes = SIMULATION_DIR / "routes.rou.xml"
    if not SIMULATION_DIR.exists():
        default_routes = ROOT_DIR / "routes.rou.xml"
    output_path = prompt_path(
        "Output path for routes.rou.xml",
        default=default_routes,
        must_exist=False,
    )
    if output_path is None:
        print("Operation canceled.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    duarouter_bin = find_sumo_binary("duarouter")
    if duarouter_bin is None:
        print(
            "❌ 'duarouter' was not found. Set the SUMO_HOME variable or add the SUMO binaries to your PATH."
        )
        return

    cmd = [
        str(duarouter_bin),
        "-n",
        str(net_path),
        "-r",
        str(trips_path),
        "-o",
        str(output_path),
        "--departlane",
        "best",
        "--unsorted-input",
        "true",
        "--ignore-errors",
    ]
    print("\n🚧 Running duarouter...")
    try:
        subprocess.run(cmd, check=True, env=_build_sumo_subprocess_env())
    except subprocess.CalledProcessError as exc:
        print(f"❌ duarouter exited with error (code {exc.returncode}).")
        return

    print(f"✔️ Generated routes.rou.xml at: {output_path}")


def handle_generate_tripinfo(session: AnalysisSession) -> None:
    sumo_bin = find_sumo_binary("sumo")
    if sumo_bin is None:
        print(
            "❌ The 'sumo' executable was not found. Set SUMO_HOME or ensure SUMO is on the PATH."
        )
        return

    default_cfg = SIMULATION_DIR / "sample.sumocfg"
    if not default_cfg.exists():
        default_cfg = None
    cfg_path = prompt_path(
        "SUMO configuration (.sumocfg) file to run",
        default=default_cfg,
        must_exist=True,
    )
    if cfg_path is None:
        print("Operation canceled.")
        return

    default_tripinfo = SIMULATION_DIR / "tripinfo.xml"
    if not SIMULATION_DIR.exists():
        default_tripinfo = ROOT_DIR / "tripinfo.xml"
    output_path = prompt_path(
        "Output path for tripinfo.xml",
        default=default_tripinfo,
        must_exist=False,
    )
    if output_path is None:
        print("Operation canceled.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(sumo_bin),
        "-c",
        str(cfg_path),
        "--tripinfo-output",
        str(output_path),
        "--no-step-log",
        "true",
        "--duration-log.disable",
        "true",
    ]
    print("\n🚦 Running SUMO...")
    try:
        subprocess.run(cmd, check=True, env=_build_sumo_subprocess_env())
    except subprocess.CalledProcessError as exc:
        print(f"❌ SUMO exited with error (code {exc.returncode}).")
        return

    print(f"✔️ Generated tripinfo.xml at: {output_path}")


def handle_flow_database_admin(session: AnalysisSession) -> None:
    def _fmt(ts: Optional[pd.Timestamp]) -> str:
        if ts is None:
            return "-"
        return ts.strftime("%Y-%m-%d %H:%M")

    while True:
        try:
            summary = get_flow_db_summary()
        except ImportError as exc:
            print(f"❌ {exc}")
            return

        print(
            "\n=== Flow database (DuckDB) ===\n"
            f"File: {summary.db_path}\n"
            f"Rows stored: {summary.row_count:,}\n"
            f"Time range: {_fmt(summary.min_timestamp)} → {_fmt(summary.max_timestamp)}\n"
        )
        print(
            "Options:\n"
            "  [1] Import CSV and append to the database\n"
            "  [2] Replace the database with a CSV\n"
            "  [3] Empty the flow table entirely\n"
            "  [4] View the columns and a sample of the table\n"
            "  [q] Return to the main menu\n"
        )
        choice = input("Select an option: ").strip().lower()
        if choice in {"q", "quit", "salir"}:
            break

        if choice == "1":
            inserted = import_flujos_to_duckdb()
            if inserted:
                print(f"✔️ Added {inserted:,} rows to the database.")
                session.flujos_df = None
                session.somu_result = None
            else:
                print("⚠️ No rows were added.")
        elif choice == "2":
            confirm = input(
                "This action will fully replace the table. Type 'REPLACE' to continue: "
            ).strip().lower()
            if confirm != "replace":
                print("Operation canceled.")
                continue
            inserted = import_flujos_to_duckdb(replace=True)
            print(f"✔️ Loaded {inserted:,} new rows.")
            session.flujos_df = None
            session.somu_result = None
        elif choice == "3":
            confirm = input(
                "Type 'DELETE' to remove all records: "
            ).strip().upper()
            if confirm != "DELETE":
                print("Operation canceled.")
                continue
            removed = clear_flow_table()
            print(f"🗑️ Removed {removed:,} rows.")
            session.flujos_df = None
            session.somu_result = None
        elif choice == "4":
            columns = get_flow_table_columns()
            if not columns:
                print("⚠️ The table has no defined columns.")
            else:
                print("\n📋 Current columns:")
                for col in columns:
                    flags = []
                    if col["notnull"]:
                        flags.append("NOT NULL")
                    if col["pk"]:
                        flags.append("PK")
                    extra = f" ({', '.join(flags)})" if flags else ""
                    print(f"  - {col['name']} :: {col['type']}{extra}")

                default_limit = 5
                limit_input = input(
                    f"\nHow many sample rows would you like to see? [Enter={default_limit}]: "
                ).strip()
                try:
                    limit = int(limit_input) if limit_input else default_limit
                except ValueError:
                    print("Invalid value; the default sample will be used.")
                    limit = default_limit
                sample_df = get_flow_table_sample(limit=max(1, limit))
                if sample_df.empty:
                    print("⚠️ The table is empty or the sample could not be retrieved.")
                else:
                    print(f"\n👀 Sample of {len(sample_df)} rows:")
                    with pd.option_context("display.max_rows", None, "display.max_columns", None):
                        print(sample_df.to_string(index=False))
        else:
            print("Invalid option.")


def show_main_menu() -> None:
    lines = [
        "\n=== Main menu ===\n",
        "1) Manage the flow database (DuckDB)\n",
        "2) SUMO\n",
        "3) Clustering\n",
    ]
    lines.append("q) Exit\n")
    print("".join(lines))


def show_sumo_menu() -> None:
    lines = [
        "\n=== SUMO ===\n",
        "1) Run the SUMO pipeline (trajectories and metrics)\n",
        "2) Generate routes.rou.xml with duarouter\n",
        "3) Run SUMO to produce tripinfo.xml\n",
        "b) Back\n",
        "q) Exit\n",
    ]
    print("".join(lines))


def show_clustering_menu(option_labels: List[str]) -> None:
    lines = ["\n=== Clustering ===\n"]
    for idx, label in enumerate(option_labels, start=1):
        lines.append(f"{idx}) {label}\n")
    lines.extend(["b) Back\n", "q) Exit\n"])
    print("".join(lines))


def handle_sumo_menu(session: AnalysisSession) -> bool:
    while True:
        show_sumo_menu()
        choice = input("Select an option: ").strip().lower()
        if choice in {"b", "back"}:
            return False
        if choice in {"q", "quit", "salir"}:
            return True
        if choice == "1":
            handle_somu_pipeline(session)
        elif choice == "2":
            handle_generate_routes_file(session)
        elif choice == "3":
            handle_generate_tripinfo(session)
        else:
            print("Invalid option.")


def handle_clustering_menu(session: AnalysisSession) -> bool:
    while True:
        has_cluster_stats = bool(list_cluster_summary_files())
        has_cluster_visual = bool(list_cluster_label_files())
        has_cluster_features = has_cluster_features_db()
        options = [
            (
                "Cluster vehicles (K-means / GMM / HDBSCAN)",
                lambda: handle_clusterization(session),
            )
        ]
        if has_cluster_stats:
            options.append(("Cluster statistics from existing results", handle_cluster_statistics))
        if has_cluster_visual:
            options.append(("Visualize clusters (Streamlit)", handle_cluster_visualization))
        if has_cluster_features:
            options.append(("Visualize features for clustering (Streamlit)", handle_cluster_features_visualization))
        show_clustering_menu([label for label, _action in options])
        choice = input("Select an option: ").strip().lower()
        if choice in {"b", "back"}:
            return False
        if choice in {"q", "quit", "salir"}:
            return True
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                options[idx][1]()
            else:
                print("Invalid option.")
        else:
            print("Invalid option.")


def main() -> None:
    session = AnalysisSession()

    try:
        while True:
            show_main_menu()
            choice = input("Select an option: ").strip().lower()
            if choice in {"q", "quit", "salir"}:
                print("Goodbye.")
                break
            if choice == "1":
                handle_flow_database_admin(session)
                continue
            if choice == "2":
                should_exit = handle_sumo_menu(session)
            elif choice == "3":
                should_exit = handle_clustering_menu(session)
            else:
                print("Invalid option.")
                continue
            if should_exit:
                print("Goodbye.")
                break
    except KeyboardInterrupt:
        print("\nInterrupted by the user.")


if __name__ == "__main__":
    os.system('cls' if os.name == 'nt' else 'clear')
    main()
