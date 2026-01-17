#!/usr/bin/env python3
"""
Streamlit app to visualize and manage experiment files.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "Resultados"

_TIMESTAMP_RE = re.compile(r"(.+)_\d{8}_\d{6}(?:_\d+)?$")


def _list_result_roots() -> List[Path]:
    roots = [
        path
        for path in ROOT_DIR.iterdir()
        if path.is_dir() and path.name.lower().startswith("resultados")
    ]
    if RESULTS_DIR.exists() and RESULTS_DIR not in roots:
        roots.append(RESULTS_DIR)
    return sorted(roots, key=lambda path: path.name)


def _format_size(num_bytes: int) -> str:
    step = 1024.0
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < step:
            return f"{num_bytes:.0f} {unit}"
        num_bytes /= step
    return f"{num_bytes:.0f} PB"


def _infer_experiment_id(name: str) -> str:
    base = name
    if base.endswith("_importance"):
        base = base[: -len("_importance")]
    if base.endswith("_trials"):
        base = base[: -len("_trials")]
    if base.startswith("feature_selection_"):
        base = base[len("feature_selection_") :]
    if base.startswith("optuna_"):
        base = base[len("optuna_") :]
    match = _TIMESTAMP_RE.match(base)
    if match:
        base = match.group(1)
    return base


def _experiment_label(path: Path, root: Path) -> str:
    stem = _infer_experiment_id(path.stem)
    try:
        rel_parent = path.parent.relative_to(root)
    except ValueError:
        rel_parent = path.parent
    if rel_parent == Path("."):
        return stem
    return f"{rel_parent.as_posix()}/{stem}"


@st.cache_data(show_spinner=False)
def _scan_files(root_paths: Tuple[str, ...]) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for root_str in root_paths:
        root = Path(root_str)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            records.append(
                {
                    "root": root.name,
                    "experiment": _experiment_label(path, root),
                    "name": path.name,
                    "relative_path": str(path.relative_to(root)),
                    "path": str(path),
                    "size_bytes": int(stat.st_size),
                    "size": _format_size(int(stat.st_size)),
                    "modified": datetime.fromtimestamp(stat.st_mtime),
                    "extension": path.suffix.lower() or "(sin extension)",
                }
            )
    if not records:
        return pd.DataFrame(
            columns=[
                "root",
                "experiment",
                "name",
                "relative_path",
                "path",
                "size_bytes",
                "size",
                "modified",
                "extension",
            ]
        )
    df = pd.DataFrame.from_records(records)
    df = df.sort_values(["experiment", "name"]).reset_index(drop=True)
    return df


def _read_jsonl_preview(
    path: Path, max_rows: Optional[int]
) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if max_rows is not None and len(rows) >= max_rows:
                break
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _read_json_preview(path: Path, max_rows: Optional[int]) -> pd.DataFrame:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        if max_rows is None:
            return pd.DataFrame(data)
        return pd.DataFrame(data[:max_rows])
    if isinstance(data, dict):
        return pd.DataFrame([data])
    return pd.DataFrame()


def _preview_table(df: pd.DataFrame) -> None:
    st.dataframe(df, width="stretch")
    if df.empty:
        return
    st.subheader("Estadistica descriptiva")
    st.dataframe(df.describe(include="all").transpose(), width="stretch")


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    search_text = st.text_input("Busqueda (texto)", value="")
    filtered = df.copy()
    if search_text:
        mask = (
            filtered.astype(str)
            .apply(lambda col: col.str.contains(search_text, case=False, na=False))
            .any(axis=1)
        )
        filtered = filtered.loc[mask]

    filter_col = st.selectbox(
        "Filtrar por columna",
        options=["(ninguna)"] + filtered.columns.tolist(),
        index=0,
    )
    if filter_col != "(ninguna)":
        series = filtered[filter_col]
        if pd.api.types.is_numeric_dtype(series):
            min_val = float(series.min())
            max_val = float(series.max())
            if min_val != max_val:
                range_vals = st.slider(
                    "Rango",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                )
                filtered = filtered[
                    (series >= range_vals[0]) & (series <= range_vals[1])
                ]
        else:
            unique_vals = (
                series.dropna().astype(str).unique().tolist()
            )
            if len(unique_vals) <= 50:
                selected_vals = st.multiselect(
                    "Valores",
                    options=sorted(unique_vals),
                    default=sorted(unique_vals),
                )
                if selected_vals:
                    filtered = filtered[
                        series.astype(str).isin(selected_vals)
                    ]
            else:
                text_filter = st.text_input(
                    "Contiene", value=""
                )
                if text_filter:
                    filtered = filtered[
                        series.astype(str).str.contains(
                            text_filter, case=False, na=False
                        )
                    ]
    st.caption(f"Filas despues de filtros: {len(filtered):,}")
    return filtered


def _render_visualization(path: Path) -> None:
    st.subheader("Visualizar")
    ext = path.suffix.lower()
    size_mb = path.stat().st_size / (1024 * 1024)
    st.caption(f"Tamano: {size_mb:.2f} MB")

    load_all = st.checkbox("Cargar todas las filas", value=False)
    if load_all:
        max_rows = None
    else:
        max_rows = st.slider(
            "Filas a cargar",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
        )

    if ext in {".csv", ".txt"}:
        sep_label = st.selectbox(
            "Separador",
            options=[",", ";", "tab", "|"],
            index=0,
        )
        sep = "\t" if sep_label == "tab" else sep_label
        try:
            read_kwargs = {"sep": sep}
            if max_rows is not None:
                read_kwargs["nrows"] = int(max_rows)
            df = pd.read_csv(path, **read_kwargs)
        except Exception as exc:
            st.error(f"No se pudo leer el archivo: {exc}")
            return
        filtered = _apply_filters(df)
        _preview_table(filtered)
        return

    if ext == ".parquet":
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            st.error(f"No se pudo leer el archivo: {exc}")
            return
        if max_rows is not None:
            df = df.head(max_rows)
        filtered = _apply_filters(df)
        _preview_table(filtered)
        return

    if ext == ".jsonl":
        try:
            df = _read_jsonl_preview(path, max_rows)
        except Exception as exc:
            st.error(f"No se pudo leer el archivo: {exc}")
            return
        filtered = _apply_filters(df)
        _preview_table(filtered)
        return

    if ext == ".json":
        if size_mb > 200:
            st.warning("Archivo JSON muy grande para cargar completo.")
            return
        try:
            df = _read_json_preview(path, max_rows)
        except Exception as exc:
            st.error(f"No se pudo leer el archivo: {exc}")
            return
        filtered = _apply_filters(df)
        _preview_table(filtered)
        return

    if ext == ".duckdb":
        try:
            import duckdb
        except ImportError:
            st.error("duckdb no esta instalado.")
            return
        try:
            con = duckdb.connect(str(path), read_only=True)
        except Exception as exc:
            st.error(f"No se pudo abrir DuckDB: {exc}")
            return
        try:
            tables = con.execute("SHOW TABLES").fetchdf()
        except Exception as exc:
            con.close()
            st.error(f"No se pudieron leer las tablas: {exc}")
            return
        if tables.empty:
            st.info("No hay tablas en la base.")
            con.close()
            return
        table_name = st.selectbox(
            "Tabla",
            options=tables["name"].tolist(),
        )
        # Get columns for filter
        try:
            col_info = con.execute(f"PRAGMA table_info('{table_name}')").fetchdf()
            col_names = col_info["name"].tolist()
        except Exception:
            col_names = []
            col_info = pd.DataFrame()

        c1, c2 = st.columns(2)
        with c1:
            filter_col = st.selectbox(
                "Filtrar por columna",
                options=["(ninguna)"] + col_names,
            )
        with c2:
            filter_val = st.text_input(
                "Valor del filtro",
                placeholder="Escribe el valor...",
                help="Texto: busca 'Contiene'. Numeros: busca 'Igual a'.",
                disabled=(filter_col == "(ninguna)")
            )

        sql_where = ""
        if filter_col != "(ninguna)" and filter_val:
            # Determine type
            ctype = ""
            if not col_info.empty:
                matches = col_info[col_info["name"] == filter_col]
                if not matches.empty:
                    ctype = str(matches.iloc[0]["type"]).upper()
            
            is_text = any(t in ctype for t in ["VARCHAR", "STRING", "TEXT", "CHAR"])
            
            # Simple SQL generation
            # Check if user typed an operator
            if any(filter_val.strip().startswith(op) for op in [">", "<", "=", "!"]):
                 sql_where = f"{filter_col} {filter_val}"
            else:
                 # Default to generalized contains search
                 sql_where = f"CAST({filter_col} AS VARCHAR) ILIKE '%{filter_val}%'"

        try:
            query = f"SELECT * FROM {table_name}"
            if sql_where:
                query += f" WHERE {sql_where}"

                
            if max_rows is not None:
                query = f"{query} LIMIT {int(max_rows)}"
            
            df = con.execute(query).fetchdf()
        except Exception as exc:
            st.error(f"Error en consulta SQL: {exc}")
            con.close()
            return
        con.close()

        # Direct preview without extra in-memory filters (since SQL handles it)
        _preview_table(df)
        return

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            lines = []
            if max_rows is None:
                for line in handle:
                    lines.append(line.rstrip("\n"))
            else:
                for _ in range(int(max_rows)):
                    line = handle.readline()
                    if not line:
                        break
                    lines.append(line.rstrip("\n"))
    except Exception as exc:
        st.error(f"No se pudo leer el archivo: {exc}")
        return
    if lines:
        st.text("\n".join(lines))
    else:
        st.info("No hay contenido para mostrar.")


def _render_export(path: Path) -> None:
    st.subheader("Exportar")
    export_name = st.text_input(
        "Nombre de archivo",
        value=path.name,
    )
    destination = st.selectbox(
        "Destino",
        ["Misma carpeta", "Resultados/exports", "Otro (ruta manual)"],
    )
    if destination == "Resultados/exports":
        dest_dir = RESULTS_DIR / "exports"
    elif destination == "Otro (ruta manual)":
        dest_text = st.text_input("Ruta destino", value=str(path.parent))
        dest_dir = Path(dest_text)
    else:
        dest_dir = path.parent

    if st.button("Exportar archivo"):
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / export_name
            shutil.copy2(path, dest_path)
        except Exception as exc:
            st.error(f"No se pudo exportar: {exc}")
        else:
            st.success(f"Exportado en {dest_path}")


def _render_delete(path: Path) -> None:
    st.subheader("Borrar")
    st.warning("Esta accion no se puede deshacer.")
    confirm = st.checkbox("Confirmo borrar este archivo.")
    keyword = st.text_input("Escriba BORRAR para confirmar", value="")
    if st.button("Borrar archivo"):
        if not confirm or keyword.strip().upper() != "BORRAR":
            st.warning("Confirme el borrado para continuar.")
            return
        try:
            path.unlink()
        except Exception as exc:
            st.error(f"No se pudo borrar: {exc}")
            return
        st.success("Archivo borrado.")
        st.cache_data.clear()


def main(*, set_page_config: bool = True, show_exit_button: bool = True) -> None:
    if set_page_config:
        st.set_page_config(page_title="Files", layout="wide")
    st.title("Files")

    if show_exit_button and st.sidebar.button("Cerrar app"):
        os._exit(0)

    roots = _list_result_roots()
    if not roots:
        st.warning("No se encontraron carpetas Resultados.")
        return

    root_names = [root.name for root in roots]
    selected_roots = st.multiselect(
        "Carpetas",
        options=root_names,
        default=root_names,
    )
    if st.button("Actualizar listado"):
        st.cache_data.clear()

    chosen_roots = tuple(
        str(root) for root in roots if root.name in selected_roots
    )
    if not chosen_roots:
        st.info("Seleccione al menos una carpeta.")
        return

    files_df = _scan_files(chosen_roots)
    if files_df.empty:
        st.info("No se encontraron archivos.")
        return

    extension_filter = st.multiselect(
        "Extensiones",
        options=sorted(files_df["extension"].unique().tolist()),
        default=sorted(files_df["extension"].unique().tolist()),
    )
    filtered_files = files_df[files_df["extension"].isin(extension_filter)]
    search_exp = st.text_input("Buscar experimento/archivo", value="")
    if search_exp:
        mask = (
            filtered_files["experiment"].str.contains(
                search_exp, case=False, na=False
            )
            | filtered_files["name"].str.contains(
                search_exp, case=False, na=False
            )
            | filtered_files["relative_path"].str.contains(
                search_exp, case=False, na=False
            )
        )
        filtered_files = filtered_files.loc[mask]

    if filtered_files.empty:
        st.info("No hay archivos para los filtros aplicados.")
        return

    summary = (
        filtered_files.groupby(["root", "experiment"])
        .agg(
            archivos=("name", "count"),
            total_mb=("size_bytes", lambda x: float(x.sum()) / (1024 * 1024)),
        )
        .reset_index()
    )
    st.subheader("Experimentos")
    summary["total_mb"] = summary["total_mb"].round(2)
    st.dataframe(summary, width="stretch")

    experiment_options = (
        summary.apply(lambda row: f"{row['root']} | {row['experiment']}", axis=1)
        .tolist()
    )
    selected_exp = st.selectbox(
        "Experimento",
        options=experiment_options,
    )
    selected_root, _, selected_experiment = selected_exp.partition(" | ")
    exp_files = filtered_files[
        (filtered_files["root"] == selected_root)
        & (filtered_files["experiment"] == selected_experiment)
    ].copy()
    exp_files = exp_files.sort_values("name")
    st.subheader("Archivos")
    st.dataframe(
        exp_files[["name", "relative_path", "size", "modified"]],
        width="stretch",
    )

    file_choice = st.selectbox(
        "Archivo",
        options=exp_files["relative_path"].tolist(),
    )
    file_row = exp_files[exp_files["relative_path"] == file_choice].iloc[0]
    file_path = Path(file_row["path"])
    st.caption(f"Ruta: {file_path}")

    tab_visual, tab_export, tab_delete = st.tabs(
        ["Visualizar", "Exportar", "Borrar"]
    )
    with tab_visual:
        _render_visualization(file_path)
    with tab_export:
        _render_export(file_path)
    with tab_delete:
        _render_delete(file_path)


if __name__ == "__main__":
    main()
