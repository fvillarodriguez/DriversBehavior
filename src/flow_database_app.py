#!/usr/bin/env python3
"""
Streamlit app to manage the flow database (DuckDB).
"""
from __future__ import annotations

import fnmatch
import inspect
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import (  # noqa: E402
    clear_flow_table,
    get_flow_db_summary,
    get_flow_table_columns,
    get_flow_table_sample,
    import_flujos_to_duckdb,
)

DATA_DIR = ROOT_DIR / "Datos"
DB_PATH = DATA_DIR / "flujos.duckdb"
FLOW_PATTERNS = ("flujo*.csv",)


def _supports_progress() -> bool:
    try:
        sig = inspect.signature(import_flujos_to_duckdb)
    except (ValueError, TypeError):
        return False
    return "progress_callback" in sig.parameters


def _run_import(
    *,
    csv_path: Path,
    replace: bool,
    progress_callback: Optional[object],
) -> int:
    kwargs = {
        "csv_path": str(csv_path),
        "db_path": DB_PATH,
        "replace": replace,
    }
    if progress_callback is not None and _supports_progress():
        kwargs["progress_callback"] = progress_callback
    try:
        return import_flujos_to_duckdb(**kwargs)
    except TypeError as exc:
        if "progress_callback" in str(exc):
            kwargs.pop("progress_callback", None)
            return import_flujos_to_duckdb(**kwargs)
        raise


def _format_ts(value) -> str:
    if value is None:
        return "-"
    try:
        return value.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(value)


def _list_flow_csv_files(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []
    files = []
    for entry in base_dir.iterdir():
        if not entry.is_file() or entry.suffix.lower() != ".csv":
            continue
        if any(fnmatch.fnmatch(entry.name.lower(), pattern) for pattern in FLOW_PATTERNS):
            files.append(entry)
    return sorted(files)


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix or ".csv"
    for idx in range(1, 1000):
        candidate = path.with_name(f"{stem}_{idx}{suffix}")
        if not candidate.exists():
            return candidate
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{stem}_{stamp}{suffix}")


def _save_uploaded_csv(uploaded_file, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    raw_name = Path(uploaded_file.name or "flujos_upload.csv").name
    filename = raw_name if raw_name else "flujos_upload.csv"
    destination = _unique_path(target_dir / filename)
    with open(destination, "wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return destination


def _select_csv_source(key_prefix: str) -> Tuple[str, Optional[Path], Optional[object]]:
    source = st.radio(
        "Origen del CSV",
        ["Archivo en Datos/", "Subir CSV"],
        key=f"{key_prefix}_source",
        horizontal=True,
    )
    if source == "Archivo en Datos/":
        files = _list_flow_csv_files(DATA_DIR)
        if not files:
            st.warning("No se encontraron archivos flujo*.csv en Datos/.")
            return source, None, None
        names = [path.name for path in files]
        selected = st.selectbox(
            "Archivo CSV",
            names,
            key=f"{key_prefix}_file",
        )
        return source, DATA_DIR / selected, None
    uploaded = st.file_uploader(
        "Subir CSV de flujos",
        type=["csv"],
        key=f"{key_prefix}_upload",
    )
    return source, None, uploaded


def _resolve_csv_path(
    source: str, selected_path: Optional[Path], uploaded
) -> Optional[Path]:
    if source == "Archivo en Datos/":
        if selected_path is None:
            st.error("Seleccione un archivo CSV valido.")
            return None
        return selected_path
    if uploaded is None:
        st.error("Suba un archivo CSV para continuar.")
        return None
    saved_path = _save_uploaded_csv(uploaded, DATA_DIR)
    st.caption(f"Archivo guardado en: {saved_path}")
    return saved_path


def _render_summary() -> None:
    try:
        summary = get_flow_db_summary(db_path=DB_PATH)
    except ImportError as exc:
        st.error(str(exc))
        st.stop()
    col1, col2, col3 = st.columns(3)
    col1.metric("Filas", f"{summary.row_count:,}")
    col2.metric("Fecha min", _format_ts(summary.min_timestamp))
    col3.metric("Fecha max", _format_ts(summary.max_timestamp))
    st.caption(f"Archivo DuckDB: {summary.db_path}")


def main(*, set_page_config: bool = True, show_exit_button: bool = True) -> None:
    if set_page_config:
        st.set_page_config(page_title="Flow database", layout="wide")
    st.title("Flow database (DuckDB)")

    if show_exit_button and st.sidebar.button("Cerrar app"):
        os._exit(0)

    summary_container = st.empty()
    with summary_container.container():
        _render_summary()

    tabs = st.tabs(
        [
            "Importar / Anexar",
            "Reemplazar",
            "Vaciar tabla",
            "Esquema y muestra",
        ]
    )

    with tabs[0]:
        st.write("Importa un CSV y lo anexa a la tabla actual.")
        source, selected_path, uploaded = _select_csv_source("append")
        progress_placeholder = st.empty()
        progress_status = st.empty()
        if st.button("Importar y anexar", key="append_button"):
            csv_path = _resolve_csv_path(source, selected_path, uploaded)
            if csv_path is None:
                st.stop()
            progress_bar = progress_placeholder.progress(0.0)

            def _update_progress(ratio: float, message: str) -> None:
                progress_bar.progress(ratio)
                progress_status.caption(message)

            with st.spinner("Importando datos..."):
                inserted = _run_import(
                    csv_path=csv_path,
                    replace=False,
                    progress_callback=_update_progress,
                )
            progress_bar.progress(1.0)
            if inserted:
                st.success(f"Se agregaron {inserted:,} filas.")
            else:
                st.warning("No se agregaron filas.")
            with summary_container.container():
                _render_summary()

    with tabs[1]:
        st.write("Reemplaza la tabla completa con un CSV.")
        source, selected_path, uploaded = _select_csv_source("replace")
        confirm_text = st.text_input(
            "Escriba REPLACE para confirmar",
            key="replace_confirm",
        )
        can_replace = confirm_text.strip().lower() == "replace"
        if st.button(
            "Reemplazar tabla",
            key="replace_button",
            disabled=not can_replace,
        ):
            csv_path = _resolve_csv_path(source, selected_path, uploaded)
            if csv_path is None:
                st.stop()
            progress_placeholder = st.empty()
            progress_status = st.empty()
            progress_bar = progress_placeholder.progress(0.0)

            def _update_progress(ratio: float, message: str) -> None:
                progress_bar.progress(ratio)
                progress_status.caption(message)

            with st.spinner("Reemplazando datos..."):
                inserted = _run_import(
                    csv_path=csv_path,
                    replace=True,
                    progress_callback=_update_progress,
                )
            progress_bar.progress(1.0)
            st.success(f"Se cargaron {inserted:,} filas nuevas.")
            with summary_container.container():
                _render_summary()

    with tabs[2]:
        st.write("Elimina todos los registros de la tabla.")
        confirm_text = st.text_input(
            "Escriba DELETE para confirmar",
            key="delete_confirm",
        )
        can_delete = confirm_text.strip().upper() == "DELETE"
        if st.button(
            "Vaciar tabla",
            key="delete_button",
            disabled=not can_delete,
        ):
            with st.spinner("Eliminando registros..."):
                removed = clear_flow_table(db_path=DB_PATH)
            st.success(f"Se eliminaron {removed:,} filas.")
            with summary_container.container():
                _render_summary()

    with tabs[3]:
        st.write("Explora el esquema y una muestra de la tabla.")
        try:
            columns = get_flow_table_columns(db_path=DB_PATH)
        except ImportError as exc:
            st.error(str(exc))
            st.stop()
        if columns:
            st.subheader("Columnas")
            st.dataframe(pd.DataFrame(columns), width="stretch")
        else:
            st.info("La tabla no tiene columnas definidas.")

        sample_limit = st.slider(
            "Filas de muestra",
            min_value=1,
            max_value=200,
            value=5,
            step=1,
        )
        try:
            sample_df = get_flow_table_sample(
                limit=int(sample_limit), db_path=DB_PATH
            )
        except ImportError as exc:
            st.error(str(exc))
            st.stop()
        if sample_df.empty:
            st.warning("La tabla esta vacia o no hay datos disponibles.")
        else:
            st.subheader("Muestra")
            st.dataframe(sample_df, width="stretch")


if __name__ == "__main__":
    main()
