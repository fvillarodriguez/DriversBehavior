from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT_DIR / "docs"
BUILD_DIR = DOCS_DIR / "_latex_build"


def render_tex_to_pdf(
    tex_path: str | Path,
    out_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    tex_path = Path(tex_path).resolve()
    out_dir = Path(out_dir).resolve() if out_dir else tex_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "latexmk",
        "-pdf",
        "-interaction=nonstopmode",
        "-halt-on-error",
    ]
    if force:
        cmd.append("-f")
    cmd.extend(
        [
            f"-outdir={out_dir}",
            str(tex_path),
        ]
    )
    subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(tex_path.parent),
    )
    return out_dir / (tex_path.stem + ".pdf")


def open_pdf(pdf_path: Path) -> None:
    if sys.platform.startswith("darwin"):
        subprocess.run(["open", str(pdf_path)], check=False)
    elif os.name == "nt":
        os.startfile(str(pdf_path))  # type: ignore[attr-defined]
    else:
        subprocess.run(["xdg-open", str(pdf_path)], check=False)


def _list_tex_files() -> list[Path]:
    if not DOCS_DIR.exists():
        return []
    return sorted(DOCS_DIR.rglob("*.tex"))


def _clean_build_dir(build_dir: Path) -> None:
    if build_dir.exists():
        for item in build_dir.iterdir():
            try:
                if item.is_dir():
                    for sub in item.rglob("*"):
                        if sub.is_file():
                            sub.unlink()
                    item.rmdir()
                else:
                    item.unlink()
            except Exception:
                continue


def Latex(set_page_config: bool = True, show_exit_button: bool = True) -> None:
    if set_page_config:
        st.set_page_config(page_title="LaTeX Viewer", layout="wide")

    st.title("LaTeX Viewer")
    st.caption("Compila archivos .tex dentro de `docs/` usando `latexmk` y abre el PDF.")

    if not DOCS_DIR.exists():
        st.error(f"No existe el directorio: {DOCS_DIR}")
        return

    latexmk_path = shutil.which("latexmk")
    if not latexmk_path:
        st.error("No se encontró `latexmk`. Instale latexmk + MacTeX. -> pip install latexmk & brew install --cask mactex")
        return

    tex_files = _list_tex_files()
    if not tex_files:
        st.warning("No se encontraron archivos .tex en `docs/`.")
        return

    options = [str(p.relative_to(DOCS_DIR)) for p in tex_files]
    selected = st.selectbox("Archivo .tex en `docs/`", options=options, index=0)
    out_dir = BUILD_DIR

    col1, col2 = st.columns([2, 1])
    with col1:
        custom_out = st.text_input("Carpeta de salida (opcional)", value=str(out_dir))
        force_compile = st.checkbox("Forzar recompilación (latexmk -f)", value=False)
    with col2:
        open_after = st.checkbox("Abrir al finalizar", value=True)
        if st.button("Limpiar build"):
            _clean_build_dir(BUILD_DIR)
            st.success("Build limpio.")

    if st.button("Compilar y abrir", type="primary"):
        tex_path = DOCS_DIR / selected
        try:
            pdf_path = render_tex_to_pdf(tex_path, out_dir=custom_out or None, force=force_compile)
        except subprocess.CalledProcessError as e:
            st.error("Falló la compilación con latexmk.")
            if e.stdout:
                st.code(e.stdout, language="text")
            if e.stderr:
                st.code(e.stderr, language="text")
            return
        except Exception as e:
            st.error(f"Error compilando: {e}")
            return

        st.success(f"PDF generado: {pdf_path}")
        if open_after:
            open_pdf(pdf_path)

        try:
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "Descargar PDF",
                    data=f,
                    file_name=pdf_path.name,
                    mime="application/pdf",
                )
        except Exception:
            st.info("No se pudo preparar el archivo para descarga.")


if __name__ == "__main__":
    Latex(set_page_config=True)
