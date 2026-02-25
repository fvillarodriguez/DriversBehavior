from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

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


def _list_pdf_files() -> list[Path]:
    if not DOCS_DIR.exists():
        return []
    return sorted(DOCS_DIR.rglob("*.pdf"))


def _list_viewer_files() -> list[Path]:
    return sorted(_list_tex_files() + _list_pdf_files())


def _embed_pdf(pdf_path: Path, height: int = 900) -> None:
    data = pdf_path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    html = f"""
    <div id="pdf-root" style="border:1px solid #ddd; border-radius:8px; overflow:auto; height:{height}px; padding:8px; background:#0b0b0b;">
      <div id="pdf-status" style="color:#ddd; font-family:sans-serif; padding:8px;">Cargando PDF...</div>
      <div id="pdf-pages"></div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.16.105/pdf.min.js"></script>
    <script>
      (async () => {{
        const statusEl = document.getElementById("pdf-status");
        const pagesEl = document.getElementById("pdf-pages");
        try {{
          if (!window.pdfjsLib) {{
            throw new Error("pdf.js no cargó");
          }}

          window.pdfjsLib.GlobalWorkerOptions.workerSrc =
            "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.16.105/pdf.worker.min.js";

          const b64 = "{b64}";
          const raw = atob(b64);
          const bytes = new Uint8Array(raw.length);
          for (let i = 0; i < raw.length; i++) {{
            bytes[i] = raw.charCodeAt(i);
          }}

          const loadingTask = window.pdfjsLib.getDocument({{ data: bytes }});
          const pdf = await loadingTask.promise;
          const pageCount = pdf.numPages;
          statusEl.textContent = `PDF cargado (${{pageCount}} página(s))`;

          for (let pageNum = 1; pageNum <= pageCount; pageNum++) {{
            const page = await pdf.getPage(pageNum);
            const viewport = page.getViewport({{ scale: 1.0 }});
            const maxWidth = pagesEl.clientWidth > 0 ? pagesEl.clientWidth - 16 : 1000;
            const scale = Math.max(1.0, Math.min(2.5, maxWidth / viewport.width));
            const scaledViewport = page.getViewport({{ scale }});

            const wrap = document.createElement("div");
            wrap.style.margin = "0 0 14px 0";
            wrap.style.padding = "8px";
            wrap.style.background = "#111";
            wrap.style.border = "1px solid #2a2a2a";
            wrap.style.borderRadius = "6px";

            const label = document.createElement("div");
            label.textContent = "Página " + pageNum;
            label.style.color = "#ddd";
            label.style.fontFamily = "sans-serif";
            label.style.fontSize = "12px";
            label.style.margin = "0 0 8px 0";
            wrap.appendChild(label);

            const canvas = document.createElement("canvas");
            const context = canvas.getContext("2d");
            canvas.width = scaledViewport.width;
            canvas.height = scaledViewport.height;
            canvas.style.width = "100%";
            canvas.style.height = "auto";
            canvas.style.display = "block";
            canvas.style.background = "#fff";
            wrap.appendChild(canvas);
            pagesEl.appendChild(wrap);

            await page.render({{ canvasContext: context, viewport: scaledViewport }}).promise;
          }}
        }} catch (err) {{
          if (statusEl) {{
            statusEl.style.color = "#ff6b6b";
            statusEl.textContent = "Error al renderizar PDF embebido: " + String(err);
          }}
        }}
      }})();
    </script>
    """
    components.html(html, height=height + 24, scrolling=True)


def _candidate_pdf_for_tex(tex_path: Path) -> Path | None:
    candidates = [
        BUILD_DIR / f"{tex_path.stem}.pdf",
        tex_path.with_suffix(".pdf"),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


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
    st.caption("Selecciona `.tex` o `.pdf` dentro de `docs/`, y visualiza el resultado embebido en la UI.")

    if not DOCS_DIR.exists():
        st.error(f"No existe el directorio: {DOCS_DIR}")
        return

    latexmk_path = shutil.which("latexmk")

    viewer_files = _list_viewer_files()
    if not viewer_files:
        st.warning("No se encontraron archivos `.tex` o `.pdf` en `docs/`.")
        return

    options = [str(p.relative_to(DOCS_DIR)) for p in viewer_files]
    selected = st.selectbox("Archivo en `docs/`", options=options, index=0)
    selected_path = DOCS_DIR / selected
    out_dir = BUILD_DIR

    col1, col2, col3 = st.columns([2, 1, 1])
    custom_out = str(out_dir)
    force_compile = False
    open_after = False

    with col1:
        if selected_path.suffix.lower() == ".tex":
            custom_out = st.text_input("Carpeta de salida (opcional)", value=str(out_dir))
            force_compile = st.checkbox("Forzar recompilación (latexmk -f)", value=False)
    with col2:
        if selected_path.suffix.lower() == ".tex":
            open_after = st.checkbox("Abrir al finalizar", value=False)
        if st.button("Limpiar build"):
            _clean_build_dir(BUILD_DIR)
            st.success("Build limpio.")
    with col3:
        preview_height = st.slider("Alto visor", min_value=480, max_value=1400, value=900, step=40)

    can_compile = selected_path.suffix.lower() == ".tex" and bool(latexmk_path)
    if selected_path.suffix.lower() == ".tex" and not latexmk_path:
        st.warning("`latexmk` no está disponible. Puedes abrir `.pdf` embebidos, pero no compilar `.tex` hasta instalar latexmk.")

    if can_compile and st.button("Compilar", type="primary"):
        try:
            tex_path = selected_path
            pdf_path = render_tex_to_pdf(tex_path, out_dir=custom_out or None, force=force_compile)
            st.session_state["latex_viewer_last_pdf"] = str(pdf_path)
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

    if selected_path.suffix.lower() == ".pdf":
        st.subheader(f"Vista embebida: `{selected}`")
        try:
            _embed_pdf(selected_path, height=preview_height)
            with open(selected_path, "rb") as f:
                st.download_button(
                    "Descargar PDF",
                    data=f,
                    file_name=selected_path.name,
                    mime="application/pdf",
                )
        except Exception as e:
            st.error(f"No se pudo renderizar el PDF seleccionado: {e}")
        return

    if selected_path.suffix.lower() == ".tex":
        st.subheader(f"Vista embebida: `{selected}`")

        pdf_path: Path | None = None
        last_pdf = st.session_state.get("latex_viewer_last_pdf")
        if isinstance(last_pdf, str):
            last_pdf_path = Path(last_pdf)
            if last_pdf_path.exists() and last_pdf_path.stem == selected_path.stem:
                pdf_path = last_pdf_path
        if pdf_path is None:
            pdf_path = _candidate_pdf_for_tex(selected_path)

        tab_preview, tab_source = st.tabs(["Preview PDF", "Fuente .tex"])
        with tab_preview:
            if pdf_path and pdf_path.exists():
                _embed_pdf(pdf_path, height=preview_height)
                try:
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            "Descargar PDF",
                            data=f,
                            file_name=pdf_path.name,
                            mime="application/pdf",
                            key=f"dl-{pdf_path}",
                        )
                except Exception:
                    st.info("No se pudo preparar el PDF para descarga.")
            else:
                st.info("No hay PDF para previsualizar. Presiona `Compilar`.")

        with tab_source:
            try:
                tex_text = selected_path.read_text(encoding="utf-8")
                st.code(tex_text, language="latex")
            except Exception as e:
                st.error(f"No se pudo leer el archivo `.tex`: {e}")
        return


if __name__ == "__main__":
    Latex(set_page_config=True)
