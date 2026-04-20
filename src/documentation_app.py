from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT_DIR = Path(__file__).resolve().parents[1]
HTML_DIR = ROOT_DIR / "docs" / "html"

LANGUAGES = {
    "Español": "index_es.html",
    "English": "index_en.html",
}


def _read_html(filename: str) -> str | None:
    path = HTML_DIR / filename
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def Documentation(set_page_config: bool = True, show_exit_button: bool = True) -> None:
    if set_page_config:
        st.set_page_config(page_title="Documentación", layout="wide")

    if not HTML_DIR.exists():
        st.error(f"No existe el directorio de documentación: {HTML_DIR}")
        return

    with st.sidebar:
        lang = st.radio(
            "Idioma",
            options=list(LANGUAGES.keys()),
            index=0,
            horizontal=True,
            key="documentation_lang",
        )

    filename = LANGUAGES[lang]
    html = _read_html(filename)
    if html is None:
        st.error(f"No se encontró `{filename}` en `{HTML_DIR}`.")
        return

    components.html(html, height=2200, scrolling=True)


def main(set_page_config: bool = True, show_exit_button: bool = True) -> None:
    Documentation(set_page_config=set_page_config, show_exit_button=show_exit_button)


if __name__ == "__main__":
    main()
