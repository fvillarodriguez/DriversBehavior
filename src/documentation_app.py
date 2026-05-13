from __future__ import annotations

import html as _html
import json as _json
import re as _re
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import streamlit as st
import streamlit.components.v1 as components

ROOT_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT_DIR / "src" / "documentation_web"
LEGACY_DIR = ROOT_DIR / "docs" / "html"
STYLES_CSS = WEB_DIR / "styles.css"
THEME_JS = WEB_DIR / "theme.js"
HERO_DIR = WEB_DIR / "hero"
SECTIONS_DIR = WEB_DIR / "sections"
LEGACY_MATHJAX_COMPONENT = "tex-mjx-chtml.js"
MATHJAX_COMPONENT = "tex-mml-chtml.js"

VERSION = "2026.05"
TITLE_ES = "Crash Prediction · Manuscrito Técnico"
TITLE_EN = "Crash Prediction · Technical Manuscript"


@dataclass(frozen=True)
class SectionDef:
    sid: str
    label_es: str
    label_en: str
    short_es: str
    short_en: str
    height: int


SECTIONS: tuple[SectionDef, ...] = (
    SectionDef("hero",                    "Inicio",                 "Home",                 "Resumen y guía",                  "Overview and guide",                  2100),
    SectionDef("01_clustering",           "§ 1 · Clustering",       "§ 1 · Clustering",     "Agrupación de conductores",       "Driver behaviour grouping",          22000),
    SectionDef("02_accident_prediction",  "§ 2 · Predicción",       "§ 2 · Prediction",     "Modelos supervisados",            "Supervised models",                  17000),
    SectionDef("03_nlp_severity",         "§ 3 · NLP severidad",    "§ 3 · NLP severity",   "Modelado multimodal de severidad", "Multimodal severity modelling",       19000),
    SectionDef("04_drift_detection",      "§ 4 · Drift detection",  "§ 4 · Drift detection","Recalibración batch vs adaptiva", "Batch vs adaptive recalibration",     8800),
    SectionDef("06_gnn_pipeline",         "§ 6 · Graph Neural Network", "§ 6 · Graph Neural Network", "Grafo espacio-temporal", "Spatio-temporal graph",              18000),
    SectionDef("07_mairl",                "§ 7 · MA-AIRL",          "§ 7 · MA-AIRL",        "Multi-agent adversarial IRL",     "Multi-agent adversarial IRL",         3500),
)


# ---------- HTML helpers ----------

PLACEHOLDER_TEMPLATE = """<!doctype html>
<html lang="__LANG__" data-theme="__THEME__">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>__TITLE__</title>
<style>__CSS__</style>
</head>
<body>
<div class="lab-page">
  <main class="lab-main">
    <section class="lab-section reveal">
      <header class="lab-section-head" data-stagger>
        <div class="num reveal">__NUM__</div>
        <div>
          <span class="meta reveal">__META__</span>
          <h2 class="reveal">__TITLE__</h2>
        </div>
      </header>
      <div class="lab-callout ochre reveal">
        <div class="label">__SOON_LABEL__</div>
        <p>__SOON_BODY__</p>
      </div>
      <div class="lab-callout sage reveal">
        <div class="label">__INTERIM_LABEL__</div>
        <p>__INTERIM_BODY__</p>
      </div>
    </section>
  </main>
  <aside class="lab-margin" aria-label="Notas">
    <div class="pinned">
      <span class="label">__MARGIN_LABEL__</span>
      <p>__MARGIN_BODY__</p>
    </div>
  </aside>
</div>
<script>__JS__</script>
</body>
</html>
"""


def _read(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _section_anchor_id(section: SectionDef) -> str:
    match = _re.match(r"^0*(\d+)_", section.sid)
    return f"s{match.group(1)}" if match else section.sid


def _doc_nav_config() -> dict:
    return {
        "sections": {
            section.sid: {
                "anchor": _section_anchor_id(section),
                "labels": {
                    "es": section.label_es,
                    "en": section.label_en,
                },
            }
            for section in SECTIONS
            if section.sid != "hero"
        }
    }


def _doc_nav_config_script() -> str:
    payload = _json.dumps(_doc_nav_config(), ensure_ascii=False, separators=(",", ":"))
    return f"<script>window.LAB_DOC_NAV={payload};</script>"


def _inject_assets(html: str, css: str, js: str, theme: str, query: str = "") -> str:
    out = (
        html
        .replace(LEGACY_MATHJAX_COMPONENT, MATHJAX_COMPONENT)
        .replace('<link rel="stylesheet" href="{{CSS_HREF}}" />', f"<style>\n{css}\n</style>")
        .replace('<script src="{{JS_HREF}}"></script>', f"{_doc_nav_config_script()}\n<script>\n{js}\n</script>")
        .replace("{{THEME}}", theme)
        .replace("{{VERSION}}", VERSION)
    )
    if query:
        meta = f'<meta name="lab-query" content="{_html.escape(query, quote=True)}">'
        out = out.replace("</head>", f"  {meta}\n</head>", 1)
    return out


# ---------- Search index ----------

_SCRIPT_RE = _re.compile(r"<script[^>]*>.*?</script>", _re.S | _re.I)
_STYLE_RE = _re.compile(r"<style[^>]*>.*?</style>", _re.S | _re.I)
_TAG_RE = _re.compile(r"<[^>]+>")
_WS_RE = _re.compile(r"\s+")


def _strip_html_to_text(html: str) -> str:
    """Coarse plain-text extraction for search indexing."""
    s = _SCRIPT_RE.sub(" ", html)
    s = _STYLE_RE.sub(" ", s)
    s = _TAG_RE.sub(" ", s)
    s = (s.replace("&nbsp;", " ").replace("&amp;", "&")
         .replace("&lt;", "<").replace("&gt;", ">"))
    return _WS_RE.sub(" ", s).strip()


@st.cache_data(show_spinner=False)
def _search_index() -> dict:
    """Map (sid, lang) -> plain text for full-text search."""
    idx: dict = {}
    for s in SECTIONS:
        if s.sid == "hero":
            continue  # hero is just navigation, no need to index
        for lang in ("es", "en"):
            path = SECTIONS_DIR / lang / f"{s.sid}.html"
            if not path.exists():
                continue
            idx[(s.sid, lang)] = _strip_html_to_text(path.read_text(encoding="utf-8"))
    return idx


def _search_counts(query: str, lang: str) -> list[tuple[SectionDef, int]]:
    if not query or len(query.strip()) < 2:
        return []
    q = query.lower().strip()
    idx = _search_index()
    out: list[tuple[SectionDef, int]] = []
    for s in SECTIONS:
        if s.sid == "hero":
            continue
        text = idx.get((s.sid, lang), "")
        if not text:
            continue
        n = text.lower().count(q)
        if n:
            out.append((s, n))
    out.sort(key=lambda x: -x[1])
    return out


def _placeholder_html(section: SectionDef, lang_code: str, theme: str, css: str, js: str) -> str:
    is_es = lang_code == "es"
    title = section.label_es if is_es else section.label_en
    short = section.short_es if is_es else section.short_en
    num = section.sid.split("_")[0].lstrip("0") or "0"
    if section.sid == "hero":
        num = "§"

    soon_label = "Próximamente" if is_es else "Coming soon"
    soon_body = (
        "Esta sección está siendo migrada al nuevo diseño <em>Editorial Lab</em>. "
        "Mantendremos las fórmulas, tablas y callouts del manuscrito original e "
        "iremos añadiendo gráficos dinámicos y resultados a medida que estén disponibles."
        if is_es else
        "This section is being migrated to the new <em>Editorial Lab</em> design. "
        "We will keep the formulas, tables and callouts from the original manuscript "
        "and add dynamic charts and results as they become available."
    )
    interim_label = "Mientras tanto" if is_es else "In the meantime"
    interim_body = (
        "El documento completo (todas las secciones consolidadas) sigue disponible "
        "en <code>docs/html/index_es.html</code>. Puedes abrirlo directamente o "
        "esperar a que cada capítulo se migre individualmente."
        if is_es else
        "The complete document (all sections consolidated) remains available at "
        "<code>docs/html/index_en.html</code>. You can open it directly or wait "
        "until each chapter is migrated individually."
    )
    margin_label = "Roadmap" if is_es else "Roadmap"
    margin_body = (
        "F1 ✓ Shell · F2 ✓ Diseño · F3 → migración por capítulos · F4 → buscador · "
        "F5 → resultados desde DuckDB y modelos."
        if is_es else
        "F1 ✓ Shell · F2 ✓ Design · F3 → chapter migration · F4 → search · "
        "F5 → live results from DuckDB and models."
    )

    return (
        PLACEHOLDER_TEMPLATE
        .replace("__LANG__", lang_code)
        .replace("__THEME__", theme)
        .replace("__TITLE__", title)
        .replace("__NUM__", num)
        .replace("__META__", short)
        .replace("__SOON_LABEL__", soon_label)
        .replace("__SOON_BODY__", soon_body)
        .replace("__INTERIM_LABEL__", interim_label)
        .replace("__INTERIM_BODY__", interim_body)
        .replace("__MARGIN_LABEL__", margin_label)
        .replace("__MARGIN_BODY__", margin_body)
        .replace("__CSS__", css)
        .replace("<script>__JS__</script>", f"{_doc_nav_config_script()}\n<script>{js}</script>")
    )


def _section_path(sid: str, lang_code: str) -> Path | None:
    if sid == "hero":
        path = HERO_DIR / f"{lang_code}.html"
    else:
        path = SECTIONS_DIR / lang_code / f"{sid}.html"
    return path if path.exists() else None


# ---------- Sidebar UI ----------

_SIDEBAR_MARKER_CLASS = "lab-doc-sidebar-scope"

_SIDEBAR_CSS = """
<style>
/* Scoped to the documentation page only; we tag the sidebar with a marker
   class via the first markdown call, then chain selectors off it. */
section[data-testid="stSidebar"]:has(.lab-doc-sidebar-scope) [data-testid="stMarkdownContainer"] h3 {
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 500;
    letter-spacing: -0.01em;
    color: #0e0e0e;
    margin-top: 0;
}
section[data-testid="stSidebar"]:has(.lab-doc-sidebar-scope) [data-testid="stCaptionContainer"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10.5px;
    letter-spacing: 0.08em;
    color: #565656;
}
section[data-testid="stSidebar"]:has(.lab-doc-sidebar-scope) .stRadio [role="radiogroup"] label p {
    font-family: 'Newsreader', Georgia, serif !important;
    font-size: 14.5px !important;
}
.lab-doc-sidebar-scope { display: none !important; }
</style>
<span class="lab-doc-sidebar-scope" aria-hidden="true"></span>
"""


def _sidebar() -> Tuple[str, str, SectionDef, str]:
    with st.sidebar:
        st.markdown(_SIDEBAR_CSS, unsafe_allow_html=True)

        st.markdown("### Manuscrito Técnico")
        st.caption(f"v {VERSION} · ES / EN")

        st.divider()

        lang = st.radio(
            "Idioma · Language",
            options=["ES", "EN"],
            horizontal=True,
            index=0,
            key="doc_lang",
        )
        theme = st.radio(
            "Tema · Theme",
            options=["Claro · Light", "Oscuro · Dark"],
            horizontal=True,
            index=0,
            key="doc_theme",
        )

        st.divider()

        is_es = lang == "ES"
        lang_code = "es" if is_es else "en"

        search_label = "Buscar · Search"
        search_placeholder = (
            "Headway, ADWIN, SHAP…" if is_es else "Headway, ADWIN, SHAP…"
        )
        query = st.text_input(
            search_label,
            value="",
            key="doc_search",
            placeholder=search_placeholder,
        ).strip()

        # Cross-section match counts
        if query and len(query) >= 2:
            hits = _search_counts(query, lang_code)
            if hits:
                lines = []
                for s, n in hits:
                    name = s.label_es if is_es else s.label_en
                    lines.append(f"`{n:>3}`  ·  {name}")
                header = "Coincidencias" if is_es else "Matches"
                st.markdown(
                    f"**{header}**  \n" + "  \n".join(lines)
                )
            else:
                st.caption(
                    "Sin coincidencias en ningún capítulo." if is_es
                    else "No matches across chapters."
                )

        st.divider()

        labels = [s.label_es if is_es else s.label_en for s in SECTIONS]
        if st.session_state.get("doc_section") not in labels:
            st.session_state["doc_section"] = labels[0]
        nav_label = "Capítulos" if is_es else "Chapters"
        choice = st.radio(
            nav_label,
            options=labels,
            index=0,
            key="doc_section",
            label_visibility="visible",
        )
        idx = labels.index(choice) if choice in labels else 0
        section = SECTIONS[idx]

        st.divider()
        legend = (
            "Pasa el cursor sobre las ecuaciones numeradas para abrir el inspector. "
            "Usa el buscador para resaltar coincidencias en el capítulo actual."
            if is_es else
            "Hover the numbered equations to open the inspector. "
            "Use the search box to highlight matches in the current chapter."
        )
        st.caption(legend)

    theme_code = "light" if "Claro" in theme else "dark"
    return lang_code, theme_code, section, query


# ---------- Public API ----------

def Documentation(set_page_config: bool = True, show_exit_button: bool = True) -> None:
    if set_page_config:
        st.set_page_config(page_title=TITLE_ES, layout="wide")

    if not WEB_DIR.exists():
        st.error(f"No existe el directorio de documentación: {WEB_DIR}")
        return

    css = _read(STYLES_CSS)
    js = _read(THEME_JS)
    if not css:
        st.warning(f"Falta {STYLES_CSS.name}; el diseño puede verse degradado.")

    lang_code, theme_code, section, query = _sidebar()

    path = _section_path(section.sid, lang_code)
    if path is None:
        html = _placeholder_html(section, lang_code, theme_code, css, js)
    else:
        html = _inject_assets(_read(path), css, js, theme_code, query=query)

    components.html(html, height=section.height, scrolling=True)


def main(set_page_config: bool = True, show_exit_button: bool = True) -> None:
    Documentation(set_page_config=set_page_config, show_exit_button=show_exit_button)


if __name__ == "__main__":
    main()
