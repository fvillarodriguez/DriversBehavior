from __future__ import annotations

import json
import re
from html.parser import HTMLParser

import src.documentation_app as documentation_app


class _HeroLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attr_map = {key: value or "" for key, value in attrs}
        if "data-section" in attr_map:
            self.links.append(attr_map)


def _hero_links(lang: str) -> list[dict[str, str]]:
    parser = _HeroLinkParser()
    parser.feed((documentation_app.HERO_DIR / f"{lang}.html").read_text(encoding="utf-8"))
    return parser.links


def _chapter_sections() -> tuple[documentation_app.SectionDef, ...]:
    return tuple(section for section in documentation_app.SECTIONS if section.sid != "hero")


def test_web_documentation_assets_live_under_src() -> None:
    assert documentation_app.WEB_DIR == documentation_app.ROOT_DIR / "src" / "documentation_web"
    assert documentation_app.STYLES_CSS.exists()
    assert documentation_app.THEME_JS.exists()
    assert documentation_app.HERO_DIR.exists()
    assert documentation_app.SECTIONS_DIR.exists()


def test_hero_toc_links_target_existing_chapter_anchors() -> None:
    expected = {
        section.sid: f"#{documentation_app._section_anchor_id(section)}"
        for section in _chapter_sections()
    }

    for lang in ("es", "en"):
        links = _hero_links(lang)

        assert [link["data-section"] for link in links] == list(expected)
        for link in links:
            section_id = link["data-section"]
            assert link["href"] == expected[section_id]

            section_path = documentation_app.SECTIONS_DIR / lang / f"{section_id}.html"
            assert section_path.exists()
            section_html = section_path.read_text(encoding="utf-8")
            assert f'id="{expected[section_id].lstrip("#")}"' in section_html


def test_doc_nav_config_script_covers_all_chapters() -> None:
    script = documentation_app._doc_nav_config_script()
    match = re.fullmatch(r"<script>window\.LAB_DOC_NAV=(.*);</script>", script)

    assert match is not None
    config = json.loads(match.group(1))
    sections = config["sections"]
    assert set(sections) == {section.sid for section in _chapter_sections()}

    for section in _chapter_sections():
        entry = sections[section.sid]
        assert entry["anchor"] == documentation_app._section_anchor_id(section)
        assert entry["labels"] == {"es": section.label_es, "en": section.label_en}


def test_clustering_feature_detail_is_nested_under_clustering() -> None:
    assert "05_clustering_features" not in {section.sid for section in documentation_app.SECTIONS}

    for lang in ("es", "en"):
        links = _hero_links(lang)
        assert "05_clustering_features" not in {link["data-section"] for link in links}

    clustering_es = (
        documentation_app.SECTIONS_DIR / "es" / "01_clustering.html"
    ).read_text(encoding="utf-8")
    clustering_en = (
        documentation_app.SECTIONS_DIR / "en" / "01_clustering.html"
    ).read_text(encoding="utf-8")

    assert "Detalle de variables conductuales" in clustering_es
    assert "Behavioural feature detail" in clustering_en


def test_gnn_chapter_is_named_graph_neural_network() -> None:
    section = next(section for section in documentation_app.SECTIONS if section.sid == "06_gnn_pipeline")

    assert section.label_es == "§ 6 · Graph Neural Network"
    assert section.label_en == "§ 6 · Graph Neural Network"

    for lang in ("es", "en"):
        section_html = (
            documentation_app.SECTIONS_DIR / lang / "06_gnn_pipeline.html"
        ).read_text(encoding="utf-8")
        hero_html = (documentation_app.HERO_DIR / f"{lang}.html").read_text(encoding="utf-8")

        assert "<title>§ 6 · Graph Neural Network</title>" in section_html
        assert '<h2 class="reveal">Graph Neural Network</h2>' in section_html
        assert '<strong class="title">Graph Neural Network</strong>' in hero_html


def test_clustering_ttc_formula_matches_implementation_contract() -> None:
    for lang in ("es", "en"):
        html = (
            documentation_app.SECTIONS_DIR / lang / "01_clustering.html"
        ).read_text(encoding="utf-8")

        assert r"\mathcal{K}_i" in html
        assert r"\Delta v_j&gt;0" in html
        assert r"\mathrm{TTC}_j&lt;\tau_{p_j}" in html
        assert "0<h_j" not in html
        assert "0&lt;h_j" in html


def test_injected_assets_place_nav_config_before_runtime_script() -> None:
    html = '<html><head></head><body><script src="{{JS_HREF}}"></script></body></html>'

    injected = documentation_app._inject_assets(
        html,
        css="",
        js="window.__themeRuntimeLoaded = true;",
        theme="light",
    )

    assert injected.index("window.LAB_DOC_NAV=") < injected.index("window.__themeRuntimeLoaded")


def test_injected_assets_uses_valid_mathjax_component() -> None:
    html = (
        '<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/'
        'tex-mjx-chtml.js"></script>'
        '<script src="{{JS_HREF}}"></script>'
    )

    injected = documentation_app._inject_assets(html, css="", js="", theme="light")

    assert documentation_app.MATHJAX_COMPONENT in injected
    assert documentation_app.LEGACY_MATHJAX_COMPONENT not in injected


def test_equation_inspector_stays_above_document_content() -> None:
    css = (documentation_app.WEB_DIR / "styles.css").read_text(encoding="utf-8")
    js = (documentation_app.WEB_DIR / "theme.js").read_text(encoding="utf-8")

    assert ".lab-eq.has-inspector.is-inspector-open" in css
    assert '.lab-eq:has(> .eq-inspector[data-open="true"])' in css
    assert "z-index: 1000;" in css
    assert "z-index: 1001;" in css
    assert "max-height: min(70vh, 560px);" in css
    assert "eq.classList.add('is-inspector-open')" in js
    assert "classList.remove('is-inspector-open')" in js
