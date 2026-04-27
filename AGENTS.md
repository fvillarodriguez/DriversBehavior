# Drivers Behavior Modeling and Simulation - Agent Guide

## Project Purpose

This repository is a Python/Streamlit suite for traffic-flow analysis and
simulation. It covers flow database management, driver-behavior clustering,
crash prediction, GNN pipelines, NLP severity modeling, drift detection,
SUMO simulation, experiment monitoring, and local Ray/Dask orchestration.

The main app launcher is `streamlit_main.py`. Most business logic and
Streamlit pages live in `src/`.

## Quick Start

Use Python 3.12 when possible. Python 3.13+ is not recommended because parts of
the PyTorch Geometric stack used by NeighborLoader are not compatible.

```bash
python venv_start.py --python python3.12
source .venv/bin/activate
```

On macOS, the preferred app launcher is:

```bash
./start_app.command
```

Manual Streamlit launch:

```bash
streamlit run streamlit_main.py
```

## Repository Map

- `streamlit_main.py` - multipage Streamlit navigation and app entrypoint.
- `src/` - application modules, ML pipelines, utilities, and page renderers.
- `tests/` - pytest test suite for feature engineering, models, pipelines, and apps.
- `Datos/` - persistent input data such as DuckDB/CSV sources.
- `Resultados/` - generated artifacts, models, reports, and intermediate outputs.
- `simulación/` - SUMO network/configuration files and generated SUMO outputs.
- `dask/` - separate local Dask cluster app package with its own agent guide.

## Common Commands

```bash
source .venv/bin/activate
pytest
pytest tests/<test_file>.py -q
streamlit run streamlit_main.py
./start_app.command
```

Use focused tests for narrow changes. Run the broader `pytest` suite when a
change affects shared feature engineering, model-training protocols, persistence,
or app-wide behavior.

## Development Rules

- Keep changes small and aligned with the existing module boundaries.
- Do not commit large data, generated artifacts, local virtualenvs, caches, or
  private configuration. The ignored data/output roots include `Datos/`,
  `Resultados/`, `simulación/`, `docs/`, `NLP/`, and `DRIFT/`.
- Preserve expected input/output paths used by the apps, especially
  `Datos/flujos.duckdb`, `Datos/Porticos.csv`, event files under `Datos/`,
  generated files under `Resultados/`, and SUMO files under `simulación/`.
- Streamlit modules are often runnable both from `streamlit_main.py` and
  standalone. Preserve existing `main(..., set_page_config=..., show_exit_button=...)`
  conventions when editing pages.
- Prefer existing helpers in `src/` over introducing new abstractions.
- Be careful with long-running ML, SUMO, Ray, and Dask workflows; add or run
  lightweight tests first when possible.

## Testing Notes

The root test suite uses `pytest`. Many tests use `pytest.importorskip` for
optional heavy dependencies, so targeted tests are usually the fastest way to
validate a local change.

For documentation-only edits, no application tests are required; verify the
Markdown and cited commands instead.

## Dask Subproject

When working inside `dask/`, also read and follow `dask/AGENTS.md`. That guide
owns the Dask cluster app package conventions, including its FastAPI/CLI layout,
`pyproject.toml` settings, Dask-specific test commands, and Ruff configuration.
