"""
TRC paper pipeline — dynamic latent class analysis of driver behaviour.

Subpackage for the umbrella paper targeting Transportation Research Part C.
The pipeline lives entirely under src/trc_paper/. Inputs come from the project
data folders (Datos/, Resultados/). All outputs are persisted under
Resultados/trc_paper/.

Step order (matches the Snakefile DAG):
    validate_data         → Resultados/trc_paper/validation/*.json
    run_dynamic_gmm       → Resultados/trc_paper/dynamic_gmm/*assignments.duckdb
    compute_entropy       → Resultados/trc_paper/entropy/*.parquet
    markov_matrix         → Resultados/trc_paper/markov/P_global.parquet
    homogeneity_test      → Resultados/trc_paper/markov/homogeneity.json
    stationary_asymmetry  → Resultados/trc_paper/markov/stationary.json
    covid_decomposition   → Resultados/trc_paper/covid/decomposition.json
    event_matching        → Resultados/trc_paper/events/matched_pairs.parquet
    integration_h_bound   → Resultados/trc_paper/integration/h_bound.json

Run from the Streamlit UI (Página TRC Paper) or from the CLI through Snakemake.
Manuscript material lives in papers/dynamic_clusters_trc/.
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent
CONFIG_DIR = PACKAGE_DIR / "config"
WORKFLOW_DIR = PACKAGE_DIR / "workflow"
RESULTS_ROOT = PROJECT_ROOT / "Resultados" / "trc_paper"
LOGS_ROOT = RESULTS_ROOT / "logs"

__all__ = [
    "PACKAGE_DIR",
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "WORKFLOW_DIR",
    "RESULTS_ROOT",
    "LOGS_ROOT",
]
