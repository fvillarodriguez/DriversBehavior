# Drivers Behavior — Traffic Accident Prediction & Driver Clustering

A Streamlit-based toolkit to **analyze highway traffic detections**, **cluster driver behavior** (by license plate), and **predict near-future accidents** at gantries (“pórticos”) using aggregated flow features and optional cluster-derived features.

This repository bundles:

* A modular **Streamlit UI** (multi-page app)
* A **DuckDB-backed** workflow for flow data
* Driver-behavior **clustering** (plate-level features → KMeans/GMM/HDBSCAN)
* Accident prediction models with **temporal splits**, **imbalance handling**, and **threshold calibration** to meet a target **False Alarm Rate (FAR)**
* Experiment tooling and a **live dashboard** reading results from SQLite

---

## Key Ideas

### 1) What is predicted?

Each row in the modeling dataset represents a pair:

**(gantry p, time interval τ)**

The binary target is defined so that:

> `target = 1` means: “an accident will occur in the *next* interval at this gantry”.

Concretely, an accident at time `t_a` mapped to gantry `p_a` is labeled at the **previous** discrete interval:

* `τ(t) = floor(t / Δ) * Δ`
* `τ_a = τ(t_a) − Δ`

---

## Main Modules

### Flow database (DuckDB)

Flow detections are stored in **DuckDB** and queried by time range. Each detection typically includes:

* timestamp `t`
* gantry `p`
* lane `ℓ`
* vehicle category `c`
* speed `v`
* license plate `plate`

### Clustering (driver behavior)

Builds **plate-level feature vectors** from detections (counts, mean speed, lane usage, relative speed, headway, TTC/conflict proxies, lane-change rate, etc.) and clusters drivers using:

* **K-Means**
* **Gaussian Mixture Models (GMM)** with AIC/BIC support
* **HDBSCAN** (noise label = -1)

Optionally supports a **frequent vs. rare** driver flow:

* Train clustering on frequent plates
* Assign rare plates with confidence thresholds (distance/probability), otherwise label as unknown (-1)

Outputs include:

* `Resultados/cluster_features*.duckdb`
* `Resultados/cluster_kmeans_kK.csv`, `Resultados/cluster_gmm_kK.csv`, `Resultados/cluster_hdbscan.csv`
* summaries/descriptives CSVs

### Accident prediction

Builds a dataset at **(gantry, interval)** resolution using:

**Macro flow features** (by gantry, interval, category), e.g.:

* counts, mean speeds
* flow per hour, density
* optional temporal deltas (first differences)

**Optional cluster-aggregated features** (by gantry, interval, cluster label):

* cluster shares, cluster-level flow/speed/density, temporal deltas
* mixture entropy: `H = −Σ share * log(share)`

Models supported:

* Random Forest (with class weights)
* XGBoost (binary logistic)
* SVM (scaled, probability output)

Imbalance handling:

* class weighting
* optional **SMOTE applied only on the training set**

Threshold calibration for operational control:

* Choose a threshold `θ` on validation such that **FAR ≤ α**, maximizing sensitivity under that constraint.

### Experiments (Optuna + live monitoring)

Includes an optimization loop that can tune:

* SMOTE parameters
* model hyperparameters
* optionally the threshold (or use FAR-calibrated threshold strategy)

Artifacts typically written to `Resultados/`, e.g.:

* feature CSV/DuckDB outputs
* balanced datasets
* Optuna JSON + trials CSV
* iterative experiments CSV
* live experiment SQLite DB files: `Resultados/experiment_live_*.sqlite`

A dedicated Streamlit page (“Experiments Live”) monitors those SQLite files and updates automatically.

---

## Streamlit App

The main menu provides pages such as:

* Flow database
* Clustering
* Crash prediction
* Experiments Live
* Events map visualization
* Files browser
* Test page

---

## Project Structure (typical)

```text
.
├── streamlit_main.py
├── src/
│   ├── flow_database_app.py
│   ├── clustering_tabs_app.py
│   ├── cluster_accident_app.py
│   ├── experiments_live_app.py
│   ├── events_map_app.py
│   └── ...
├── tests/
├── Datos/
│   └── Porticos.csv
└── Resultados/
```

> Note: `Datos/Porticos.csv` is used to map accident locations (e.g., km/eje/calzada) to nearby gantries.

---

## Installation

### Requirements

* Python (recommended: 3.10+)
* DuckDB
* Streamlit
* Common ML stack (pandas, numpy, scikit-learn)
* Optional depending on features:

  * imbalanced-learn (SMOTE)
  * xgboost
  * optuna
  * altair and/or plotly (visualizations)
  * hdbscan

### Setup (example)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

---

## Quickstart

Run the Streamlit UI:

```bash
streamlit run streamlit_main.py
```

---

## Data Inputs

This project expects (at least):

1. **Flow detections** accessible via DuckDB tables (queried by time range).
2. **Accident events files** filtered to “Accidente” (and optionally by road type, e.g., “vía expresa”).
3. Gantry metadata (e.g., `Datos/Porticos.csv`) for location-to-gantry mapping.

---

## Outputs

Generated artifacts are written under `Resultados/`, including:

* flow features and cluster features datasets
* labels per plate (clustering)
* balanced training data (if SMOTE used)
* Optuna optimization logs
* iterative experiment result tables
* live monitoring SQLite databases for the dashboard

---

## Reproducibility Notes

* Dataset splits are **temporal** to reduce leakage.
* FAR-constrained threshold selection is performed on validation and then evaluated on test.
* SMOTE is applied only to training data.
