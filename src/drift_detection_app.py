#!/usr/bin/env python3
"""
Streamlit app and reusable utilities to replicate the paper:
"Evaluating recalibration strategies for real-time crash prediction:
adaptive drift detection vs. cumulative retraining".

The module is self-contained by design to satisfy the requirement of hosting the
full "Drift detection" section in a single source file.
"""
from __future__ import annotations

import gc
import hashlib
import importlib
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
import warnings
import zipfile
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from joblib import Parallel as JoblibParallel
from joblib import delayed as joblib_delayed
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
try:
    from sklearn.utils.parallel import Parallel as SklearnParallel
    from sklearn.utils.parallel import delayed as sklearn_delayed
except Exception:
    SklearnParallel = JoblibParallel
    sklearn_delayed = joblib_delayed

try:
    import duckdb  # type: ignore
except Exception:
    duckdb = None

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

try:
    import optuna  # type: ignore
    from optuna.samplers import TPESampler  # type: ignore
except Exception:
    optuna = None
    TPESampler = None

try:
    from imblearn.over_sampling import SMOTE  # type: ignore
except Exception:
    SMOTE = None

try:
    from river import drift as river_drift  # type: ignore
    from river import forest as river_forest  # type: ignore
    from river import metrics as river_metrics  # type: ignore
except Exception:
    river_drift = None
    river_forest = None
    river_metrics = None

from src.utils import (
    find_candidate_porticos,
    get_portico_segments,
    load_porticos,
    process_accidentes_df,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "Resultados"
DATA_DIR = ROOT_DIR / "Datos"
FLOW_DB_PATH = DATA_DIR / "flujos.duckdb"
FLOW_TABLE_NAME = "flujos_duckdb"
DRIFT_FOCUS_PORTICOS = ["11", "12", "14", "15"]
DRIFT_ARTICLE_PORTICO_ORDER = ["15", "14", "12", "11"]
DRIFT_FOCUS_TRAMO = ("__DRIFT_FIXED_PORTICOS__", "", "", "")
DRIFT_ARTICLE_CATEGORY_MAP = {1: "Light", 2: "Heavy", 3: "Heavy", 4: "Motorcycle"}
DRIFT_ARTICLE_CATEGORIES = ["Light", "Heavy", "Motorcycle"]
DRIFT_ARTICLE_CATEGORY_SLUGS = {
    "Light": "light",
    "Heavy": "heavy",
    "Motorcycle": "motorcycle",
}
DRIFT_ARTICLE_MIN_REFERENCE_SPEED_KMH = 20.0
DRIFT_ARTICLE_MIN_MATCH_WINDOW_MINUTES = 30.0

PAPER_TITLE = (
    "Evaluating recalibration strategies for real-time crash prediction: "
    "adaptive drift detection vs. cumulative retraining"
)
PAPER_TIME_SPAN = "2018-01-01 to 2024-09-30"
NNET_TRAINING_POLICY_VERSION = "scaled_early_stopping_v1"
NNET_MAX_ITER_RETRY_CAP = 1000
WINDOW_TUNING_POLICY_VERSION = "window_scoped_tuning_v1"
RF_SMOTE_BALANCE_POLICY_VERSION = "rf_disable_class_weight_when_smote_v1"
THRESHOLD_CALIBRATION_POLICY_VERSION = "platt_youden_v1"
DEFAULT_CALIBRATION_METHOD = "platt"
DEFAULT_THRESHOLD_POLICY = "youden"
DEFAULT_THRESHOLD_FN_COST = 5.0
DEFAULT_THRESHOLD_FP_COST = 1.0


class ModelTrainingSkipped(RuntimeError):
    def __init__(self, message: str, *, metadata: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.metadata = dict(metadata or {})


def _import_external_xgboost():
    src_dir = Path(__file__).resolve().parent
    original_sys_path = list(sys.path)
    existing_module = sys.modules.get("xgboost")
    removed_local_module = None
    try:
        if existing_module is not None:
            module_file = Path(str(getattr(existing_module, "__file__", "") or "")).resolve()
            if module_file == (src_dir / "xgboost.py").resolve():
                removed_local_module = sys.modules.pop("xgboost")
        sys.path = [
            entry
            for entry in original_sys_path
            if str(Path(entry or ".").resolve()) != str(src_dir)
        ]
        xgb = importlib.import_module("xgboost")  # type: ignore
    finally:
        sys.path = original_sys_path
        if removed_local_module is not None:
            sys.modules["xgboost"] = removed_local_module

    module_path = Path(str(getattr(xgb, "__file__", "") or "")).resolve()
    if module_path == (src_dir / "xgboost.py").resolve():
        raise ImportError(
            "Se importo el modulo local `src/xgboost.py` en lugar del paquete externo `xgboost`."
        )
    if not hasattr(xgb, "XGBClassifier"):
        raise ImportError(
            "El paquete `xgboost` importado no expone `XGBClassifier`. "
            f"Modulo cargado: {module_path}"
        )
    return xgb


def _recalibration_policy_signature() -> Dict[str, Any]:
    return {
        "window_tuning_policy_version": WINDOW_TUNING_POLICY_VERSION,
        "rf_smote_balance_policy_version": RF_SMOTE_BALANCE_POLICY_VERSION,
        "threshold_calibration_policy_version": THRESHOLD_CALIBRATION_POLICY_VERSION,
        "calibration_method": DEFAULT_CALIBRATION_METHOD,
        "threshold_policy": DEFAULT_THRESHOLD_POLICY,
        "fn_cost": float(DEFAULT_THRESHOLD_FN_COST),
        "fp_cost": float(DEFAULT_THRESHOLD_FP_COST),
    }

ARTICLE_SECTIONS: List[Tuple[str, str]] = [
    ("1", "Introduction"),
    ("2", "Literature Review"),
    ("2.1", "Crash Prediction Models"),
    ("2.2", "Recalibration"),
    ("2.3", "Implementation of Drift Detection in Transport"),
    ("2.4", "Research gap"),
    ("3", "Data Description"),
    ("3.1", "AVI Data"),
    ("3.2", "Accident Data"),
    ("4", "Methodology"),
    ("4.1", "Feature engineering"),
    ("4.2", "Data preparation"),
    ("4.3", "Variable selection"),
    ("4.3.1", "Correlation Matrix"),
    ("4.3.2", "Random Forest for feature selection"),
    ("4.4", "Drift"),
    ("4.5", "Prediction Models"),
    ("4.6", "Performance Indicators"),
    ("4.7", "Experimental design"),
    ("5", "Experimental Results"),
    ("6", "Concluding Remarks"),
]

ARTICLE_FIGURES: List[Tuple[str, str]] = [
    ("Figure 1", "Autopista Central contextual map"),
    ("Figure 2", "Autopista Central gate layout"),
    ("Figure 3", "Accidents per hour"),
    ("Figure 4", "Data preparation flowchart"),
    ("Figure 5", "Density of |rho| pairwise correlations"),
    ("Figure 6", "Top-20 Random Forest feature importances"),
    ("Figure 7", "Average ROC curves by strategy and model"),
]

ARTICLE_TABLES: List[Tuple[str, str]] = [
    ("Table 1", "Related work on concept/data drift in transportation"),
    ("Table 2", "Sample of AVI vehicle records"),
    ("Table 3", "Annual summary of traffic volume/composition/speed"),
    ("Table 4", "Feature engineering variable catalog"),
    ("Table 5", "Hyperparameter grids"),
    ("Table A.6", "Static strategy results"),
    ("Table A.7", "Period-aligned strategy results"),
    ("Table A.8", "Cumulative strategy results"),
    ("Table A.9", "Adaptive ADWIN drift-iteration results"),
]

REPLICATION_ANALYSES: List[Dict[str, str]] = [
    {
        "analysis": "Traffic and accidents descriptive analysis",
        "paper_ref": "Sections 3.1-3.2, Table 3, Figure 3",
        "implementation": "compute_annual_traffic_summary + compute_accident_hour_distribution",
    },
    {
        "analysis": "Feature engineering and target alignment",
        "paper_ref": "Section 4.1, Table 4",
        "implementation": "feature_catalog_table + target-ready dataset interface",
    },
    {
        "analysis": "Data preparation and anomaly interval filtering",
        "paper_ref": "Section 4.2, Figure 4",
        "implementation": "build_data_preparation_summary",
    },
    {
        "analysis": "Correlation screening",
        "paper_ref": "Section 4.3.1, Figure 5",
        "implementation": "compute_abs_correlations + drop_highly_correlated_features",
    },
    {
        "analysis": "Feature ranking with Random Forest",
        "paper_ref": "Section 4.3.2, Figure 6",
        "implementation": "rank_features_random_forest",
    },
    {
        "analysis": "Threshold calibration with Youden criterion",
        "paper_ref": "Section 4.5",
        "implementation": "youden_threshold + train_model_with_internal_validation",
    },
    {
        "analysis": "Metric computation",
        "paper_ref": "Section 4.6",
        "implementation": "compute_classification_metrics",
    },
    {
        "analysis": "Experimental design with four recalibration strategies",
        "paper_ref": "Section 4.7",
        "implementation": "run_recalibration_experiments",
    },
    {
        "analysis": "Average ROC synthesis",
        "paper_ref": "Section 5, Figure 7",
        "implementation": "build_average_roc_curves",
    },
    {
        "analysis": "Drift-event reporting",
        "paper_ref": "Table A.9",
        "implementation": "run_adaptive_strategy + format_appendix_tables",
    },
]

PYTHON_MIGRATION_REVIEW_PLAN: List[Dict[str, str]] = [
    {
        "step": "1. Auditar paridad articulo vs. R",
        "objective": "Reemplazar bugs estructurales de main.R por una sola ruta Python verificable.",
        "python_artifacts": "build_python_migration_review_plan + run_recalibration_experiments",
        "addresses": "df_pred_vs_real inexistente, loops duplicados, NNet faltante",
        "review_check": "Verificar que cada estrategia/modelo corre desde una unica funcion y que NNet aparece en la corrida.",
    },
    {
        "step": "2. Congelar pipeline de datos",
        "objective": "Asegurar que la base usada por los experimentos sea determinista y trazable.",
        "python_artifacts": "run_configurable_preparation_pipeline + run_manifest",
        "addresses": "preparacion secuencial y reproducible",
        "review_check": "Confirmar filas iniciales/finales, features activas y configuracion guardada en el manifiesto.",
    },
    {
        "step": "3. Fijar seleccion final de variables",
        "objective": "Aplicar correlacion + ranking RF + top-N efectivo antes de entrenar.",
        "python_artifacts": "drop_highly_correlated_features + rank_features_random_forest",
        "addresses": "alineacion con las 15 variables mas relevantes del articulo",
        "review_check": "Comprobar que la pestaña de experimentos usa exactamente el top-N seleccionado.",
    },
    {
        "step": "4. Ejecutar yearly strategies por semilla",
        "objective": "Correr static, period-aligned y cumulative secuencialmente con umbral Youden por entrenamiento.",
        "python_artifacts": "run_yearly_strategy + train_model_with_internal_validation",
        "addresses": "umbral hardcodeado 0.0023 y ambiguedad de resultados",
        "review_check": "Revisar threshold, best_params y metricas por ano/modelo/semilla.",
    },
    {
        "step": "5. Ejecutar ADWIN secuencial",
        "objective": "Procesar el stream punto a punto, registrar drift y reentrenar solo con W1.",
        "python_artifacts": "run_adaptive_strategy + SimpleADWIN",
        "addresses": "recalibracion adaptativa reproducible",
        "review_check": "Validar drift_date, W/W0/W1 y cada reentreno en el log.",
    },
    {
        "step": "6. Agregar resultados y evidencias",
        "objective": "Separar resultados crudos por corrida de los promedios reportables.",
        "python_artifacts": "summarize_results + format_appendix_tables_mean + build_average_roc_curves",
        "addresses": "resultados finales del articulo como promedio sobre semillas",
        "review_check": "Comparar tablas raw vs. mean y revisar Figure 7/Appendix A.6-A.9.",
    },
    {
        "step": "7. Revisar auditoria de ejecucion",
        "objective": "Tener trazabilidad completa de cada paso para reproducir o depurar.",
        "python_artifacts": "execution_log + run_manifest",
        "addresses": "falta de log estructurado",
        "review_check": "Inspeccionar eventos ok/skipped/error antes de aceptar los resultados.",
    },
]


def build_related_work_table() -> pd.DataFrame:
    """Returns a compact replica of paper Table 1."""
    rows = [
        ("Wibisono et al. (2016)", "Traffic flow forecasting", "FIMT-DD built-in drift"),
        ("Saadallah et al. (2018)", "Taxi demand forecasting", "Adaptive ensemble"),
        ("Pan et al. (2020)", "User targeting distribution shift", "Adversarial validation"),
        ("Laha and Verma (2021)", "Mobility mode classification", "OT-based drift detector"),
        ("Manias et al. (2021)", "Streaming adaptation", "Windowed ensemble adaptation"),
        ("Andresini et al. (2021)", "Edge/fog traffic streams", "Window-based incremental learning"),
        ("Malekghaini et al. (2022)", "Vehicular network monitoring", "Spectral-entropy drift"),
        ("Lee and Park (2022)", "Autonomous vehicle control", "OSWDD"),
        ("Hossain et al. (2024)", "Safety-critical crash risk", "Empirical data drift analysis"),
        ("This paper", "RTCP recalibration timing", "ADWIN-guided adaptive retraining"),
    ]
    return pd.DataFrame(rows, columns=["paper", "objective", "drift_method"])


def feature_catalog_table() -> pd.DataFrame:
    """Tabla tipo Table 4, corregida y traducida al español."""
    rows = [
        {
            "bloque": "Velocidad promedio por pórtico y tipo de vehículo",
            "equivalente_articulo": "Vel_x^Y",
            "que_mide": "La velocidad media observada en un pórtico para una categoría de vehículo en un intervalo de 5 minutos.",
            "como_se_calcula": "Se agrupan los registros AVI por intervalo, pórtico y tipo de vehículo, y se promedia la velocidad registrada.",
        },
        {
            "bloque": "Desviación estándar de velocidad por pórtico y tipo de vehículo",
            "equivalente_articulo": "Sd_x^Y",
            "que_mide": "La dispersión de velocidades dentro del mismo pórtico, intervalo y tipo de vehículo.",
            "como_se_calcula": "Se usa la desviación estándar muestral de las velocidades de los vehículos observados en ese grupo.",
        },
        {
            "bloque": "Flujo por pórtico y tipo de vehículo",
            "equivalente_articulo": "Flow_x^Y",
            "que_mide": "La cantidad de vehículos de una categoría que pasan por un pórtico en un intervalo.",
            "como_se_calcula": "Se cuenta el número de registros AVI del grupo.",
        },
        {
            "bloque": "Densidad por pórtico y tipo de vehículo",
            "equivalente_articulo": "Den_x^Y",
            "que_mide": "Una aproximación de la ocupación del tráfico en el pórtico para una categoría.",
            "como_se_calcula": "Se divide el flujo del grupo por su velocidad promedio. Si no hay velocidad válida, la densidad queda vacía.",
        },
        {
            "bloque": "Cambios temporales por pórtico y tipo de vehículo",
            "equivalente_articulo": "DeltaFlow_x^Y, DeltaVel_x^Y, DeltaDen_x^Y, DeltaSd_x^Y",
            "que_mide": "Cómo cambia el flujo, la velocidad, la densidad o la dispersión de velocidad entre intervalos consecutivos.",
            "como_se_calcula": "Para cada variable base del pórtico, se resta el valor del intervalo anterior dentro del mismo pórtico y tipo de vehículo.",
        },
        {
            "bloque": "Fracciones de motos y pesados por pórtico",
            "equivalente_articulo": "Ft_Motorcycle^Y, Ft_Heavy^Y",
            "que_mide": "Qué proporción del flujo total del pórtico corresponde a motos o a vehículos pesados.",
            "como_se_calcula": "Se divide el flujo de la categoría por el flujo total del pórtico en el mismo intervalo.",
        },
        {
            "bloque": "Variables por segmento y tipo de vehículo",
            "equivalente_articulo": "Vel_x^{Y,Z}, Sd_x^{Y,Z}, Flow_x^{Y,Z}, Den_x^{Y,Z}, DeltaFlow_x^{Y,Z}, DeltaVel_x^{Y,Z}, DeltaDen_x^{Y,Z}",
            "que_mide": "El estado y la evolución del tráfico entre dos pórticos consecutivos para cada tipo de vehículo.",
            "como_se_calcula": "Se emparejan patentes entre pórticos consecutivos dentro de una ventana de tiempo razonable. A partir del tiempo de viaje y la distancia entre pórticos se calcula la velocidad espacial; luego se agregan flujo, dispersión, densidad y sus deltas por intervalo y categoría.",
        },
        {
            "bloque": "Cambio de pista por segmento y tipo de vehículo",
            "equivalente_articulo": "CL_x^{Y,Z}",
            "que_mide": "La proporción de vehículos emparejados que cambian de carril entre el pórtico de origen y el de destino.",
            "como_se_calcula": "Dentro de los vehículos emparejados del segmento, se compara el carril de entrada y salida y se calcula la fracción con cambio.",
        },
        {
            "bloque": "Variables globales por segmento",
            "equivalente_articulo": "Vel^{Y,Z}, Sd^{Y,Z}, Flow^{Y,Z}, Den^{Y,Z}, DeltaFlow^{Y,Z}, DeltaVel^{Y,Z}, DeltaDen^{Y,Z}",
            "que_mide": "La condición agregada del tráfico del segmento sin separar por tipo de vehículo.",
            "como_se_calcula": "Se repite el mismo cálculo del segmento, pero agrupando todos los vehículos juntos.",
        },
        {
            "bloque": "Variables por carril en el segmento",
            "equivalente_articulo": "Vel_lane_l^{Y,Z}, Sd_lane_l^{Y,Z}, Flow_lane_l^{Y,Z}, Den_lane_l^{Y,Z}, DeltaFlow_lane_l^{Y,Z}, DeltaVel_lane_l^{Y,Z}, DeltaDen_lane_l^{Y,Z}",
            "que_mide": "La condición del segmento vista desde el carril de origen del vehículo. Permite detectar diferencias entre carriles y coincide con lo observado en la Figure 6.",
            "como_se_calcula": "Sobre los vehículos emparejados del segmento, se filtra por carril de origen en Y y se agregan velocidad, flujo, densidad, dispersión y deltas para cada lane.",
        },
        {
            "bloque": "Target de accidente",
            "equivalente_articulo": "Acc(t)",
            "que_mide": "Si en el siguiente intervalo ocurre al menos un accidente en el corredor analizado.",
            "como_se_calcula": "Los accidentes se alinean a la grilla de 5 minutos y luego se desplazan un intervalo hacia atrás, de modo que las features de t predicen accidentes en t+1.",
        },
    ]
    return pd.DataFrame(rows)


def feature_engineering_reference_table() -> pd.DataFrame:
    """Referencia detallada en español para la UI de Drift detection."""
    rows = [
        {
            "bloque": "Estado del tráfico en cada pórtico por tipo de vehículo",
            "columnas_generadas": "vel_*, sd_*, flow_*, den_*, delta_flow_*, delta_vel_*, delta_den_*, delta_sd_*",
            "nivel": "Pórtico",
            "desagregacion": "Light / Heavy / Motorcycle",
            "que_mide": "Describe cómo se comporta el tráfico en un pórtico específico para cada tipo de vehículo.",
            "como_se_calcula": "Se agrupan los registros AVI por intervalo de 5 minutos, pórtico y categoría. El flujo es un conteo, la velocidad es un promedio, la desviación estándar mide dispersión, la densidad es flujo dividido por velocidad y los deltas son diferencias contra el intervalo anterior.",
            "ejemplos": "vel_light_15, flow_heavy_12, delta_sd_motorcycle_14",
        },
        {
            "bloque": "Composición del flujo en cada pórtico",
            "columnas_generadas": "ft_motorcycle_*, ft_heavy_*",
            "nivel": "Pórtico",
            "desagregacion": "Total del pórtico",
            "que_mide": "Indica qué porcentaje del flujo corresponde a motos o a vehículos pesados.",
            "como_se_calcula": "Se divide el flujo de la categoría por el flujo total del pórtico en el mismo intervalo.",
            "ejemplos": "ft_motorcycle_15, ft_heavy_12",
        },
        {
            "bloque": "Estado del tráfico en cada segmento por tipo de vehículo",
            "columnas_generadas": "vel_*_15_14, sd_*_15_14, flow_*_15_14, den_*_15_14, delta_flow_*_15_14, delta_vel_*_15_14, delta_den_*_15_14, cl_*_15_14",
            "nivel": "Segmento entre pórticos consecutivos",
            "desagregacion": "Light / Heavy / Motorcycle",
            "que_mide": "Captura cómo se mueve cada tipo de vehículo entre dos pórticos consecutivos del corredor.",
            "como_se_calcula": "Se emparejan patentes entre pórticos consecutivos dentro de una ventana razonable. Con la distancia del segmento y el tiempo de viaje se calcula la velocidad espacial. Luego se agregan flujo, velocidad, dispersión, densidad, deltas y cambio de pista por intervalo y categoría.",
            "ejemplos": "flow_heavy_15_14, delta_vel_light_14_12, cl_motorcycle_12_11",
        },
        {
            "bloque": "Estado global del segmento",
            "columnas_generadas": "vel_15_14, sd_15_14, flow_15_14, den_15_14, delta_flow_15_14, delta_vel_15_14, delta_den_15_14",
            "nivel": "Segmento entre pórticos consecutivos",
            "desagregacion": "Todos los vehículos juntos",
            "que_mide": "Resume la condición general del segmento sin separar por tipo de vehículo.",
            "como_se_calcula": "Se usa el mismo matching entre pórticos, pero agregando todos los vehículos en un solo grupo.",
            "ejemplos": "vel_15_14, den_14_12, delta_den_12_11",
        },
        {
            "bloque": "Estado del segmento por carril",
            "columnas_generadas": "vel_lane1_15_14, sd_lane2_14_12, flow_lane3_12_11, den_lane*_*, delta_flow_lane*_*, delta_vel_lane*_*, delta_den_lane*_*",
            "nivel": "Segmento entre pórticos consecutivos",
            "desagregacion": "Lane 1 / Lane 2 / Lane 3",
            "que_mide": "Permite ver si el segmento se comporta distinto según el carril de origen del vehículo.",
            "como_se_calcula": "Se toma el matching del segmento y se vuelve a agregar, pero separando por el carril en el que el vehículo fue observado en el pórtico de origen. Este bloque se incluyó porque la Figure 6 destaca variables por lane entre las más relevantes.",
            "ejemplos": "vel_lane2_15_14, flow_lane1_14_12, delta_den_lane3_12_11",
        },
        {
            "bloque": "Target de accidente",
            "columnas_generadas": "target",
            "nivel": "Intervalo",
            "desagregacion": "Corredor completo",
            "que_mide": "Indica si en el siguiente intervalo ocurre al menos un accidente dentro del corredor analizado.",
            "como_se_calcula": "Los accidentes se alinean a la grilla temporal de 5 minutos y se desplazan un intervalo hacia atrás. Así, las variables del intervalo t intentan anticipar accidentes en t+1.",
            "ejemplos": "target",
        },
    ]
    return pd.DataFrame(rows)


def hyperparameter_reference_table() -> pd.DataFrame:
    """Returns Table 5 grids exactly as described in the paper."""
    rows = [
        (
            "Random Forest",
            "mtry={2,4,6}; splitrule={gini,extratrees}; min.node.size={1,5,10}; 5-fold CV",
        ),
        (
            "XGBoost",
            "max_depth={3,6,9}; eta={0.01,0.1,0.3}; gamma={0,1,5}; "
            "colsample_bytree={0.5,0.8,1}; min_child_weight={1,5,10}; "
            "subsample={0.5,0.7,1}; nrounds=100; 5-fold CV",
        ),
        (
            "AdaBoost",
            "iter={50,100,150}; maxdepth={1,2,3}; nu={0.1,0.5,1}; 5-fold CV",
        ),
        (
            "Neural Network (NNet)",
            "size={6,7,10,12,13,14,15,16}; decay={0.095,0.1,0.15,0.2}; "
            "maxit={190,200,300,500}; 5-fold CV",
        ),
    ]
    return pd.DataFrame(rows, columns=["model", "grid"])


def build_article_coverage_matrix() -> pd.DataFrame:
    """
    Declares full article coverage and validates each item is mapped to a
    concrete implementation artifact in this module.
    """
    mappings: List[Dict[str, str]] = []

    for sec_num, sec_title in ARTICLE_SECTIONS:
        mappings.append(
            {
                "item_id": f"Section {sec_num}",
                "item_type": "section",
                "title": sec_title,
                "artifact": "REPLICATION_ANALYSES",
            }
        )

    figure_artifacts = {
        "Figure 1": "build_gate_layout_from_porticos",
        "Figure 2": "build_gate_layout_from_porticos",
        "Figure 3": "compute_accident_hour_distribution",
        "Figure 4": "build_data_preparation_summary",
        "Figure 5": "compute_abs_correlations",
        "Figure 6": "rank_features_random_forest",
        "Figure 7": "build_average_roc_curves",
    }
    for fig_id, fig_title in ARTICLE_FIGURES:
        mappings.append(
            {
                "item_id": fig_id,
                "item_type": "figure",
                "title": fig_title,
                "artifact": figure_artifacts[fig_id],
            }
        )

    table_artifacts = {
        "Table 1": "build_related_work_table",
        "Table 2": "sample_avi_records",
        "Table 3": "compute_annual_traffic_summary",
        "Table 4": "feature_catalog_table",
        "Table 5": "hyperparameter_reference_table",
        "Table A.6": "format_appendix_tables",
        "Table A.7": "format_appendix_tables",
        "Table A.8": "format_appendix_tables",
        "Table A.9": "format_appendix_tables",
    }
    for tab_id, tab_title in ARTICLE_TABLES:
        mappings.append(
            {
                "item_id": tab_id,
                "item_type": "table",
                "title": tab_title,
                "artifact": table_artifacts[tab_id],
            }
        )

    matrix = pd.DataFrame(mappings)
    implemented: List[bool] = []
    for artifact in matrix["artifact"]:
        if artifact == "REPLICATION_ANALYSES":
            implemented.append(len(REPLICATION_ANALYSES) > 0)
            continue
        obj = globals().get(artifact)
        implemented.append(callable(obj))
    matrix["implemented"] = implemented
    return matrix


def article_coverage_percentage(matrix: Optional[pd.DataFrame] = None) -> float:
    matrix = build_article_coverage_matrix() if matrix is None else matrix
    if matrix.empty:
        return 0.0
    return float(matrix["implemented"].mean() * 100.0)


def article_replication_blueprint() -> Dict[str, pd.DataFrame]:
    analyses_df = pd.DataFrame(REPLICATION_ANALYSES)
    figures_df = pd.DataFrame(ARTICLE_FIGURES, columns=["figure", "description"])
    tables_df = pd.DataFrame(ARTICLE_TABLES, columns=["table", "description"])
    return {
        "analyses": analyses_df,
        "figures": figures_df,
        "tables": tables_df,
    }


def build_python_migration_review_plan() -> pd.DataFrame:
    """Ordered checklist to review the R -> Python migration work."""
    return pd.DataFrame(PYTHON_MIGRATION_REVIEW_PLAN)


def sample_avi_records(flows_df: pd.DataFrame, n_rows: int = 6) -> pd.DataFrame:
    """Builds a paper-like sample table (Table 2 style)."""
    if flows_df is None or flows_df.empty:
        return pd.DataFrame()
    expected = ["FECHA", "VELOCIDAD", "CATEGORIA", "MATRICULA", "PORTICO", "CARRIL"]
    available = [c for c in expected if c in flows_df.columns]
    if not available:
        return pd.DataFrame()
    out = flows_df[available].head(max(1, int(n_rows))).copy()
    return out.reset_index(drop=True)


def compute_annual_traffic_summary(
    flows_df: pd.DataFrame,
    *,
    time_col: str = "FECHA",
    speed_col: str = "VELOCIDAD",
    category_col: str = "CATEGORIA",
    lane_col: str = "CARRIL",
    plate_col: str = "MATRICULA",
    category_map: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """
    Computes annual summary aligned with paper Table 3 intent.
    """
    if flows_df is None or flows_df.empty:
        return pd.DataFrame()

    work = flows_df.copy()
    if time_col not in work.columns:
        return pd.DataFrame()

    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.dropna(subset=[time_col])
    if work.empty:
        return pd.DataFrame()

    work["year"] = work[time_col].dt.year.astype(int)

    if category_map is None:
        category_map = {1: "Light", 2: "Heavy", 3: "Heavy", 4: "Motorcycle"}

    if category_col in work.columns:
        work[category_col] = pd.to_numeric(work[category_col], errors="coerce")
        work["category_name"] = work[category_col].map(category_map).fillna("Other")
    else:
        work["category_name"] = "Unknown"

    if speed_col in work.columns:
        work[speed_col] = pd.to_numeric(work[speed_col], errors="coerce")

    rows: List[Dict[str, Any]] = []
    for year, grp in work.groupby("year"):
        row: Dict[str, Any] = {
            "year": int(year),
            "vehicular_flow": int(len(grp)),
        }
        if plate_col in grp.columns:
            row["distinct_plates"] = int(grp[plate_col].astype(str).nunique())
        else:
            row["distinct_plates"] = np.nan

        for cat in ["Light", "Heavy", "Motorcycle"]:
            share = float((grp["category_name"] == cat).mean() * 100.0)
            row[f"share_{cat.lower()}"] = share

        if lane_col in grp.columns:
            lane_vals = pd.to_numeric(grp[lane_col], errors="coerce")
            for lane in [1, 2, 3]:
                row[f"lane_{lane}_share"] = float((lane_vals == lane).mean() * 100.0)

        if speed_col in grp.columns:
            for cat in ["Light", "Heavy", "Motorcycle"]:
                cat_speed = pd.to_numeric(
                    grp.loc[grp["category_name"] == cat, speed_col], errors="coerce"
                )
                row[f"speed_{cat.lower()}"] = float(cat_speed.mean()) if not cat_speed.empty else np.nan

        rows.append(row)

    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def compute_accident_hour_distribution(
    accidents_df: pd.DataFrame,
    *,
    accident_time_col: str = "accidente_time",
) -> pd.DataFrame:
    """Replicates Figure 3 style hourly accident distribution."""
    if accidents_df is None or accidents_df.empty:
        return pd.DataFrame(columns=["hour", "accidents", "share_pct"])
    if accident_time_col not in accidents_df.columns:
        return pd.DataFrame(columns=["hour", "accidents", "share_pct"])

    work = accidents_df.copy()
    work[accident_time_col] = pd.to_datetime(work[accident_time_col], errors="coerce")
    work = work.dropna(subset=[accident_time_col])
    if work.empty:
        return pd.DataFrame(columns=["hour", "accidents", "share_pct"])

    work["hour"] = work[accident_time_col].dt.hour.astype(int)
    counts = work.groupby("hour").size().reindex(range(24), fill_value=0)
    total = float(counts.sum())
    out = pd.DataFrame(
        {
            "hour": counts.index.astype(int),
            "accidents": counts.values.astype(int),
        }
    )
    out["share_pct"] = np.where(total > 0, out["accidents"] / total * 100.0, 0.0)
    return out


def compute_accident_hour_distribution_from_target(
    features_df: pd.DataFrame,
    *,
    time_col: str = "interval_start",
    target_col: str = "target",
) -> pd.DataFrame:
    """Alternative hourly distribution when only interval-level target is available."""
    if features_df is None or features_df.empty:
        return pd.DataFrame(columns=["hour", "accidents", "share_pct"])
    if time_col not in features_df.columns or target_col not in features_df.columns:
        return pd.DataFrame(columns=["hour", "accidents", "share_pct"])

    work = features_df.copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.dropna(subset=[time_col])
    if work.empty:
        return pd.DataFrame(columns=["hour", "accidents", "share_pct"])

    positives = work.loc[pd.to_numeric(work[target_col], errors="coerce").fillna(0).astype(int) == 1]
    if positives.empty:
        return pd.DataFrame({"hour": list(range(24)), "accidents": [0] * 24, "share_pct": [0.0] * 24})

    positives["hour"] = positives[time_col].dt.hour.astype(int)
    counts = positives.groupby("hour").size().reindex(range(24), fill_value=0)
    total = float(counts.sum())
    out = pd.DataFrame({"hour": counts.index.astype(int), "accidents": counts.values.astype(int)})
    out["share_pct"] = np.where(total > 0, out["accidents"] / total * 100.0, 0.0)
    return out


def _count_positive_target_rows(
    df: Optional[pd.DataFrame],
    *,
    target_col: str = "target",
) -> int:
    """Counts rows labeled as accidents in the final dataset."""
    if not isinstance(df, pd.DataFrame) or df.empty or target_col not in df.columns:
        return 0
    target = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)
    return int(target.eq(1).sum())


@dataclass
class DataPreparationSummary:
    initial_rows: int
    initial_features: int
    removed_high_missing_features: int
    remaining_features_after_stage1: int
    rows_after_missing_drop: int
    removed_zero_run_rows: int
    final_rows: int
    zero_run_windows_detected: int


def _infer_interval_minutes_from_feature_df(
    df: Optional[pd.DataFrame],
    *,
    time_col: str = "interval_start",
    default_minutes: int = 5,
) -> int:
    if not isinstance(df, pd.DataFrame) or df.empty or time_col not in df.columns:
        return int(default_minutes)
    timestamps = pd.to_datetime(df[time_col], errors="coerce").dropna().drop_duplicates().sort_values()
    if len(timestamps) < 2:
        return int(default_minutes)
    deltas = timestamps.diff().dropna()
    deltas = deltas[deltas > pd.Timedelta(0)]
    if deltas.empty:
        return int(default_minutes)
    try:
        mode_delta = deltas.mode().iloc[0]
        minutes = int(round(pd.Timedelta(mode_delta).total_seconds() / 60.0))
        return max(1, minutes)
    except Exception:
        return int(default_minutes)


def _rebuild_preparation_artifacts_from_payload(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    *,
    missing_threshold: float = 0.01,
    target_col: str = "target",
    time_col: str = "interval_start",
    min_zero_days: int = 7,
) -> Tuple[DataPreparationSummary, Dict[str, Any], pd.DataFrame]:
    """Reconstructs Stage 1/2 counts from raw_features + clean_features."""
    if raw_df is None or raw_df.empty:
        empty_summary = DataPreparationSummary(0, 0, 0, 0, 0, 0, int(len(clean_df)) if isinstance(clean_df, pd.DataFrame) else 0, 0)
        return empty_summary, {"removed_cols": [], "remaining_features": [], "rows_after_drop": 0}, pd.DataFrame()

    raw_features = _feature_columns_for_modeling(raw_df, target_col=target_col)
    missing_ratio = raw_df[raw_features].isna().mean() if raw_features else pd.Series(dtype=float)
    removed_cols = missing_ratio[missing_ratio > float(missing_threshold)].index.tolist()
    remaining_features = [col for col in raw_features if col not in set(removed_cols)]

    stage1_df = raw_df.copy()
    if remaining_features:
        stage1_df = stage1_df.dropna(subset=remaining_features).copy()
    if time_col in stage1_df.columns:
        stage1_df[time_col] = pd.to_datetime(stage1_df[time_col], errors="coerce")

    interval_minutes = _infer_interval_minutes_from_feature_df(stage1_df, time_col=time_col, default_minutes=5)
    removed_zero_run_rows = max(0, int(len(stage1_df) - len(clean_df)))
    zero_runs = pd.DataFrame(columns=["start", "end", "length_intervals", "length_days"])
    if removed_zero_run_rows > 0:
        zero_runs = identify_long_zero_accident_runs(
            stage1_df,
            target_col=target_col,
            time_col=time_col,
            min_days=min_zero_days,
            interval_minutes=interval_minutes,
        )

    summary = DataPreparationSummary(
        initial_rows=int(len(raw_df)),
        initial_features=int(len(raw_features)),
        removed_high_missing_features=int(len(removed_cols)),
        remaining_features_after_stage1=int(len(remaining_features)),
        rows_after_missing_drop=int(len(stage1_df)),
        removed_zero_run_rows=removed_zero_run_rows,
        final_rows=int(len(clean_df)),
        zero_run_windows_detected=int(len(zero_runs)) if removed_zero_run_rows > 0 else 0,
    )
    stage1_info = {
        "removed_cols": removed_cols,
        "remaining_features": remaining_features,
        "rows_after_drop": int(len(stage1_df)),
        "missing_ratio": missing_ratio.sort_values(ascending=False),
    }
    return summary, stage1_info, zero_runs


@dataclass
class FlowSampleSelection:
    date_start: Optional[pd.Timestamp] = None
    date_end: Optional[pd.Timestamp] = None
    row_limit: Optional[int] = None


def _format_bytes(value: Optional[float]) -> str:
    if value is None:
        return "-"
    size = float(value)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:,.2f} {unit}"
        size /= 1024.0
    return f"{size:,.2f} TB"


def _get_feature_engineering_ram_text() -> str:
    process_bytes: Optional[float] = None
    system_used_bytes: Optional[float] = None
    system_total_bytes: Optional[float] = None
    system_percent: Optional[float] = None

    if psutil is not None:
        try:
            proc = psutil.Process(os.getpid())
            process_bytes = float(proc.memory_info().rss)
        except Exception:
            process_bytes = None
        try:
            vm = psutil.virtual_memory()
            system_used_bytes = float(vm.used)
            system_total_bytes = float(vm.total)
            system_percent = float(vm.percent)
        except Exception:
            system_used_bytes = None
            system_total_bytes = None
            system_percent = None
    else:
        try:
            import resource

            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                process_bytes = float(rss)
            else:
                process_bytes = float(rss) * 1024.0
        except Exception:
            process_bytes = None
        if hasattr(os, "sysconf"):
            try:
                page_size = float(os.sysconf("SC_PAGE_SIZE"))
                phys_pages = float(os.sysconf("SC_PHYS_PAGES"))
                system_total_bytes = page_size * phys_pages
            except Exception:
                system_total_bytes = None

    parts: List[str] = []
    if process_bytes is not None:
        parts.append(f"RSS proceso: {_format_bytes(process_bytes)}")
    if system_used_bytes is not None and system_total_bytes is not None and system_percent is not None:
        parts.append(
            f"RAM sistema: {_format_bytes(system_used_bytes)} / {_format_bytes(system_total_bytes)} ({system_percent:.1f}%)"
        )
    elif system_total_bytes is not None and process_bytes is not None:
        percent = process_bytes / system_total_bytes * 100.0 if system_total_bytes > 0 else 0.0
        parts.append(f"RAM total: {_format_bytes(system_total_bytes)} | RSS {percent:.1f}%")
    return " | ".join(parts) if parts else "RAM -"


class _FeatureEngineeringProgress:
    def __init__(self) -> None:
        self._bar = st.progress(0.0)
        self._status = st.empty()
        self._ram = st.empty()
        self.set_progress(0.0, "Preparando pipeline...")

    def set_progress(self, ratio: float, label: str, *, detail: Optional[str] = None) -> None:
        bounded = max(0.0, min(float(ratio), 1.0))
        self._bar.progress(bounded)
        pct = int(round(bounded * 100.0))
        self._status.caption(f"{pct}% | {label}")
        ram_text = _get_feature_engineering_ram_text()
        if detail:
            self._ram.caption(f"{detail} | {ram_text}")
        else:
            self._ram.caption(ram_text)

    def clear(self) -> None:
        self._bar.empty()
        self._status.empty()
        self._ram.empty()


class _ExperimentProgress:
    def __init__(self) -> None:
        self._bar = st.progress(0.0)
        self._status = st.empty()
        self._detail = st.empty()
        self.update(
            {
                "completed_units": 0,
                "total_units": 1,
                "label": "Preparando experimentos...",
                "detail": "",
            }
        )

    def update(self, payload: Dict[str, Any]) -> None:
        total_units = max(1, int(payload.get("total_units", 1)))
        completed_units = float(payload.get("completed_units", 0))
        ratio = max(0.0, min(completed_units / float(total_units), 1.0))
        self._bar.progress(ratio)
        pct = int(round(ratio * 100.0))
        label = str(payload.get("label", "Ejecutando experimentos..."))
        detail = str(payload.get("detail", "") or "")
        self._status.caption(f"{pct}% | {label}")
        if detail:
            self._detail.caption(detail)
        else:
            if total_units == 1 and abs(completed_units) < 1e-9:
                self._detail.caption("Inicializando plan de ejecucion...")
                return
            if abs(completed_units - round(completed_units)) < 1e-9:
                self._detail.caption(f"{int(round(completed_units))} / {total_units} bloques completados")
            else:
                self._detail.caption(f"{completed_units:.2f} / {total_units} bloques completados")


def _update_feature_progress(
    progress_bar: Optional[Any],
    ratio: float,
    label: str,
    *,
    detail: Optional[str] = None,
) -> None:
    if progress_bar is None:
        return
    setter = getattr(progress_bar, "set_progress", None)
    if callable(setter):
        setter(ratio, label, detail=detail)
        return
    progress_fn = getattr(progress_bar, "progress", None)
    if callable(progress_fn):
        progress_fn(max(0.0, min(float(ratio), 1.0)))


def _emit_experiment_progress(
    progress_callback: Optional[Callable[[Dict[str, Any]], None]],
    *,
    ratio: float,
    label: str,
    detail: str = "",
) -> None:
    if progress_callback is None:
        return
    progress_callback(
        {
            "ratio": max(0.0, min(float(ratio), 1.0)),
            "label": str(label),
            "detail": str(detail),
        }
    )


def _canonical_drift_category_name(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().lower()
    aliases = {
        "light": "Light",
        "lights": "Light",
        "heavy": "Heavy",
        "heavies": "Heavy",
        "motorcycle": "Motorcycle",
        "motorcycles": "Motorcycle",
        "moto": "Motorcycle",
        "motos": "Motorcycle",
        "bike": "Motorcycle",
        "bikes": "Motorcycle",
    }
    return aliases.get(text)


def _resolve_drift_article_categories(
    categories: Optional[Sequence[str]] = None,
) -> List[str]:
    if not categories:
        return list(DRIFT_ARTICLE_CATEGORIES)
    normalized = [
        _canonical_drift_category_name(item)
        for item in categories
    ]
    normalized = [item for item in normalized if item is not None]
    if not normalized:
        return list(DRIFT_ARTICLE_CATEGORIES)
    return [item for item in DRIFT_ARTICLE_CATEGORIES if item in normalized]


def _unique_preserve_order(values: Iterable[object]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _resolve_drift_article_portico_order(
    allowed_porticos: Optional[Sequence[str]] = None,
) -> List[str]:
    base = allowed_porticos if allowed_porticos is not None else DRIFT_FOCUS_PORTICOS
    normalized = [
        _normalize_portico_code(item)
        for item in base
    ]
    normalized = [item for item in normalized if item is not None]
    ordered = _unique_preserve_order(normalized)
    article_first = [item for item in DRIFT_ARTICLE_PORTICO_ORDER if item in ordered]
    remainder = [item for item in ordered if item not in article_first]
    return article_first + remainder if article_first else ordered


def _resolve_drift_article_segments(
    portico_order: Sequence[str],
) -> List[Tuple[str, str]]:
    normalized = _resolve_drift_article_portico_order(portico_order)
    return [
        (normalized[idx], normalized[idx + 1])
        for idx in range(len(normalized) - 1)
    ]


def _parse_lat_lon_pair(value: object) -> Optional[Tuple[float, float]]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except Exception:
        return None


def _haversine_km(
    coord_a: Tuple[float, float],
    coord_b: Tuple[float, float],
) -> float:
    lat1, lon1 = coord_a
    lat2, lon2 = coord_b
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * 6371.0 * math.atan2(math.sqrt(a), math.sqrt(max(1e-12, 1.0 - a)))


def _lookup_article_segment_distance_km(
    start_portico: str,
    end_portico: str,
    *,
    porticos_df: Optional[pd.DataFrame] = None,
) -> Optional[float]:
    try:
        meta = load_porticos() if porticos_df is None else porticos_df.copy()
    except Exception:
        return None
    if meta is None or meta.empty or "portico" not in meta.columns:
        return None

    meta = meta.copy()
    meta["portico_norm"] = _normalize_portico_series(meta["portico"])
    candidates: List[float] = []

    try:
        segments_df = get_portico_segments(meta)
    except Exception:
        segments_df = pd.DataFrame()
    if not segments_df.empty:
        segments_df = segments_df.copy()
        segments_df["portico_last"] = _normalize_portico_series(segments_df["portico_last"])
        segments_df["portico_next"] = _normalize_portico_series(segments_df["portico_next"])
        exact = segments_df[
            (segments_df["portico_last"] == str(start_portico))
            & (segments_df["portico_next"] == str(end_portico))
        ]
        if not exact.empty:
            dist = pd.to_numeric(exact["km_next"], errors="coerce") - pd.to_numeric(
                exact["km_last"], errors="coerce"
            )
            dist = dist.abs().replace(0, np.nan).dropna()
            if not dist.empty:
                candidates.append(float(dist.max()))

    start_rows = meta.loc[meta["portico_norm"] == str(start_portico)].copy()
    end_rows = meta.loc[meta["portico_norm"] == str(end_portico)].copy()
    if start_rows.empty or end_rows.empty:
        return max(candidates) if candidates else None

    start_km = pd.to_numeric(start_rows.get("km"), errors="coerce").dropna().tolist()
    end_km = pd.to_numeric(end_rows.get("km"), errors="coerce").dropna().tolist()
    for km_start in start_km:
        for km_end in end_km:
            dist = abs(float(km_end) - float(km_start))
            if dist > 0:
                candidates.append(dist)

    if "lat-lon" in start_rows.columns and "lat-lon" in end_rows.columns:
        start_coords = [
            coord
            for coord in start_rows["lat-lon"].map(_parse_lat_lon_pair).tolist()
            if coord is not None
        ]
        end_coords = [
            coord
            for coord in end_rows["lat-lon"].map(_parse_lat_lon_pair).tolist()
            if coord is not None
        ]
        for coord_start in start_coords:
            for coord_end in end_coords:
                dist = _haversine_km(coord_start, coord_end)
                if dist > 0:
                    candidates.append(dist)

    if not candidates:
        return None
    return float(max(candidates))


def _article_match_tolerance_minutes(distance_km: Optional[float]) -> float:
    if distance_km is None or distance_km <= 0:
        return float(DRIFT_ARTICLE_MIN_MATCH_WINDOW_MINUTES)
    derived = distance_km / max(DRIFT_ARTICLE_MIN_REFERENCE_SPEED_KMH, 1e-9) * 60.0
    return float(max(DRIFT_ARTICLE_MIN_MATCH_WINDOW_MINUTES, derived))


def _rename_gate_article_columns(
    columns: Iterable[Tuple[str, str]],
    prefix: str,
) -> List[str]:
    return [
        f"{prefix}_{DRIFT_ARTICLE_CATEGORY_SLUGS.get(category, str(category).lower())}_{portico}"
        for portico, category in columns
    ]


def _compute_article_segment_matches(
    work: pd.DataFrame,
    *,
    start_portico: str,
    end_portico: str,
    distance_km: Optional[float],
    tolerance_minutes: Optional[float] = None,
) -> pd.DataFrame:
    needed = {
        "plate_clean",
        "portico_norm",
        "FECHA",
        "interval_start",
        "category_name",
        "lane_clean",
    }
    if work is None or work.empty or not needed.issubset(set(work.columns)):
        return pd.DataFrame(
            columns=["interval_start", "category_name", "segment_speed", "lane_change", "lane_origin"]
        )

    subset = work.loc[
        work["plate_clean"].notna()
        & work["portico_norm"].isin([str(start_portico), str(end_portico)])
    ].copy()
    if subset.empty:
        return pd.DataFrame(
            columns=["interval_start", "category_name", "segment_speed", "lane_change", "lane_origin"]
        )

    start_df = subset.loc[
        subset["portico_norm"] == str(start_portico),
        ["plate_clean", "FECHA", "interval_start", "category_name", "lane_clean"],
    ].rename(
        columns={
            "FECHA": "time_start",
            "category_name": "category_start",
            "lane_clean": "lane_start",
        }
    )
    end_df = subset.loc[
        subset["portico_norm"] == str(end_portico),
        ["plate_clean", "FECHA", "category_name", "lane_clean"],
    ].rename(
        columns={
            "FECHA": "time_end",
            "category_name": "category_end",
            "lane_clean": "lane_end",
        }
    )
    if start_df.empty or end_df.empty:
        return pd.DataFrame(
            columns=["interval_start", "category_name", "segment_speed", "lane_change", "lane_origin"]
        )

    tolerance = pd.Timedelta(
        minutes=float(
            tolerance_minutes
            if tolerance_minutes is not None
            else _article_match_tolerance_minutes(distance_km)
        )
    )

    start_df = start_df.sort_values(["time_start", "plate_clean"])
    end_df = end_df.sort_values(["time_end", "plate_clean"])

    matches = pd.merge_asof(
        end_df,
        start_df,
        by="plate_clean",
        left_on="time_end",
        right_on="time_start",
        direction="backward",
        tolerance=tolerance,
    )
    matches = matches.dropna(subset=["time_start", "interval_start"])
    if matches.empty:
        return pd.DataFrame(
            columns=["interval_start", "category_name", "segment_speed", "lane_change", "lane_origin"]
        )

    matches = matches.loc[
        matches["category_start"].eq(matches["category_end"])
        | matches["category_end"].isna()
    ].copy()
    if matches.empty:
        return pd.DataFrame(
            columns=["interval_start", "category_name", "segment_speed", "lane_change", "lane_origin"]
        )

    matches = matches.sort_values(["plate_clean", "time_end"])
    matches = matches.drop_duplicates(subset=["plate_clean", "time_start"], keep="first")
    travel_minutes = (matches["time_end"] - matches["time_start"]).dt.total_seconds() / 60.0
    positive_mask = travel_minutes > 0
    matches = matches.loc[positive_mask].copy()
    if matches.empty:
        return pd.DataFrame(
            columns=["interval_start", "category_name", "segment_speed", "lane_change", "lane_origin"]
        )

    travel_hours = travel_minutes.loc[matches.index] / 60.0
    if distance_km is None or distance_km <= 0:
        matches["segment_speed"] = np.nan
    else:
        matches["segment_speed"] = float(distance_km) / travel_hours

    lane_valid = matches["lane_start"].notna() & matches["lane_end"].notna()
    matches["lane_change"] = np.where(
        lane_valid,
        matches["lane_start"].astype(str) != matches["lane_end"].astype(str),
        np.nan,
    )
    matches["category_name"] = matches["category_start"].fillna(matches["category_end"])
    matches["lane_origin"] = (
        matches["lane_start"]
        .astype("string")
        .str.extract(r"(\d+)", expand=False)
    )
    return matches[["interval_start", "category_name", "segment_speed", "lane_change", "lane_origin"]]


def _compute_drift_article_features(
    flows_df: pd.DataFrame,
    *,
    interval_minutes: int = 5,
    categories: Optional[Sequence[str]] = None,
    allowed_porticos: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    required = {"FECHA", "VELOCIDAD", "CATEGORIA", "MATRICULA", "PORTICO", "CARRIL"}
    if flows_df is None or flows_df.empty or not required.issubset(set(flows_df.columns)):
        return pd.DataFrame()

    portico_order = _resolve_drift_article_portico_order(allowed_porticos)
    if not portico_order:
        return pd.DataFrame()
    category_order = _resolve_drift_article_categories(categories)
    segment_order = _resolve_drift_article_segments(portico_order)

    work = flows_df[list(required)].copy()
    work["FECHA"] = pd.to_datetime(work["FECHA"], errors="coerce")
    work["VELOCIDAD"] = pd.to_numeric(work["VELOCIDAD"], errors="coerce")
    work["CATEGORIA"] = pd.to_numeric(work["CATEGORIA"], errors="coerce")
    work["portico_norm"] = _normalize_portico_series(work["PORTICO"])
    work["category_raw"] = pd.to_numeric(work["CATEGORIA"], errors="coerce").astype("Int64")
    work["category_name"] = work["category_raw"].map(DRIFT_ARTICLE_CATEGORY_MAP)
    work["plate_clean"] = (
        work["MATRICULA"]
        .astype("string")
        .str.strip()
        .str.upper()
        .replace({"": pd.NA, "NAN": pd.NA, "NULL": pd.NA, "NONE": pd.NA})
    )
    work["lane_clean"] = (
        work["CARRIL"]
        .astype("string")
        .str.strip()
        .replace({"": pd.NA, "NAN": pd.NA, "NULL": pd.NA, "NONE": pd.NA})
    )
    work = work.dropna(subset=["FECHA", "portico_norm", "category_name"])
    work = work.loc[
        work["portico_norm"].isin(portico_order)
        & work["category_name"].isin(category_order)
    ].copy()
    if work.empty:
        return pd.DataFrame()

    work["interval_start"] = work["FECHA"].dt.floor(f"{int(interval_minutes)}min")
    intervals = (
        pd.Index(work["interval_start"].dropna().sort_values().unique(), name="interval_start")
        .sort_values()
    )
    if len(intervals) == 0:
        return pd.DataFrame()

    interval_frame = pd.DataFrame({"interval_start": intervals})
    gate_columns = pd.MultiIndex.from_product(
        [portico_order, category_order],
        names=["portico", "category_name"],
    )

    gate_stats = (
        work.groupby(["interval_start", "portico_norm", "category_name"], dropna=False)
        .agg(
            flow=("category_name", "size"),
            vel=("VELOCIDAD", "mean"),
            sd=("VELOCIDAD", "std"),
        )
        .reset_index()
    )
    gate_stats["den"] = np.where(
        gate_stats["vel"] > 0,
        gate_stats["flow"] / gate_stats["vel"],
        np.nan,
    )

    def _gate_pivot(value_col: str, prefix: str, *, fill_zero: bool) -> pd.DataFrame:
        wide = gate_stats.pivot(
            index="interval_start",
            columns=["portico_norm", "category_name"],
            values=value_col,
        )
        wide = wide.reindex(index=intervals, columns=gate_columns)
        if fill_zero:
            wide = wide.fillna(0.0)
        wide = wide.astype(float)
        wide.columns = _rename_gate_article_columns(list(wide.columns), prefix)
        return wide

    gate_flow = _gate_pivot("flow", "flow", fill_zero=True)
    gate_vel = _gate_pivot("vel", "vel", fill_zero=False)
    gate_sd = _gate_pivot("sd", "sd", fill_zero=False)
    gate_den = _gate_pivot("den", "den", fill_zero=False)

    def _delta_frame(frame: pd.DataFrame, *, prefix_from: str, prefix_to: str) -> pd.DataFrame:
        delta = frame.diff()
        if not delta.empty:
            first_valid = frame.iloc[0].notna()
            delta.iloc[0, first_valid.to_numpy()] = 0.0
        delta.columns = [
            col.replace(f"{prefix_from}_", f"{prefix_to}_", 1)
            for col in frame.columns
        ]
        return delta

    gate_delta_flow = _delta_frame(gate_flow, prefix_from="flow", prefix_to="delta_flow")
    gate_delta_vel = _delta_frame(gate_vel, prefix_from="vel", prefix_to="delta_vel")
    gate_delta_den = _delta_frame(gate_den, prefix_from="den", prefix_to="delta_den")
    gate_delta_sd = _delta_frame(gate_sd, prefix_from="sd", prefix_to="delta_sd")

    result = interval_frame.set_index("interval_start")
    for frame in [
        gate_flow,
        gate_vel,
        gate_sd,
        gate_den,
        gate_delta_flow,
        gate_delta_vel,
        gate_delta_den,
        gate_delta_sd,
    ]:
        result = result.join(frame, how="left")

    for portico in portico_order:
        total = (
            result.get(f"flow_light_{portico}", pd.Series(0.0, index=result.index)).fillna(0.0)
            + result.get(f"flow_heavy_{portico}", pd.Series(0.0, index=result.index)).fillna(0.0)
            + result.get(f"flow_motorcycle_{portico}", pd.Series(0.0, index=result.index)).fillna(0.0)
        )
        motorcycle_flow = result.get(
            f"flow_motorcycle_{portico}",
            pd.Series(0.0, index=result.index),
        ).fillna(0.0)
        heavy_flow = result.get(
            f"flow_heavy_{portico}",
            pd.Series(0.0, index=result.index),
        ).fillna(0.0)
        result[f"ft_motorcycle_{portico}"] = np.where(total > 0, motorcycle_flow / total, np.nan)
        result[f"ft_heavy_{portico}"] = np.where(total > 0, heavy_flow / total, np.nan)

    porticos_meta: Optional[pd.DataFrame]
    try:
        porticos_meta = load_porticos()
    except Exception:
        porticos_meta = None

    for start_portico, end_portico in segment_order:
        distance_km = _lookup_article_segment_distance_km(
            start_portico,
            end_portico,
            porticos_df=porticos_meta,
        )
        matches = _compute_article_segment_matches(
            work,
            start_portico=start_portico,
            end_portico=end_portico,
            distance_km=distance_km,
        )
        lane_order = ["1", "2", "3"]
        if matches.empty:
            segment_stats = pd.DataFrame(
                columns=["interval_start", "category_name", "flow", "vel", "sd", "cl", "den"]
            )
            overall_stats = pd.DataFrame(columns=["interval_start", "flow", "vel", "sd", "den"])
            lane_stats = pd.DataFrame(columns=["interval_start", "lane_origin", "flow", "vel", "sd", "den"])
        else:
            segment_stats = (
                matches.groupby(["interval_start", "category_name"], dropna=False)
                .agg(
                    flow=("category_name", "size"),
                    vel=("segment_speed", "mean"),
                    sd=("segment_speed", "std"),
                    cl=("lane_change", "mean"),
                )
                .reset_index()
            )
            segment_stats["den"] = np.where(
                segment_stats["vel"] > 0,
                segment_stats["flow"] / segment_stats["vel"],
                np.nan,
            )
            overall_stats = (
                matches.groupby("interval_start", dropna=False)
                .agg(
                    flow=("category_name", "size"),
                    vel=("segment_speed", "mean"),
                    sd=("segment_speed", "std"),
                )
                .reset_index()
            )
            overall_stats["den"] = np.where(
                overall_stats["vel"] > 0,
                overall_stats["flow"] / overall_stats["vel"],
                np.nan,
            )
            lane_stats = (
                matches.dropna(subset=["lane_origin"])
                .groupby(["interval_start", "lane_origin"], dropna=False)
                .agg(
                    flow=("lane_origin", "size"),
                    vel=("segment_speed", "mean"),
                    sd=("segment_speed", "std"),
                )
                .reset_index()
            )
            lane_stats["den"] = np.where(
                lane_stats["vel"] > 0,
                lane_stats["flow"] / lane_stats["vel"],
                np.nan,
            )

        def _segment_pivot(value_col: str, prefix: str, *, fill_zero: bool) -> pd.DataFrame:
            wide = segment_stats.pivot(
                index="interval_start",
                columns="category_name",
                values=value_col,
            )
            wide = wide.reindex(index=intervals, columns=category_order)
            if fill_zero:
                wide = wide.fillna(0.0)
            wide = wide.astype(float)
            wide.columns = [
                f"{prefix}_{DRIFT_ARTICLE_CATEGORY_SLUGS.get(category, str(category).lower())}_{start_portico}_{end_portico}"
                for category in wide.columns
            ]
            return wide

        def _lane_pivot(value_col: str, prefix: str, *, fill_zero: bool) -> pd.DataFrame:
            wide = lane_stats.pivot(
                index="interval_start",
                columns="lane_origin",
                values=value_col,
            )
            wide = wide.reindex(index=intervals, columns=lane_order)
            if fill_zero:
                wide = wide.fillna(0.0)
            wide = wide.astype(float)
            wide.columns = [
                f"{prefix}_lane{lane}_{start_portico}_{end_portico}"
                for lane in wide.columns
            ]
            return wide

        seg_flow = _segment_pivot("flow", "flow", fill_zero=True)
        seg_vel = _segment_pivot("vel", "vel", fill_zero=False)
        seg_sd = _segment_pivot("sd", "sd", fill_zero=False)
        seg_den = _segment_pivot("den", "den", fill_zero=False)
        seg_cl = _segment_pivot("cl", "cl", fill_zero=False)
        seg_delta_flow = _delta_frame(seg_flow, prefix_from="flow", prefix_to="delta_flow")
        seg_delta_vel = _delta_frame(seg_vel, prefix_from="vel", prefix_to="delta_vel")
        seg_delta_den = _delta_frame(seg_den, prefix_from="den", prefix_to="delta_den")

        lane_flow = _lane_pivot("flow", "flow", fill_zero=True)
        lane_vel = _lane_pivot("vel", "vel", fill_zero=False)
        lane_sd = _lane_pivot("sd", "sd", fill_zero=False)
        lane_den = _lane_pivot("den", "den", fill_zero=False)
        lane_delta_flow = _delta_frame(lane_flow, prefix_from="flow", prefix_to="delta_flow")
        lane_delta_vel = _delta_frame(lane_vel, prefix_from="vel", prefix_to="delta_vel")
        lane_delta_den = _delta_frame(lane_den, prefix_from="den", prefix_to="delta_den")

        overall_stats = overall_stats.set_index("interval_start").reindex(intervals)
        overall_flow = pd.DataFrame(
            {
                f"flow_{start_portico}_{end_portico}": pd.to_numeric(
                    overall_stats["flow"],
                    errors="coerce",
                )
                .fillna(0.0)
                .astype(float)
            },
            index=overall_stats.index,
        )
        overall_vel = pd.DataFrame(
            {
                f"vel_{start_portico}_{end_portico}": pd.to_numeric(
                    overall_stats["vel"],
                    errors="coerce",
                ).astype(float)
            },
            index=overall_stats.index,
        )
        overall_sd = pd.DataFrame(
            {
                f"sd_{start_portico}_{end_portico}": pd.to_numeric(
                    overall_stats["sd"],
                    errors="coerce",
                ).astype(float)
            },
            index=overall_stats.index,
        )
        overall_den = pd.DataFrame(
            {
                f"den_{start_portico}_{end_portico}": pd.to_numeric(
                    overall_stats["den"],
                    errors="coerce",
                ).astype(float)
            },
            index=overall_stats.index,
        )
        overall_delta_flow = _delta_frame(overall_flow, prefix_from="flow", prefix_to="delta_flow")
        overall_delta_vel = _delta_frame(overall_vel, prefix_from="vel", prefix_to="delta_vel")
        overall_delta_den = _delta_frame(overall_den, prefix_from="den", prefix_to="delta_den")

        for frame in [
            seg_flow,
            seg_vel,
            seg_sd,
            seg_den,
            seg_cl,
            seg_delta_flow,
            seg_delta_vel,
            seg_delta_den,
            overall_flow,
            overall_vel,
            overall_sd,
            overall_den,
            overall_delta_flow,
            overall_delta_vel,
            overall_delta_den,
            lane_flow,
            lane_vel,
            lane_sd,
            lane_den,
            lane_delta_flow,
            lane_delta_vel,
            lane_delta_den,
        ]:
            result = result.join(frame, how="left")

    out = result.reset_index().sort_values("interval_start").reset_index(drop=True)
    value_cols = [col for col in out.columns if col != "interval_start"]
    for col in value_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    return out


def _filter_accidents_for_allowed_porticos(
    accidents_df: pd.DataFrame,
    allowed_porticos: Optional[Sequence[str]],
) -> pd.DataFrame:
    if accidents_df is None or accidents_df.empty:
        return pd.DataFrame()
    if not allowed_porticos:
        return accidents_df.copy()

    allowed = {
        item
        for item in (
            _normalize_portico_code(portico)
            for portico in allowed_porticos
        )
        if item is not None
    }
    if not allowed:
        return accidents_df.copy()

    work = accidents_df.copy()
    if "ultimo_portico" in work.columns and "proximo_portico" in work.columns:
        allowed_segments = {
            (str(start), str(end))
            for start, end in _resolve_drift_article_segments(list(allowed))
        }
        if allowed_segments:
            ultimo = _normalize_portico_series(work["ultimo_portico"])
            proximo = _normalize_portico_series(work["proximo_portico"])
            mask = pd.Series(False, index=work.index)
            for start_portico, end_portico in allowed_segments:
                mask |= ultimo.eq(start_portico) & proximo.eq(end_portico)
            return work.loc[mask].copy()

    masks: List[pd.Series] = []
    if "ultimo_portico" in work.columns:
        ultimo = _normalize_portico_series(work["ultimo_portico"])
        masks.append(ultimo.isin(allowed))
    if "proximo_portico" in work.columns:
        proximo = _normalize_portico_series(work["proximo_portico"])
        masks.append(proximo.isin(allowed))
    if not masks:
        return work

    mask = masks[0].copy()
    for extra in masks[1:]:
        mask |= extra
    return work.loc[mask].copy()


def _add_interval_accident_target(
    features_df: pd.DataFrame,
    accidents_df: pd.DataFrame,
    *,
    interval_minutes: int = 5,
    interval_col: str = "interval_start",
    accident_time_col: str = "accidente_time",
) -> pd.DataFrame:
    if features_df is None or features_df.empty:
        return pd.DataFrame()

    df = features_df.copy()
    if interval_col not in df.columns:
        return pd.DataFrame()
    df[interval_col] = pd.to_datetime(df[interval_col], errors="coerce")
    df = df.dropna(subset=[interval_col])
    if df.empty:
        return pd.DataFrame()

    if accidents_df is None or accidents_df.empty or accident_time_col not in accidents_df.columns:
        df["target"] = 0
        return df

    acc = accidents_df[[accident_time_col]].copy()
    acc[accident_time_col] = pd.to_datetime(acc[accident_time_col], errors="coerce")
    acc = acc.dropna(subset=[accident_time_col])
    if acc.empty:
        df["target"] = 0
        return df

    acc[interval_col] = acc[accident_time_col].dt.floor(f"{int(interval_minutes)}min") - pd.Timedelta(
        minutes=int(interval_minutes)
    )
    acc_pairs = acc[[interval_col]].drop_duplicates().copy()
    acc_pairs["target"] = 1
    merged = df.merge(acc_pairs, how="left", on=interval_col)
    merged["target"] = merged["target"].fillna(0).astype(int)
    return merged


def _infer_feature_columns(
    df: pd.DataFrame,
    *,
    exclude: Sequence[str],
) -> List[str]:
    candidates = [c for c in df.columns if c not in set(exclude)]
    numeric_candidates = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    return numeric_candidates


def apply_missing_data_policy(
    df: pd.DataFrame,
    *,
    feature_cols: Optional[Sequence[str]] = None,
    missing_threshold: float = 0.01,
    target_col: str = "target",
    time_col: str = "interval_start",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Paper Section 4.2 Stage 1:
    - remove variables with >1% missing values
    - remove intervals with any missing predictor value
    """
    if df is None or df.empty:
        return pd.DataFrame(), {
            "removed_cols": [],
            "remaining_features": [],
            "rows_after_drop": 0,
        }

    work = df.copy()
    if feature_cols is None:
        feature_cols = _infer_feature_columns(work, exclude=[target_col])
    feature_cols = [c for c in feature_cols if c in work.columns]

    missing_ratio = work[feature_cols].isna().mean()
    removed_cols = missing_ratio[missing_ratio > float(missing_threshold)].index.tolist()
    remaining = [c for c in feature_cols if c not in removed_cols]

    if remaining:
        cleaned = work.dropna(subset=remaining).copy()
    else:
        cleaned = work.copy()

    if time_col in cleaned.columns:
        cleaned[time_col] = pd.to_datetime(cleaned[time_col], errors="coerce")

    info = {
        "removed_cols": removed_cols,
        "remaining_features": remaining,
        "rows_after_drop": int(len(cleaned)),
        "missing_ratio": missing_ratio.sort_values(ascending=False),
    }
    return cleaned.reset_index(drop=True), info


def identify_long_zero_accident_runs(
    df: pd.DataFrame,
    *,
    target_col: str = "target",
    time_col: str = "interval_start",
    min_days: int = 7,
    interval_minutes: int = 5,
) -> pd.DataFrame:
    """
    Finds windows of >= min_days with no accident records (target==0).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["start", "end", "length_intervals", "length_days"])
    if time_col not in df.columns or target_col not in df.columns:
        return pd.DataFrame(columns=["start", "end", "length_intervals", "length_days"])

    work = df[[time_col, target_col]].copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce").fillna(0).astype(int)
    work = work.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    if work.empty:
        return pd.DataFrame(columns=["start", "end", "length_intervals", "length_days"])

    zero_mask = work[target_col].eq(0)
    group_id = (zero_mask != zero_mask.shift(fill_value=False)).cumsum()

    min_intervals = int(math.ceil((int(min_days) * 24 * 60) / float(interval_minutes)))

    rows: List[Dict[str, Any]] = []
    for gid, grp in work.groupby(group_id):
        del gid
        if not bool(grp[target_col].iloc[0] == 0):
            continue
        length = int(len(grp))
        if length < min_intervals:
            continue
        start_ts = pd.Timestamp(grp[time_col].iloc[0])
        end_ts = pd.Timestamp(grp[time_col].iloc[-1])
        days = float(length * float(interval_minutes) / (24.0 * 60.0))
        rows.append(
            {
                "start": start_ts,
                "end": end_ts,
                "length_intervals": length,
                "length_days": days,
            }
        )

    return pd.DataFrame(rows)


def filter_long_zero_accident_runs(
    df: pd.DataFrame,
    *,
    target_col: str = "target",
    time_col: str = "interval_start",
    min_days: int = 7,
    interval_minutes: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Drops rows that belong to long no-accident windows."""
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()

    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    runs = identify_long_zero_accident_runs(
        work,
        target_col=target_col,
        time_col=time_col,
        min_days=min_days,
        interval_minutes=interval_minutes,
    )
    if runs.empty:
        return work.reset_index(drop=True), runs

    drop_mask = np.zeros(len(work), dtype=bool)
    for run in runs.itertuples(index=False):
        drop_mask |= (work[time_col] >= run.start) & (work[time_col] <= run.end)

    filtered = work.loc[~drop_mask].reset_index(drop=True)
    return filtered, runs


def build_data_preparation_summary(
    df: pd.DataFrame,
    *,
    feature_cols: Optional[Sequence[str]] = None,
    missing_threshold: float = 0.01,
    target_col: str = "target",
    time_col: str = "interval_start",
    min_zero_days: int = 7,
    interval_minutes: int = 5,
) -> Tuple[pd.DataFrame, DataPreparationSummary, Dict[str, Any], pd.DataFrame]:
    """
    Runs the full Section 4.2 pipeline and returns clean dataset + stage statistics.
    """
    if df is None or df.empty:
        empty_summary = DataPreparationSummary(0, 0, 0, 0, 0, 0, 0, 0)
        return pd.DataFrame(), empty_summary, {}, pd.DataFrame()

    work = df.copy()
    if feature_cols is None:
        feature_cols = _infer_feature_columns(work, exclude=[target_col])

    initial_rows = int(len(work))
    initial_features = int(len(feature_cols))

    stage1_df, stage1_info = apply_missing_data_policy(
        work,
        feature_cols=feature_cols,
        missing_threshold=missing_threshold,
        target_col=target_col,
        time_col=time_col,
    )

    filtered_df, zero_runs = filter_long_zero_accident_runs(
        stage1_df,
        target_col=target_col,
        time_col=time_col,
        min_days=min_zero_days,
        interval_minutes=interval_minutes,
    )

    summary = DataPreparationSummary(
        initial_rows=initial_rows,
        initial_features=initial_features,
        removed_high_missing_features=int(len(stage1_info.get("removed_cols", []))),
        remaining_features_after_stage1=int(len(stage1_info.get("remaining_features", []))),
        rows_after_missing_drop=int(len(stage1_df)),
        removed_zero_run_rows=int(len(stage1_df) - len(filtered_df)),
        final_rows=int(len(filtered_df)),
        zero_run_windows_detected=int(len(zero_runs)),
    )

    return filtered_df, summary, stage1_info, zero_runs


def run_configurable_preparation_pipeline(
    df: pd.DataFrame,
    *,
    feature_cols: Optional[Sequence[str]] = None,
    apply_stage1: bool = True,
    missing_threshold: float = 0.01,
    apply_stage2: bool = True,
    min_zero_days: int = 7,
    target_col: str = "target",
    time_col: str = "interval_start",
    interval_minutes: int = 5,
) -> Tuple[pd.DataFrame, DataPreparationSummary, Dict[str, Any], pd.DataFrame, List[str]]:
    """
    Applies Stage 1 / Stage 2 with explicit toggles for the Feature engineering UI.
    """
    if df is None or df.empty:
        empty_summary = DataPreparationSummary(0, 0, 0, 0, 0, 0, 0, 0)
        return pd.DataFrame(), empty_summary, {}, pd.DataFrame(), []

    work = df.copy()
    if feature_cols is None:
        feature_cols = _infer_feature_columns(work, exclude=[target_col])
    feature_cols = [c for c in feature_cols if c in work.columns]

    initial_rows = int(len(work))
    initial_features = int(len(feature_cols))
    steps: List[str] = []

    stage1_info: Dict[str, Any]
    if apply_stage1:
        stage1_df, stage1_info = apply_missing_data_policy(
            work,
            feature_cols=feature_cols,
            missing_threshold=missing_threshold,
            target_col=target_col,
            time_col=time_col,
        )
        steps.append(
            "Stage 1 aplicado: se eliminaron variables con mas de "
            f"{missing_threshold * 100:.2f}% de missing y luego intervalos incompletos."
        )
    else:
        stage1_df = work.copy()
        stage1_info = {
            "removed_cols": [],
            "remaining_features": list(feature_cols),
            "rows_after_drop": int(len(stage1_df)),
            "missing_ratio": stage1_df[feature_cols].isna().mean()
            if feature_cols
            else pd.Series(dtype=float),
        }
        steps.append("Stage 1 omitido por configuracion.")

    zero_runs = pd.DataFrame(columns=["start", "end", "length_intervals", "length_days"])
    if apply_stage2:
        final_df, zero_runs = filter_long_zero_accident_runs(
            stage1_df,
            target_col=target_col,
            time_col=time_col,
            min_days=min_zero_days,
            interval_minutes=interval_minutes,
        )
        steps.append(
            "Stage 2 aplicado: se removieron ventanas multidiarias sin accidentes "
            f"(>= {int(min_zero_days)} dias)."
        )
    else:
        final_df = stage1_df.copy()
        steps.append("Stage 2 omitido por configuracion.")

    summary = DataPreparationSummary(
        initial_rows=initial_rows,
        initial_features=initial_features,
        removed_high_missing_features=int(len(stage1_info.get("removed_cols", []))),
        remaining_features_after_stage1=int(len(stage1_info.get("remaining_features", []))),
        rows_after_missing_drop=int(len(stage1_df)),
        removed_zero_run_rows=int(len(stage1_df) - len(final_df)),
        final_rows=int(len(final_df)),
        zero_run_windows_detected=int(len(zero_runs)),
    )
    return final_df, summary, stage1_info, zero_runs, steps


def _build_preparation_detail_table(
    summary: DataPreparationSummary,
    *,
    prep_config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    prep_config = prep_config or {}
    missing_threshold = prep_config.get("missing_threshold")
    min_zero_days = prep_config.get("min_zero_days")
    stage1_interval_drops = max(0, int(summary.initial_rows - summary.rows_after_missing_drop))

    if isinstance(missing_threshold, (int, float)):
        stage1_rule = f"Variables removidas por Stage 1 (> {float(missing_threshold) * 100:.2f}% missing)"
    else:
        stage1_rule = "Variables removidas por Stage 1 (missing sobre el umbral configurado)"

    if isinstance(min_zero_days, (int, float)):
        stage2_rule = (
            "Filas eliminadas por Stage 2 "
            f"(ventanas de >={int(min_zero_days)} dias consecutivos sin accidentes)"
        )
        stage2_windows_rule = (
            "Ventanas detectadas por Stage 2 "
            f"(>={int(min_zero_days)} dias consecutivos sin accidentes)"
        )
    else:
        stage2_rule = "Filas eliminadas por Stage 2 (ventanas largas sin accidentes)"
        stage2_windows_rule = "Ventanas detectadas por Stage 2 (ventanas largas sin accidentes)"

    rows = [
        {"stage": "Stage 1", "detalle": stage1_rule, "cantidad": int(summary.removed_high_missing_features)},
        {
            "stage": "Stage 1",
            "detalle": "Intervalos eliminados por Stage 1 (any missing value en predictors restantes)",
            "cantidad": stage1_interval_drops,
        },
        {"stage": "Stage 2", "detalle": stage2_rule, "cantidad": int(summary.removed_zero_run_rows)},
        {"stage": "Stage 2", "detalle": stage2_windows_rule, "cantidad": int(summary.zero_run_windows_detected)},
    ]
    return pd.DataFrame(rows)


def compute_abs_correlations(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.Series:
    """Returns absolute pairwise correlations (upper triangle) as a 1D series."""
    if df is None or df.empty:
        return pd.Series(dtype=float)

    cols = [c for c in feature_cols if c in df.columns]
    if len(cols) < 2:
        return pd.Series(dtype=float)

    corr = df[cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = upper.stack().astype(float)
    return pairs.sort_values(ascending=False)


def drop_highly_correlated_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    threshold: float = 0.95,
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    Drops features with |rho| > threshold using upper-triangle greedy strategy.
    """
    cols = [c for c in feature_cols if c in df.columns]
    if len(cols) < 2:
        return cols, [], pd.DataFrame()

    corr = df[cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    kept = [c for c in cols if c not in set(to_drop)]
    return kept, to_drop, corr


def rank_features_random_forest(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    target_col: str = "target",
    n_estimators: int = 300,
    random_state: int = 42,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Section 4.3.2 feature importance ranking.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["feature", "importance", "rank"])

    cols = [c for c in feature_cols if c in df.columns]
    if not cols or target_col not in df.columns:
        return pd.DataFrame(columns=["feature", "importance", "rank"])

    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)
    if y.nunique() < 2:
        return pd.DataFrame(columns=["feature", "importance", "rank"])

    X = df[cols].copy()
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        random_state=int(random_state),
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    model.fit(X, y)

    imp = pd.DataFrame(
        {
            "feature": cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    imp["rank"] = np.arange(1, len(imp) + 1)
    return imp.head(max(1, int(top_n))).reset_index(drop=True)


def youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """
    Returns threshold maximizing Youden J = sensitivity + specificity - 1.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)

    if np.unique(y).size < 2:
        return {
            "threshold": 0.5,
            "youden": 0.0,
            "sensitivity": float("nan"),
            "specificity": float("nan"),
        }

    fpr, tpr, thr = roc_curve(y, s)
    youden = tpr - fpr
    idx = int(np.nanargmax(youden))
    threshold = float(thr[idx])
    sensitivity = float(tpr[idx])
    specificity = float(1.0 - fpr[idx])
    return {
        "threshold": threshold,
        "youden": float(youden[idx]),
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    clipped = np.clip(arr, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_platt_calibrator(
    y_true: np.ndarray,
    raw_scores: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any], Optional[LogisticRegression]]:
    y = np.asarray(y_true).astype(int)
    scores = np.asarray(raw_scores, dtype=float)
    mask = np.isfinite(scores)
    metadata: Dict[str, Any] = {
        "method": DEFAULT_CALIBRATION_METHOD,
        "policy_version": THRESHOLD_CALIBRATION_POLICY_VERSION,
        "kind": "identity",
        "applied": False,
        "coef": 1.0,
        "intercept": 0.0,
    }
    if mask.sum() < 2 or np.unique(y[mask]).size < 2:
        calibrated = np.full(len(scores), np.nan, dtype=float)
        calibrated[mask] = np.clip(scores[mask], 0.0, 1.0)
        metadata["reason"] = "insufficient_classes_or_scores"
        return calibrated, metadata, None

    model = LogisticRegression(max_iter=1000)
    try:
        model.fit(scores[mask].reshape(-1, 1), y[mask])
    except Exception as exc:
        calibrated = np.full(len(scores), np.nan, dtype=float)
        calibrated[mask] = np.clip(scores[mask], 0.0, 1.0)
        metadata["reason"] = f"fit_failed:{exc}"
        return calibrated, metadata, None

    calibrated = np.full(len(scores), np.nan, dtype=float)
    calibrated[mask] = model.predict_proba(scores[mask].reshape(-1, 1))[:, 1]
    metadata.update(
        {
            "kind": "platt",
            "applied": True,
            "coef": float(np.asarray(model.coef_, dtype=float).reshape(-1)[0]),
            "intercept": float(np.asarray(model.intercept_, dtype=float).reshape(-1)[0]),
        }
    )
    return calibrated, metadata, model


def _apply_probability_calibration(
    raw_scores: np.ndarray,
    *,
    calibration_model: Optional[LogisticRegression] = None,
    calibration_metadata: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    scores = np.asarray(raw_scores, dtype=float)
    mask = np.isfinite(scores)
    calibrated = np.full(len(scores), np.nan, dtype=float)
    metadata = dict(calibration_metadata or {})
    kind = str(metadata.get("kind", "identity"))

    if calibration_model is not None and kind == "platt":
        calibrated[mask] = calibration_model.predict_proba(scores[mask].reshape(-1, 1))[:, 1]
        return calibrated

    if kind == "platt":
        coef = float(metadata.get("coef", 1.0))
        intercept = float(metadata.get("intercept", 0.0))
        calibrated[mask] = _sigmoid(coef * scores[mask] + intercept)
        return calibrated

    calibrated[mask] = np.clip(scores[mask], 0.0, 1.0)
    return calibrated


def _expected_cost_threshold(
    y_true: np.ndarray,
    calibrated_scores: np.ndarray,
    *,
    fn_cost: float = DEFAULT_THRESHOLD_FN_COST,
    fp_cost: float = DEFAULT_THRESHOLD_FP_COST,
) -> Dict[str, float]:
    y = np.asarray(y_true).astype(int)
    scores = np.asarray(calibrated_scores, dtype=float)
    mask = np.isfinite(scores)
    if mask.sum() < 2 or np.unique(y[mask]).size < 2:
        threshold = 0.5
        return {
            "threshold": float(threshold),
            "expected_cost": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "fn_cost": float(fn_cost),
            "fp_cost": float(fp_cost),
        }

    work = pd.DataFrame({"score": scores[mask], "y": y[mask]}).sort_values("score", ascending=False)
    grouped = (
        work.groupby("score", sort=False)["y"]
        .agg(total="size", positives="sum")
        .reset_index()
    )
    grouped["negatives"] = grouped["total"] - grouped["positives"]
    grouped["tp"] = grouped["positives"].cumsum()
    grouped["fp"] = grouped["negatives"].cumsum()
    total_pos = float(grouped["positives"].sum())
    total_neg = float(grouped["negatives"].sum())
    grouped["fn"] = total_pos - grouped["tp"]
    grouped["tn"] = total_neg - grouped["fp"]
    grouped["expected_cost"] = float(fn_cost) * grouped["fn"] + float(fp_cost) * grouped["fp"]
    grouped["sensitivity"] = np.where(total_pos > 0, grouped["tp"] / total_pos, np.nan)
    grouped["specificity"] = np.where(total_neg > 0, grouped["tn"] / total_neg, np.nan)

    grouped = grouped.sort_values(
        ["expected_cost", "sensitivity", "specificity", "score"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    best = grouped.iloc[0]
    return {
        "threshold": float(best["score"]),
        "expected_cost": float(best["expected_cost"]),
        "sensitivity": float(best["sensitivity"]),
        "specificity": float(best["specificity"]),
        "fn_cost": float(fn_cost),
        "fp_cost": float(fp_cost),
    }


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)
    if np.unique(y).size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def _safe_pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)
    if np.unique(y).size < 2:
        return float("nan")
    try:
        return float(average_precision_score(y, s))
    except Exception:
        return float("nan")


def compute_classification_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    decision_scores: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)
    decision = s if decision_scores is None else np.asarray(decision_scores).astype(float)
    preds = (decision >= float(threshold)).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    error_rate = float((fp + fn) / max(1, (tp + tn + fp + fn)))

    return {
        "auc": _safe_auc(y, s),
        "pr_auc": _safe_pr_auc(y, s),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "error_rate": error_rate,
        "threshold": float(threshold),
    }


def _compute_calibration_stage_metrics(
    y_true: np.ndarray,
    raw_scores: np.ndarray,
    *,
    raw_threshold: float,
    calibrated_scores: Optional[np.ndarray] = None,
    calibrated_threshold: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    y = np.asarray(y_true).astype(int)
    raw = np.asarray(raw_scores).astype(float)
    before_metrics = compute_classification_metrics(
        y,
        raw,
        threshold=float(raw_threshold),
    )
    calibrated = raw if calibrated_scores is None else np.asarray(calibrated_scores).astype(float)
    after_metrics = compute_classification_metrics(
        y,
        raw,
        threshold=float(raw_threshold if calibrated_threshold is None else calibrated_threshold),
        decision_scores=calibrated,
    )
    return {
        "before_calibration": before_metrics,
        "after_calibration": after_metrics,
    }


def _calibration_rate_fields(
    before_metrics: Dict[str, float],
    after_metrics: Dict[str, float],
) -> Dict[str, float]:
    return {
        "sensitivity_before_calibration": float(before_metrics["sensitivity"]),
        "specificity_before_calibration": float(before_metrics["specificity"]),
        "sensitivity_after_calibration": float(after_metrics["sensitivity"]),
        "specificity_after_calibration": float(after_metrics["specificity"]),
    }


MODEL_NAMES = ["XGBoost", "AdaBoost", "Random Forest", "NNet"]
YEARLY_STRATEGIES = ["static", "period_aligned", "cumulative"]
ADAPTIVE_ADWIN_STRATEGY = "adaptive_adwin"
ADAPTIVE_ARF_STRATEGY = "adaptive_arf"
ADAPTIVE_KSWIN_STRATEGY = "adaptive_kswin"
EXPERIMENT_STRATEGIES = YEARLY_STRATEGIES + [ADAPTIVE_ADWIN_STRATEGY, ADAPTIVE_ARF_STRATEGY, ADAPTIVE_KSWIN_STRATEGY]
ARF_VARIANT_NAMES = ["ARFmoderate", "ARFmaj", "ARFfast"]
ARF_DEFAULT_VARIANTS: Tuple[str, ...] = ("ARFmoderate", "ARFmaj")
KSWIN_VARIANT_NAMES = ["KSWINpaper", "KSWINseasonal"]
KSWIN_DEFAULT_VARIANTS: Tuple[str, ...] = ("KSWINpaper",)
DEFAULT_REPETITION_SEEDS: Tuple[int, ...] = (42, 52, 62)
BALANCE_MODE_NONE = "none"
BALANCE_MODE_SMOTE = "smote"
BALANCE_MODE_NOT_APPLICABLE = "not_applicable"
DEFAULT_BATCH_BALANCE_MODES: Tuple[str, ...] = (BALANCE_MODE_NONE, BALANCE_MODE_SMOTE)
EXPERIMENT_RESOURCE_MODE_RESTRICTED = "restricted"
EXPERIMENT_RESOURCE_MODE_FULL_MEMORY = "full_memory"
DEFAULT_EXPERIMENT_RESOURCE_MODE = EXPERIMENT_RESOURCE_MODE_RESTRICTED
EXPERIMENT_RESOURCE_POLICY_VERSION = "v1"
EXPERIMENT_RESOURCE_MODE_CONFIGS: Dict[str, Dict[str, Any]] = {
    EXPERIMENT_RESOURCE_MODE_RESTRICTED: {
        "label": "Modo restringido",
        "description": (
            "Replica la politica actual: Random Forest hasta 2 hilos, XGBoost hasta 4 hilos, "
            "CV paralela solo para AdaBoost y reutilizacion de checkpoints/caches separada por perfil."
        ),
        "estimator_n_jobs": {
            "Random Forest": 2,
            "XGBoost": 4,
        },
        "parallel_cv_models": ["AdaBoost"],
        "high_risk_thresholds": {
            "folds": 5,
            "trials": 20,
        },
    },
    EXPERIMENT_RESOURCE_MODE_FULL_MEMORY: {
        "label": "Modo full memoria",
        "description": (
            "Relaja los caps internos de CPU para aprovechar toda la memoria disponible: "
            "Random Forest y XGBoost pueden usar todos los hilos visibles. Mantiene "
            "checkpoints/caches separados para no mezclar corridas entre perfiles."
        ),
        "estimator_n_jobs": {
            "Random Forest": None,
            "XGBoost": None,
        },
        "parallel_cv_models": ["AdaBoost"],
        "high_risk_thresholds": {
            "folds": 3,
            "trials": 10,
        },
    },
}
DEFAULT_EXPERIMENT_PRESET = "Full Base"
EXPERIMENT_PRESET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Full Base": {
        "description": (
            "Replica la configuracion del ultimo experimento completo: cuatro modelos, "
            "todas las estrategias, `none` + `smote`, seed 42 y KSWINpaper."
        ),
        "resource_mode": DEFAULT_EXPERIMENT_RESOURCE_MODE,
        "models": list(MODEL_NAMES),
        "strategies": list(EXPERIMENT_STRATEGIES),
        "fast_mode": True,
        "optuna_trials": 30,
        "folds": 3,
        "validation_size": 0.2,
        "adwin_delta": 0.002,
        "min_window": 45000,
        "min_retrain_size": 4500,
        "arf_variants": ["ARFmoderate"],
        "kswin_variants": ["KSWINpaper"],
        "kswin_top_k_features": 3,
        "kswin_vote_threshold": 3,
        "kswin_retrain_days": 30,
        "kswin_min_retrain_rows": 50,
        "balance_modes": [BALANCE_MODE_NONE, BALANCE_MODE_SMOTE],
        "repetition_seeds": [42],
    },
    "KSWIN": {
        "description": (
            "Preset acotado para relanzar `adaptive_kswin` sin irse a dias: "
            "solo AdaBoost + Random Forest, balance `none`, seed 42, top-k=5, folds=2 y 8 trials."
        ),
        "resource_mode": DEFAULT_EXPERIMENT_RESOURCE_MODE,
        "models": ["AdaBoost", "Random Forest"],
        "strategies": [ADAPTIVE_KSWIN_STRATEGY],
        "fast_mode": True,
        "optuna_trials": 8,
        "folds": 2,
        "validation_size": 0.2,
        "adwin_delta": 0.002,
        "min_window": 45000,
        "min_retrain_size": 4500,
        "arf_variants": ["ARFmoderate"],
        "kswin_variants": ["KSWINpaper"],
        "kswin_top_k_features": 5,
        "kswin_vote_threshold": 2,
        "kswin_retrain_days": 90,
        "kswin_min_retrain_rows": 100,
        "balance_modes": [BALANCE_MODE_NONE],
        "repetition_seeds": [42],
    },
}

FAST_SMOTE_SEARCH_SPACE: Dict[str, Sequence[Any]] = {
    "sampling_strategy": [0.5, 1.0],
    "k_neighbors": [3, 5],
}

FULL_SMOTE_SEARCH_SPACE: Dict[str, Sequence[Any]] = {
    "sampling_strategy": [0.3, 0.5, 0.75, 1.0],
    "k_neighbors": [3, 5, 7],
}


def _normalize_experiment_resource_mode(value: Optional[str]) -> str:
    mode = str(value or DEFAULT_EXPERIMENT_RESOURCE_MODE).strip().lower()
    if mode not in EXPERIMENT_RESOURCE_MODE_CONFIGS:
        return DEFAULT_EXPERIMENT_RESOURCE_MODE
    return mode


def _experiment_resource_mode_label(resource_mode: Optional[str]) -> str:
    mode = _normalize_experiment_resource_mode(resource_mode)
    return str(EXPERIMENT_RESOURCE_MODE_CONFIGS[mode]["label"])


def _resolve_experiment_resource_policy(resource_mode: Optional[str]) -> Dict[str, Any]:
    mode = _normalize_experiment_resource_mode(resource_mode)
    config = dict(EXPERIMENT_RESOURCE_MODE_CONFIGS[mode])
    cpu_count = max(1, int(os.cpu_count() or 1))
    estimator_caps: Dict[str, int] = {}
    for model_name, cap in dict(config.get("estimator_n_jobs") or {}).items():
        resolved_cap = cpu_count if cap is None else int(cap)
        estimator_caps[str(model_name)] = max(1, min(int(resolved_cap), cpu_count))
    return {
        "mode": mode,
        "label": str(config.get("label", mode)),
        "description": str(config.get("description", "")).strip(),
        "cpu_count": int(cpu_count),
        "estimator_n_jobs": estimator_caps,
        "parallel_cv_models": [
            str(model_name)
            for model_name in (config.get("parallel_cv_models") or [])
        ],
        "high_risk_thresholds": {
            "folds": max(2, int((config.get("high_risk_thresholds") or {}).get("folds", 5))),
            "trials": max(1, int((config.get("high_risk_thresholds") or {}).get("trials", 20))),
        },
        "policy_version": EXPERIMENT_RESOURCE_POLICY_VERSION,
    }


def _resource_policy_signature(resource_mode: Optional[str]) -> Dict[str, Any]:
    policy = _resolve_experiment_resource_policy(resource_mode)
    return {
        "mode": str(policy["mode"]),
        "policy_version": str(policy["policy_version"]),
        "estimator_n_jobs": dict(policy.get("estimator_n_jobs") or {}),
        "parallel_cv_models": list(policy.get("parallel_cv_models") or []),
    }


def _apply_experiment_preset(preset_name: str, *, feature_count: int) -> None:
    preset = EXPERIMENT_PRESET_CONFIGS.get(preset_name) or EXPERIMENT_PRESET_CONFIGS[DEFAULT_EXPERIMENT_PRESET]
    bounded_feature_count = max(1, int(feature_count))
    bounded_vote_max = max(1, min(10, bounded_feature_count))

    st.session_state["drift_exp_resource_mode"] = _normalize_experiment_resource_mode(
        str(preset.get("resource_mode", DEFAULT_EXPERIMENT_RESOURCE_MODE))
    )
    st.session_state["drift_exp_models"] = list(preset["models"])
    st.session_state["drift_exp_strategies"] = list(preset["strategies"])
    st.session_state["drift_exp_fast_mode"] = bool(preset["fast_mode"])
    st.session_state["drift_exp_optuna_trials"] = int(preset["optuna_trials"])
    st.session_state["drift_exp_folds"] = int(preset["folds"])
    st.session_state["drift_exp_validation_size"] = float(preset["validation_size"])
    st.session_state["drift_exp_adwin_delta"] = float(preset["adwin_delta"])
    st.session_state["drift_exp_min_window"] = int(preset["min_window"])
    st.session_state["drift_exp_min_retrain_size"] = int(preset["min_retrain_size"])
    st.session_state["drift_exp_arf_variants"] = list(preset["arf_variants"])
    st.session_state["drift_exp_kswin_variants"] = list(preset["kswin_variants"])
    st.session_state["drift_exp_kswin_top_k_features"] = min(
        int(preset["kswin_top_k_features"]),
        bounded_feature_count,
    )
    st.session_state["drift_exp_kswin_vote_threshold"] = min(
        int(preset["kswin_vote_threshold"]),
        bounded_vote_max,
    )
    st.session_state["drift_exp_kswin_retrain_days"] = int(preset["kswin_retrain_days"])
    st.session_state["drift_exp_kswin_min_retrain_rows"] = int(preset["kswin_min_retrain_rows"])
    st.session_state["drift_exp_balance_modes"] = list(preset["balance_modes"])
    st.session_state["drift_exp_seed_text"] = ", ".join(
        str(seed) for seed in preset["repetition_seeds"]
    )


def _sync_experiment_control_bounds(feature_count: int) -> None:
    bounded_feature_count = max(1, int(feature_count))
    bounded_vote_max = max(1, min(10, bounded_feature_count))

    if "drift_exp_kswin_top_k_features" in st.session_state:
        st.session_state["drift_exp_kswin_top_k_features"] = min(
            max(1, int(st.session_state["drift_exp_kswin_top_k_features"])),
            bounded_feature_count,
        )
    if "drift_exp_kswin_vote_threshold" in st.session_state:
        st.session_state["drift_exp_kswin_vote_threshold"] = min(
            max(1, int(st.session_state["drift_exp_kswin_vote_threshold"])),
            bounded_vote_max,
        )


def _on_experiment_preset_change() -> None:
    preset_name = str(st.session_state.get("drift_experiment_preset", DEFAULT_EXPERIMENT_PRESET))
    feature_count = int(st.session_state.get("drift_experiment_feature_count", 1))
    _apply_experiment_preset(preset_name, feature_count=feature_count)
    st.session_state["drift_experiment_preset_applied"] = preset_name

ARF_VARIANT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ARFmoderate": {
        "warning_delta": 1e-4,
        "drift_delta": 1e-5,
        "disable_weighted_vote": False,
        "description": "Configuracion base del paper con ADWIN conservador y weighted vote.",
    },
    "ARFmaj": {
        "warning_delta": 1e-4,
        "drift_delta": 1e-5,
        "disable_weighted_vote": True,
        "description": "Misma configuracion moderate, pero con majority vote para reducir sesgo por accuracy.",
    },
    "ARFfast": {
        "warning_delta": 1e-2,
        "drift_delta": 1e-3,
        "disable_weighted_vote": False,
        "description": "Version mas reactiva con ADWIN agresivo para drift abrupto.",
    },
}

KSWIN_VARIANT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "KSWINpaper": {
        "alpha": 0.0001,
        "window_size": 300,
        "stat_size": 30,
        "transform": "raw",
        "description": "Configuracion del paper: KSWIN sobre variables crudas con n=300, r=30 y alpha=0.0001.",
    },
    "KSWINseasonal": {
        "alpha": 0.0001,
        "window_size": 300,
        "stat_size": 30,
        "transform": "seasonal_zscore",
        "description": "KSWIN sobre z-scores por estacionalidad base (hora y dia), para reducir falsos positivos de trafico.",
    },
}

FULL_HYPERPARAM_GRIDS: Dict[str, Dict[str, Sequence[Any]]] = {
    "Random Forest": {
        "mtry": [2, 4, 6],
        "splitrule": ["gini", "extratrees"],
        "min_node_size": [1, 5, 10],
    },
    "XGBoost": {
        "max_depth": [3, 6, 9],
        "eta": [0.01, 0.1, 0.3],
        "gamma": [0.0, 1.0, 5.0],
        "colsample_bytree": [0.5, 0.8, 1.0],
        "min_child_weight": [1.0, 5.0, 10.0],
        "subsample": [0.5, 0.7, 1.0],
        "nrounds": [100],
    },
    "AdaBoost": {
        "iter": [50, 100, 150],
        "maxdepth": [1, 2, 3],
        "nu": [0.1, 0.5, 1.0],
    },
    "NNet": {
        "size": [6, 7, 10, 12, 13, 14, 15, 16],
        "decay": [0.095, 0.1, 0.15, 0.2],
        "maxit": [300, 500, 800],
    },
}

FAST_HYPERPARAM_GRIDS: Dict[str, Dict[str, Sequence[Any]]] = {
    "Random Forest": {
        "mtry": [2, 6],
        "splitrule": ["gini", "extratrees"],
        "min_node_size": [1, 10],
    },
    "XGBoost": {
        "max_depth": [3, 6],
        "eta": [0.1],
        "gamma": [0.0, 1.0],
        "colsample_bytree": [0.8],
        "min_child_weight": [1.0, 5.0],
        "subsample": [0.7],
        "nrounds": [100],
    },
    "AdaBoost": {
        "iter": [50, 100],
        "maxdepth": [1, 2],
        "nu": [0.1, 0.5],
    },
    "NNet": {
        "size": [10, 14],
        "decay": [0.1, 0.2],
        "maxit": [300, 500],
    },
}


def parse_repetition_seeds(raw_value: str) -> List[int]:
    """
    Parses the repetition seed input used in the Streamlit UI.
    Empty input falls back to the article-aligned default seed list.
    """
    if raw_value is None or not str(raw_value).strip():
        return list(DEFAULT_REPETITION_SEEDS)

    seeds: List[int] = []
    seen = set()
    tokens = [tok for tok in re.split(r"[\s,;]+", str(raw_value).strip()) if tok]
    for token in tokens:
        seed = int(token)
        if seed < 0:
            raise ValueError("Seeds must be non-negative integers.")
        if seed not in seen:
            seeds.append(seed)
            seen.add(seed)

    if not seeds:
        return list(DEFAULT_REPETITION_SEEDS)
    return seeds


def _normalize_log_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json.dumps(np.asarray(value).tolist(), ensure_ascii=True)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str, ensure_ascii=True)
    if isinstance(value, (list, tuple, set)):
        return json.dumps(list(value), default=str, ensure_ascii=True)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _json_default(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def _to_json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default, ensure_ascii=True))


def _slugify_token(value: Any, *, max_len: int = 160) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    if not text:
        text = "item"
    return text[:max_len]


def _current_rss_mb() -> Optional[float]:
    if psutil is None:
        return None
    try:
        proc = psutil.Process(os.getpid())
        return float(proc.memory_info().rss / (1024.0 ** 2))
    except Exception:
        return None


def _cleanup_recalibration_memory() -> Optional[float]:
    gc.collect()
    return _current_rss_mb()


def _update_memory_summary(
    manifest: Optional[Dict[str, Any]],
    *,
    phase: str,
    rss_before_mb: Optional[float] = None,
    rss_after_mb: Optional[float] = None,
) -> None:
    if manifest is None:
        return
    summary = dict(manifest.get("memory_summary") or {})
    if rss_before_mb is not None:
        summary["rss_before_mb"] = float(rss_before_mb)
        peak = summary.get("rss_peak_mb")
        if peak is None or float(rss_before_mb) > float(peak):
            summary["rss_peak_mb"] = float(rss_before_mb)
    if rss_after_mb is not None:
        summary["rss_after_mb"] = float(rss_after_mb)
        peak = summary.get("rss_peak_mb")
        if peak is None or float(rss_after_mb) > float(peak):
            summary["rss_peak_mb"] = float(rss_after_mb)
    summary["last_phase"] = str(phase)
    summary["updated_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["memory_summary"] = summary


def _atomic_write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}_", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_to_json_safe(payload), handle, ensure_ascii=True, indent=2)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def _load_json_file(path: Path, *, default: Any = None) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        return default
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _hash_series_signature(series: pd.Series) -> bytes:
    hashed = pd.util.hash_pandas_object(series, index=False).to_numpy(dtype=np.uint64, copy=False)
    return hashed.tobytes()


def _frame_signature(
    df: pd.DataFrame,
    *,
    columns: Sequence[str],
    include_index: bool = True,
) -> str:
    if df is None or df.empty:
        return "empty"
    sha = hashlib.sha256()
    work = df.copy()
    if include_index:
        sha.update(np.asarray(work.index.to_numpy(), dtype=np.int64).tobytes())
    for col in columns:
        if col not in work.columns:
            sha.update(f"missing::{col}".encode("utf-8"))
            continue
        series = work[col]
        if isinstance(series.dtype, pd.DatetimeTZDtype) or pd.api.types.is_datetime64_any_dtype(series):
            normalized = pd.to_datetime(series, errors="coerce").astype("string").fillna("<NA>")
        else:
            normalized = series.astype("string").fillna("<NA>")
        sha.update(str(col).encode("utf-8"))
        sha.update(_hash_series_signature(normalized))
    sha.update(str(len(work)).encode("utf-8"))
    return sha.hexdigest()


def _model_execution_priority(model_name: str) -> int:
    priority = {
        "NNet": 0,
        "AdaBoost": 1,
        "Random Forest": 2,
        "XGBoost": 3,
    }
    return int(priority.get(str(model_name), 99))


def _feature_export_signature(feature_selection_context: Dict[str, Any]) -> str:
    export_path = feature_selection_context.get("feature_export_path")
    if export_path:
        return str(export_path)
    return ",".join(str(col) for col in feature_selection_context.get("selected_features", []))


def _portable_path_name(raw_path: Any) -> str:
    raw = str(raw_path or "").strip()
    if not raw:
        return ""
    if "\\" in raw:
        win_name = str(PureWindowsPath(raw).name).strip()
        if win_name:
            return win_name
    posix_name = str(PurePosixPath(raw).name).strip()
    if posix_name:
        return posix_name
    default_name = str(Path(raw).name).strip()
    return default_name or raw


def _resolve_existing_feature_export_path(raw_path: Any) -> Optional[Path]:
    raw = str(raw_path or "").strip()
    if not raw:
        return None

    candidates: List[Path] = []
    literal_path = Path(raw).expanduser()
    candidates.append(literal_path)
    if not literal_path.is_absolute():
        candidates.append((ROOT_DIR / literal_path).expanduser())
        candidates.append((RESULTS_DIR / literal_path).expanduser())

    portable_name = _portable_path_name(raw)
    if portable_name:
        candidates.extend(
            [
                (RESULTS_DIR / portable_name).expanduser(),
                (ROOT_DIR / portable_name).expanduser(),
                Path.cwd() / portable_name,
            ]
        )

    seen: set[str] = set()
    for candidate in candidates:
        candidate_key = str(candidate)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        if candidate.exists():
            return candidate.resolve()
    return None


def _build_recalibration_run_id(
    *,
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    time_col: str,
    model_names: Sequence[str],
    strategies: Sequence[str],
    arf_variants: Sequence[str],
    kswin_variants: Sequence[str],
    balance_modes: Sequence[str],
    repetition_seeds: Sequence[int],
    base_year: Optional[int],
    validation_size: float,
    folds: int,
    random_state: int,
    fast_mode: bool,
    resource_mode: str,
    grid_limit: Optional[int],
    adwin_delta: float,
    min_window: int,
    min_retrain_size: Optional[int],
    kswin_top_k_features: int,
    kswin_vote_threshold: int,
    kswin_retrain_days: int,
    kswin_min_retrain_rows: int,
    continue_on_block_error: bool,
    custom_grids: Optional[Dict[str, Dict[str, Sequence[Any]]]],
    feature_selection_context: Dict[str, Any],
) -> str:
    signature_cols = [col for col in [time_col, target_col] + list(feature_cols) if col in df.columns]
    dataset_signature = _frame_signature(df, columns=signature_cols, include_index=True)
    payload = {
        "dataset_signature": dataset_signature,
        "feature_export_signature": _feature_export_signature(feature_selection_context),
        "recalibration_policy": _recalibration_policy_signature(),
        "feature_cols": list(feature_cols),
        "target_col": str(target_col),
        "time_col": str(time_col),
        "models": list(model_names),
        "strategies": list(strategies),
        "arf_variants": list(arf_variants),
        "kswin_variants": list(kswin_variants),
        "balance_modes": list(balance_modes),
        "repetition_seeds": [int(seed) for seed in repetition_seeds],
        "base_year": None if base_year is None else int(base_year),
        "validation_size": float(validation_size),
        "folds": int(folds),
        "random_state": int(random_state),
        "fast_mode": bool(fast_mode),
        "grid_limit": None if grid_limit is None else int(grid_limit),
        "adwin_delta": float(adwin_delta),
        "min_window": int(min_window),
        "min_retrain_size": None if min_retrain_size is None else int(min_retrain_size),
        "kswin_top_k_features": int(kswin_top_k_features),
        "kswin_vote_threshold": int(kswin_vote_threshold),
        "kswin_retrain_days": int(kswin_retrain_days),
        "kswin_min_retrain_rows": int(kswin_min_retrain_rows),
        "continue_on_block_error": bool(continue_on_block_error),
        "custom_grids": _to_json_safe(custom_grids or {}),
    }
    if _normalize_experiment_resource_mode(resource_mode) != DEFAULT_EXPERIMENT_RESOURCE_MODE:
        payload["resource_policy"] = _resource_policy_signature(resource_mode)
    if "NNet" in [str(name) for name in model_names]:
        payload["nnet_training_policy_version"] = NNET_TRAINING_POLICY_VERSION
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    return f"run_{digest[:16]}"


def _build_recalibration_block_id(block: Dict[str, Any], *, run_seed: int, run_order: int) -> str:
    payload = {
        "run_seed": int(run_seed),
        "run_order": int(run_order),
        "strategy": str(block.get("strategy", "")),
        "model": str(block.get("model", "")),
        "detector_variant": str(block.get("detector_variant", "")),
        "balance_mode": str(block.get("balance_mode", BALANCE_MODE_NOT_APPLICABLE)),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    return f"block_{digest[:16]}"


def _build_global_tuning_key(
    *,
    model_name: str,
    balance_mode: str,
    feature_cols: Sequence[str],
    target_col: str,
    time_col: str,
    canonical_train_df: pd.DataFrame,
    validation_size: float,
    folds: int,
    random_state: int,
    fast_mode: bool,
    resource_mode: str,
    grid_limit: Optional[int],
    custom_grid: Optional[Dict[str, Sequence[Any]]],
) -> str:
    signature_cols = [col for col in [time_col, target_col] + list(feature_cols) if col in canonical_train_df.columns]
    payload = {
        "model_name": str(model_name),
        "balance_mode": str(balance_mode),
        "recalibration_policy": _recalibration_policy_signature(),
        "feature_cols": list(feature_cols),
        "target_col": str(target_col),
        "time_col": str(time_col),
        "train_signature": _frame_signature(canonical_train_df, columns=signature_cols, include_index=True),
        "validation_size": float(validation_size),
        "folds": int(folds),
        "random_state": int(random_state),
        "fast_mode": bool(fast_mode),
        "grid_limit": None if grid_limit is None else int(grid_limit),
        "custom_grid": _to_json_safe(custom_grid or {}),
    }
    if _normalize_experiment_resource_mode(resource_mode) != DEFAULT_EXPERIMENT_RESOURCE_MODE:
        payload["resource_policy"] = _resource_policy_signature(resource_mode)
    policy_version = _model_policy_version(model_name)
    if policy_version is not None:
        payload["model_policy_version"] = str(policy_version)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    return f"tuning_{digest[:16]}"


def _build_smote_key(
    *,
    run_id: str,
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    time_col: Optional[str],
    sampling_strategy: float,
    k_neighbors: int,
) -> str:
    signature_cols = [str(target_col)] + list(feature_cols)
    if time_col and time_col in train_df.columns:
        signature_cols = [str(time_col)] + signature_cols
    payload = {
        "run_id": str(run_id),
        "train_signature": _frame_signature(train_df, columns=signature_cols, include_index=True),
        "feature_cols": list(feature_cols),
        "target_col": str(target_col),
        "time_col": None if time_col is None else str(time_col),
        "sampling_strategy": float(sampling_strategy),
        "k_neighbors": int(k_neighbors),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    return f"smote_{digest[:16]}"


def _recalibration_run_dir(run_id: str, *, checkpoint_root: Optional[Path] = None) -> Path:
    root = _recalibration_checkpoint_root(checkpoint_root=checkpoint_root)
    return root / str(run_id)


def _recalibration_run_paths(run_dir: Path) -> Dict[str, Path]:
    return {
        "run_dir": run_dir,
        "manifest": run_dir / "manifest.json",
        "blocks_dir": run_dir / "blocks",
        "tuning_dir": run_dir / "tuning",
        "smote_dir": run_dir / "smote",
        "live_status": run_dir / "live_status.json",
        "live_events": run_dir / "live_events.jsonl",
    }


def _ensure_recalibration_run_dirs(paths: Dict[str, Path]) -> None:
    for key in ["run_dir", "blocks_dir", "tuning_dir", "smote_dir"]:
        Path(paths[key]).mkdir(parents=True, exist_ok=True)


def _recalibration_checkpoint_root(*, checkpoint_root: Optional[Path] = None) -> Path:
    return Path(checkpoint_root) if checkpoint_root is not None else RESULTS_DIR / "drift_recalibration_runs"


def _persist_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _atomic_write_json(path, manifest)


def _persist_recalibration_live_status(path: Path, payload: Dict[str, Any]) -> None:
    _atomic_write_json(path, payload)


def _append_recalibration_live_event(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_to_json_safe(payload), ensure_ascii=True))
        handle.write("\n")


def _reset_recalibration_live_artifacts(paths: Dict[str, Path]) -> None:
    for key in ["live_status", "live_events"]:
        live_path = Path(paths[key])
        if not live_path.exists():
            continue
        try:
            live_path.unlink()
        except Exception:
            pass


def _load_manifest(path: Path) -> Optional[Dict[str, Any]]:
    payload = _load_json_file(path, default=None)
    if isinstance(payload, dict):
        return payload
    return None


def _normalized_checkpoint_archive_member(name: str) -> Path:
    raw_path = Path(str(name).strip())
    if raw_path.is_absolute():
        raise ValueError("Checkpoint archive contains absolute paths.")
    parts = [part for part in raw_path.parts if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        raise ValueError("Checkpoint archive contains unsafe relative paths.")
    return Path(*parts)


def _build_recalibration_checkpoint_archive(run_dir: Path) -> bytes:
    run_path = Path(run_dir)
    paths = _recalibration_run_paths(run_path)
    manifest = _load_manifest(paths["manifest"])
    if manifest is None:
        raise FileNotFoundError(f"Checkpoint manifest not found at {paths['manifest']}")

    zip_buffer = io.BytesIO()
    added_files = 0
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(run_path.rglob("*")):
            if not file_path.is_file():
                continue
            archive_name = Path(run_path.name) / file_path.relative_to(run_path)
            archive.write(file_path, arcname=str(archive_name))
            added_files += 1
    if added_files == 0:
        raise ValueError("Checkpoint run directory has no files to export.")
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def _import_recalibration_checkpoint_archive(
    archive_bytes: bytes,
    *,
    checkpoint_root: Optional[Path] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    checkpoint_root_path = _recalibration_checkpoint_root(checkpoint_root=checkpoint_root)
    checkpoint_root_path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
        members: List[Tuple[zipfile.ZipInfo, Path]] = []
        for info in archive.infolist():
            if info.is_dir():
                continue
            normalized_member = _normalized_checkpoint_archive_member(info.filename)
            if normalized_member.parts[0] == "__MACOSX":
                continue
            members.append((info, normalized_member))

        if not members:
            raise ValueError("Checkpoint archive is empty.")

        top_level_dirs = {member.parts[0] for _, member in members}
        if len(top_level_dirs) != 1:
            raise ValueError("Checkpoint archive must contain a single run directory.")

        run_id = str(next(iter(top_level_dirs)))
        manifest_member = Path(run_id) / "manifest.json"
        if manifest_member not in {member for _, member in members}:
            raise ValueError("Checkpoint archive does not contain run manifest.json.")

        destination_run_dir = checkpoint_root_path / run_id
        if destination_run_dir.exists():
            if not overwrite:
                raise FileExistsError(f"Checkpoint {run_id} already exists in {checkpoint_root_path}")
            shutil.rmtree(destination_run_dir)

        with tempfile.TemporaryDirectory(prefix="drift_ckpt_import_", dir=str(checkpoint_root_path)) as tmp_dir:
            tmp_root = Path(tmp_dir)
            for info, member in members:
                target_path = tmp_root / member
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info, "r") as source_handle, target_path.open("wb") as target_handle:
                    shutil.copyfileobj(source_handle, target_handle)

            imported_run_dir = tmp_root / run_id
            imported_paths = _recalibration_run_paths(imported_run_dir)
            manifest = _load_manifest(imported_paths["manifest"])
            if manifest is None:
                raise ValueError("Imported checkpoint manifest is invalid.")
            shutil.move(str(imported_run_dir), str(destination_run_dir))

    final_paths = _recalibration_run_paths(destination_run_dir)
    manifest = _load_manifest(final_paths["manifest"])
    if manifest is not None:
        manifest = _reconcile_recalibration_manifest(manifest, paths=final_paths)
        _persist_manifest(final_paths["manifest"], manifest)
    return {
        "run_id": str(run_id),
        "run_dir": str(destination_run_dir),
        "manifest_path": str(final_paths["manifest"]),
        "status": str((manifest or {}).get("status", "missing")),
    }


def _reset_recalibration_blocks_for_rerun(
    manifest: Dict[str, Any],
    *,
    paths: Dict[str, Path],
    checkpoint_status: str,
) -> Dict[str, Any]:
    for block_path in sorted(Path(paths["blocks_dir"]).glob("*.json")):
        try:
            block_path.unlink()
        except Exception:
            continue

    progress = dict(manifest.get("progress") or {})
    completed_tuning_tasks = int(
        sum(1 for item in (manifest.get("tuning_index") or {}).values() if str(item.get("status")) == "completed")
    )
    progress["completed_units"] = float(completed_tuning_tasks)
    progress["completed_blocks"] = 0
    progress["skipped_failed_blocks"] = 0
    progress["total_tuning_tasks"] = max(
        int(progress.get("total_tuning_tasks", 0)),
        int(len(manifest.get("tuning_index") or {})),
    )

    manifest["status"] = "running"
    manifest["completed_block_ids"] = []
    manifest["skipped_failed_block_ids"] = []
    manifest["pending_block_ids"] = []
    manifest["failed_block_id"] = None
    manifest["last_error"] = None
    manifest["nonfatal_block_errors"] = []
    manifest["block_index"] = {}
    manifest["progress"] = progress
    manifest["global_execution_log"] = []
    manifest["memory_summary"] = {}
    manifest["restarted_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["resume"] = {
        "auto_resumed": True,
        "checkpoint_status": str(checkpoint_status),
        "checkpoint_mode": "recompute_blocks",
    }
    return manifest


def _preview_recalibration_checkpoint(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    target_col: str = "target",
    time_col: str = "interval_start",
    model_names: Optional[Sequence[str]] = None,
    strategies: Optional[Sequence[str]] = None,
    base_year: Optional[int] = None,
    validation_size: float = 0.2,
    folds: int = 3,
    random_state: int = 42,
    fast_mode: bool = True,
    resource_mode: str = DEFAULT_EXPERIMENT_RESOURCE_MODE,
    grid_limit: Optional[int] = 30,
    adwin_delta: float = 0.002,
    min_window: int = 45_000,
    min_retrain_size: Optional[int] = None,
    arf_variants: Optional[Sequence[str]] = None,
    kswin_variants: Optional[Sequence[str]] = None,
    kswin_top_k_features: int = 10,
    kswin_vote_threshold: int = 2,
    kswin_retrain_days: int = 90,
    kswin_min_retrain_rows: int = 100,
    continue_on_block_error: bool = False,
    custom_grids: Optional[Dict[str, Dict[str, Sequence[Any]]]] = None,
    repetition_seeds: Optional[Sequence[int]] = None,
    balance_modes: Optional[Sequence[str]] = None,
    feature_selection_context: Optional[Dict[str, Any]] = None,
    checkpoint_root: Optional[Path] = None,
) -> Dict[str, Any]:
    if model_names is None:
        model_names = MODEL_NAMES
    if strategies is None:
        strategies = EXPERIMENT_STRATEGIES
    if arf_variants is None:
        arf_variants = ARF_DEFAULT_VARIANTS
    if kswin_variants is None:
        kswin_variants = KSWIN_DEFAULT_VARIANTS
    if repetition_seeds is None:
        repetition_seeds = DEFAULT_REPETITION_SEEDS
    if balance_modes is None:
        balance_modes = DEFAULT_BATCH_BALANCE_MODES

    normalized_feature_selection_context = _normalize_feature_selection_context(
        feature_selection_context,
        feature_cols=feature_cols,
    )
    effective_base_year = _canonical_base_year(df, time_col=time_col, base_year=base_year)
    run_id = _build_recalibration_run_id(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        time_col=time_col,
        model_names=list(model_names),
        strategies=list(strategies),
        arf_variants=list(arf_variants),
        kswin_variants=list(kswin_variants),
        balance_modes=list(balance_modes),
        repetition_seeds=list(repetition_seeds),
        base_year=effective_base_year,
        validation_size=float(validation_size),
        folds=int(folds),
        random_state=int(random_state),
        fast_mode=bool(fast_mode),
        resource_mode=_normalize_experiment_resource_mode(resource_mode),
        grid_limit=grid_limit,
        adwin_delta=float(adwin_delta),
        min_window=int(min_window),
        min_retrain_size=min_retrain_size,
        kswin_top_k_features=int(kswin_top_k_features),
        kswin_vote_threshold=int(kswin_vote_threshold),
        kswin_retrain_days=int(kswin_retrain_days),
        kswin_min_retrain_rows=int(kswin_min_retrain_rows),
        continue_on_block_error=bool(continue_on_block_error),
        custom_grids=custom_grids,
        feature_selection_context=normalized_feature_selection_context,
    )
    run_dir = _recalibration_run_dir(run_id, checkpoint_root=checkpoint_root)
    paths = _recalibration_run_paths(run_dir)
    manifest = _load_manifest(paths["manifest"])
    if manifest is not None:
        manifest = _reconcile_recalibration_manifest(manifest, paths=paths)
    progress = dict((manifest or {}).get("progress") or {})
    status = str((manifest or {}).get("status", "missing"))
    checkpoint_available = manifest is not None
    return {
        "run_id": str(run_id),
        "run_dir": str(run_dir),
        "manifest_path": str(paths["manifest"]),
        "status": status,
        "checkpoint_available": bool(checkpoint_available),
        "can_resume": bool(checkpoint_available and status != "completed"),
        "can_load_completed": bool(checkpoint_available and status == "completed"),
        "can_recompute_trainings": bool(checkpoint_available and int(progress.get("completed_tuning_tasks", 0)) > 0),
        "updated_at": None if manifest is None else manifest.get("updated_at"),
        "completed_tuning_tasks": int(progress.get("completed_tuning_tasks", 0)),
        "total_tuning_tasks": int(progress.get("total_tuning_tasks", 0)),
        "completed_blocks": int(progress.get("completed_blocks", 0)),
        "skipped_failed_blocks": int(progress.get("skipped_failed_blocks", 0)),
        "total_blocks": int(progress.get("total_blocks", 0)),
        "smote_artifacts": len((manifest or {}).get("smote_index") or {}),
        "manifest": manifest,
        "feature_selection_context": normalized_feature_selection_context,
        "effective_base_year": effective_base_year,
    }


def _preview_recalibration_checkpoint_run(
    run_id: str,
    *,
    checkpoint_root: Optional[Path] = None,
) -> Dict[str, Any]:
    resolved_run_id = str(run_id).strip()
    if not resolved_run_id:
        return {
            "run_id": "",
            "run_dir": str(_recalibration_run_dir("", checkpoint_root=checkpoint_root)),
            "manifest_path": str(_recalibration_run_paths(_recalibration_run_dir("", checkpoint_root=checkpoint_root))["manifest"]),
            "status": "missing",
            "checkpoint_available": False,
            "can_resume": False,
            "can_load_completed": False,
            "can_recompute_trainings": False,
            "updated_at": None,
            "completed_tuning_tasks": 0,
            "total_tuning_tasks": 0,
            "completed_blocks": 0,
            "skipped_failed_blocks": 0,
            "total_blocks": 0,
            "smote_artifacts": 0,
            "manifest": None,
        }

    run_dir = _recalibration_run_dir(resolved_run_id, checkpoint_root=checkpoint_root)
    paths = _recalibration_run_paths(run_dir)
    manifest = _load_manifest(paths["manifest"])
    if manifest is not None:
        manifest = _reconcile_recalibration_manifest(manifest, paths=paths)
    progress = dict((manifest or {}).get("progress") or {})
    status = str((manifest or {}).get("status", "missing"))
    checkpoint_available = manifest is not None
    return {
        "run_id": resolved_run_id,
        "run_dir": str(run_dir),
        "manifest_path": str(paths["manifest"]),
        "status": status,
        "checkpoint_available": bool(checkpoint_available),
        "can_resume": bool(checkpoint_available and status != "completed"),
        "can_load_completed": bool(checkpoint_available and status == "completed"),
        "can_recompute_trainings": bool(checkpoint_available and int(progress.get("completed_tuning_tasks", 0)) > 0),
        "updated_at": None if manifest is None else manifest.get("updated_at"),
        "completed_tuning_tasks": int(progress.get("completed_tuning_tasks", 0)),
        "total_tuning_tasks": int(progress.get("total_tuning_tasks", 0)),
        "completed_blocks": int(progress.get("completed_blocks", 0)),
        "skipped_failed_blocks": int(progress.get("skipped_failed_blocks", 0)),
        "total_blocks": int(progress.get("total_blocks", 0)),
        "smote_artifacts": len((manifest or {}).get("smote_index") or {}),
        "manifest": manifest,
        "run_manifest": dict((manifest or {}).get("run_manifest") or {}),
        "feature_selection_context": dict((manifest or {}).get("feature_selection_context") or {}),
    }


def _persist_recalibration_block(path: Path, payload: Dict[str, Any]) -> None:
    _atomic_write_json(path, payload)


def _load_recalibration_block(path: Path) -> Optional[Dict[str, Any]]:
    payload = _load_json_file(path, default=None)
    if isinstance(payload, dict):
        return payload
    return None


def _persist_tuning_artifact(path: Path, payload: Dict[str, Any]) -> None:
    _atomic_write_json(path, payload)


def _load_tuning_artifact(path: Path) -> Optional[Dict[str, Any]]:
    payload = _load_json_file(path, default=None)
    if isinstance(payload, dict):
        return payload
    return None


def _write_smote_payload_to_duckdb(
    path: Path,
    *,
    augmented_df: pd.DataFrame,
    metadata: Dict[str, Any],
) -> None:
    if duckdb is None:
        raise ImportError("duckdb is not installed.")
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(f".{target_path.stem}_{time.time_ns()}.tmp")
    con = duckdb.connect(str(temp_path))
    try:
        con.register("augmented_view", augmented_df)
        con.execute("CREATE OR REPLACE TABLE augmented_train AS SELECT * FROM augmented_view")
        meta_df = pd.DataFrame(
            {
                "metadata_json": [
                    json.dumps(_to_json_safe(metadata), ensure_ascii=True, sort_keys=True)
                ]
            }
        )
        con.register("metadata_view", meta_df)
        con.execute("CREATE OR REPLACE TABLE metadata AS SELECT * FROM metadata_view")
    finally:
        con.close()
    os.replace(temp_path, target_path)


def _load_smote_payload_from_duckdb(path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if duckdb is None:
        raise ImportError("duckdb is not installed.")
    con = duckdb.connect(str(path), read_only=True)
    try:
        augmented_df = con.execute("SELECT * FROM augmented_train").df()
        metadata_json = con.execute("SELECT metadata_json FROM metadata LIMIT 1").fetchone()
    finally:
        con.close()
    metadata = {}
    if metadata_json and metadata_json[0]:
        try:
            metadata = json.loads(str(metadata_json[0]))
        except Exception:
            metadata = {"raw_metadata": str(metadata_json[0])}
    return augmented_df, metadata


def _tuning_search_df_from_artifact(
    artifact: Optional[Dict[str, Any]],
    *,
    model_name: str,
    balance_mode: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    trials = list((artifact or {}).get("trials") or [])
    for row in trials:
        rows.append(
            {
                "trial_number": int(row.get("trial_number", len(rows))),
                "model": str(model_name),
                "cv_auc": row.get("cv_auc"),
                "params": json.dumps(row.get("params") or {}, sort_keys=True, ensure_ascii=True),
                "state": str(row.get("state", "")),
                "balance_mode": str(balance_mode),
                "converged": bool(row.get("converged", True)),
                "n_non_converged_folds": int(row.get("n_non_converged_folds", 0)),
                "n_valid_folds": int(row.get("n_valid_folds", 0)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["trial_number", "model", "cv_auc", "params", "state", "balance_mode", "converged", "n_non_converged_folds", "n_valid_folds"])
    return pd.DataFrame(rows).sort_values(["converged", "cv_auc"], ascending=[False, False], na_position="last").reset_index(drop=True)


def _serialize_block_payload(
    *,
    block_id: str,
    block: Dict[str, Any],
    run_seed: int,
    run_order: int,
    yearly_rows: pd.DataFrame,
    adaptive_rows: pd.DataFrame,
    roc_payload: Sequence[Dict[str, Any]],
    execution_log: Sequence[Dict[str, Any]],
    tuning_refs: Sequence[str],
    smote_refs: Sequence[str],
) -> Dict[str, Any]:
    return {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "block_id": str(block_id),
        "block": _to_json_safe(block),
        "run_seed": int(run_seed),
        "run_order": int(run_order),
        "yearly_rows": [] if yearly_rows is None or yearly_rows.empty else yearly_rows.to_dict(orient="records"),
        "adaptive_rows": [] if adaptive_rows is None or adaptive_rows.empty else adaptive_rows.to_dict(orient="records"),
        "roc_payload": _to_json_safe(list(roc_payload)),
        "execution_log": _to_json_safe(list(execution_log)),
        "tuning_refs": [str(ref) for ref in tuning_refs],
        "smote_refs": [str(ref) for ref in smote_refs],
    }


def _load_completed_block_payloads(manifest: Dict[str, Any], *, blocks_dir: Path) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for block_id in manifest.get("completed_block_ids", []):
        block_info = (manifest.get("block_index") or {}).get(str(block_id), {})
        block_path = blocks_dir / str(block_info.get("filename") or f"{block_id}.json")
        payload = _load_recalibration_block(block_path)
        if payload:
            payloads.append(payload)
    strategy_priority = {
        "static": 0,
        "period_aligned": 1,
        "cumulative": 2,
        ADAPTIVE_ADWIN_STRATEGY: 3,
        ADAPTIVE_ARF_STRATEGY: 4,
        ADAPTIVE_KSWIN_STRATEGY: 5,
    }
    payloads.sort(
        key=lambda payload: (
            int(payload.get("run_order", 1)),
            int(payload.get("run_seed", 0)),
            int(strategy_priority.get(str((payload.get("block") or {}).get("strategy", "")), 99)),
            0 if str((payload.get("block") or {}).get("balance_mode", BALANCE_MODE_NOT_APPLICABLE)) != BALANCE_MODE_SMOTE else 1,
            _model_execution_priority(str((payload.get("block") or {}).get("model", ""))),
            str((payload.get("block") or {}).get("model", "")),
            str((payload.get("block") or {}).get("detector_variant", "")),
        )
    )
    return payloads


def _reconcile_recalibration_manifest(manifest: Dict[str, Any], *, paths: Dict[str, Path]) -> Dict[str, Any]:
    completed_ids = {str(item) for item in manifest.get("completed_block_ids", [])}
    block_index = dict(manifest.get("block_index") or {})
    for block_path in sorted(paths["blocks_dir"].glob("*.json")):
        payload = _load_recalibration_block(block_path)
        if not payload:
            continue
        block_id = str(payload.get("block_id") or block_path.stem)
        block_index[block_id] = {
            **dict(block_index.get(block_id) or {}),
            "filename": block_path.name,
            "status": "completed",
        }
        completed_ids.add(block_id)
    tuning_index = dict(manifest.get("tuning_index") or {})
    for tuning_path in sorted(paths["tuning_dir"].glob("*.json")):
        payload = _load_tuning_artifact(tuning_path)
        if not payload:
            continue
        tuning_key = str(payload.get("tuning_key") or tuning_path.stem)
        tuning_index[tuning_key] = {
            **dict(tuning_index.get(tuning_key) or {}),
            "filename": tuning_path.name,
            "status": "completed",
            "study_id": payload.get("study_id"),
        }
    smote_index = dict(manifest.get("smote_index") or {})
    for smote_path in sorted(paths["smote_dir"].glob("*.duckdb")):
        if smote_path.name in {".DS_Store"}:
            continue
        smote_key = smote_path.stem
        smote_index[smote_key] = {
            **dict(smote_index.get(smote_key) or {}),
            "filename": smote_path.name,
            "status": "available",
        }
    manifest["block_index"] = block_index
    manifest["tuning_index"] = tuning_index
    manifest["smote_index"] = smote_index
    manifest["completed_block_ids"] = sorted(completed_ids)
    manifest["skipped_failed_block_ids"] = sorted({str(item) for item in manifest.get("skipped_failed_block_ids", [])})
    manifest["nonfatal_block_errors"] = list(manifest.get("nonfatal_block_errors") or [])
    return manifest


def _assemble_recalibration_outputs_from_checkpoints(
    manifest: Dict[str, Any],
    *,
    paths: Dict[str, Path],
    feature_selection_context: Dict[str, Any],
) -> Dict[str, Any]:
    block_payloads = _load_completed_block_payloads(manifest, blocks_dir=paths["blocks_dir"])
    yearly_frames: List[pd.DataFrame] = []
    adaptive_frames: List[pd.DataFrame] = []
    roc_payload: List[Dict[str, Any]] = []
    execution_log_rows: List[Dict[str, Any]] = list(manifest.get("global_execution_log") or [])
    for payload in block_payloads:
        yearly_rows = payload.get("yearly_rows") or []
        adaptive_rows = payload.get("adaptive_rows") or []
        if yearly_rows:
            yearly_frames.append(pd.DataFrame(yearly_rows))
        if adaptive_rows:
            adaptive_frames.append(pd.DataFrame(adaptive_rows))
        roc_payload.extend(list(payload.get("roc_payload") or []))
        execution_log_rows.extend(list(payload.get("execution_log") or []))

    yearly_results = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    adaptive_results = pd.concat(adaptive_frames, ignore_index=True) if adaptive_frames else pd.DataFrame()
    roc_curves = build_average_roc_curves(roc_payload)
    summary = summarize_results(yearly_results, adaptive_results)
    appendix = format_appendix_tables(yearly_results, adaptive_results)
    appendix_mean = format_appendix_tables_mean(yearly_results, adaptive_results)

    studies: List[Dict[str, Any]] = []
    for tuning_info in (manifest.get("tuning_index") or {}).values():
        tuning_path = paths["tuning_dir"] / str(tuning_info.get("filename", ""))
        payload = _load_tuning_artifact(tuning_path)
        if payload:
            studies.append(payload)

    for idx, row in enumerate(execution_log_rows, start=1):
        row["order"] = idx
    execution_log_df = pd.DataFrame(execution_log_rows)

    return {
        "yearly_results": yearly_results,
        "adaptive_results": adaptive_results,
        "roc_payload": roc_payload,
        "average_roc": roc_curves,
        "summary": summary,
        "appendix_tables": appendix,
        "appendix_tables_mean": appendix_mean,
        "execution_log": execution_log_df,
        "run_manifest": dict(manifest.get("run_manifest") or {}),
        "checkpoint_manifest": manifest,
        "studies": studies,
        "feature_selection_context": feature_selection_context,
    }


def _estimate_recalibration_workload(
    *,
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    time_col: str,
    model_names: Sequence[str],
    strategies: Sequence[str],
    arf_variants: Sequence[str],
    kswin_variants: Sequence[str],
    balance_modes: Sequence[str],
    repetition_seeds: Sequence[int],
    base_year: Optional[int],
    validation_size: float,
    grid_limit: Optional[int],
    folds: int,
    resource_mode: str = DEFAULT_EXPERIMENT_RESOURCE_MODE,
    random_state: int = 42,
    fast_mode: bool = True,
    custom_grids: Optional[Dict[str, Dict[str, Sequence[Any]]]] = None,
) -> Dict[str, Any]:
    resource_policy = _resolve_experiment_resource_policy(resource_mode)
    risk_thresholds = dict(resource_policy.get("high_risk_thresholds") or {})
    years = _available_prediction_years(df, time_col)
    pred_years = max(0, len(years) - 1)
    batch_strategies = [s for s in strategies if s in YEARLY_STRATEGIES + [ADAPTIVE_ADWIN_STRATEGY, ADAPTIVE_KSWIN_STRATEGY]]
    tuning_tasks = _batch_tuning_tasks(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        time_col=time_col,
        model_names=model_names,
        strategies=batch_strategies,
        balance_modes=balance_modes,
        base_year=base_year,
        validation_size=float(validation_size),
        folds=int(folds),
        random_state=int(random_state),
        fast_mode=bool(fast_mode),
        resource_mode=str(resource_mode),
        grid_limit=grid_limit,
        custom_grids=custom_grids,
    )
    experiment_blocks = _build_experiment_blocks(
        model_names=list(model_names),
        strategies=list(strategies),
        arf_variants=list(arf_variants),
        kswin_variants=list(kswin_variants),
        balance_modes=list(balance_modes),
    )
    lower_bound_fits = 0
    for block in experiment_blocks:
        strategy = str(block.get("strategy", ""))
        if strategy in YEARLY_STRATEGIES:
            lower_bound_fits += pred_years
        elif strategy in {ADAPTIVE_ADWIN_STRATEGY, ADAPTIVE_KSWIN_STRATEGY}:
            lower_bound_fits += 1
    lower_bound_fits *= max(1, len(repetition_seeds))
    high_risk = (
        "Random Forest" in model_names
        and BALANCE_MODE_SMOTE in balance_modes
        and int(folds) >= int(risk_thresholds.get("folds", 5))
        and (grid_limit is None or int(grid_limit) >= int(risk_thresholds.get("trials", 20)))
    )
    return {
        "n_tuning_tasks": int(len(tuning_tasks)),
        "n_blocks": int(len(experiment_blocks) * max(1, len(repetition_seeds))),
        "estimated_batch_fits": int(lower_bound_fits),
        "high_risk_memory": bool(high_risk),
        "resource_mode": str(resource_policy["mode"]),
        "resource_mode_label": str(resource_policy["label"]),
        "resource_policy": _to_json_safe(resource_policy),
    }


def _append_execution_log(
    log_rows: Optional[List[Dict[str, Any]]],
    *,
    phase: str,
    status: str,
    message: str,
    **context: Any,
) -> None:
    if log_rows is None:
        return

    entry: Dict[str, Any] = {
        "order": len(log_rows) + 1,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "phase": str(phase),
        "status": str(status),
        "message": str(message),
    }
    for key, value in context.items():
        entry[str(key)] = _normalize_log_value(value)
    log_rows.append(entry)


def _group_seed_metadata(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=list(group_cols) + ["n_repetitions", "seed_list", "run_orders"])

    grouped = df.groupby(list(group_cols), dropna=False)
    return grouped.apply(
        lambda grp: pd.Series(
            {
                "n_repetitions": int(
                    grp["run_seed"].dropna().astype(int).nunique()
                    if "run_seed" in grp.columns and grp["run_seed"].notna().any()
                    else 1
                ),
                "seed_list": ",".join(
                    str(int(seed))
                    for seed in sorted(grp["run_seed"].dropna().astype(int).unique())
                )
                if "run_seed" in grp.columns and grp["run_seed"].notna().any()
                else "",
                "run_orders": ",".join(
                    str(int(order))
                    for order in sorted(grp["run_order"].dropna().astype(int).unique())
                )
                if "run_order" in grp.columns and grp["run_order"].notna().any()
                else "",
            }
        ),
        include_groups=False,
    ).reset_index()


def _ensure_optuna_available() -> None:
    if optuna is None or TPESampler is None:
        raise ImportError(
            "La busqueda de hiperparametros requiere `optuna`. "
            "Revise el entorno activo antes de ejecutar Drift detection."
        )


def _ensure_smote_available() -> None:
    if SMOTE is None:
        raise ImportError(
            "El balanceo con SMOTE requiere `imbalanced-learn`. "
            "Revise el entorno activo antes de ejecutar Drift detection con SMOTE."
        )


def _normalize_param_choice(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _prepare_xy_raw(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    cols = [c for c in feature_cols if c in df.columns]
    X = df[cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)
    return X, y


def _fit_fill_values(X: pd.DataFrame) -> Dict[str, float]:
    if X is None or X.empty:
        return {}
    medians = X.median(numeric_only=True)
    out: Dict[str, float] = {}
    for col in X.columns:
        value = medians.get(col, 0.0)
        value = 0.0 if pd.isna(value) else float(value)
        out[str(col)] = value
    return out


def _apply_fill_values(X: pd.DataFrame, fill_values: Optional[Dict[str, float]]) -> pd.DataFrame:
    work = X.copy()
    values = dict(fill_values or {})
    for col in work.columns:
        fill_value = float(values.get(str(col), 0.0))
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(fill_value).astype(float)
    return work


def _default_search_params(search_space: Dict[str, Sequence[Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, values in search_space.items():
        options = list(values)
        if not options:
            continue
        out[str(name)] = _normalize_param_choice(options[0])
    return out


def _search_space_size(search_space: Dict[str, Sequence[Any]]) -> int:
    size = 1
    for values in search_space.values():
        options = list(values)
        if not options:
            continue
        size *= len(options)
    return max(1, int(size))


def _parallel_cv_jobs(
    model_name: str,
    effective_folds: int,
    *,
    resource_mode: str = DEFAULT_EXPERIMENT_RESOURCE_MODE,
) -> int:
    resource_policy = _resolve_experiment_resource_policy(resource_mode)
    if str(model_name) not in set(resource_policy.get("parallel_cv_models") or []):
        return 1
    cpu_count = int(resource_policy.get("cpu_count", os.cpu_count() or 1))
    return max(1, min(int(effective_folds), int(cpu_count)))


def _bounded_estimator_n_jobs(
    model_name: str,
    *,
    resource_mode: str = DEFAULT_EXPERIMENT_RESOURCE_MODE,
) -> int:
    resource_policy = _resolve_experiment_resource_policy(resource_mode)
    cpu_count = max(1, int(resource_policy.get("cpu_count", os.cpu_count() or 1)))
    env_var = ""
    default_cap = 1

    if model_name == "Random Forest":
        env_var = "SUMO_RF_N_JOBS"
        default_cap = int((resource_policy.get("estimator_n_jobs") or {}).get("Random Forest", 2))
    elif model_name == "XGBoost":
        env_var = "SUMO_XGB_N_JOBS"
        default_cap = int((resource_policy.get("estimator_n_jobs") or {}).get("XGBoost", 4))
    else:
        return 1

    raw_value = os.getenv(env_var, "").strip()
    if raw_value:
        try:
            requested = int(raw_value)
        except ValueError:
            requested = default_cap
        return max(1, min(requested, cpu_count))
    return max(1, min(default_cap, cpu_count))


def _build_model_search_space(
    model_name: str,
    *,
    fast_mode: bool,
    custom_grid: Optional[Dict[str, Sequence[Any]]] = None,
    balance_mode: str = BALANCE_MODE_NONE,
) -> Dict[str, Sequence[Any]]:
    if custom_grid is not None:
        search_space = {str(key): list(values) for key, values in custom_grid.items()}
    else:
        base = FAST_HYPERPARAM_GRIDS[model_name] if fast_mode else FULL_HYPERPARAM_GRIDS[model_name]
        search_space = {str(key): list(values) for key, values in base.items()}

    if balance_mode == BALANCE_MODE_SMOTE:
        smote_space = FAST_SMOTE_SEARCH_SPACE if fast_mode else FULL_SMOTE_SEARCH_SPACE
        for key, values in smote_space.items():
            search_space[str(key)] = list(values)
    return search_space


def _split_model_and_smote_params(
    params: Dict[str, Any],
    *,
    balance_mode: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    smote_keys = {"sampling_strategy", "k_neighbors"}
    clean = {str(key): _normalize_param_choice(value) for key, value in dict(params).items()}
    model_params = {key: value for key, value in clean.items() if key not in smote_keys}
    smote_params = {}
    if balance_mode == BALANCE_MODE_SMOTE:
        smote_params = {
            key: clean[key]
            for key in smote_keys
            if key in clean
        }
    return model_params, smote_params


def _model_policy_version(model_name: str) -> Optional[str]:
    if str(model_name) == "NNet":
        return NNET_TRAINING_POLICY_VERSION
    return None


def _requires_feature_transform(model_name: str) -> bool:
    return str(model_name) == "NNet"


def _fit_feature_transform(
    X: pd.DataFrame,
    *,
    model_name: str,
) -> Optional[Dict[str, Any]]:
    if not _requires_feature_transform(model_name):
        return None
    scaler = StandardScaler()
    scaler.fit(X)
    scale = np.asarray(getattr(scaler, "scale_", np.ones(X.shape[1], dtype=float)), dtype=float)
    scale[~np.isfinite(scale)] = 1.0
    scale[np.isclose(scale, 0.0)] = 1.0
    mean = np.asarray(getattr(scaler, "mean_", np.zeros(X.shape[1], dtype=float)), dtype=float)
    mean[~np.isfinite(mean)] = 0.0
    return {
        "kind": "standard_scaler",
        "policy_version": NNET_TRAINING_POLICY_VERSION,
        "columns": list(X.columns),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
    }


def _apply_feature_transform(
    X: pd.DataFrame,
    feature_transform: Optional[Dict[str, Any]],
) -> pd.DataFrame:
    if not feature_transform:
        return X.copy()
    if str(feature_transform.get("kind", "")) != "standard_scaler":
        return X.copy()
    columns = [str(col) for col in feature_transform.get("columns", list(X.columns))]
    work = X.copy()
    for col in columns:
        if col not in work.columns:
            work[col] = 0.0
    work = work[columns].copy()
    mean = np.asarray(feature_transform.get("mean", [0.0] * len(columns)), dtype=float)
    scale = np.asarray(feature_transform.get("scale", [1.0] * len(columns)), dtype=float)
    scale[~np.isfinite(scale)] = 1.0
    scale[np.isclose(scale, 0.0)] = 1.0
    values = work.to_numpy(dtype=float, copy=True)
    transformed = (values - mean) / scale
    return pd.DataFrame(transformed, columns=columns, index=work.index)


def _fit_nnet_warning_safe(model, X: pd.DataFrame, y: Sequence[int]) -> Dict[str, Any]:
    warnings_list: List[warnings.WarningMessage] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X, y)
        warnings_list = list(caught)
    convergence_items = [
        item for item in warnings_list
        if issubclass(item.category, ConvergenceWarning)
    ]
    for item in warnings_list:
        if issubclass(item.category, ConvergenceWarning):
            continue
        warnings.warn(item.message, item.category, stacklevel=2)
    n_iter_value = getattr(model, "n_iter_", None)
    if isinstance(n_iter_value, np.generic):
        n_iter_value = n_iter_value.item()
    return {
        "converged": len(convergence_items) == 0,
        "fit_warning_count": int(len(convergence_items)),
        "n_iter_final": None if n_iter_value is None else int(n_iter_value),
        "warning_messages": [str(item.message) for item in convergence_items],
    }


def _fit_model_with_diagnostics(
    model_name: str,
    model_params: Dict[str, Any],
    *,
    random_state: int,
    n_features: int,
    X_fit: pd.DataFrame,
    y_fit: Sequence[int],
    allow_retry: bool = False,
    resource_mode: str = DEFAULT_EXPERIMENT_RESOURCE_MODE,
    balance_mode: str = BALANCE_MODE_NONE,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    params_used = {str(key): _normalize_param_choice(value) for key, value in dict(model_params).items()}
    model = _build_model(
        model_name,
        params_used,
        random_state=random_state,
        n_features=n_features,
        resource_mode=resource_mode,
        balance_mode=balance_mode,
    )
    diagnostics: Dict[str, Any]
    if str(model_name) == "NNet":
        diagnostics = _fit_nnet_warning_safe(model, X_fit, y_fit)
        diagnostics["fit_retry_applied"] = False
        diagnostics["retry_initial_converged"] = bool(diagnostics.get("converged", True))
        if allow_retry and not bool(diagnostics.get("converged", True)):
            retry_params = dict(params_used)
            current_maxit = int(retry_params.get("maxit", 200))
            retry_params["maxit"] = int(min(max(current_maxit * 2, current_maxit + 1), NNET_MAX_ITER_RETRY_CAP))
            if int(retry_params["maxit"]) != int(current_maxit):
                retry_model = _build_model(
                    model_name,
                    retry_params,
                    random_state=random_state,
                    n_features=n_features,
                    resource_mode=resource_mode,
                    balance_mode=balance_mode,
                )
                retry_diagnostics = _fit_nnet_warning_safe(retry_model, X_fit, y_fit)
                retry_diagnostics["fit_retry_applied"] = True
                retry_diagnostics["retry_initial_converged"] = bool(diagnostics.get("converged", True))
                model = retry_model
                params_used = retry_params
                diagnostics = retry_diagnostics
    else:
        model.fit(X_fit, y_fit)
        diagnostics = {
            "converged": True,
            "fit_warning_count": 0,
            "n_iter_final": None,
            "fit_retry_applied": False,
            "retry_initial_converged": True,
            "warning_messages": [],
        }
    diagnostics["model_name"] = str(model_name)
    diagnostics["params_used"] = dict(params_used)
    diagnostics["balance_mode"] = str(balance_mode)
    return model, diagnostics, params_used


def _apply_smote_balance(
    X: pd.DataFrame,
    y: Sequence[int],
    *,
    sampling_strategy: float,
    k_neighbors: int,
    random_state: int,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    _ensure_smote_available()
    y_np = np.asarray(y).astype(int)
    train_rows = int(len(X))
    if X.empty or np.unique(y_np).size < 2:
        return X.copy(), y_np, {
            "applied": False,
            "train_rows": train_rows,
            "original_rows": train_rows,
            "balanced_rows": train_rows,
            "sampling_strategy": float(sampling_strategy),
            "k_neighbors": int(k_neighbors),
        }

    class_counts = np.bincount(y_np)
    positive_count = int(class_counts.min()) if class_counts.size else 0
    majority_count = int(class_counts.max()) if class_counts.size else 0
    if positive_count < 2:
        return X.copy(), y_np, {
            "applied": False,
            "train_rows": train_rows,
            "original_rows": train_rows,
            "balanced_rows": train_rows,
            "sampling_strategy": float(sampling_strategy),
            "k_neighbors": int(k_neighbors),
        }
    current_ratio = float(positive_count / majority_count) if majority_count > 0 else 1.0
    if current_ratio >= float(sampling_strategy):
        return X.copy(), y_np, {
            "applied": False,
            "train_rows": train_rows,
            "original_rows": train_rows,
            "balanced_rows": train_rows,
            "sampling_strategy": float(sampling_strategy),
            "k_neighbors": int(k_neighbors),
            "reason": "already_at_or_above_target_ratio",
        }

    effective_k = max(1, min(int(k_neighbors), positive_count - 1))
    sampler = SMOTE(
        sampling_strategy=float(sampling_strategy),
        k_neighbors=int(effective_k),
        random_state=int(random_state),
    )
    X_res, y_res = sampler.fit_resample(X, y_np)
    X_bal = pd.DataFrame(X_res, columns=list(X.columns))
    y_bal = np.asarray(y_res).astype(int)
    return X_bal, y_bal, {
        "applied": True,
        "train_rows": train_rows,
        "sampling_strategy": float(sampling_strategy),
        "k_neighbors": int(effective_k),
        "original_rows": train_rows,
        "balanced_rows": int(len(X_bal)),
    }


def _load_or_create_smote_artifact(
    *,
    run_id: str,
    train_df: pd.DataFrame,
    X: pd.DataFrame,
    y: Sequence[int],
    feature_cols: Sequence[str],
    target_col: str,
    time_col: Optional[str],
    sampling_strategy: float,
    k_neighbors: int,
    random_state: int,
    smote_cache_dir: Optional[Path],
    smote_cache_registry: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    if smote_cache_dir is None:
        return _apply_smote_balance(
            X,
            y,
            sampling_strategy=float(sampling_strategy),
            k_neighbors=int(k_neighbors),
            random_state=int(random_state),
        )

    smote_cache_dir = Path(smote_cache_dir)
    smote_cache_dir.mkdir(parents=True, exist_ok=True)
    smote_key = _build_smote_key(
        run_id=run_id,
        train_df=train_df,
        feature_cols=feature_cols,
        target_col=target_col,
        time_col=time_col,
        sampling_strategy=float(sampling_strategy),
        k_neighbors=int(k_neighbors),
    )
    artifact_path = smote_cache_dir / f"{smote_key}.duckdb"
    if artifact_path.exists():
        augmented_df, metadata = _load_smote_payload_from_duckdb(artifact_path)
        y_bal = pd.to_numeric(augmented_df[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
        X_bal = augmented_df[[col for col in feature_cols if col in augmented_df.columns]].copy()
        fit_info = dict(metadata.get("fit_info") or {})
        fit_info.update(
            {
                "artifact_key": smote_key,
                "artifact_path": str(artifact_path),
                "artifact_reused": True,
            }
        )
        if smote_cache_registry is not None:
            smote_cache_registry[smote_key] = {
                "smote_key": smote_key,
                "path": str(artifact_path),
                "metadata": metadata,
                "status": "available",
                "reused": True,
            }
        return X_bal, y_bal, fit_info

    X_bal, y_bal, fit_info = _apply_smote_balance(
        X,
        y,
        sampling_strategy=float(sampling_strategy),
        k_neighbors=int(k_neighbors),
        random_state=int(random_state),
    )
    augmented_df = X_bal.copy()
    augmented_df[target_col] = np.asarray(y_bal).astype(int)
    metadata = {
        "smote_key": smote_key,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "feature_cols": list(feature_cols),
        "target_col": str(target_col),
        "time_col": None if time_col is None else str(time_col),
        "sampling_strategy": float(sampling_strategy),
        "k_neighbors": int(k_neighbors),
        "fit_info": {
            **dict(fit_info),
            "artifact_key": smote_key,
            "artifact_path": str(artifact_path),
            "artifact_reused": False,
        },
    }
    _write_smote_payload_to_duckdb(
        artifact_path,
        augmented_df=augmented_df,
        metadata=metadata,
    )
    if smote_cache_registry is not None:
        smote_cache_registry[smote_key] = {
            "smote_key": smote_key,
            "path": str(artifact_path),
            "metadata": metadata,
            "status": "available",
            "reused": False,
        }
    fit_info = dict(metadata["fit_info"])
    return X_bal, y_bal, fit_info


def _normalized_optuna_trials(
    study: Any,
    *,
    balance_mode: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for trial in study.trials:
        state_name = str(getattr(trial.state, "name", trial.state))
        trial_value = None
        if trial.value is not None and float(trial.value) >= 0:
            trial_value = float(trial.value)
        model_params, smote_params = _split_model_and_smote_params(
            dict(trial.params),
            balance_mode=balance_mode,
        )
        cv_diagnostics = dict(getattr(trial, "user_attrs", {}).get("cv_diagnostics") or {})
        rows.append(
            {
                "trial_number": int(trial.number),
                "state": state_name,
                "cv_auc": trial_value,
                "params": {key: _normalize_param_choice(value) for key, value in trial.params.items()},
                "model_params": model_params,
                "smote_params": smote_params,
                "converged": bool(cv_diagnostics.get("converged", True)),
                "n_non_converged_folds": int(cv_diagnostics.get("n_non_converged_folds", 0)),
                "n_valid_folds": int(cv_diagnostics.get("n_valid_folds", 0)),
                "fold_diagnostics": _to_json_safe(cv_diagnostics.get("fold_diagnostics") or []),
                "n_iter_final": cv_diagnostics.get("n_iter_final"),
            }
        )
    return rows


def _build_tuning_study_id(
    *,
    stage: str,
    strategy: str,
    model_name: str,
    balance_mode: str,
    run_seed: int,
    run_order: int,
    detector_variant: Optional[str] = None,
    prediction_year: Optional[int] = None,
    training_year: Optional[str] = None,
    drift: Optional[int] = None,
) -> str:
    parts = [
        stage,
        strategy,
        model_name,
        balance_mode,
        f"seed_{int(run_seed)}",
        f"run_{int(run_order)}",
    ]
    if detector_variant:
        parts.append(str(detector_variant))
    if prediction_year is not None:
        parts.append(f"pred_{int(prediction_year)}")
    if training_year:
        parts.append(f"train_{training_year}")
    if drift is not None:
        parts.append(f"drift_{int(drift)}")
    return _slugify_token("__".join(parts))


def _append_tuning_study(
    study_rows: Optional[List[Dict[str, Any]]],
    artifact: Optional[Dict[str, Any]],
    **context: Any,
) -> None:
    if study_rows is None or not artifact:
        return
    entry = dict(artifact)
    entry.update(context)
    study_rows.append(_to_json_safe(entry))


def _build_model(
    model_name: str,
    params: Dict[str, Any],
    *,
    random_state: int,
    n_features: int,
    resource_mode: str = DEFAULT_EXPERIMENT_RESOURCE_MODE,
    balance_mode: str = BALANCE_MODE_NONE,
):
    if model_name == "Random Forest":
        splitrule = str(params.get("splitrule", "gini"))
        mtry = int(params.get("mtry", min(6, max(1, n_features))))
        class_weight = None if str(balance_mode) == BALANCE_MODE_SMOTE else "balanced_subsample"
        common_kwargs = {
            "n_estimators": 400,
            "max_features": max(1, min(mtry, n_features)),
            "min_samples_leaf": int(params.get("min_node_size", 1)),
            "random_state": random_state,
            "n_jobs": _bounded_estimator_n_jobs(model_name, resource_mode=resource_mode),
        }
        if splitrule == "extratrees":
            return ExtraTreesClassifier(
                criterion="gini",
                class_weight=None if str(balance_mode) == BALANCE_MODE_SMOTE else "balanced",
                **common_kwargs,
            )
        return RandomForestClassifier(
            criterion="gini",
            class_weight=class_weight,
            **common_kwargs,
        )

    if model_name == "AdaBoost":
        base = DecisionTreeClassifier(
            max_depth=int(params.get("maxdepth", 1)),
            random_state=random_state,
        )
        return AdaBoostClassifier(
            estimator=base,
            n_estimators=int(params.get("iter", 100)),
            learning_rate=float(params.get("nu", 0.1)),
            random_state=random_state,
        )

    if model_name == "NNet":
        return MLPClassifier(
            hidden_layer_sizes=(int(params.get("size", 10)),),
            alpha=float(params.get("decay", 0.1)),
            max_iter=int(params.get("maxit", 200)),
            random_state=random_state,
            solver="adam",
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            tol=1e-4,
        )

    if model_name == "XGBoost":
        try:
            xgb = _import_external_xgboost()
        except ImportError as exc:
            raise ImportError(
                "No se pudo cargar el paquete externo `xgboost`. "
                "Revise el entorno o el sombreado con `src/xgboost.py`."
            ) from exc
        return xgb.XGBClassifier(
            n_estimators=int(params.get("nrounds", 100)),
            max_depth=int(params.get("max_depth", 6)),
            learning_rate=float(params.get("eta", 0.1)),
            gamma=float(params.get("gamma", 0.0)),
            colsample_bytree=float(params.get("colsample_bytree", 0.8)),
            min_child_weight=float(params.get("min_child_weight", 1.0)),
            subsample=float(params.get("subsample", 0.7)),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=_bounded_estimator_n_jobs(model_name, resource_mode=resource_mode),
            verbosity=0,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def _model_scores(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def _cv_auc(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_name: str,
    params: Dict[str, Any],
    folds: int,
    random_state: int,
    balance_mode: str = BALANCE_MODE_NONE,
    return_diagnostics: bool = False,
    resource_mode: str = DEFAULT_EXPERIMENT_RESOURCE_MODE,
) -> Any:
    effective_folds = _effective_stratified_folds(y, folds)
    if effective_folds < 2:
        empty_payload = {
            "cv_auc": float("nan"),
            "converged": False,
            "n_non_converged_folds": 0,
            "n_valid_folds": 0,
            "fold_diagnostics": [],
        }
        return empty_payload if return_diagnostics else float("nan")

    y_np = np.asarray(y).astype(int)
    skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=random_state)
    splits = list(enumerate(skf.split(X, y_np), start=1))

    def evaluate_fold(fold_idx: int, tr_idx: np.ndarray, va_idx: np.ndarray) -> Dict[str, Any]:
        X_tr_raw = X.iloc[tr_idx]
        X_va_raw = X.iloc[va_idx]
        y_tr = y_np[tr_idx]
        y_va = y_np[va_idx]

        try:
            fill_values = _fit_fill_values(X_tr_raw)
            X_tr_filled = _apply_fill_values(X_tr_raw, fill_values)
            X_va_filled = _apply_fill_values(X_va_raw, fill_values)
            feature_transform = _fit_feature_transform(X_tr_filled, model_name=model_name)
            X_tr = _apply_feature_transform(X_tr_filled, feature_transform)
            X_va = _apply_feature_transform(X_va_filled, feature_transform)
            model_params, smote_params = _split_model_and_smote_params(
                params,
                balance_mode=balance_mode,
            )
            if balance_mode == BALANCE_MODE_SMOTE:
                X_fit, y_fit, _ = _apply_smote_balance(
                    X_tr,
                    y_tr,
                    sampling_strategy=float(smote_params.get("sampling_strategy", 1.0)),
                    k_neighbors=int(smote_params.get("k_neighbors", 5)),
                    random_state=int(random_state) + int(fold_idx),
                )
            else:
                X_fit, y_fit = X_tr, y_tr
            model, fit_diagnostics, _ = _fit_model_with_diagnostics(
                model_name,
                model_params,
                random_state=random_state,
                n_features=X.shape[1],
                X_fit=X_fit,
                y_fit=y_fit,
                allow_retry=False,
                resource_mode=resource_mode,
                balance_mode=balance_mode,
            )
            if str(model_name) == "NNet" and not bool(fit_diagnostics.get("converged", True)):
                return {
                    "auc": float("nan"),
                    "converged": False,
                    "n_iter_final": fit_diagnostics.get("n_iter_final"),
                    "fit_warning_count": int(fit_diagnostics.get("fit_warning_count", 0)),
                }
            scores = _model_scores(model, X_va)
            auc = _safe_auc(y_va, scores)
            if not np.isnan(auc):
                return {
                    "auc": float(auc),
                    "converged": bool(fit_diagnostics.get("converged", True)),
                    "n_iter_final": fit_diagnostics.get("n_iter_final"),
                    "fit_warning_count": int(fit_diagnostics.get("fit_warning_count", 0)),
                }
        except Exception:
            return {
                "auc": float("nan"),
                "converged": False if str(model_name) == "NNet" else True,
                "n_iter_final": None,
                "fit_warning_count": 0,
            }
        return {
            "auc": float("nan"),
            "converged": False if str(model_name) == "NNet" else True,
            "n_iter_final": None,
            "fit_warning_count": 0,
        }

    parallel_jobs = _parallel_cv_jobs(model_name, effective_folds, resource_mode=resource_mode)
    if parallel_jobs > 1:
        fold_payloads = SklearnParallel(n_jobs=parallel_jobs, prefer="threads")(
            sklearn_delayed(evaluate_fold)(fold_idx, tr_idx, va_idx)
            for fold_idx, (tr_idx, va_idx) in splits
        )
    else:
        fold_payloads = [evaluate_fold(fold_idx, tr_idx, va_idx) for fold_idx, (tr_idx, va_idx) in splits]
    aucs = [float(payload.get("auc", float("nan"))) for payload in fold_payloads if np.isfinite(payload.get("auc", float("nan")))]
    non_converged_folds = int(
        sum(
            1
            for payload in fold_payloads
            if str(model_name) == "NNet" and not bool(payload.get("converged", True))
        )
    )
    fold_n_iters = [
        int(payload["n_iter_final"])
        for payload in fold_payloads
        if payload.get("n_iter_final") is not None
    ]
    diagnostics = {
        "cv_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "converged": non_converged_folds == 0,
        "n_non_converged_folds": non_converged_folds,
        "n_valid_folds": int(len(aucs)),
        "n_iter_final": max(fold_n_iters) if fold_n_iters else None,
        "fold_diagnostics": _to_json_safe(fold_payloads),
    }
    if return_diagnostics:
        return diagnostics
    if not aucs:
        return float("nan")
    return float(np.mean(aucs))


def _effective_stratified_folds(y: Sequence[int], folds: int) -> int:
    y_np = np.asarray(y).astype(int)
    if np.unique(y_np).size < 2:
        return 0

    min_class = int(np.bincount(y_np).min())
    effective_folds = max(2, min(int(folds), min_class))
    if effective_folds < 2:
        return 0
    return int(effective_folds)


def _cross_validated_scores(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_name: str,
    params: Dict[str, Any],
    folds: int,
    random_state: int,
    balance_mode: str = BALANCE_MODE_NONE,
    return_diagnostics: bool = False,
    resource_mode: str = DEFAULT_EXPERIMENT_RESOURCE_MODE,
) -> Any:
    effective_folds = _effective_stratified_folds(y, folds)
    scores = np.full(len(X), np.nan, dtype=float)
    if effective_folds < 2:
        payload = {
            "scores": scores,
            "converged": False,
            "n_non_converged_folds": 0,
            "n_valid_folds": 0,
            "fold_diagnostics": [],
        }
        return payload if return_diagnostics else scores

    y_np = np.asarray(y).astype(int)
    skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=random_state)
    splits = list(enumerate(skf.split(X, y_np), start=1))
    model_params, smote_params = _split_model_and_smote_params(
        params,
        balance_mode=balance_mode,
    )

    def score_fold(
        fold_idx: int,
        tr_idx: np.ndarray,
        va_idx: np.ndarray,
    ) -> Dict[str, Any]:
        X_tr_raw = X.iloc[tr_idx]
        X_va_raw = X.iloc[va_idx]
        y_tr = y_np[tr_idx]

        try:
            fill_values = _fit_fill_values(X_tr_raw)
            X_tr_filled = _apply_fill_values(X_tr_raw, fill_values)
            X_va_filled = _apply_fill_values(X_va_raw, fill_values)
            feature_transform = _fit_feature_transform(X_tr_filled, model_name=model_name)
            X_tr = _apply_feature_transform(X_tr_filled, feature_transform)
            X_va = _apply_feature_transform(X_va_filled, feature_transform)
            if balance_mode == BALANCE_MODE_SMOTE:
                X_fit, y_fit, _ = _apply_smote_balance(
                    X_tr,
                    y_tr,
                    sampling_strategy=float(smote_params.get("sampling_strategy", 1.0)),
                    k_neighbors=int(smote_params.get("k_neighbors", 5)),
                    random_state=int(random_state) + int(fold_idx),
                )
            else:
                X_fit, y_fit = X_tr, y_tr
            model, fit_diagnostics, _ = _fit_model_with_diagnostics(
                model_name,
                model_params,
                random_state=random_state,
                n_features=X.shape[1],
                X_fit=X_fit,
                y_fit=y_fit,
                allow_retry=False,
                resource_mode=resource_mode,
                balance_mode=balance_mode,
            )
            if str(model_name) == "NNet" and not bool(fit_diagnostics.get("converged", True)):
                return {
                    "va_idx": np.asarray([], dtype=int),
                    "scores": np.asarray([], dtype=float),
                    "converged": False,
                    "n_iter_final": fit_diagnostics.get("n_iter_final"),
                    "fit_warning_count": int(fit_diagnostics.get("fit_warning_count", 0)),
                }
            return {
                "va_idx": np.asarray(va_idx, dtype=int),
                "scores": np.asarray(_model_scores(model, X_va), dtype=float),
                "converged": bool(fit_diagnostics.get("converged", True)),
                "n_iter_final": fit_diagnostics.get("n_iter_final"),
                "fit_warning_count": int(fit_diagnostics.get("fit_warning_count", 0)),
            }
        except Exception:
            return {
                "va_idx": np.asarray([], dtype=int),
                "scores": np.asarray([], dtype=float),
                "converged": False if str(model_name) == "NNet" else True,
                "n_iter_final": None,
                "fit_warning_count": 0,
            }

    parallel_jobs = _parallel_cv_jobs(model_name, effective_folds, resource_mode=resource_mode)
    if parallel_jobs > 1:
        fold_results = SklearnParallel(n_jobs=parallel_jobs, prefer="threads")(
            sklearn_delayed(score_fold)(fold_idx, tr_idx, va_idx)
            for fold_idx, (tr_idx, va_idx) in splits
        )
    else:
        fold_results = [score_fold(fold_idx, tr_idx, va_idx) for fold_idx, (tr_idx, va_idx) in splits]

    for payload in fold_results:
        va_idx = np.asarray(payload.get("va_idx", []), dtype=int)
        fold_scores = np.asarray(payload.get("scores", []), dtype=float)
        if len(va_idx) and len(fold_scores) == len(va_idx):
            scores[va_idx] = fold_scores
    non_converged_folds = int(
        sum(1 for item in fold_results if str(model_name) == "NNet" and not bool(item.get("converged", True)))
    )
    n_valid_folds = int(sum(1 for item in fold_results if len(np.asarray(item.get("va_idx", []), dtype=int))))
    n_iter_final = max(
        [
            int(item["n_iter_final"])
            for item in fold_results
            if item.get("n_iter_final") is not None
        ],
        default=None,
    )
    if not return_diagnostics:
        return scores
    payload = {
        "scores": scores,
        "converged": non_converged_folds == 0,
        "n_non_converged_folds": non_converged_folds,
        "n_valid_folds": n_valid_folds,
        "n_iter_final": n_iter_final,
        "fold_diagnostics": _to_json_safe(fold_results),
    }
    return payload if return_diagnostics else scores


def tune_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_name: str,
    folds: int = 5,
    random_state: int = 42,
    fast_mode: bool = True,
    grid_limit: Optional[int] = None,
    custom_grid: Optional[Dict[str, Sequence[Any]]] = None,
    balance_mode: str = BALANCE_MODE_NONE,
    study_id: Optional[str] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    resource_mode: str = DEFAULT_EXPERIMENT_RESOURCE_MODE,
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, Any]]:
    _ensure_optuna_available()
    search_space = _build_model_search_space(
        model_name,
        fast_mode=fast_mode,
        custom_grid=custom_grid,
        balance_mode=balance_mode,
    )
    search_space_size = _search_space_size(search_space)
    requested_trials = int(grid_limit) if grid_limit is not None and int(grid_limit) > 0 else search_space_size
    n_trials = max(1, min(int(requested_trials), int(search_space_size)))
    if study_id is None:
        study_id = _build_tuning_study_id(
            stage="generic",
            strategy="generic",
            model_name=model_name,
            balance_mode=balance_mode,
            run_seed=int(random_state),
            run_order=1,
        )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=int(random_state)),
        study_name=study_id,
    )
    _emit_experiment_progress(
        progress_callback,
        ratio=0.0,
        label="Sintonizando hiperparametros...",
        detail=f"Modelo {model_name} | 0 / {int(n_trials)} trials",
    )

    def objective(trial: Any) -> float:
        params = {
            name: _normalize_param_choice(
                trial.suggest_categorical(name, list(values))
            )
            for name, values in search_space.items()
        }
        cv_payload = _cv_auc(
            X,
            y,
            model_name=model_name,
            params=params,
            folds=folds,
            random_state=random_state,
            balance_mode=balance_mode,
            return_diagnostics=True,
            resource_mode=resource_mode,
        )
        trial.set_user_attr("cv_diagnostics", _to_json_safe(cv_payload))
        auc = float(cv_payload.get("cv_auc", float("nan")))
        if str(model_name) == "NNet" and not bool(cv_payload.get("converged", True)):
            return -1.0
        return float(auc) if np.isfinite(auc) else -1.0

    def optuna_progress(study_obj: Any, _trial: Any) -> None:
        completed = len(study_obj.trials)
        best_value = getattr(study_obj, "best_value", None)
        best_text = "n/a" if best_value is None or not np.isfinite(float(best_value)) else f"{float(best_value):.4f}"
        _emit_experiment_progress(
            progress_callback,
            ratio=float(completed) / float(max(1, int(n_trials))),
            label="Sintonizando hiperparametros...",
            detail=f"Modelo {model_name} | Trial {completed} / {int(n_trials)} | Mejor AUC CV {best_text}",
        )

    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False, callbacks=[optuna_progress])
    trials = _normalized_optuna_trials(study, balance_mode=balance_mode)
    rows: List[Dict[str, Any]] = []
    for row in trials:
        rows.append(
            {
                "trial_number": int(row["trial_number"]),
                "model": model_name,
                "cv_auc": row["cv_auc"],
                "params": json.dumps(row["params"], sort_keys=True, ensure_ascii=True),
                "state": str(row["state"]),
                "balance_mode": balance_mode,
                "converged": bool(row.get("converged", True)),
                "n_non_converged_folds": int(row.get("n_non_converged_folds", 0)),
                "n_valid_folds": int(row.get("n_valid_folds", 0)),
            }
        )

    best_params = _default_search_params(search_space)
    best_value: Optional[float] = None
    has_valid_trial = False
    if study.trials and study.best_value is not None and float(study.best_value) >= 0:
        best_params = {key: _normalize_param_choice(value) for key, value in study.best_trial.params.items()}
        best_value = float(study.best_value)
        has_valid_trial = True
    invalid_trial_count = int(sum(1 for row in trials if not bool(row.get("converged", True))))

    tuning_artifact = {
        "study_id": study_id,
        "model_name": model_name,
        "balance_mode": balance_mode,
        "best_value": best_value,
        "best_params": best_params,
        "has_valid_trial": bool(has_valid_trial),
        "invalid_trial_count": invalid_trial_count,
        "policy_version": _model_policy_version(model_name),
        "model_params": _split_model_and_smote_params(best_params, balance_mode=balance_mode)[0],
        "smote_params": _split_model_and_smote_params(best_params, balance_mode=balance_mode)[1],
        "n_trials": int(n_trials),
        "requested_trials": int(requested_trials),
        "search_space_size": int(search_space_size),
        "search_space": {key: list(values) for key, values in search_space.items()},
        "trials": trials,
    }
    search_df = pd.DataFrame(rows).sort_values(
        ["converged", "cv_auc"],
        ascending=[False, False],
        na_position="last",
    )
    return best_params, search_df, tuning_artifact


def _prepare_xy(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    X_raw, y = _prepare_xy_raw(df, feature_cols, target_col)
    return _apply_fill_values(X_raw, _fit_fill_values(X_raw)), y


def _require_river_streaming() -> None:
    if river_forest is None or river_drift is None or river_metrics is None:
        raise ImportError(
            "Las estrategias online basadas en `river` requieren esa libreria instalada. "
            "Instale las dependencias del proyecto o revise el entorno activo antes de ejecutar "
            "`adaptive_arf` o `adaptive_kswin`."
        )


def _build_arf_classifier(
    variant_name: str,
    *,
    random_state: int,
):
    _require_river_streaming()
    if variant_name not in ARF_VARIANT_CONFIGS:
        raise ValueError(f"Unsupported ARF variant: {variant_name}")

    config = ARF_VARIANT_CONFIGS[variant_name]
    return river_forest.ARFClassifier(
        n_models=10,
        max_features="sqrt",
        lambda_value=6,
        metric=river_metrics.Accuracy(),
        disable_weighted_vote=bool(config["disable_weighted_vote"]),
        warning_detector=river_drift.ADWIN(delta=float(config["warning_delta"])),
        drift_detector=river_drift.ADWIN(delta=float(config["drift_delta"])),
        grace_period=50,
        leaf_prediction="nba",
        seed=int(random_state),
    )


def _row_to_online_feature_dict(row: pd.Series) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in row.items():
        out[str(key)] = 0.0 if pd.isna(value) else float(value)
    return out


def _score_online_binary_classifier(model: Any, x_dict: Dict[str, float]) -> float:
    if hasattr(model, "predict_proba_one"):
        try:
            proba = model.predict_proba_one(x_dict) or {}
        except Exception:
            proba = {}
        if not proba:
            return 0.5
        if 1 in proba:
            return float(proba.get(1, 0.0))
        if "1" in proba:
            return float(proba.get("1", 0.0))

    if hasattr(model, "predict_one"):
        pred = model.predict_one(x_dict)
        return 1.0 if str(pred) in {"1", "True"} else 0.0

    return 0.5


def _online_counter_value(model: Any, attr_name: str) -> int:
    value = getattr(model, attr_name, 0)
    if callable(value):
        value = value()
    try:
        return int(value)
    except Exception:
        return 0


def _select_kswin_monitor_columns(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    top_k: int,
) -> List[str]:
    if df is None or df.empty:
        return []

    out: List[str] = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        out.append(str(col))
        if len(out) >= max(1, int(top_k)):
            break
    return out


def _build_kswin_seasonal_profiles(
    df: pd.DataFrame,
    *,
    monitor_cols: Sequence[str],
    time_col: str,
) -> Dict[str, Any]:
    if df is None or df.empty or time_col not in df.columns:
        return {"global": {}, "hour": {}, "dow_hour": {}}

    work = df[[time_col] + [c for c in monitor_cols if c in df.columns]].copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.dropna(subset=[time_col]).reset_index(drop=True)
    if work.empty:
        return {"global": {}, "hour": {}, "dow_hour": {}}

    work["hour"] = work[time_col].dt.hour.astype(int)
    work["dow"] = work[time_col].dt.dayofweek.astype(int)

    global_stats: Dict[str, Dict[str, float]] = {}
    hour_stats: Dict[str, Dict[int, Dict[str, float]]] = {}
    dow_hour_stats: Dict[str, Dict[Tuple[int, int], Dict[str, float]]] = {}

    for col in monitor_cols:
        if col not in work.columns:
            continue
        series = pd.to_numeric(work[col], errors="coerce")
        mean = float(series.mean()) if series.notna().any() else 0.0
        std = float(series.std(ddof=0)) if series.notna().any() else 1.0
        if not np.isfinite(std) or std <= 0:
            std = 1.0
        global_stats[str(col)] = {"mean": mean, "std": std}

        hour_grp = (
            work.groupby("hour", dropna=False)[col]
            .agg(["mean", "std"])
            .replace([np.inf, -np.inf], np.nan)
            .fillna({"std": std, "mean": mean})
        )
        hour_stats[str(col)] = {
            int(hour): {
                "mean": float(vals["mean"]) if np.isfinite(vals["mean"]) else mean,
                "std": float(vals["std"]) if np.isfinite(vals["std"]) and vals["std"] > 0 else std,
            }
            for hour, vals in hour_grp.iterrows()
        }

        dow_hour_grp = (
            work.groupby(["dow", "hour"], dropna=False)[col]
            .agg(["mean", "std"])
            .replace([np.inf, -np.inf], np.nan)
            .fillna({"std": np.nan, "mean": np.nan})
        )
        dow_hour_stats[str(col)] = {}
        for key, vals in dow_hour_grp.iterrows():
            dow = int(key[0])
            hour = int(key[1])
            mean_val = float(vals["mean"]) if np.isfinite(vals["mean"]) else hour_stats[str(col)].get(hour, global_stats[str(col)])["mean"]
            std_val = float(vals["std"]) if np.isfinite(vals["std"]) and vals["std"] > 0 else hour_stats[str(col)].get(hour, global_stats[str(col)])["std"]
            dow_hour_stats[str(col)][(dow, hour)] = {"mean": mean_val, "std": std_val}

    return {"global": global_stats, "hour": hour_stats, "dow_hour": dow_hour_stats}


def _transform_kswin_value(
    *,
    timestamp: pd.Timestamp,
    value: float,
    feature_name: str,
    variant_name: str,
    seasonal_profiles: Optional[Dict[str, Any]] = None,
) -> float:
    numeric = float(value)
    if not np.isfinite(numeric):
        return float("nan")
    if variant_name != "KSWINseasonal":
        return numeric

    profiles = seasonal_profiles or {}
    global_stats = (profiles.get("global") or {}).get(feature_name, {"mean": 0.0, "std": 1.0})
    hour_stats = (profiles.get("hour") or {}).get(feature_name, {})
    dow_hour_stats = (profiles.get("dow_hour") or {}).get(feature_name, {})

    ts = pd.Timestamp(timestamp)
    key = (int(ts.dayofweek), int(ts.hour))
    ref = dow_hour_stats.get(key)
    if ref is None:
        ref = hour_stats.get(int(ts.hour))
    if ref is None:
        ref = global_stats

    mean = float(ref.get("mean", global_stats.get("mean", 0.0)))
    std = float(ref.get("std", global_stats.get("std", 1.0)))
    if not np.isfinite(std) or std <= 0:
        std = float(global_stats.get("std", 1.0) or 1.0)
    return float((numeric - mean) / std)


def _build_kswin_detectors(
    *,
    variant_name: str,
    monitor_cols: Sequence[str],
    warm_df: pd.DataFrame,
    time_col: str,
    seasonal_profiles: Optional[Dict[str, Any]],
    random_state: int,
) -> Dict[str, Any]:
    _require_river_streaming()
    if variant_name not in KSWIN_VARIANT_CONFIGS:
        raise ValueError(f"Unsupported KSWIN variant: {variant_name}")

    config = KSWIN_VARIANT_CONFIGS[variant_name]
    warm = warm_df.copy() if isinstance(warm_df, pd.DataFrame) else pd.DataFrame()
    if not warm.empty and time_col in warm.columns:
        warm[time_col] = pd.to_datetime(warm[time_col], errors="coerce")
        warm = warm.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    detectors: Dict[str, Any] = {}
    for idx, col in enumerate(monitor_cols):
        warm_window: List[float] = []
        if not warm.empty and col in warm.columns and time_col in warm.columns:
            subset = warm[[time_col, col]].copy()
            subset[col] = pd.to_numeric(subset[col], errors="coerce")
            subset = subset.dropna(subset=[col])
            for row in subset.itertuples(index=False):
                warm_window.append(
                    _transform_kswin_value(
                        timestamp=getattr(row, time_col),
                        value=float(getattr(row, col)),
                        feature_name=str(col),
                        variant_name=variant_name,
                        seasonal_profiles=seasonal_profiles,
                    )
                )
            warm_window = [float(v) for v in warm_window if np.isfinite(v)]
            if len(warm_window) > int(config["window_size"]):
                warm_window = warm_window[-int(config["window_size"]):]

        detectors[str(col)] = river_drift.KSWIN(
            alpha=float(config["alpha"]),
            window_size=int(config["window_size"]),
            stat_size=int(config["stat_size"]),
            seed=int(random_state) + int(idx),
            window=warm_window if warm_window else None,
        )
    return detectors


def _select_kswin_retrain_window(
    observed_df: pd.DataFrame,
    *,
    current_ts: pd.Timestamp,
    time_col: str,
    retrain_days: int,
    min_rows: int,
    target_col: str,
) -> pd.DataFrame:
    if observed_df is None or observed_df.empty or time_col not in observed_df.columns:
        return pd.DataFrame()

    work = observed_df.copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    if work.empty:
        return work

    start_ts = pd.Timestamp(current_ts) - pd.Timedelta(days=max(1, int(retrain_days)))
    candidate = work.loc[work[time_col] >= start_ts].copy()
    min_rows = max(10, int(min_rows))

    if len(candidate) < min_rows:
        candidate = work.tail(min_rows).copy()

    if target_col in candidate.columns:
        y = pd.to_numeric(candidate[target_col], errors="coerce").fillna(0).astype(int)
        if y.nunique() < 2:
            candidate = work.copy()

    return candidate.reset_index(drop=True)


def train_arf_with_internal_validation(
    train_df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    target_col: str,
    time_col: str = "interval_start",
    variant_name: str,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Calibrates an ARF threshold on the trailing validation block of the base year
    and returns a model already warmed on the full base-year data.
    """
    work = train_df.copy()
    if time_col in work.columns:
        work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
        work = work.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    else:
        work = work.reset_index(drop=True)

    if len(work) < 2:
        raise ValueError("ARF training requires at least two base rows.")

    X_all_raw, y_all = _prepare_xy_raw(work, feature_cols, target_col)
    if y_all.nunique() < 2:
        raise ValueError("Training data has a single target class.")

    n_total = int(len(work))
    n_val = max(1, int(round(n_total * float(validation_size))))
    n_val = min(n_val, n_total - 1)
    n_warm = max(1, n_total - n_val)
    fill_values = _fit_fill_values(X_all_raw.iloc[:n_warm])
    X_all = _apply_fill_values(X_all_raw, fill_values)

    model = _build_arf_classifier(variant_name, random_state=random_state)

    for idx in range(n_warm):
        x_dict = _row_to_online_feature_dict(X_all.iloc[idx])
        y_i = int(y_all.iloc[idx])
        model.learn_one(x_dict, y_i)

    val_scores: List[float] = []
    val_y: List[int] = []
    for idx in range(n_warm, n_total):
        x_dict = _row_to_online_feature_dict(X_all.iloc[idx])
        y_i = int(y_all.iloc[idx])
        val_scores.append(_score_online_binary_classifier(model, x_dict))
        val_y.append(y_i)
        model.learn_one(x_dict, y_i)

    youden = youden_threshold(np.asarray(val_y), np.asarray(val_scores))
    val_metrics = compute_classification_metrics(
        np.asarray(val_y),
        np.asarray(val_scores),
        threshold=float(youden["threshold"]),
    )
    config = ARF_VARIANT_CONFIGS[variant_name]

    return {
        "model": model,
        "threshold": float(youden["threshold"]),
        "youden": youden,
        "val_metrics": val_metrics,
        "best_params": {
            "variant": variant_name,
            "lambda_value": 6,
            "max_features": "sqrt",
            "warning_delta": float(config["warning_delta"]),
            "drift_delta": float(config["drift_delta"]),
            "disable_weighted_vote": bool(config["disable_weighted_vote"]),
        },
        "fill_values": fill_values,
        "feature_cols": list(X_all.columns),
        "warm_rows": int(n_warm),
        "validation_rows": int(n_val),
    }


def train_model_with_internal_validation(
    train_df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    target_col: str,
    time_col: Optional[str] = None,
    model_name: str,
    validation_size: float = 0.2,
    folds: int = 5,
    random_state: int = 42,
    fast_mode: bool = True,
    grid_limit: Optional[int] = None,
    custom_grid: Optional[Dict[str, Sequence[Any]]] = None,
    balance_mode: str = BALANCE_MODE_NONE,
    study_id: Optional[str] = None,
    fixed_params: Optional[Dict[str, Any]] = None,
    fixed_tuning_artifact: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
    smote_cache_dir: Optional[Path] = None,
    smote_cache_registry: Optional[Dict[str, Dict[str, Any]]] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    resource_mode: str = DEFAULT_EXPERIMENT_RESOURCE_MODE,
) -> Dict[str, Any]:
    """
    Section 4.5 replication:
    - internal 20% validation split
    - threshold calibration with out-of-fold scores on the full training split
    - final refit on all available training rows
    """
    X_all_raw, y_all = _prepare_xy_raw(train_df, feature_cols, target_col)
    if y_all.nunique() < 2:
        raise ValueError("Training data has a single target class.")
    _emit_experiment_progress(
        progress_callback,
        ratio=0.02,
        label="Preparando entrenamiento batch...",
        detail=f"Modelo {model_name} | filas {len(train_df):,}",
    )
    if (
        str(model_name) == "NNet"
        and fixed_tuning_artifact is not None
        and not bool((fixed_tuning_artifact or {}).get("has_valid_trial", True))
    ):
        raise ModelTrainingSkipped(
            "No converged NNet trials were available in the persisted tuning artifact.",
            metadata={
                "model_name": str(model_name),
                "balance_mode": str(balance_mode),
                "study_id": str((fixed_tuning_artifact or {}).get("study_id", "")),
                "reason": "no_valid_tuning_trial",
            },
        )

    X_tr_raw, X_va_raw, y_tr, y_va = train_test_split(
        X_all_raw,
        y_all,
        test_size=float(validation_size),
        random_state=int(random_state),
        stratify=y_all,
    )

    if fixed_params is not None:
        best_params = {str(key): _normalize_param_choice(value) for key, value in dict(fixed_params).items()}
        tuning_artifact = _to_json_safe(
            {
                **dict(fixed_tuning_artifact or {}),
                "study_id": str((fixed_tuning_artifact or {}).get("study_id") or study_id or ""),
                "model_name": model_name,
                "balance_mode": balance_mode,
                "best_params": best_params,
                "model_params": _split_model_and_smote_params(best_params, balance_mode=balance_mode)[0],
                "smote_params": _split_model_and_smote_params(best_params, balance_mode=balance_mode)[1],
            }
        )
        search_df = _tuning_search_df_from_artifact(
            tuning_artifact,
            model_name=model_name,
            balance_mode=balance_mode,
        )
        _emit_experiment_progress(
            progress_callback,
            ratio=0.70,
            label="Reutilizando hiperparametros persistidos...",
            detail=f"Modelo {model_name} | study {str(tuning_artifact.get('study_id', '') or 'cached')}",
        )
    else:
        best_params, search_df, tuning_artifact = tune_hyperparameters(
            X_tr_raw,
            y_tr,
            model_name=model_name,
            folds=folds,
            random_state=random_state,
            fast_mode=fast_mode,
            grid_limit=grid_limit,
            custom_grid=custom_grid,
            balance_mode=balance_mode,
            study_id=study_id,
            resource_mode=resource_mode,
            progress_callback=(
                None
                if progress_callback is None
                else lambda payload: _emit_experiment_progress(
                    progress_callback,
                    ratio=0.05 + 0.65 * max(0.0, min(float(payload.get("ratio", 0.0)), 1.0)),
                    label=str(payload.get("label", "Sintonizando hiperparametros...")),
                    detail=str(payload.get("detail", "")),
                )
            ),
        )
    _emit_experiment_progress(
        progress_callback,
        ratio=0.75,
        label="Calibrando threshold...",
        detail=f"Modelo {model_name} | validacion interna",
    )

    model_params, smote_params = _split_model_and_smote_params(
        best_params,
        balance_mode=balance_mode,
    )
    raw_calibration_scores = np.asarray(
        _cross_validated_scores(
            X_all_raw,
            y_all,
            model_name=model_name,
            params=best_params,
            folds=folds,
            random_state=random_state,
            balance_mode=balance_mode,
            return_diagnostics=False,
            resource_mode=resource_mode,
        ),
        dtype=float,
    )
    calibration_mask = np.isfinite(raw_calibration_scores)

    if calibration_mask.sum() < 2 or np.unique(y_all.to_numpy()[calibration_mask]).size < 2:
        holdout_fill_values = _fit_fill_values(X_tr_raw)
        X_tr_filled = _apply_fill_values(X_tr_raw, holdout_fill_values)
        X_va_filled = _apply_fill_values(X_va_raw, holdout_fill_values)
        holdout_feature_transform = _fit_feature_transform(X_tr_filled, model_name=model_name)
        X_tr = _apply_feature_transform(X_tr_filled, holdout_feature_transform)
        X_va = _apply_feature_transform(X_va_filled, holdout_feature_transform)
        if balance_mode == BALANCE_MODE_SMOTE:
            X_holdout_fit, y_holdout_fit, _ = _apply_smote_balance(
                X_tr,
                y_tr.to_numpy(),
                sampling_strategy=float(smote_params.get("sampling_strategy", 1.0)),
                k_neighbors=int(smote_params.get("k_neighbors", 5)),
                random_state=int(random_state),
            )
        else:
            X_holdout_fit, y_holdout_fit = X_tr, y_tr.to_numpy()
        calibration_model, calibration_fit_diagnostics, _ = _fit_model_with_diagnostics(
            model_name,
            model_params,
            random_state=random_state,
            n_features=X_tr.shape[1],
            X_fit=X_holdout_fit,
            y_fit=y_holdout_fit,
            allow_retry=False,
            resource_mode=resource_mode,
            balance_mode=balance_mode,
        )
        if str(model_name) == "NNet" and not bool(calibration_fit_diagnostics.get("converged", True)):
            raise ModelTrainingSkipped(
                "NNet threshold calibration did not converge on the internal holdout fallback.",
                metadata={
                    "model_name": str(model_name),
                    "balance_mode": str(balance_mode),
                    "study_id": str((tuning_artifact or {}).get("study_id", study_id or "")),
                    "reason": "calibration_non_converged",
                    "fit_warning_count": int(calibration_fit_diagnostics.get("fit_warning_count", 0)),
                    "n_iter_final": calibration_fit_diagnostics.get("n_iter_final"),
                },
            )
        calibration_y = y_va.to_numpy()
        raw_calibration_scores = _model_scores(calibration_model, X_va)
    else:
        calibration_y = y_all.to_numpy()[calibration_mask]
        raw_calibration_scores = raw_calibration_scores[calibration_mask]

    raw_youden = youden_threshold(calibration_y, raw_calibration_scores)
    calibrated_scores, calibration_metadata, probability_calibrator = _fit_platt_calibrator(
        calibration_y,
        raw_calibration_scores,
    )
    calibrated_youden = youden_threshold(
        calibration_y,
        calibrated_scores,
    )
    calibration_stage_metrics = _compute_calibration_stage_metrics(
        calibration_y,
        raw_calibration_scores,
        raw_threshold=float(raw_youden["threshold"]),
        calibrated_scores=calibrated_scores,
        calibrated_threshold=float(calibrated_youden["threshold"]),
    )
    val_metrics = calibration_stage_metrics["after_calibration"]
    _emit_experiment_progress(
        progress_callback,
        ratio=0.9,
        label="Reentrenando modelo final...",
        detail=f"Modelo {model_name} | refit con datos completos",
    )

    fill_values = _fit_fill_values(X_all_raw)
    X_all_filled = _apply_fill_values(X_all_raw, fill_values)
    feature_transform = _fit_feature_transform(X_all_filled, model_name=model_name)
    X_all = _apply_feature_transform(X_all_filled, feature_transform)
    smote_fit_info: Dict[str, Any] = {
        "applied": False,
        "train_rows": int(len(X_all)),
        "balanced_rows": int(len(X_all)),
    }
    if balance_mode == BALANCE_MODE_SMOTE:
        X_fit, y_fit, smote_fit_info = _load_or_create_smote_artifact(
            run_id=str(run_id or "local_run"),
            train_df=train_df,
            X=X_all,
            y=y_all.to_numpy(),
            feature_cols=list(X_all.columns),
            target_col=target_col,
            time_col=time_col,
            sampling_strategy=float(smote_params.get("sampling_strategy", 1.0)),
            k_neighbors=int(smote_params.get("k_neighbors", 5)),
            random_state=int(random_state),
            smote_cache_dir=smote_cache_dir,
            smote_cache_registry=smote_cache_registry,
        )
    else:
        X_fit, y_fit = X_all, y_all.to_numpy()

    model, fit_diagnostics, final_model_params = _fit_model_with_diagnostics(
        model_name,
        model_params,
        random_state=random_state,
        n_features=X_all.shape[1],
        X_fit=X_fit,
        y_fit=y_fit,
        allow_retry=True,
        resource_mode=resource_mode,
        balance_mode=balance_mode,
    )
    if str(model_name) == "NNet" and not bool(fit_diagnostics.get("converged", True)):
        raise ModelTrainingSkipped(
            "NNet final refit did not converge after retry.",
            metadata={
                "model_name": str(model_name),
                "balance_mode": str(balance_mode),
                "study_id": str((tuning_artifact or {}).get("study_id", study_id or "")),
                "reason": "final_refit_non_converged",
                "fit_warning_count": int(fit_diagnostics.get("fit_warning_count", 0)),
                "n_iter_final": fit_diagnostics.get("n_iter_final"),
                "fit_retry_applied": bool(fit_diagnostics.get("fit_retry_applied", False)),
                "best_params": dict(best_params),
            },
        )
    _emit_experiment_progress(
        progress_callback,
        ratio=1.0,
        label="Entrenamiento batch completo.",
        detail=f"Modelo {model_name} | threshold {float(calibrated_youden['threshold']):.4f}",
    )

    return {
        "model": model,
        "threshold": float(calibrated_youden["threshold"]),
        "operating_threshold": float(calibrated_youden["threshold"]),
        "threshold_policy": DEFAULT_THRESHOLD_POLICY,
        "fn_cost": float(DEFAULT_THRESHOLD_FN_COST),
        "fp_cost": float(DEFAULT_THRESHOLD_FP_COST),
        "raw_youden_threshold": float(raw_youden["threshold"]),
        "youden": raw_youden,
        "raw_youden": raw_youden,
        "calibrated_youden_threshold": float(calibrated_youden["threshold"]),
        "calibrated_youden": calibrated_youden,
        "calibration_model": probability_calibrator,
        "calibration_metadata": calibration_metadata,
        "calibration_method": DEFAULT_CALIBRATION_METHOD,
        "val_metrics": val_metrics,
        "val_metrics_before_calibration": calibration_stage_metrics["before_calibration"],
        "val_metrics_after_calibration": calibration_stage_metrics["after_calibration"],
        "best_params": best_params,
        "model_params": final_model_params,
        "smote_params": smote_params,
        "balance_mode": balance_mode,
        "search_df": search_df,
        "feature_cols": list(X_all_raw.columns),
        "fill_values": fill_values,
        "feature_transform": feature_transform,
        "smote_fit_info": smote_fit_info,
        "converged": bool(fit_diagnostics.get("converged", True)),
        "fit_warning_count": int(fit_diagnostics.get("fit_warning_count", 0)),
        "n_iter_final": fit_diagnostics.get("n_iter_final"),
        "fit_retry_applied": bool(fit_diagnostics.get("fit_retry_applied", False)),
        "tuning_artifact": {
            **tuning_artifact,
            "best_params": best_params,
            "model_params": final_model_params,
            "smote_params": smote_params,
            "calibration_method": DEFAULT_CALIBRATION_METHOD,
            "threshold_policy": DEFAULT_THRESHOLD_POLICY,
            "fn_cost": float(DEFAULT_THRESHOLD_FN_COST),
            "fp_cost": float(DEFAULT_THRESHOLD_FP_COST),
            "operating_threshold": float(calibrated_youden["threshold"]),
            "raw_youden_threshold": float(raw_youden["threshold"]),
            "calibrated_youden_threshold": float(calibrated_youden["threshold"]),
            "calibration_metadata": _to_json_safe(calibration_metadata),
            "val_metrics_before_calibration": calibration_stage_metrics["before_calibration"],
            "val_metrics_after_calibration": calibration_stage_metrics["after_calibration"],
            "smote_fit_info": smote_fit_info,
            "feature_transform": _to_json_safe(feature_transform),
            "converged": bool(fit_diagnostics.get("converged", True)),
            "fit_warning_count": int(fit_diagnostics.get("fit_warning_count", 0)),
            "n_iter_final": fit_diagnostics.get("n_iter_final"),
            "fit_retry_applied": bool(fit_diagnostics.get("fit_retry_applied", False)),
            "has_valid_trial": bool((tuning_artifact or {}).get("has_valid_trial", True)),
        },
    }


def _evaluate_split(
    model,
    test_df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    target_col: str,
    threshold: float,
    raw_threshold: Optional[float] = None,
    fill_values: Optional[Dict[str, float]] = None,
    feature_transform: Optional[Dict[str, Any]] = None,
    calibration_model: Optional[LogisticRegression] = None,
    calibration_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    X_te_raw, y_te = _prepare_xy_raw(test_df, feature_cols, target_col)
    X_te_filled = _apply_fill_values(X_te_raw, fill_values or _fit_fill_values(X_te_raw))
    X_te = _apply_feature_transform(X_te_filled, feature_transform)
    raw_scores = _model_scores(model, X_te)
    calibrated_scores = _apply_probability_calibration(
        raw_scores,
        calibration_model=calibration_model,
        calibration_metadata=calibration_metadata,
    )
    stage_metrics = _compute_calibration_stage_metrics(
        y_te.to_numpy(),
        raw_scores,
        raw_threshold=float(threshold if raw_threshold is None else raw_threshold),
        calibrated_scores=calibrated_scores,
        calibrated_threshold=float(threshold),
    )
    metrics = stage_metrics["after_calibration"]
    return {
        "metrics": metrics,
        "metrics_before_calibration": stage_metrics["before_calibration"],
        "metrics_after_calibration": stage_metrics["after_calibration"],
        "y_true": y_te.to_numpy(),
        "scores": raw_scores,
        "calibrated_scores": calibrated_scores,
    }


def _available_prediction_years(df: pd.DataFrame, time_col: str) -> List[int]:
    if df is None or df.empty or time_col not in df.columns:
        return []
    years = pd.to_datetime(df[time_col], errors="coerce").dt.year.dropna().astype(int).unique()
    return sorted(years.tolist())


def _as_year_mask(df: pd.DataFrame, time_col: str, year: int) -> pd.Series:
    times = pd.to_datetime(df[time_col], errors="coerce")
    return times.dt.year.eq(int(year))


def run_yearly_strategy(
    df: pd.DataFrame,
    *,
    strategy: str,
    feature_cols: Sequence[str],
    target_col: str = "target",
    time_col: str = "interval_start",
    model_names: Optional[Sequence[str]] = None,
    base_year: Optional[int] = None,
    validation_size: float = 0.2,
    folds: int = 5,
    random_state: int = 42,
    fast_mode: bool = True,
    resource_mode: str = DEFAULT_EXPERIMENT_RESOURCE_MODE,
    grid_limit: Optional[int] = None,
    custom_grids: Optional[Dict[str, Dict[str, Sequence[Any]]]] = None,
    run_seed: Optional[int] = None,
    run_order: int = 1,
    execution_log: Optional[List[Dict[str, Any]]] = None,
    balance_mode: str = BALANCE_MODE_NONE,
    study_collector: Optional[List[Dict[str, Any]]] = None,
    fixed_params: Optional[Dict[str, Any]] = None,
    fixed_tuning_artifact: Optional[Dict[str, Any]] = None,
    tuning_resolver: Optional[Callable[..., Dict[str, Any]]] = None,
    run_id: Optional[str] = None,
    smote_cache_dir: Optional[Path] = None,
    smote_cache_registry: Optional[Dict[str, Dict[str, Any]]] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Runs static / period-aligned / cumulative yearly experiments.
    """
    if model_names is None:
        model_names = MODEL_NAMES

    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.dropna(subset=[time_col])
    years = _available_prediction_years(work, time_col)
    if len(years) < 2:
        return pd.DataFrame(), []

    if base_year is None:
        base_year = years[0]

    pred_years = [y for y in years if y > int(base_year)]

    effective_seed = int(random_state if run_seed is None else run_seed)
    rows: List[Dict[str, Any]] = []
    roc_payload: List[Dict[str, Any]] = []
    training_bundle_cache: Dict[Tuple[Any, ...], Tuple[Dict[str, Any], float]] = {}
    total_splits = max(1, len(model_names) * len(pred_years))
    split_counter = 0

    for model_name in model_names:
        for pred_year in pred_years:
            split_index = split_counter
            split_counter += 1
            split_start_ratio = float(split_index) / float(total_splits)
            split_end_ratio = float(split_index + 1) / float(total_splits)
            split_span = split_end_ratio - split_start_ratio
            split_detail_prefix = (
                f"Estrategia {strategy} | Modelo {model_name} | "
                f"Split {split_index + 1} / {total_splits} | Ano prediccion {int(pred_year)}"
            )
            _emit_experiment_progress(
                progress_callback,
                ratio=split_start_ratio,
                label="Preparando split anual...",
                detail=split_detail_prefix,
            )
            if strategy == "static":
                train_mask = _as_year_mask(work, time_col, int(base_year))
                train_label = f"{int(base_year)}"
            elif strategy == "period_aligned":
                train_mask = _as_year_mask(work, time_col, int(pred_year) - 1)
                train_label = f"{int(pred_year) - 1}"
            elif strategy == "cumulative":
                train_mask = pd.to_datetime(work[time_col], errors="coerce").dt.year.le(int(pred_year) - 1)
                train_label = f"[<= {int(pred_year) - 1}]"
            else:
                raise ValueError(f"Unsupported yearly strategy: {strategy}")

            test_mask = _as_year_mask(work, time_col, int(pred_year))
            train_df = work.loc[train_mask].copy()
            test_df = work.loc[test_mask].copy()

            if train_df.empty or test_df.empty:
                _append_execution_log(
                    execution_log,
                    phase="yearly_split_skip",
                    status="skipped",
                    message="Training or test split is empty.",
                    strategy=strategy,
                    model=model_name,
                    prediction_year=int(pred_year),
                    training_year=train_label,
                    run_seed=effective_seed,
                    run_order=run_order,
                    balance_mode=balance_mode,
                    n_train=int(len(train_df)),
                    n_test=int(len(test_df)),
                )
                continue
            if pd.to_numeric(train_df[target_col], errors="coerce").fillna(0).astype(int).nunique() < 2:
                _append_execution_log(
                    execution_log,
                    phase="yearly_split_skip",
                    status="skipped",
                    message="Training split contains a single target class.",
                    strategy=strategy,
                    model=model_name,
                    prediction_year=int(pred_year),
                    training_year=train_label,
                    run_seed=effective_seed,
                    run_order=run_order,
                    balance_mode=balance_mode,
                    n_train=int(len(train_df)),
                    n_test=int(len(test_df)),
                )
                continue
            if pd.to_numeric(test_df[target_col], errors="coerce").fillna(0).astype(int).nunique() < 2:
                _append_execution_log(
                    execution_log,
                    phase="yearly_split_skip",
                    status="skipped",
                    message="Test split contains a single target class.",
                    strategy=strategy,
                    model=model_name,
                    prediction_year=int(pred_year),
                    training_year=train_label,
                    run_seed=effective_seed,
                    run_order=run_order,
                    balance_mode=balance_mode,
                    n_train=int(len(train_df)),
                    n_test=int(len(test_df)),
                )
                continue

            study_id = _build_tuning_study_id(
                stage="yearly_train",
                strategy=strategy,
                model_name=model_name,
                balance_mode=balance_mode,
                run_seed=effective_seed,
                run_order=run_order,
                prediction_year=int(pred_year),
                training_year=train_label,
            )
            _append_execution_log(
                execution_log,
                phase="yearly_train_start",
                status="started",
                message="Training model for yearly strategy split.",
                strategy=strategy,
                model=model_name,
                prediction_year=int(pred_year),
                training_year=train_label,
                run_seed=effective_seed,
                run_order=run_order,
                balance_mode=balance_mode,
                study_id=study_id,
                n_train=int(len(train_df)),
                n_test=int(len(test_df)),
            )
            cache_key: Optional[Tuple[Any, ...]] = None
            resolved_fixed_params = fixed_params
            resolved_fixed_tuning_artifact = fixed_tuning_artifact
            if tuning_resolver is not None:
                resolved_fixed_tuning_artifact = tuning_resolver(
                    train_df=train_df,
                    model_name=model_name,
                    balance_mode=balance_mode,
                    strategy_context={
                        "stage": "yearly_train",
                        "strategy": strategy,
                        "window_kind": strategy,
                        "training_year": train_label,
                        "prediction_year": int(pred_year),
                        "run_seed": int(effective_seed),
                        "run_order": int(run_order),
                    },
                )
                resolved_fixed_params = dict((resolved_fixed_tuning_artifact or {}).get("best_params") or {})
            if strategy == "static":
                cache_key = (
                    strategy,
                    model_name,
                    balance_mode,
                    int(base_year),
                    int(effective_seed),
                        float(validation_size),
                        int(folds),
                        bool(fast_mode),
                        str(_normalize_experiment_resource_mode(resource_mode)),
                        None if grid_limit is None else int(grid_limit),
                        json.dumps((custom_grids or {}).get(model_name), sort_keys=True, default=str, ensure_ascii=True),
                        str((resolved_fixed_tuning_artifact or {}).get("tuning_key", "")),
                    )

            cached_entry = training_bundle_cache.get(cache_key) if cache_key is not None else None
            if cached_entry is not None:
                bundle, train_time = cached_entry
                _append_execution_log(
                    execution_log,
                    phase="yearly_train_reuse",
                    status="ok",
                    message="Reused cached training bundle for identical static training split.",
                    strategy=strategy,
                    model=model_name,
                    prediction_year=int(pred_year),
                    training_year=train_label,
                    run_seed=effective_seed,
                    run_order=run_order,
                    balance_mode=balance_mode,
                    study_id=study_id,
                    n_train=int(len(train_df)),
                    n_test=int(len(test_df)),
                    cached_training_time_sec=float(train_time),
                )
            else:
                t0 = time.perf_counter()
                try:
                    bundle = train_model_with_internal_validation(
                        train_df,
                        feature_cols=feature_cols,
                        target_col=target_col,
                        time_col=time_col,
                        model_name=model_name,
                        validation_size=validation_size,
                        folds=folds,
                        random_state=effective_seed,
                        fast_mode=fast_mode,
                        resource_mode=resource_mode,
                        grid_limit=grid_limit,
                        custom_grid=(custom_grids or {}).get(model_name),
                        balance_mode=balance_mode,
                        study_id=study_id,
                        fixed_params=resolved_fixed_params,
                        fixed_tuning_artifact=resolved_fixed_tuning_artifact,
                        run_id=run_id,
                        smote_cache_dir=smote_cache_dir,
                        smote_cache_registry=smote_cache_registry,
                        progress_callback=(
                            None
                            if progress_callback is None
                            else lambda payload, _start=split_start_ratio, _span=split_span, _detail=split_detail_prefix: _emit_experiment_progress(
                                progress_callback,
                                ratio=_start + _span * max(0.0, min(float(payload.get("ratio", 0.0)), 1.0)) * 0.9,
                                label=str(payload.get("label", "Entrenando split anual...")),
                                detail=f"{_detail} | {str(payload.get('detail', '') or '')}".strip(" |"),
                            )
                        ),
                    )
                except ModelTrainingSkipped as exc:
                    skip_metadata = dict(getattr(exc, "metadata", {}) or {})
                    skip_phase = "nnet_final_refit_skipped"
                    if str(skip_metadata.get("reason", "")) == "no_valid_tuning_trial":
                        skip_phase = "nnet_trial_invalid"
                    _append_execution_log(
                        execution_log,
                        phase=skip_phase,
                        status="skipped",
                        message="Skipped yearly strategy split because NNet did not produce a stable fitted model.",
                        strategy=strategy,
                        model=model_name,
                        prediction_year=int(pred_year),
                        training_year=train_label,
                        run_seed=effective_seed,
                        run_order=run_order,
                        balance_mode=balance_mode,
                        study_id=study_id,
                        n_train=int(len(train_df)),
                        n_test=int(len(test_df)),
                        **skip_metadata,
                    )
                    continue
                except Exception as exc:
                    _append_execution_log(
                        execution_log,
                        phase="yearly_train_error",
                        status="error",
                        message="Model training failed for yearly strategy split.",
                        strategy=strategy,
                        model=model_name,
                        prediction_year=int(pred_year),
                        training_year=train_label,
                        run_seed=effective_seed,
                        run_order=run_order,
                        balance_mode=balance_mode,
                        study_id=study_id,
                        n_train=int(len(train_df)),
                        n_test=int(len(test_df)),
                        error=str(exc),
                    )
                    continue
                train_time = time.perf_counter() - t0
                if cache_key is not None:
                    training_bundle_cache[cache_key] = (bundle, float(train_time))
                if str(model_name) == "NNet" and bool(bundle.get("fit_retry_applied", False)):
                    _append_execution_log(
                        execution_log,
                        phase="nnet_final_refit_retry",
                        status="warning",
                        message="NNet final refit converged only after increasing max_iter.",
                        strategy=strategy,
                        model=model_name,
                        prediction_year=int(pred_year),
                        training_year=train_label,
                        run_seed=effective_seed,
                        run_order=run_order,
                        balance_mode=balance_mode,
                        study_id=study_id,
                        n_iter_final=bundle.get("n_iter_final"),
                        fit_warning_count=int(bundle.get("fit_warning_count", 0)),
                    )
            _emit_experiment_progress(
                progress_callback,
                ratio=split_start_ratio + split_span * 0.95,
                label="Evaluando split anual...",
                detail=split_detail_prefix,
            )

            eval_payload = _evaluate_split(
                bundle["model"],
                test_df,
                feature_cols=feature_cols,
                target_col=target_col,
                threshold=float(bundle["threshold"]),
                raw_threshold=float(bundle.get("raw_youden_threshold", bundle["threshold"])),
                fill_values=bundle.get("fill_values"),
                feature_transform=bundle.get("feature_transform"),
                calibration_model=bundle.get("calibration_model"),
                calibration_metadata=bundle.get("calibration_metadata"),
            )

            _append_tuning_study(
                study_collector,
                bundle.get("tuning_artifact"),
                stage="yearly_train",
                strategy=strategy,
                model=model_name,
                balance_mode=balance_mode,
                run_seed=effective_seed,
                run_order=int(run_order),
                prediction_year=int(pred_year),
                training_year=train_label,
                n_train=int(len(train_df)),
                n_test=int(len(test_df)),
            )

            metrics = eval_payload["metrics"]
            metrics_before_calibration = eval_payload["metrics_before_calibration"]
            metrics_after_calibration = eval_payload["metrics_after_calibration"]
            operating_context = _bundle_operating_context(bundle)
            row = {
                "strategy": strategy,
                "iteration": int(pred_year - years[0]),
                "training_year": train_label,
                "prediction_year": int(pred_year),
                "model": model_name,
                "balance_mode": balance_mode,
                "auc": float(metrics["auc"]),
                "pr_auc": float(metrics["pr_auc"]),
                "f1": float(metrics["f1"]),
                "sensitivity": float(metrics["sensitivity"]),
                "specificity": float(metrics["specificity"]),
                "error_rate": float(metrics["error_rate"]),
                "threshold": float(operating_context["threshold"]),
                "operating_threshold": float(operating_context["operating_threshold"]),
                "raw_youden_threshold": float(operating_context["raw_youden_threshold"]),
                "threshold_policy": str(operating_context["threshold_policy"]),
                "calibration_method": str(operating_context["calibration_method"]),
                "fn_cost": float(operating_context["fn_cost"]),
                "fp_cost": float(operating_context["fp_cost"]),
                "training_time_sec": float(train_time),
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "best_params": json.dumps(bundle["best_params"], sort_keys=True),
                "run_seed": effective_seed,
                "run_order": int(run_order),
            }
            row.update(_calibration_rate_fields(metrics_before_calibration, metrics_after_calibration))
            rows.append(row)

            _append_execution_log(
                execution_log,
                phase="yearly_eval_complete",
                status="ok",
                message="Finished yearly strategy training and evaluation.",
                strategy=strategy,
                model=model_name,
                prediction_year=int(pred_year),
                training_year=train_label,
                run_seed=effective_seed,
                run_order=run_order,
                balance_mode=balance_mode,
                study_id=study_id,
                threshold=float(operating_context["threshold"]),
                operating_threshold=float(operating_context["operating_threshold"]),
                raw_youden_threshold=float(operating_context["raw_youden_threshold"]),
                threshold_policy=str(operating_context["threshold_policy"]),
                calibration_method=str(operating_context["calibration_method"]),
                auc=float(metrics["auc"]),
                sensitivity=float(metrics["sensitivity"]),
                specificity=float(metrics["specificity"]),
                sensitivity_before_calibration=float(metrics_before_calibration["sensitivity"]),
                specificity_before_calibration=float(metrics_before_calibration["specificity"]),
                sensitivity_after_calibration=float(metrics_after_calibration["sensitivity"]),
                specificity_after_calibration=float(metrics_after_calibration["specificity"]),
                error_rate=float(metrics["error_rate"]),
                best_params=bundle["best_params"],
                n_train=int(len(train_df)),
                n_test=int(len(test_df)),
                training_time_sec=float(train_time),
            )

            roc_payload.append(
                {
                    "strategy": strategy,
                    "model": model_name,
                    "balance_mode": balance_mode,
                    "segment": str(pred_year),
                    "y_true": eval_payload["y_true"],
                    "scores": eval_payload["scores"],
                    "calibrated_scores": eval_payload["calibrated_scores"],
                    "raw_threshold": float(bundle.get("raw_youden_threshold", bundle["threshold"])),
                    "calibrated_threshold": float(bundle["threshold"]),
                    "run_seed": effective_seed,
                    "run_order": int(run_order),
                }
            )
            _emit_experiment_progress(
                progress_callback,
                ratio=split_end_ratio,
                label="Split anual completado.",
                detail=split_detail_prefix,
            )

    return pd.DataFrame(rows), roc_payload


class SimpleADWIN:
    """
    Lightweight ADWIN detector for binary error streams.
    """

    def __init__(
        self,
        *,
        delta: float = 0.002,
        min_window: int = 45_000,
        min_subwindow: int = 32,
        max_splits: int = 128,
    ) -> None:
        self.delta = float(delta)
        self.min_window = int(min_window)
        self.min_subwindow = int(min_subwindow)
        self.max_splits = int(max_splits)
        self.window: deque[float] = deque()

    def _hoeffding_cut_threshold(self, n0: int, n1: int, n: int) -> float:
        """
        Canonical ADWIN Hoeffding cut from Bifet and Gavaldà (2007).
        """
        m = 1.0 / ((1.0 / n0) + (1.0 / n1))
        delta_prime = max(self.delta, 1e-9) / max(1.0, float(n))
        return math.sqrt((1.0 / (2.0 * m)) * math.log(4.0 / delta_prime))

    def __len__(self) -> int:
        return len(self.window)

    def update(self, value: float) -> Tuple[bool, Optional[Dict[str, float]]]:
        self.window.append(float(value))
        n = len(self.window)

        if n < max(self.min_window, 2 * self.min_subwindow):
            return False, None

        arr = np.fromiter(self.window, dtype=float, count=n)
        prefix = np.cumsum(arr)

        step = max(1, n // max(1, self.max_splits))
        best: Optional[Dict[str, float]] = None

        for cut in range(self.min_subwindow, n - self.min_subwindow + 1, step):
            n0 = cut
            n1 = n - cut
            sum0 = prefix[cut - 1]
            sum1 = prefix[-1] - sum0
            mu0 = float(sum0 / n0)
            mu1 = float(sum1 / n1)
            gap = abs(mu0 - mu1)

            eps = self._hoeffding_cut_threshold(n0, n1, n)

            if gap > eps:
                candidate = {
                    "cut": float(cut),
                    "gap": float(gap),
                    "eps": float(eps),
                    "n": float(n),
                }
                if best is None or candidate["gap"] > best["gap"]:
                    best = candidate

        if best is None:
            return False, None

        n0 = int(best["cut"])
        n1 = n - n0
        for _ in range(n0):
            self.window.popleft()

        return True, {
            "n": float(n),
            "n0": float(n0),
            "n1": float(n1),
            "gap": float(best["gap"]),
            "eps": float(best["eps"]),
        }


def run_adaptive_strategy(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    target_col: str = "target",
    time_col: str = "interval_start",
    model_names: Optional[Sequence[str]] = None,
    base_year: Optional[int] = None,
    random_state: int = 42,
    validation_size: float = 0.2,
    folds: int = 5,
    fast_mode: bool = True,
    resource_mode: str = DEFAULT_EXPERIMENT_RESOURCE_MODE,
    grid_limit: Optional[int] = None,
    adwin_delta: float = 0.002,
    min_window: int = 45_000,
    min_retrain_size: Optional[int] = None,
    custom_grids: Optional[Dict[str, Dict[str, Sequence[Any]]]] = None,
    run_seed: Optional[int] = None,
    run_order: int = 1,
    execution_log: Optional[List[Dict[str, Any]]] = None,
    balance_mode: str = BALANCE_MODE_NONE,
    study_collector: Optional[List[Dict[str, Any]]] = None,
    fixed_params: Optional[Dict[str, Any]] = None,
    fixed_tuning_artifact: Optional[Dict[str, Any]] = None,
    tuning_resolver: Optional[Callable[..., Dict[str, Any]]] = None,
    run_id: Optional[str] = None,
    smote_cache_dir: Optional[Path] = None,
    smote_cache_registry: Optional[Dict[str, Dict[str, Any]]] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Runs Section 4.7 adaptive retraining with ADWIN drift detection.
    """
    if model_names is None:
        model_names = MODEL_NAMES

    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    years = _available_prediction_years(work, time_col)
    if len(years) < 2:
        return pd.DataFrame(), []

    if base_year is None:
        base_year = years[0]

    base_train = work.loc[_as_year_mask(work, time_col, int(base_year))].copy()
    stream = work.loc[pd.to_datetime(work[time_col], errors="coerce").dt.year.gt(int(base_year))].copy()

    if base_train.empty or stream.empty:
        return pd.DataFrame(), []

    effective_seed = int(random_state if run_seed is None else run_seed)
    effective_min_retrain_size = max(20, int(min_window) // 10)
    if min_retrain_size is not None:
        effective_min_retrain_size = max(2, int(min_retrain_size))
    rows: List[Dict[str, Any]] = []
    roc_payload: List[Dict[str, Any]] = []

    total_models = max(1, len(model_names))
    for model_idx, model_name in enumerate(model_names):
        model_start_ratio = float(model_idx) / float(total_models)
        model_end_ratio = float(model_idx + 1) / float(total_models)
        model_span = model_end_ratio - model_start_ratio
        model_detail_prefix = f"Estrategia adaptive_adwin | Modelo {model_name} | {model_idx + 1} / {total_models}"
        _emit_experiment_progress(
            progress_callback,
            ratio=model_start_ratio,
            label="Preparando bloque adaptive_adwin...",
            detail=model_detail_prefix,
        )
        if pd.to_numeric(base_train[target_col], errors="coerce").fillna(0).astype(int).nunique() < 2:
            _append_execution_log(
                execution_log,
                phase="adaptive_base_skip",
                status="skipped",
                message="Base training split contains a single target class.",
                strategy="adaptive_adwin",
                model=model_name,
                training_year=int(base_year),
                run_seed=effective_seed,
                run_order=run_order,
                balance_mode=balance_mode,
                n_train=int(len(base_train)),
                n_stream=int(len(stream)),
            )
            continue

        base_study_id = _build_tuning_study_id(
            stage="adaptive_base",
            strategy=ADAPTIVE_ADWIN_STRATEGY,
            model_name=model_name,
            balance_mode=balance_mode,
            run_seed=effective_seed,
            run_order=run_order,
            training_year=str(int(base_year)),
        )
        try:
            t0 = time.perf_counter()
            resolved_base_tuning_artifact = fixed_tuning_artifact
            resolved_base_params = fixed_params
            if tuning_resolver is not None:
                resolved_base_tuning_artifact = tuning_resolver(
                    train_df=base_train,
                    model_name=model_name,
                    balance_mode=balance_mode,
                    strategy_context={
                        "stage": "adaptive_base",
                        "strategy": ADAPTIVE_ADWIN_STRATEGY,
                        "window_kind": "base_year",
                        "training_year": str(int(base_year)),
                        "run_seed": int(effective_seed),
                        "run_order": int(run_order),
                    },
                )
                resolved_base_params = dict((resolved_base_tuning_artifact or {}).get("best_params") or {})
            bundle = train_model_with_internal_validation(
                base_train,
                feature_cols=feature_cols,
                target_col=target_col,
                time_col=time_col,
                model_name=model_name,
                validation_size=validation_size,
                folds=folds,
                random_state=effective_seed,
                fast_mode=fast_mode,
                resource_mode=resource_mode,
                grid_limit=grid_limit,
                custom_grid=(custom_grids or {}).get(model_name),
                balance_mode=balance_mode,
                study_id=base_study_id,
                fixed_params=resolved_base_params,
                fixed_tuning_artifact=resolved_base_tuning_artifact,
                run_id=run_id,
                smote_cache_dir=smote_cache_dir,
                smote_cache_registry=smote_cache_registry,
                progress_callback=(
                    None
                    if progress_callback is None
                    else lambda payload, _start=model_start_ratio, _span=model_span, _detail=model_detail_prefix: _emit_experiment_progress(
                        progress_callback,
                        ratio=_start + _span * (0.05 + 0.30 * max(0.0, min(float(payload.get("ratio", 0.0)), 1.0))),
                        label=str(payload.get("label", "Entrenando base adaptive_adwin...")),
                        detail=f"{_detail} | {str(payload.get('detail', '') or '')}".strip(" |"),
                    )
                ),
            )
            fit_time = time.perf_counter() - t0
            if str(model_name) == "NNet" and bool(bundle.get("fit_retry_applied", False)):
                _append_execution_log(
                    execution_log,
                    phase="nnet_final_refit_retry",
                    status="warning",
                    message="NNet base adaptive model converged only after increasing max_iter.",
                    strategy="adaptive_adwin",
                    model=model_name,
                    training_year=int(base_year),
                    run_seed=effective_seed,
                    run_order=run_order,
                    balance_mode=balance_mode,
                    study_id=base_study_id,
                    n_iter_final=bundle.get("n_iter_final"),
                    fit_warning_count=int(bundle.get("fit_warning_count", 0)),
                )
            _append_execution_log(
                execution_log,
                phase="adaptive_base_train_complete",
                status="ok",
                message="Finished base training for adaptive strategy.",
                strategy="adaptive_adwin",
                model=model_name,
                training_year=int(base_year),
                run_seed=effective_seed,
                run_order=run_order,
                balance_mode=balance_mode,
                study_id=base_study_id,
                threshold=float(bundle["threshold"]),
                best_params=bundle["best_params"],
                n_train=int(len(base_train)),
                n_stream=int(len(stream)),
                training_time_sec=float(fit_time),
            )
        except ModelTrainingSkipped as exc:
            skip_metadata = dict(getattr(exc, "metadata", {}) or {})
            skip_phase = "nnet_final_refit_skipped"
            if str(skip_metadata.get("reason", "")) == "no_valid_tuning_trial":
                skip_phase = "nnet_trial_invalid"
            _append_execution_log(
                execution_log,
                phase=skip_phase,
                status="skipped",
                message="Skipped adaptive ADWIN block because NNet did not produce a stable base model.",
                strategy="adaptive_adwin",
                model=model_name,
                training_year=int(base_year),
                run_seed=effective_seed,
                run_order=run_order,
                balance_mode=balance_mode,
                study_id=base_study_id,
                n_train=int(len(base_train)),
                n_stream=int(len(stream)),
                **skip_metadata,
            )
            continue
        except Exception as exc:
            _append_execution_log(
                execution_log,
                phase="adaptive_base_train_error",
                status="error",
                message="Base training failed for adaptive strategy.",
                strategy="adaptive_adwin",
                model=model_name,
                training_year=int(base_year),
                run_seed=effective_seed,
                run_order=run_order,
                balance_mode=balance_mode,
                study_id=base_study_id,
                n_train=int(len(base_train)),
                n_stream=int(len(stream)),
                error=str(exc),
            )
            continue

        _append_tuning_study(
            study_collector,
            bundle.get("tuning_artifact"),
            stage="adaptive_base",
            strategy=ADAPTIVE_ADWIN_STRATEGY,
            model=model_name,
            balance_mode=balance_mode,
            run_seed=effective_seed,
            run_order=int(run_order),
            training_year=int(base_year),
            n_train=int(len(base_train)),
            n_stream=int(len(stream)),
        )

        model = bundle["model"]
        operating_context = _bundle_operating_context(bundle)
        threshold = float(operating_context["threshold"])
        raw_youden_threshold = float(operating_context["raw_youden_threshold"])
        threshold_policy = str(operating_context["threshold_policy"])
        calibration_method = str(operating_context["calibration_method"])
        fn_cost = float(operating_context["fn_cost"])
        fp_cost = float(operating_context["fp_cost"])
        calibration_model = operating_context.get("calibration_model")
        calibration_metadata = operating_context.get("calibration_metadata")
        fill_values = dict(bundle.get("fill_values") or {})
        feature_transform = dict(bundle.get("feature_transform") or {}) or None
        detector = SimpleADWIN(delta=adwin_delta, min_window=min_window)

        X_stream_raw, y_stream = _prepare_xy_raw(stream, feature_cols, target_col)
        ts_stream = pd.to_datetime(stream[time_col], errors="coerce").reset_index(drop=True)
        stream_progress_step = max(1, len(stream) // 50)

        segment_y: List[int] = []
        segment_raw_scores: List[float] = []
        segment_decision_scores: List[float] = []
        training_window_rows: deque[Dict[str, Any]] = deque()
        drift_idx = 0

        for i in range(len(stream)):
            if i == 0 or (i + 1) % stream_progress_step == 0 or i == len(stream) - 1:
                _emit_experiment_progress(
                    progress_callback,
                    ratio=model_start_ratio + model_span * (0.40 + 0.55 * float(i + 1) / float(max(1, len(stream)))),
                    label="Evaluando stream adaptive_adwin...",
                    detail=f"{model_detail_prefix} | fila {i + 1:,} / {len(stream):,}",
                )
            x_i = _apply_fill_values(X_stream_raw.iloc[[i]], fill_values)
            x_i_model = _apply_feature_transform(x_i, feature_transform)
            y_i = int(y_stream.iloc[i])
            ts_i = ts_stream.iloc[i]

            raw_score_i = float(_model_scores(model, x_i_model)[0])
            decision_score_i = float(
                _apply_probability_calibration(
                    np.asarray([raw_score_i], dtype=float),
                    calibration_model=calibration_model,
                    calibration_metadata=calibration_metadata,
                )[0]
            )
            pred_i = int(decision_score_i >= threshold)
            err_i = float(int(pred_i != y_i))

            segment_y.append(y_i)
            segment_raw_scores.append(raw_score_i)
            segment_decision_scores.append(decision_score_i)
            training_window_rows.append(stream.iloc[i].to_dict())

            is_drift, info = detector.update(err_i)
            if not is_drift or info is None:
                continue

            drift_idx += 1
            stage_metrics = _compute_calibration_stage_metrics(
                np.asarray(segment_y),
                np.asarray(segment_raw_scores),
                raw_threshold=raw_youden_threshold,
                calibrated_scores=np.asarray(segment_decision_scores),
                calibrated_threshold=threshold,
            )
            seg_metrics = stage_metrics["after_calibration"]
            seg_metrics_before = stage_metrics["before_calibration"]
            seg_metrics_after = stage_metrics["after_calibration"]

            remaining = int(len(stream) - (i + 1))
            row = {
                "strategy": "adaptive_adwin",
                "drift": drift_idx,
                "drift_date": pd.Timestamp(ts_i),
                "balance_mode": balance_mode,
                "W": int(info["n"]),
                "W0": int(info["n0"]),
                "W1": int(info["n1"]),
                "remaining_periods": remaining,
                "model": model_name,
                "auc": float(seg_metrics["auc"]),
                "pr_auc": float(seg_metrics["pr_auc"]),
                "f1": float(seg_metrics["f1"]),
                "sensitivity": float(seg_metrics["sensitivity"]),
                "specificity": float(seg_metrics["specificity"]),
                "error_rate": float(seg_metrics["error_rate"]),
                "training_time_sec": float(fit_time),
                "threshold": float(threshold),
                "operating_threshold": float(threshold),
                "raw_youden_threshold": float(raw_youden_threshold),
                "threshold_policy": str(threshold_policy),
                "calibration_method": str(calibration_method),
                "fn_cost": float(fn_cost),
                "fp_cost": float(fp_cost),
                "run_seed": effective_seed,
                "run_order": int(run_order),
            }
            row.update(_calibration_rate_fields(seg_metrics_before, seg_metrics_after))
            rows.append(row)

            _append_execution_log(
                execution_log,
                phase="adaptive_drift_detected",
                status="ok",
                message="ADWIN detected drift and closed a prediction segment.",
                strategy="adaptive_adwin",
                model=model_name,
                drift=int(drift_idx),
                drift_date=pd.Timestamp(ts_i),
                run_seed=effective_seed,
                run_order=run_order,
                balance_mode=balance_mode,
                threshold=float(threshold),
                operating_threshold=float(threshold),
                raw_youden_threshold=float(raw_youden_threshold),
                threshold_policy=str(threshold_policy),
                calibration_method=str(calibration_method),
                W=int(info["n"]),
                W0=int(info["n0"]),
                W1=int(info["n1"]),
                remaining_periods=remaining,
                auc=float(seg_metrics["auc"]),
                sensitivity=float(seg_metrics["sensitivity"]),
                specificity=float(seg_metrics["specificity"]),
                sensitivity_before_calibration=float(seg_metrics_before["sensitivity"]),
                specificity_before_calibration=float(seg_metrics_before["specificity"]),
                sensitivity_after_calibration=float(seg_metrics_after["sensitivity"]),
                specificity_after_calibration=float(seg_metrics_after["specificity"]),
                error_rate=float(seg_metrics["error_rate"]),
            )

            roc_payload.append(
                {
                    "strategy": "adaptive_adwin",
                    "model": model_name,
                    "balance_mode": balance_mode,
                    "segment": f"drift_{drift_idx}",
                    "y_true": np.asarray(segment_y),
                    "scores": np.asarray(segment_raw_scores),
                    "calibrated_scores": np.asarray(segment_decision_scores),
                    "raw_threshold": float(raw_youden_threshold),
                    "calibrated_threshold": float(threshold),
                    "run_seed": effective_seed,
                    "run_order": int(run_order),
                }
            )

            while len(training_window_rows) > int(info["n1"]):
                training_window_rows.popleft()

            retrain_df = pd.DataFrame(list(training_window_rows))
            if (
                len(retrain_df) >= int(effective_min_retrain_size)
                and target_col in retrain_df.columns
                and pd.to_numeric(retrain_df[target_col], errors="coerce").fillna(0).astype(int).nunique() >= 2
            ):
                retrain_study_id = _build_tuning_study_id(
                    stage="adaptive_retrain",
                    strategy=ADAPTIVE_ADWIN_STRATEGY,
                    model_name=model_name,
                    balance_mode=balance_mode,
                    run_seed=effective_seed,
                    run_order=run_order,
                    drift=int(drift_idx),
                )
                try:
                    t_fit = time.perf_counter()
                    resolved_retrain_tuning_artifact = fixed_tuning_artifact
                    resolved_retrain_params = fixed_params
                    if tuning_resolver is not None:
                        resolved_retrain_tuning_artifact = tuning_resolver(
                            train_df=retrain_df,
                            model_name=model_name,
                            balance_mode=balance_mode,
                            strategy_context={
                                "stage": "adaptive_retrain",
                                "strategy": ADAPTIVE_ADWIN_STRATEGY,
                                "window_kind": "adaptive_retrain",
                                "drift": int(drift_idx),
                                "drift_date": pd.Timestamp(ts_i),
                                "retrain_rows": int(len(retrain_df)),
                                "training_years": sorted(
                                    pd.to_datetime(retrain_df[time_col], errors="coerce")
                                    .dt.year.dropna()
                                    .astype(int)
                                    .unique()
                                    .tolist()
                                ),
                                "run_seed": int(effective_seed),
                                "run_order": int(run_order),
                            },
                        )
                        resolved_retrain_params = dict((resolved_retrain_tuning_artifact or {}).get("best_params") or {})
                    bundle = train_model_with_internal_validation(
                        retrain_df,
                        feature_cols=feature_cols,
                        target_col=target_col,
                        time_col=time_col,
                        model_name=model_name,
                        validation_size=validation_size,
                        folds=folds,
                        random_state=effective_seed,
                        fast_mode=fast_mode,
                        resource_mode=resource_mode,
                        grid_limit=grid_limit,
                        custom_grid=(custom_grids or {}).get(model_name),
                        balance_mode=balance_mode,
                        study_id=retrain_study_id,
                        fixed_params=resolved_retrain_params,
                        fixed_tuning_artifact=resolved_retrain_tuning_artifact,
                        run_id=run_id,
                        smote_cache_dir=smote_cache_dir,
                        smote_cache_registry=smote_cache_registry,
                    )
                    fit_time = time.perf_counter() - t_fit
                    if str(model_name) == "NNet" and bool(bundle.get("fit_retry_applied", False)):
                        _append_execution_log(
                            execution_log,
                            phase="nnet_final_refit_retry",
                            status="warning",
                            message="NNet adaptive retraining converged only after increasing max_iter.",
                            strategy="adaptive_adwin",
                            model=model_name,
                            drift=int(drift_idx),
                            run_seed=effective_seed,
                            run_order=run_order,
                            balance_mode=balance_mode,
                            study_id=retrain_study_id,
                            n_iter_final=bundle.get("n_iter_final"),
                            fit_warning_count=int(bundle.get("fit_warning_count", 0)),
                        )
                    model = bundle["model"]
                    operating_context = _bundle_operating_context(bundle)
                    threshold = float(operating_context["threshold"])
                    raw_youden_threshold = float(operating_context["raw_youden_threshold"])
                    threshold_policy = str(operating_context["threshold_policy"])
                    calibration_method = str(operating_context["calibration_method"])
                    fn_cost = float(operating_context["fn_cost"])
                    fp_cost = float(operating_context["fp_cost"])
                    calibration_model = operating_context.get("calibration_model")
                    calibration_metadata = operating_context.get("calibration_metadata")
                    fill_values = dict(bundle.get("fill_values") or {})
                    feature_transform = dict(bundle.get("feature_transform") or {}) or None
                    _append_tuning_study(
                        study_collector,
                        bundle.get("tuning_artifact"),
                        stage="adaptive_retrain",
                        strategy=ADAPTIVE_ADWIN_STRATEGY,
                        model=model_name,
                        balance_mode=balance_mode,
                        run_seed=effective_seed,
                        run_order=int(run_order),
                        drift=int(drift_idx),
                        retrain_rows=int(len(retrain_df)),
                    )
                    _append_execution_log(
                        execution_log,
                        phase="adaptive_retrain_complete",
                        status="ok",
                        message="Adaptive retraining completed using ADWIN window W1.",
                        strategy="adaptive_adwin",
                        model=model_name,
                        drift=int(drift_idx),
                        run_seed=effective_seed,
                        run_order=run_order,
                        balance_mode=balance_mode,
                        study_id=retrain_study_id,
                        threshold=float(threshold),
                        operating_threshold=float(threshold),
                        raw_youden_threshold=float(raw_youden_threshold),
                        threshold_policy=str(threshold_policy),
                        calibration_method=str(calibration_method),
                        best_params=bundle["best_params"],
                        retrain_rows=int(len(retrain_df)),
                        training_time_sec=float(fit_time),
                        min_retrain_size=int(effective_min_retrain_size),
                    )
                except ModelTrainingSkipped as exc:
                    skip_metadata = dict(getattr(exc, "metadata", {}) or {})
                    skip_phase = "nnet_final_refit_skipped"
                    if str(skip_metadata.get("reason", "")) == "no_valid_tuning_trial":
                        skip_phase = "nnet_trial_invalid"
                    _append_execution_log(
                        execution_log,
                        phase=skip_phase,
                        status="skipped",
                        message="Skipped adaptive retraining because NNet did not converge on the drift window.",
                        strategy="adaptive_adwin",
                        model=model_name,
                        drift=int(drift_idx),
                        run_seed=effective_seed,
                        run_order=run_order,
                        balance_mode=balance_mode,
                        study_id=retrain_study_id,
                        retrain_rows=int(len(retrain_df)),
                        min_retrain_size=int(effective_min_retrain_size),
                        **skip_metadata,
                    )
                except Exception as exc:
                    _append_execution_log(
                        execution_log,
                        phase="adaptive_retrain_error",
                        status="error",
                        message="Adaptive retraining failed after drift detection.",
                        strategy="adaptive_adwin",
                        model=model_name,
                        drift=int(drift_idx),
                        run_seed=effective_seed,
                        run_order=run_order,
                        balance_mode=balance_mode,
                        study_id=retrain_study_id,
                        retrain_rows=int(len(retrain_df)),
                        min_retrain_size=int(effective_min_retrain_size),
                        error=str(exc),
                    )

            segment_y = []
            segment_raw_scores = []
            segment_decision_scores = []

        # Final segment row (matches A.9 style "remaining=0")
        if segment_y:
            drift_idx += 1
            stage_metrics = _compute_calibration_stage_metrics(
                np.asarray(segment_y),
                np.asarray(segment_raw_scores),
                raw_threshold=raw_youden_threshold,
                calibrated_scores=np.asarray(segment_decision_scores),
                calibrated_threshold=threshold,
            )
            seg_metrics = stage_metrics["after_calibration"]
            seg_metrics_before = stage_metrics["before_calibration"]
            seg_metrics_after = stage_metrics["after_calibration"]
            row = {
                "strategy": "adaptive_adwin",
                "drift": drift_idx,
                "drift_date": pd.Timestamp(ts_stream.iloc[-1]),
                "balance_mode": balance_mode,
                "W": int(len(detector)),
                "W0": np.nan,
                "W1": np.nan,
                "remaining_periods": 0,
                "model": model_name,
                "auc": float(seg_metrics["auc"]),
                "pr_auc": float(seg_metrics["pr_auc"]),
                "f1": float(seg_metrics["f1"]),
                "sensitivity": float(seg_metrics["sensitivity"]),
                "specificity": float(seg_metrics["specificity"]),
                "error_rate": float(seg_metrics["error_rate"]),
                "training_time_sec": float(fit_time),
                "threshold": float(threshold),
                "operating_threshold": float(threshold),
                "raw_youden_threshold": float(raw_youden_threshold),
                "threshold_policy": str(threshold_policy),
                "calibration_method": str(calibration_method),
                "fn_cost": float(fn_cost),
                "fp_cost": float(fp_cost),
                "run_seed": effective_seed,
                "run_order": int(run_order),
            }
            row.update(_calibration_rate_fields(seg_metrics_before, seg_metrics_after))
            rows.append(row)
            _append_execution_log(
                execution_log,
                phase="adaptive_final_segment",
                status="ok",
                message="Stored final adaptive segment after finishing the stream.",
                strategy="adaptive_adwin",
                model=model_name,
                drift=int(drift_idx),
                drift_date=pd.Timestamp(ts_stream.iloc[-1]),
                run_seed=effective_seed,
                run_order=run_order,
                balance_mode=balance_mode,
                threshold=float(threshold),
                operating_threshold=float(threshold),
                raw_youden_threshold=float(raw_youden_threshold),
                threshold_policy=str(threshold_policy),
                calibration_method=str(calibration_method),
                remaining_periods=0,
                auc=float(seg_metrics["auc"]),
                sensitivity=float(seg_metrics["sensitivity"]),
                specificity=float(seg_metrics["specificity"]),
                sensitivity_before_calibration=float(seg_metrics_before["sensitivity"]),
                specificity_before_calibration=float(seg_metrics_before["specificity"]),
                sensitivity_after_calibration=float(seg_metrics_after["sensitivity"]),
                specificity_after_calibration=float(seg_metrics_after["specificity"]),
                error_rate=float(seg_metrics["error_rate"]),
            )
            roc_payload.append(
                {
                    "strategy": "adaptive_adwin",
                    "model": model_name,
                    "balance_mode": balance_mode,
                    "segment": "final",
                    "y_true": np.asarray(segment_y),
                    "scores": np.asarray(segment_raw_scores),
                    "calibrated_scores": np.asarray(segment_decision_scores),
                    "raw_threshold": float(raw_youden_threshold),
                    "calibrated_threshold": float(threshold),
                    "run_seed": effective_seed,
                    "run_order": int(run_order),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["model", "drift"]).reset_index(drop=True)
    _emit_experiment_progress(
        progress_callback,
        ratio=1.0,
        label="Bloque adaptive_adwin completado.",
        detail="adaptive_adwin",
    )
    return out, roc_payload


def run_arf_strategy(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    target_col: str = "target",
    time_col: str = "interval_start",
    arf_variants: Optional[Sequence[str]] = None,
    base_year: Optional[int] = None,
    random_state: int = 42,
    validation_size: float = 0.2,
    run_seed: Optional[int] = None,
    run_order: int = 1,
    execution_log: Optional[List[Dict[str, Any]]] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Runs immediate-label Adaptive Random Forest (ARF) evaluation year by year
    after warming on the full base year.
    """
    if arf_variants is None:
        arf_variants = ARF_VARIANT_NAMES
    balance_mode = BALANCE_MODE_NOT_APPLICABLE

    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    years = _available_prediction_years(work, time_col)
    if len(years) < 2:
        return pd.DataFrame(), []

    if base_year is None:
        base_year = years[0]

    base_train = work.loc[_as_year_mask(work, time_col, int(base_year))].copy()
    stream = work.loc[pd.to_datetime(work[time_col], errors="coerce").dt.year.gt(int(base_year))].copy()

    if base_train.empty or stream.empty:
        return pd.DataFrame(), []

    effective_seed = int(random_state if run_seed is None else run_seed)
    rows: List[Dict[str, Any]] = []
    roc_payload: List[Dict[str, Any]] = []

    total_variants = max(1, len(arf_variants))
    for variant_idx, variant_name in enumerate(arf_variants):
        variant_start_ratio = float(variant_idx) / float(total_variants)
        variant_end_ratio = float(variant_idx + 1) / float(total_variants)
        variant_span = variant_end_ratio - variant_start_ratio
        variant_detail_prefix = f"Estrategia adaptive_arf | Variante {variant_name} | {variant_idx + 1} / {total_variants}"
        _emit_experiment_progress(
            progress_callback,
            ratio=variant_start_ratio,
            label="Preparando bloque adaptive_arf...",
            detail=variant_detail_prefix,
        )
        if variant_name not in ARF_VARIANT_CONFIGS:
            raise ValueError(f"Unsupported ARF variant: {variant_name}")

        if pd.to_numeric(base_train[target_col], errors="coerce").fillna(0).astype(int).nunique() < 2:
            _append_execution_log(
                execution_log,
                phase="arf_base_skip",
                status="skipped",
                message="Base training split contains a single target class.",
                strategy=ADAPTIVE_ARF_STRATEGY,
                model=variant_name,
                training_year=int(base_year),
                run_seed=effective_seed,
                run_order=run_order,
                balance_mode=balance_mode,
                n_train=int(len(base_train)),
                n_stream=int(len(stream)),
            )
            continue

        try:
            t0 = time.perf_counter()
            bundle = train_arf_with_internal_validation(
                base_train,
                feature_cols=feature_cols,
                target_col=target_col,
                time_col=time_col,
                variant_name=variant_name,
                validation_size=validation_size,
                random_state=effective_seed,
            )
            fit_time = time.perf_counter() - t0
            _emit_experiment_progress(
                progress_callback,
                ratio=variant_start_ratio + variant_span * 0.25,
                label="Entrenamiento base ARF completo.",
                detail=variant_detail_prefix,
            )
            _append_execution_log(
                execution_log,
                phase="arf_base_train_complete",
                status="ok",
                message="Finished base training for adaptive ARF strategy.",
                strategy=ADAPTIVE_ARF_STRATEGY,
                model=variant_name,
                training_year=int(base_year),
                run_seed=effective_seed,
                run_order=run_order,
                balance_mode=balance_mode,
                threshold=float(bundle["threshold"]),
                best_params=bundle["best_params"],
                warm_rows=int(bundle["warm_rows"]),
                validation_rows=int(bundle["validation_rows"]),
                n_train=int(len(base_train)),
                n_stream=int(len(stream)),
                training_time_sec=float(fit_time),
            )
        except Exception as exc:
            _append_execution_log(
                execution_log,
                phase="arf_base_train_error",
                status="error",
                message="Base training failed for adaptive ARF strategy.",
                strategy=ADAPTIVE_ARF_STRATEGY,
                model=variant_name,
                training_year=int(base_year),
                run_seed=effective_seed,
                run_order=run_order,
                balance_mode=balance_mode,
                n_train=int(len(base_train)),
                n_stream=int(len(stream)),
                error=str(exc),
            )
            continue

        model = bundle["model"]
        threshold = float(bundle["threshold"])
        fill_values = dict(bundle.get("fill_values") or {})
        X_stream_raw, y_stream = _prepare_xy_raw(stream, feature_cols, target_col)
        ts_stream = pd.to_datetime(stream[time_col], errors="coerce").reset_index(drop=True)
        stream_progress_step = max(1, len(stream) // 50)

        segment_idx = 0
        current_year: Optional[int] = None
        segment_y: List[int] = []
        segment_s: List[float] = []
        segment_drift_start = _online_counter_value(model, "n_drifts_detected")
        segment_warning_start = _online_counter_value(model, "n_warnings_detected")

        def flush_year_segment(
            prediction_year: int,
            end_ts: pd.Timestamp,
            remaining_periods: int,
        ) -> None:
            nonlocal segment_idx, segment_y, segment_s, segment_drift_start, segment_warning_start
            if not segment_y:
                return

            segment_idx += 1
            stage_metrics = _compute_calibration_stage_metrics(
                np.asarray(segment_y),
                np.asarray(segment_s),
                raw_threshold=threshold,
                calibrated_scores=np.asarray(segment_s),
                calibrated_threshold=threshold,
            )
            seg_metrics = stage_metrics["after_calibration"]
            seg_metrics_before = stage_metrics["before_calibration"]
            seg_metrics_after = stage_metrics["after_calibration"]
            current_drifts = _online_counter_value(model, "n_drifts_detected")
            current_warnings = _online_counter_value(model, "n_warnings_detected")
            segment_drifts = current_drifts - segment_drift_start
            segment_warnings = current_warnings - segment_warning_start

            row = {
                "strategy": ADAPTIVE_ARF_STRATEGY,
                "drift": segment_idx,
                "drift_date": pd.Timestamp(end_ts),
                "prediction_year": int(prediction_year),
                "balance_mode": balance_mode,
                "segment_rows": int(len(segment_y)),
                "n_internal_drifts": int(segment_drifts),
                "n_internal_warnings": int(segment_warnings),
                "W": np.nan,
                "W0": np.nan,
                "W1": np.nan,
                "remaining_periods": int(remaining_periods),
                "model": variant_name,
                "auc": float(seg_metrics["auc"]),
                "pr_auc": float(seg_metrics["pr_auc"]),
                "f1": float(seg_metrics["f1"]),
                "sensitivity": float(seg_metrics["sensitivity"]),
                "specificity": float(seg_metrics["specificity"]),
                "error_rate": float(seg_metrics["error_rate"]),
                "training_time_sec": float(fit_time),
                "threshold": float(threshold),
                "run_seed": effective_seed,
                "run_order": int(run_order),
            }
            row.update(_calibration_rate_fields(seg_metrics_before, seg_metrics_after))
            rows.append(row)

            _append_execution_log(
                execution_log,
                phase="arf_year_segment_complete",
                status="ok",
                message="Stored yearly online ARF segment.",
                strategy=ADAPTIVE_ARF_STRATEGY,
                model=variant_name,
                drift=int(segment_idx),
                prediction_year=int(prediction_year),
                drift_date=pd.Timestamp(end_ts),
                n_internal_drifts=int(segment_drifts),
                n_internal_warnings=int(segment_warnings),
                segment_rows=int(len(segment_y)),
                remaining_periods=int(remaining_periods),
                run_seed=effective_seed,
                run_order=run_order,
                balance_mode=balance_mode,
                threshold=float(threshold),
                auc=float(seg_metrics["auc"]),
                sensitivity=float(seg_metrics["sensitivity"]),
                specificity=float(seg_metrics["specificity"]),
                sensitivity_before_calibration=float(seg_metrics_before["sensitivity"]),
                specificity_before_calibration=float(seg_metrics_before["specificity"]),
                sensitivity_after_calibration=float(seg_metrics_after["sensitivity"]),
                specificity_after_calibration=float(seg_metrics_after["specificity"]),
                error_rate=float(seg_metrics["error_rate"]),
            )

            roc_payload.append(
                {
                    "strategy": ADAPTIVE_ARF_STRATEGY,
                    "model": variant_name,
                    "balance_mode": balance_mode,
                    "segment": str(prediction_year),
                    "y_true": np.asarray(segment_y),
                    "scores": np.asarray(segment_s),
                    "calibrated_scores": np.asarray(segment_s),
                    "raw_threshold": float(threshold),
                    "calibrated_threshold": float(threshold),
                    "run_seed": effective_seed,
                    "run_order": int(run_order),
                }
            )

            segment_drift_start = current_drifts
            segment_warning_start = current_warnings
            segment_y = []
            segment_s = []

        for i in range(len(stream)):
            if i == 0 or (i + 1) % stream_progress_step == 0 or i == len(stream) - 1:
                _emit_experiment_progress(
                    progress_callback,
                    ratio=variant_start_ratio + variant_span * (0.30 + 0.65 * float(i + 1) / float(max(1, len(stream)))),
                    label="Evaluando stream adaptive_arf...",
                    detail=f"{variant_detail_prefix} | fila {i + 1:,} / {len(stream):,}",
                )
            ts_i = ts_stream.iloc[i]
            year_i = int(ts_i.year)

            if current_year is None:
                current_year = year_i
            elif year_i != current_year:
                flush_year_segment(
                    int(current_year),
                    pd.Timestamp(ts_stream.iloc[i - 1]),
                    int(len(stream) - i),
                )
                current_year = year_i

            x_i = _apply_fill_values(X_stream_raw.iloc[[i]], fill_values)
            x_dict = _row_to_online_feature_dict(x_i.iloc[0])
            y_i = int(y_stream.iloc[i])
            score_i = _score_online_binary_classifier(model, x_dict)

            segment_y.append(y_i)
            segment_s.append(score_i)
            model.learn_one(x_dict, y_i)

        if current_year is not None and segment_y:
            flush_year_segment(int(current_year), pd.Timestamp(ts_stream.iloc[-1]), 0)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["model", "prediction_year", "drift"]).reset_index(drop=True)
    _emit_experiment_progress(
        progress_callback,
        ratio=1.0,
        label="Bloque adaptive_arf completado.",
        detail="adaptive_arf",
    )
    return out, roc_payload


def run_kswin_strategy(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    target_col: str = "target",
    time_col: str = "interval_start",
    model_names: Optional[Sequence[str]] = None,
    kswin_variants: Optional[Sequence[str]] = None,
    base_year: Optional[int] = None,
    random_state: int = 42,
    validation_size: float = 0.2,
    folds: int = 5,
    fast_mode: bool = True,
    resource_mode: str = DEFAULT_EXPERIMENT_RESOURCE_MODE,
    grid_limit: Optional[int] = None,
    kswin_top_k_features: int = 10,
    kswin_vote_threshold: int = 2,
    kswin_retrain_days: int = 90,
    kswin_min_retrain_rows: int = 100,
    custom_grids: Optional[Dict[str, Dict[str, Sequence[Any]]]] = None,
    run_seed: Optional[int] = None,
    run_order: int = 1,
    execution_log: Optional[List[Dict[str, Any]]] = None,
    balance_mode: str = BALANCE_MODE_NONE,
    study_collector: Optional[List[Dict[str, Any]]] = None,
    fixed_params: Optional[Dict[str, Any]] = None,
    fixed_tuning_artifact: Optional[Dict[str, Any]] = None,
    tuning_resolver: Optional[Callable[..., Dict[str, Any]]] = None,
    run_id: Optional[str] = None,
    smote_cache_dir: Optional[Path] = None,
    smote_cache_registry: Optional[Dict[str, Dict[str, Any]]] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Runs KSWIN-driven adaptive retraining using monitored feature distributions.
    """
    if model_names is None:
        model_names = MODEL_NAMES
    if kswin_variants is None:
        kswin_variants = KSWIN_DEFAULT_VARIANTS

    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    years = _available_prediction_years(work, time_col)
    if len(years) < 2:
        return pd.DataFrame(), []

    if base_year is None:
        base_year = years[0]

    base_train = work.loc[_as_year_mask(work, time_col, int(base_year))].copy()
    stream = work.loc[pd.to_datetime(work[time_col], errors="coerce").dt.year.gt(int(base_year))].copy()
    if base_train.empty or stream.empty:
        return pd.DataFrame(), []

    effective_seed = int(random_state if run_seed is None else run_seed)
    rows: List[Dict[str, Any]] = []
    roc_payload: List[Dict[str, Any]] = []
    total_blocks = max(1, len(model_names) * len(kswin_variants))
    block_counter = 0

    for model_name in model_names:
        if pd.to_numeric(base_train[target_col], errors="coerce").fillna(0).astype(int).nunique() < 2:
            _append_execution_log(
                execution_log,
                phase="kswin_base_skip",
                status="skipped",
                message="Base training split contains a single target class.",
                strategy=ADAPTIVE_KSWIN_STRATEGY,
                model=model_name,
                training_year=int(base_year),
                run_seed=effective_seed,
                run_order=run_order,
                balance_mode=balance_mode,
                n_train=int(len(base_train)),
                n_stream=int(len(stream)),
            )
            continue

        base_study_id = _build_tuning_study_id(
            stage="kswin_base",
            strategy=ADAPTIVE_KSWIN_STRATEGY,
            model_name=model_name,
            balance_mode=balance_mode,
            run_seed=effective_seed,
            run_order=run_order,
            training_year=str(int(base_year)),
        )
        try:
            t0 = time.perf_counter()
            resolved_base_tuning_artifact = fixed_tuning_artifact
            resolved_base_params = fixed_params
            if tuning_resolver is not None:
                resolved_base_tuning_artifact = tuning_resolver(
                    train_df=base_train,
                    model_name=model_name,
                    balance_mode=balance_mode,
                    strategy_context={
                        "stage": "kswin_base",
                        "strategy": ADAPTIVE_KSWIN_STRATEGY,
                        "window_kind": "base_year",
                        "training_year": str(int(base_year)),
                        "run_seed": int(effective_seed),
                        "run_order": int(run_order),
                    },
                )
                resolved_base_params = dict((resolved_base_tuning_artifact or {}).get("best_params") or {})
            bundle = train_model_with_internal_validation(
                base_train,
                feature_cols=feature_cols,
                target_col=target_col,
                time_col=time_col,
                model_name=model_name,
                validation_size=validation_size,
                folds=folds,
                random_state=effective_seed,
                fast_mode=fast_mode,
                resource_mode=resource_mode,
                grid_limit=grid_limit,
                custom_grid=(custom_grids or {}).get(model_name),
                balance_mode=balance_mode,
                study_id=base_study_id,
                fixed_params=resolved_base_params,
                fixed_tuning_artifact=resolved_base_tuning_artifact,
                run_id=run_id,
                smote_cache_dir=smote_cache_dir,
                smote_cache_registry=smote_cache_registry,
                progress_callback=(
                    None
                    if progress_callback is None
                    else lambda payload, _base=float(block_counter) / float(total_blocks), _span=1.0 / float(total_blocks), _model=model_name: _emit_experiment_progress(
                        progress_callback,
                        ratio=_base + _span * (0.05 + 0.25 * max(0.0, min(float(payload.get("ratio", 0.0)), 1.0))),
                        label=str(payload.get("label", "Entrenando base KSWIN...")),
                        detail=f"Estrategia adaptive_kswin | Modelo {_model} | {str(payload.get('detail', '') or '')}".strip(" |"),
                    )
                ),
            )
            fit_time = time.perf_counter() - t0
            if str(model_name) == "NNet" and bool(bundle.get("fit_retry_applied", False)):
                _append_execution_log(
                    execution_log,
                    phase="nnet_final_refit_retry",
                    status="warning",
                    message="NNet base KSWIN model converged only after increasing max_iter.",
                    strategy=ADAPTIVE_KSWIN_STRATEGY,
                    model=model_name,
                    training_year=int(base_year),
                    run_seed=effective_seed,
                    run_order=run_order,
                    balance_mode=balance_mode,
                    study_id=base_study_id,
                    n_iter_final=bundle.get("n_iter_final"),
                    fit_warning_count=int(bundle.get("fit_warning_count", 0)),
                )
        except ModelTrainingSkipped as exc:
            skip_metadata = dict(getattr(exc, "metadata", {}) or {})
            skip_phase = "nnet_final_refit_skipped"
            if str(skip_metadata.get("reason", "")) == "no_valid_tuning_trial":
                skip_phase = "nnet_trial_invalid"
            _append_execution_log(
                execution_log,
                phase=skip_phase,
                status="skipped",
                message="Skipped adaptive KSWIN block because NNet did not produce a stable base model.",
                strategy=ADAPTIVE_KSWIN_STRATEGY,
                model=model_name,
                training_year=int(base_year),
                run_seed=effective_seed,
                run_order=run_order,
                balance_mode=balance_mode,
                study_id=base_study_id,
                n_train=int(len(base_train)),
                n_stream=int(len(stream)),
                **skip_metadata,
            )
            continue
        except Exception as exc:
            _append_execution_log(
                execution_log,
                phase="kswin_base_train_error",
                status="error",
                message="Base training failed for adaptive KSWIN strategy.",
                strategy=ADAPTIVE_KSWIN_STRATEGY,
                model=model_name,
                training_year=int(base_year),
                run_seed=effective_seed,
                run_order=run_order,
                balance_mode=balance_mode,
                study_id=base_study_id,
                n_train=int(len(base_train)),
                n_stream=int(len(stream)),
                error=str(exc),
            )
            continue

        base_X_raw, _ = _prepare_xy_raw(base_train, feature_cols, target_col)
        base_fill_values = dict(bundle.get("fill_values") or {})
        base_X = _apply_fill_values(base_X_raw, base_fill_values)
        monitor_cols = _select_kswin_monitor_columns(
            base_X,
            feature_cols,
            top_k=kswin_top_k_features,
        )
        effective_vote_threshold = min(max(1, int(kswin_vote_threshold)), max(1, len(monitor_cols)))
        if not monitor_cols:
            _append_execution_log(
                execution_log,
                phase="kswin_monitor_skip",
                status="skipped",
                    message="No numeric monitor features available for KSWIN.",
                    strategy=ADAPTIVE_KSWIN_STRATEGY,
                    model=model_name,
                    training_year=int(base_year),
                    run_seed=effective_seed,
                    run_order=run_order,
                    balance_mode=balance_mode,
                )
            continue

        _append_tuning_study(
            study_collector,
            bundle.get("tuning_artifact"),
            stage="kswin_base",
            strategy=ADAPTIVE_KSWIN_STRATEGY,
            model=model_name,
            balance_mode=balance_mode,
            run_seed=effective_seed,
            run_order=int(run_order),
            training_year=int(base_year),
            n_train=int(len(base_train)),
            n_stream=int(len(stream)),
            monitor_cols=monitor_cols,
        )
        _append_execution_log(
            execution_log,
            phase="kswin_base_train_complete",
            status="ok",
            message="Finished base training for adaptive KSWIN strategy.",
            strategy=ADAPTIVE_KSWIN_STRATEGY,
            model=model_name,
            training_year=int(base_year),
            run_seed=effective_seed,
            run_order=run_order,
            balance_mode=balance_mode,
            study_id=base_study_id,
            threshold=float(bundle["threshold"]),
            best_params=bundle["best_params"],
            monitor_cols=monitor_cols,
            n_train=int(len(base_train)),
            n_stream=int(len(stream)),
            training_time_sec=float(fit_time),
        )

        X_stream_raw, y_stream = _prepare_xy_raw(stream, feature_cols, target_col)
        ts_stream = pd.to_datetime(stream[time_col], errors="coerce").reset_index(drop=True)

        base_monitor_df = pd.concat(
            [
                pd.to_datetime(base_train[time_col], errors="coerce").rename(time_col).reset_index(drop=True),
                base_X[monitor_cols].reset_index(drop=True),
            ],
            axis=1,
        )

        for variant_name in kswin_variants:
            variant_index = block_counter
            block_counter += 1
            variant_start_ratio = float(variant_index) / float(total_blocks)
            variant_end_ratio = float(variant_index + 1) / float(total_blocks)
            variant_span = variant_end_ratio - variant_start_ratio
            variant_detail_prefix = (
                f"Estrategia adaptive_kswin | Modelo {model_name} | Detector {variant_name} | "
                f"{variant_index + 1} / {total_blocks}"
            )
            _emit_experiment_progress(
                progress_callback,
                ratio=variant_start_ratio,
                label="Preparando bloque adaptive_kswin...",
                detail=variant_detail_prefix,
            )
            if variant_name not in KSWIN_VARIANT_CONFIGS:
                raise ValueError(f"Unsupported KSWIN variant: {variant_name}")

            seasonal_profiles = _build_kswin_seasonal_profiles(
                base_monitor_df,
                monitor_cols=monitor_cols,
                time_col=time_col,
            )
            detectors = _build_kswin_detectors(
                variant_name=variant_name,
                monitor_cols=monitor_cols,
                warm_df=base_monitor_df,
                time_col=time_col,
                seasonal_profiles=seasonal_profiles,
                random_state=effective_seed,
            )

            current_model = bundle["model"]
            operating_context = _bundle_operating_context(bundle)
            threshold = float(operating_context["threshold"])
            raw_youden_threshold = float(operating_context["raw_youden_threshold"])
            threshold_policy = str(operating_context["threshold_policy"])
            calibration_method = str(operating_context["calibration_method"])
            fn_cost = float(operating_context["fn_cost"])
            fp_cost = float(operating_context["fp_cost"])
            current_calibration_model = operating_context.get("calibration_model")
            current_calibration_metadata = operating_context.get("calibration_metadata")
            current_fill_values = dict(bundle.get("fill_values") or {})
            current_feature_transform = dict(bundle.get("feature_transform") or {}) or None
            model_label = f"{model_name} | {variant_name}"
            segment_y: List[int] = []
            segment_raw_scores: List[float] = []
            segment_decision_scores: List[float] = []
            drift_idx = 0
            stream_progress_step = max(1, len(stream) // 50)

            for i in range(len(stream)):
                if i == 0 or (i + 1) % stream_progress_step == 0 or i == len(stream) - 1:
                    _emit_experiment_progress(
                        progress_callback,
                        ratio=variant_start_ratio + variant_span * (0.35 + 0.60 * float(i + 1) / float(max(1, len(stream)))),
                        label="Evaluando stream adaptive_kswin...",
                        detail=f"{variant_detail_prefix} | fila {i + 1:,} / {len(stream):,}",
                    )
                x_i = _apply_fill_values(X_stream_raw.iloc[[i]], current_fill_values)
                x_i_model = _apply_feature_transform(x_i, current_feature_transform)
                y_i = int(y_stream.iloc[i])
                ts_i = pd.Timestamp(ts_stream.iloc[i])
                raw_score_i = float(_model_scores(current_model, x_i_model)[0])
                decision_score_i = float(
                    _apply_probability_calibration(
                        np.asarray([raw_score_i], dtype=float),
                        calibration_model=current_calibration_model,
                        calibration_metadata=current_calibration_metadata,
                    )[0]
                )

                segment_y.append(y_i)
                segment_raw_scores.append(raw_score_i)
                segment_decision_scores.append(decision_score_i)

                fired_features: List[str] = []
                for feat in monitor_cols:
                    raw_val = float(x_i.iloc[0][feat])
                    transformed = _transform_kswin_value(
                        timestamp=ts_i,
                        value=raw_val,
                        feature_name=str(feat),
                        variant_name=variant_name,
                        seasonal_profiles=seasonal_profiles,
                    )
                    if not np.isfinite(transformed):
                        continue
                    detector = detectors[str(feat)]
                    detector.update(float(transformed))
                    if bool(getattr(detector, "drift_detected", False)):
                        fired_features.append(str(feat))

                if len(fired_features) < effective_vote_threshold:
                    continue

                drift_idx += 1
                stage_metrics = _compute_calibration_stage_metrics(
                    np.asarray(segment_y),
                    np.asarray(segment_raw_scores),
                    raw_threshold=raw_youden_threshold,
                    calibrated_scores=np.asarray(segment_decision_scores),
                    calibrated_threshold=threshold,
                )
                seg_metrics = stage_metrics["after_calibration"]
                seg_metrics_before = stage_metrics["before_calibration"]
                seg_metrics_after = stage_metrics["after_calibration"]

                observed_df = pd.concat(
                    [
                        base_train,
                        stream.iloc[: i + 1].copy(),
                    ],
                    ignore_index=True,
                )
                retrain_df = _select_kswin_retrain_window(
                    observed_df,
                    current_ts=ts_i,
                    time_col=time_col,
                    retrain_days=kswin_retrain_days,
                    min_rows=kswin_min_retrain_rows,
                    target_col=target_col,
                )
                retrain_rows = int(len(retrain_df))
                retrain_positive_rows = int(
                    pd.to_numeric(retrain_df.get(target_col), errors="coerce").fillna(0).astype(int).sum()
                ) if target_col in retrain_df.columns else 0
                remaining = int(len(stream) - (i + 1))
                config = KSWIN_VARIANT_CONFIGS[variant_name]

                row = {
                    "strategy": ADAPTIVE_KSWIN_STRATEGY,
                    "drift": drift_idx,
                    "drift_date": ts_i,
                    "prediction_year": int(ts_i.year),
                    "balance_mode": balance_mode,
                    "segment_rows": int(len(segment_y)),
                    "vote_count": int(len(fired_features)),
                    "vote_threshold": int(effective_vote_threshold),
                    "monitor_feature_count": int(len(monitor_cols)),
                    "detected_features": json.dumps(fired_features, ensure_ascii=True),
                    "monitored_features": json.dumps(list(monitor_cols), ensure_ascii=True),
                    "retrain_rows": retrain_rows,
                    "retrain_positive_rows": retrain_positive_rows,
                    "W": int(config["window_size"]),
                    "W0": int(config["window_size"] - config["stat_size"]),
                    "W1": int(config["stat_size"]),
                    "remaining_periods": remaining,
                    "base_model": model_name,
                    "detector_variant": variant_name,
                    "model": model_label,
                    "auc": float(seg_metrics["auc"]),
                    "pr_auc": float(seg_metrics["pr_auc"]),
                    "f1": float(seg_metrics["f1"]),
                    "sensitivity": float(seg_metrics["sensitivity"]),
                    "specificity": float(seg_metrics["specificity"]),
                    "error_rate": float(seg_metrics["error_rate"]),
                    "training_time_sec": float(fit_time),
                    "threshold": float(threshold),
                    "operating_threshold": float(threshold),
                    "raw_youden_threshold": float(raw_youden_threshold),
                    "threshold_policy": str(threshold_policy),
                    "calibration_method": str(calibration_method),
                    "fn_cost": float(fn_cost),
                    "fp_cost": float(fp_cost),
                    "run_seed": effective_seed,
                    "run_order": int(run_order),
                }
                row.update(_calibration_rate_fields(seg_metrics_before, seg_metrics_after))
                rows.append(row)

                _append_execution_log(
                    execution_log,
                    phase="kswin_drift_detected",
                    status="ok",
                    message="KSWIN detected drift and closed a prediction segment.",
                    strategy=ADAPTIVE_KSWIN_STRATEGY,
                    model=model_label,
                    base_model=model_name,
                    detector_variant=variant_name,
                    drift=int(drift_idx),
                    prediction_year=int(ts_i.year),
                    drift_date=ts_i,
                    vote_count=int(len(fired_features)),
                    vote_threshold=int(effective_vote_threshold),
                    detected_features=fired_features,
                    monitor_feature_count=int(len(monitor_cols)),
                    monitor_cols=monitor_cols,
                    retrain_rows=retrain_rows,
                    retrain_positive_rows=retrain_positive_rows,
                    remaining_periods=remaining,
                    threshold=float(threshold),
                    operating_threshold=float(threshold),
                    raw_youden_threshold=float(raw_youden_threshold),
                    threshold_policy=str(threshold_policy),
                    calibration_method=str(calibration_method),
                    auc=float(seg_metrics["auc"]),
                    sensitivity=float(seg_metrics["sensitivity"]),
                    specificity=float(seg_metrics["specificity"]),
                    sensitivity_before_calibration=float(seg_metrics_before["sensitivity"]),
                    specificity_before_calibration=float(seg_metrics_before["specificity"]),
                    sensitivity_after_calibration=float(seg_metrics_after["sensitivity"]),
                    specificity_after_calibration=float(seg_metrics_after["specificity"]),
                    error_rate=float(seg_metrics["error_rate"]),
                    run_seed=effective_seed,
                    run_order=run_order,
                    balance_mode=balance_mode,
                )

                roc_payload.append(
                    {
                        "strategy": ADAPTIVE_KSWIN_STRATEGY,
                        "model": model_label,
                        "balance_mode": balance_mode,
                        "segment": f"drift_{drift_idx}",
                        "y_true": np.asarray(segment_y),
                        "scores": np.asarray(segment_raw_scores),
                        "calibrated_scores": np.asarray(segment_decision_scores),
                        "raw_threshold": float(raw_youden_threshold),
                        "calibrated_threshold": float(threshold),
                        "run_seed": effective_seed,
                        "run_order": int(run_order),
                    }
                )

                if (
                    not retrain_df.empty
                    and target_col in retrain_df.columns
                    and pd.to_numeric(retrain_df[target_col], errors="coerce").fillna(0).astype(int).nunique() >= 2
                ):
                    retrain_study_id = _build_tuning_study_id(
                        stage="kswin_retrain",
                        strategy=ADAPTIVE_KSWIN_STRATEGY,
                        model_name=model_name,
                        balance_mode=balance_mode,
                        run_seed=effective_seed,
                        run_order=run_order,
                        detector_variant=variant_name,
                        drift=int(drift_idx),
                    )
                    try:
                        t_fit = time.perf_counter()
                        resolved_retrain_tuning_artifact = fixed_tuning_artifact
                        resolved_retrain_params = fixed_params
                        if tuning_resolver is not None:
                            resolved_retrain_tuning_artifact = tuning_resolver(
                                train_df=retrain_df,
                                model_name=model_name,
                                balance_mode=balance_mode,
                                strategy_context={
                                    "stage": "kswin_retrain",
                                    "strategy": ADAPTIVE_KSWIN_STRATEGY,
                                    "window_kind": "kswin_retrain",
                                    "detector_variant": variant_name,
                                    "drift": int(drift_idx),
                                    "drift_date": ts_i,
                                    "prediction_year": int(ts_i.year),
                                    "retrain_rows": int(retrain_rows),
                                    "retrain_positive_rows": int(retrain_positive_rows),
                                    "training_years": sorted(
                                        pd.to_datetime(retrain_df[time_col], errors="coerce")
                                        .dt.year.dropna()
                                        .astype(int)
                                        .unique()
                                        .tolist()
                                    ),
                                    "run_seed": int(effective_seed),
                                    "run_order": int(run_order),
                                },
                            )
                            resolved_retrain_params = dict((resolved_retrain_tuning_artifact or {}).get("best_params") or {})
                        retrain_bundle = train_model_with_internal_validation(
                            retrain_df,
                            feature_cols=feature_cols,
                            target_col=target_col,
                            time_col=time_col,
                            model_name=model_name,
                            validation_size=validation_size,
                            folds=folds,
                            random_state=effective_seed,
                            fast_mode=fast_mode,
                            resource_mode=resource_mode,
                            grid_limit=grid_limit,
                            custom_grid=(custom_grids or {}).get(model_name),
                            balance_mode=balance_mode,
                            study_id=retrain_study_id,
                            fixed_params=resolved_retrain_params,
                            fixed_tuning_artifact=resolved_retrain_tuning_artifact,
                            run_id=run_id,
                            smote_cache_dir=smote_cache_dir,
                            smote_cache_registry=smote_cache_registry,
                        )
                        fit_time = time.perf_counter() - t_fit
                        if str(model_name) == "NNet" and bool(retrain_bundle.get("fit_retry_applied", False)):
                            _append_execution_log(
                                execution_log,
                                phase="nnet_final_refit_retry",
                                status="warning",
                                message="NNet KSWIN retraining converged only after increasing max_iter.",
                                strategy=ADAPTIVE_KSWIN_STRATEGY,
                                model=model_label,
                                base_model=model_name,
                                detector_variant=variant_name,
                                drift=int(drift_idx),
                                balance_mode=balance_mode,
                                study_id=retrain_study_id,
                                n_iter_final=retrain_bundle.get("n_iter_final"),
                                fit_warning_count=int(retrain_bundle.get("fit_warning_count", 0)),
                                run_seed=effective_seed,
                                run_order=run_order,
                            )
                        current_model = retrain_bundle["model"]
                        operating_context = _bundle_operating_context(retrain_bundle)
                        threshold = float(operating_context["threshold"])
                        raw_youden_threshold = float(operating_context["raw_youden_threshold"])
                        threshold_policy = str(operating_context["threshold_policy"])
                        calibration_method = str(operating_context["calibration_method"])
                        fn_cost = float(operating_context["fn_cost"])
                        fp_cost = float(operating_context["fp_cost"])
                        current_calibration_model = operating_context.get("calibration_model")
                        current_calibration_metadata = operating_context.get("calibration_metadata")
                        current_fill_values = dict(retrain_bundle.get("fill_values") or {})
                        current_feature_transform = dict(retrain_bundle.get("feature_transform") or {}) or None
                        _append_tuning_study(
                            study_collector,
                            retrain_bundle.get("tuning_artifact"),
                            stage="kswin_retrain",
                            strategy=ADAPTIVE_KSWIN_STRATEGY,
                            model=model_name,
                            base_model=model_name,
                            detector_variant=variant_name,
                            balance_mode=balance_mode,
                            run_seed=effective_seed,
                            run_order=int(run_order),
                            drift=int(drift_idx),
                            retrain_rows=retrain_rows,
                            retrain_positive_rows=retrain_positive_rows,
                        )
                        _append_execution_log(
                            execution_log,
                            phase="kswin_retrain_complete",
                            status="ok",
                            message="Adaptive retraining completed after KSWIN drift detection.",
                            strategy=ADAPTIVE_KSWIN_STRATEGY,
                            model=model_label,
                            base_model=model_name,
                            detector_variant=variant_name,
                            drift=int(drift_idx),
                            retrain_rows=retrain_rows,
                            retrain_positive_rows=retrain_positive_rows,
                            balance_mode=balance_mode,
                            study_id=retrain_study_id,
                            threshold=float(threshold),
                            operating_threshold=float(threshold),
                            raw_youden_threshold=float(raw_youden_threshold),
                            threshold_policy=str(threshold_policy),
                            calibration_method=str(calibration_method),
                            best_params=retrain_bundle["best_params"],
                            vote_threshold=int(effective_vote_threshold),
                            training_time_sec=float(fit_time),
                            run_seed=effective_seed,
                            run_order=run_order,
                        )
                    except ModelTrainingSkipped as exc:
                        skip_metadata = dict(getattr(exc, "metadata", {}) or {})
                        skip_phase = "nnet_final_refit_skipped"
                        if str(skip_metadata.get("reason", "")) == "no_valid_tuning_trial":
                            skip_phase = "nnet_trial_invalid"
                        _append_execution_log(
                            execution_log,
                            phase=skip_phase,
                            status="skipped",
                            message="Skipped adaptive retraining after KSWIN drift because NNet did not converge.",
                            strategy=ADAPTIVE_KSWIN_STRATEGY,
                            model=model_label,
                            base_model=model_name,
                            detector_variant=variant_name,
                            drift=int(drift_idx),
                            retrain_rows=retrain_rows,
                            retrain_positive_rows=retrain_positive_rows,
                            balance_mode=balance_mode,
                            study_id=retrain_study_id,
                            run_seed=effective_seed,
                            run_order=run_order,
                            **skip_metadata,
                        )
                    except Exception as exc:
                        _append_execution_log(
                            execution_log,
                            phase="kswin_retrain_error",
                            status="error",
                            message="Adaptive retraining failed after KSWIN drift detection.",
                            strategy=ADAPTIVE_KSWIN_STRATEGY,
                            model=model_label,
                            base_model=model_name,
                            detector_variant=variant_name,
                            drift=int(drift_idx),
                            retrain_rows=retrain_rows,
                            retrain_positive_rows=retrain_positive_rows,
                            balance_mode=balance_mode,
                            study_id=retrain_study_id,
                            error=str(exc),
                            run_seed=effective_seed,
                            run_order=run_order,
                        )

                warm_monitor_df = pd.concat(
                    [
                        pd.to_datetime(retrain_df[time_col], errors="coerce").rename(time_col).reset_index(drop=True),
                        _apply_fill_values(
                            _prepare_xy_raw(retrain_df, feature_cols, target_col)[0],
                            current_fill_values,
                        )[monitor_cols].reset_index(drop=True),
                    ],
                axis=1,
                ) if not retrain_df.empty else base_monitor_df

                detectors = _build_kswin_detectors(
                    variant_name=variant_name,
                    monitor_cols=monitor_cols,
                    warm_df=warm_monitor_df,
                    time_col=time_col,
                    seasonal_profiles=seasonal_profiles,
                    random_state=effective_seed + drift_idx,
                )

                segment_y = []
                segment_raw_scores = []
                segment_decision_scores = []

            if segment_y:
                drift_idx += 1
                stage_metrics = _compute_calibration_stage_metrics(
                    np.asarray(segment_y),
                    np.asarray(segment_raw_scores),
                    raw_threshold=raw_youden_threshold,
                    calibrated_scores=np.asarray(segment_decision_scores),
                    calibrated_threshold=threshold,
                )
                seg_metrics = stage_metrics["after_calibration"]
                seg_metrics_before = stage_metrics["before_calibration"]
                seg_metrics_after = stage_metrics["after_calibration"]
                row = {
                    "strategy": ADAPTIVE_KSWIN_STRATEGY,
                    "drift": drift_idx,
                    "drift_date": pd.Timestamp(ts_stream.iloc[-1]),
                    "prediction_year": int(pd.Timestamp(ts_stream.iloc[-1]).year),
                    "balance_mode": balance_mode,
                    "segment_rows": int(len(segment_y)),
                    "vote_count": 0,
                    "vote_threshold": int(effective_vote_threshold),
                    "monitor_feature_count": int(len(monitor_cols)),
                    "detected_features": json.dumps([], ensure_ascii=True),
                    "monitored_features": json.dumps(list(monitor_cols), ensure_ascii=True),
                    "retrain_rows": 0,
                    "retrain_positive_rows": 0,
                    "W": int(KSWIN_VARIANT_CONFIGS[variant_name]["window_size"]),
                    "W0": int(KSWIN_VARIANT_CONFIGS[variant_name]["window_size"] - KSWIN_VARIANT_CONFIGS[variant_name]["stat_size"]),
                    "W1": int(KSWIN_VARIANT_CONFIGS[variant_name]["stat_size"]),
                    "remaining_periods": 0,
                    "base_model": model_name,
                    "detector_variant": variant_name,
                    "model": model_label,
                    "auc": float(seg_metrics["auc"]),
                    "pr_auc": float(seg_metrics["pr_auc"]),
                    "f1": float(seg_metrics["f1"]),
                    "sensitivity": float(seg_metrics["sensitivity"]),
                    "specificity": float(seg_metrics["specificity"]),
                    "error_rate": float(seg_metrics["error_rate"]),
                    "training_time_sec": float(fit_time),
                    "threshold": float(threshold),
                    "operating_threshold": float(threshold),
                    "raw_youden_threshold": float(raw_youden_threshold),
                    "threshold_policy": str(threshold_policy),
                    "calibration_method": str(calibration_method),
                    "fn_cost": float(fn_cost),
                    "fp_cost": float(fp_cost),
                    "run_seed": effective_seed,
                    "run_order": int(run_order),
                }
                row.update(_calibration_rate_fields(seg_metrics_before, seg_metrics_after))
                rows.append(row)
                _append_execution_log(
                    execution_log,
                    phase="kswin_final_segment",
                    status="ok",
                    message="Stored final KSWIN adaptive segment after finishing the stream.",
                    strategy=ADAPTIVE_KSWIN_STRATEGY,
                    model=model_label,
                    base_model=model_name,
                    detector_variant=variant_name,
                    drift=int(drift_idx),
                    drift_date=pd.Timestamp(ts_stream.iloc[-1]),
                    remaining_periods=0,
                    threshold=float(threshold),
                    operating_threshold=float(threshold),
                    raw_youden_threshold=float(raw_youden_threshold),
                    threshold_policy=str(threshold_policy),
                    calibration_method=str(calibration_method),
                    auc=float(seg_metrics["auc"]),
                    sensitivity=float(seg_metrics["sensitivity"]),
                    specificity=float(seg_metrics["specificity"]),
                    sensitivity_before_calibration=float(seg_metrics_before["sensitivity"]),
                    specificity_before_calibration=float(seg_metrics_before["specificity"]),
                    sensitivity_after_calibration=float(seg_metrics_after["sensitivity"]),
                    specificity_after_calibration=float(seg_metrics_after["specificity"]),
                    error_rate=float(seg_metrics["error_rate"]),
                    run_seed=effective_seed,
                    run_order=run_order,
                    balance_mode=balance_mode,
                )
                roc_payload.append(
                    {
                        "strategy": ADAPTIVE_KSWIN_STRATEGY,
                        "model": model_label,
                        "balance_mode": balance_mode,
                        "segment": "final",
                        "y_true": np.asarray(segment_y),
                        "scores": np.asarray(segment_raw_scores),
                        "calibrated_scores": np.asarray(segment_decision_scores),
                        "raw_threshold": float(raw_youden_threshold),
                        "calibrated_threshold": float(threshold),
                        "run_seed": effective_seed,
                        "run_order": int(run_order),
                    }
                )
            _emit_experiment_progress(
                progress_callback,
                ratio=variant_end_ratio,
                label="Bloque adaptive_kswin completado.",
                detail=variant_detail_prefix,
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["model", "drift_date", "drift"]).reset_index(drop=True)
    return out, roc_payload


def _build_experiment_blocks(
    *,
    model_names: Sequence[str],
    strategies: Sequence[str],
    arf_variants: Sequence[str],
    kswin_variants: Sequence[str],
    balance_modes: Sequence[str],
) -> List[Dict[str, str]]:
    normalized_balance_modes = [
        str(mode)
        for mode in balance_modes
        if str(mode) in DEFAULT_BATCH_BALANCE_MODES
    ] or [BALANCE_MODE_NONE]
    blocks: List[Dict[str, str]] = []
    for strategy in strategies:
        if strategy in YEARLY_STRATEGIES or strategy == ADAPTIVE_ADWIN_STRATEGY:
            for model_name in model_names:
                for balance_mode in normalized_balance_modes:
                    blocks.append(
                        {
                            "strategy": str(strategy),
                            "model": str(model_name),
                            "balance_mode": str(balance_mode),
                        }
                    )
            continue
        if strategy == ADAPTIVE_ARF_STRATEGY:
            for variant_name in arf_variants:
                blocks.append(
                    {
                        "strategy": str(strategy),
                        "model": str(variant_name),
                        "balance_mode": BALANCE_MODE_NOT_APPLICABLE,
                    }
                )
            continue
        if strategy == ADAPTIVE_KSWIN_STRATEGY:
            for model_name in model_names:
                for variant_name in kswin_variants:
                    for balance_mode in normalized_balance_modes:
                        blocks.append(
                            {
                                "strategy": str(strategy),
                                "model": str(model_name),
                                "detector_variant": str(variant_name),
                                "balance_mode": str(balance_mode),
                            }
                        )
            continue
        raise ValueError(f"Unsupported strategy: {strategy}")
    strategy_priority = {name: idx for idx, name in enumerate(strategies)}
    balance_priority = {
        BALANCE_MODE_NONE: 0,
        BALANCE_MODE_NOT_APPLICABLE: 0,
        BALANCE_MODE_SMOTE: 1,
    }
    blocks.sort(
        key=lambda block: (
            int(strategy_priority.get(str(block.get("strategy", "")), 999)),
            int(balance_priority.get(str(block.get("balance_mode", BALANCE_MODE_NOT_APPLICABLE)), 9)),
            _model_execution_priority(str(block.get("model", ""))),
            str(block.get("model", "")),
            str(block.get("detector_variant", "")),
        )
    )
    return blocks


def _normalize_feature_selection_context(
    feature_selection_context: Optional[Dict[str, Any]],
    *,
    feature_cols: Sequence[str],
) -> Dict[str, Any]:
    context = dict(feature_selection_context or {})
    selected_features = [
        str(col)
        for col in (context.get("selected_features") or list(feature_cols))
    ]
    context["selected_features"] = selected_features
    context["feature_count"] = int(context.get("feature_count") or len(selected_features))
    if "feature_export_path" in context and context.get("feature_export_path"):
        raw_export_path = str(context["feature_export_path"]).strip()
        resolved_export_path = _resolve_existing_feature_export_path(raw_export_path)
        export_path_text = str(resolved_export_path or raw_export_path)
        context["feature_export_path"] = export_path_text
        context.setdefault("feature_export_name", _portable_path_name(export_path_text))
    return _to_json_safe(context)


def _canonical_base_year(df: pd.DataFrame, *, time_col: str, base_year: Optional[int]) -> Optional[int]:
    years = _available_prediction_years(df, time_col)
    if not years:
        return None
    if base_year is None:
        return int(years[0])
    return int(base_year)


def _batch_tuning_tasks(
    *,
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    time_col: str,
    model_names: Sequence[str],
    strategies: Sequence[str],
    balance_modes: Sequence[str],
    base_year: Optional[int],
    validation_size: float,
    folds: int,
    random_state: int,
    fast_mode: bool,
    resource_mode: str,
    grid_limit: Optional[int],
    custom_grids: Optional[Dict[str, Dict[str, Sequence[Any]]]] = None,
) -> List[Dict[str, Any]]:
    requires_batch = any(
        strategy in YEARLY_STRATEGIES + [ADAPTIVE_ADWIN_STRATEGY, ADAPTIVE_KSWIN_STRATEGY]
        for strategy in strategies
    )
    if not requires_batch:
        return []
    normalized_balance_modes = [
        str(mode)
        for mode in balance_modes
        if str(mode) in DEFAULT_BATCH_BALANCE_MODES
    ] or [BALANCE_MODE_NONE]
    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.dropna(subset=[time_col]).reset_index(drop=True)
    effective_base_year = _canonical_base_year(work, time_col=time_col, base_year=base_year)
    if effective_base_year is None:
        return []
    prediction_years = [int(year) for year in _available_prediction_years(work, time_col) if int(year) > int(effective_base_year)]
    tasks: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()

    def register_task(
        *,
        model_name: str,
        balance_mode: str,
        train_df: pd.DataFrame,
        strategy_context: Dict[str, Any],
    ) -> None:
        if train_df.empty:
            return
        tuning_key = _build_global_tuning_key(
            model_name=model_name,
            balance_mode=balance_mode,
            feature_cols=feature_cols,
            target_col=target_col,
            time_col=time_col,
            canonical_train_df=train_df,
            validation_size=float(validation_size),
            folds=int(folds),
            random_state=int(random_state),
            fast_mode=bool(fast_mode),
            resource_mode=str(resource_mode),
            grid_limit=grid_limit,
            custom_grid=(custom_grids or {}).get(str(model_name)),
        )
        if tuning_key in seen_keys:
            return
        seen_keys.add(tuning_key)
        times = pd.to_datetime(train_df[time_col], errors="coerce")
        training_years = sorted(times.dt.year.dropna().astype(int).unique().tolist())
        y_train = pd.to_numeric(train_df[target_col], errors="coerce").fillna(0).astype(int)
        tasks.append(
            {
                "tuning_key": tuning_key,
                "model_name": str(model_name),
                "balance_mode": str(balance_mode),
                "window_kind": str(strategy_context.get("window_kind", "window")),
                "base_year": int(effective_base_year),
                "training_years": training_years,
                "training_year_label": str(strategy_context.get("training_year", "")),
                "prediction_year": strategy_context.get("prediction_year"),
                "strategy_context": _to_json_safe(strategy_context),
                "positive_rows": int(y_train.sum()),
                "positive_rate": float(y_train.mean()) if len(y_train) else 0.0,
            }
        )

    for model_name in model_names:
        for balance_mode in normalized_balance_modes:
            base_train = work.loc[_as_year_mask(work, time_col, int(effective_base_year))].copy()
            if "static" in strategies or ADAPTIVE_ADWIN_STRATEGY in strategies or ADAPTIVE_KSWIN_STRATEGY in strategies:
                register_task(
                    model_name=str(model_name),
                    balance_mode=str(balance_mode),
                    train_df=base_train,
                    strategy_context={
                        "stage": "window_tuning",
                        "window_kind": "base_year",
                        "training_year": str(int(effective_base_year)),
                    },
                )
            if "period_aligned" in strategies:
                for pred_year in prediction_years:
                    prev_year = int(pred_year) - 1
                    register_task(
                        model_name=str(model_name),
                        balance_mode=str(balance_mode),
                        train_df=work.loc[_as_year_mask(work, time_col, prev_year)].copy(),
                        strategy_context={
                            "stage": "window_tuning",
                            "window_kind": "period_aligned",
                            "training_year": str(prev_year),
                            "prediction_year": int(pred_year),
                        },
                    )
            if "cumulative" in strategies:
                for pred_year in prediction_years:
                    register_task(
                        model_name=str(model_name),
                        balance_mode=str(balance_mode),
                        train_df=work.loc[pd.to_datetime(work[time_col], errors="coerce").dt.year.le(int(pred_year) - 1)].copy(),
                        strategy_context={
                            "stage": "window_tuning",
                            "window_kind": "cumulative",
                            "training_year": f"[<= {int(pred_year) - 1}]",
                            "prediction_year": int(pred_year),
                        },
                    )
    tasks.sort(
        key=lambda item: (
            0 if str(item["balance_mode"]) == BALANCE_MODE_NONE else 1,
            _model_execution_priority(str(item["model_name"])),
            len(item.get("training_years") or []),
            str(item.get("training_year_label", "")),
            str(item["model_name"]),
        )
    )
    return tasks


def _materialize_tuning_train_df(
    df: pd.DataFrame,
    *,
    time_col: str,
    task: Dict[str, Any],
) -> pd.DataFrame:
    work = df.copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.dropna(subset=[time_col]).reset_index(drop=True)
    strategy_context = dict(task.get("strategy_context") or {})
    window_kind = str(task.get("window_kind") or strategy_context.get("window_kind") or "")
    prediction_year = strategy_context.get("prediction_year", task.get("prediction_year"))
    training_years = [
        int(year)
        for year in (task.get("training_years") or strategy_context.get("training_years") or [])
        if pd.notna(year)
    ]

    if window_kind in {"base_year", "period_aligned"}:
        if not training_years:
            raise ValueError(f"Missing training years for tuning task: {task}")
        return work.loc[_as_year_mask(work, time_col, int(training_years[0]))].copy()

    if window_kind == "cumulative":
        if prediction_year is None:
            raise ValueError(f"Missing prediction year for cumulative tuning task: {task}")
        times = pd.to_datetime(work[time_col], errors="coerce")
        return work.loc[times.dt.year.le(int(prediction_year) - 1)].copy()

    if training_years:
        times = pd.to_datetime(work[time_col], errors="coerce")
        return work.loc[times.dt.year.isin(training_years)].copy()

    raise ValueError(f"Unsupported tuning task context: {task}")


def _bundle_operating_context(bundle: Dict[str, Any]) -> Dict[str, Any]:
    operating_threshold = float(bundle.get("operating_threshold", bundle.get("threshold", 0.5)))
    return {
        "threshold": operating_threshold,
        "operating_threshold": operating_threshold,
        "threshold_policy": str(bundle.get("threshold_policy", DEFAULT_THRESHOLD_POLICY)),
        "calibration_method": str(bundle.get("calibration_method", DEFAULT_CALIBRATION_METHOD)),
        "fn_cost": float(bundle.get("fn_cost", DEFAULT_THRESHOLD_FN_COST)),
        "fp_cost": float(bundle.get("fp_cost", DEFAULT_THRESHOLD_FP_COST)),
        "raw_youden_threshold": float(bundle.get("raw_youden_threshold", operating_threshold)),
        "calibration_model": bundle.get("calibration_model"),
        "calibration_metadata": bundle.get("calibration_metadata"),
    }


def _initial_recalibration_manifest(
    *,
    run_id: str,
    run_manifest: Dict[str, Any],
    feature_selection_context: Dict[str, Any],
    experiment_blocks: Sequence[Dict[str, Any]],
    tuning_tasks: Sequence[Dict[str, Any]],
    preflight: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "run_id": str(run_id),
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "run_manifest": _to_json_safe(run_manifest),
        "feature_selection_context": _to_json_safe(feature_selection_context),
        "completed_block_ids": [],
        "skipped_failed_block_ids": [],
        "pending_block_ids": [],
        "failed_block_id": None,
        "last_error": None,
        "nonfatal_block_errors": [],
        "block_index": {},
        "tuning_index": {},
        "smote_index": {},
        "progress": {
            "completed_units": 0.0,
            "total_units": int(run_manifest.get("total_progress_units", 0)),
            "completed_tuning_tasks": 0,
            "total_tuning_tasks": int(len(tuning_tasks)),
            "completed_blocks": 0,
            "skipped_failed_blocks": 0,
            "total_blocks": int(len(experiment_blocks)),
        },
        "resume": {
            "auto_resumed": False,
            "checkpoint_status": "fresh",
        },
        "preflight": _to_json_safe(preflight),
        "memory_summary": {},
        "global_execution_log": [],
    }


def _update_manifest_progress(
    manifest: Dict[str, Any],
    *,
    completed_units: float,
    pending_block_ids: Sequence[str],
) -> None:
    progress = dict(manifest.get("progress") or {})
    total_tuning_tasks = max(
        int(progress.get("total_tuning_tasks", 0)),
        int(len(manifest.get("tuning_index") or {})),
    )
    progress["completed_units"] = float(completed_units)
    progress["total_units"] = int(manifest.get("run_manifest", {}).get("total_progress_units", progress.get("total_units", 0)))
    progress["completed_tuning_tasks"] = int(
        sum(1 for item in (manifest.get("tuning_index") or {}).values() if str(item.get("status")) == "completed")
    )
    progress["total_tuning_tasks"] = int(total_tuning_tasks)
    progress["completed_blocks"] = int(len(manifest.get("completed_block_ids") or []))
    progress["skipped_failed_blocks"] = int(len(manifest.get("skipped_failed_block_ids") or []))
    progress["total_blocks"] = int(progress.get("total_blocks", 0))
    manifest["progress"] = progress
    manifest.setdefault("run_manifest", {})["total_tuning_tasks"] = int(total_tuning_tasks)
    manifest["pending_block_ids"] = [str(block_id) for block_id in pending_block_ids]


def _load_completed_tuning_cache(paths: Dict[str, Path], manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    artifacts: Dict[str, Dict[str, Any]] = {}
    for tuning_key, info in (manifest.get("tuning_index") or {}).items():
        if str(info.get("status")) != "completed":
            continue
        tuning_path = paths["tuning_dir"] / str(info.get("filename", ""))
        payload = _load_tuning_artifact(tuning_path)
        if payload:
            payload = dict(payload)
            payload["_artifact_filename"] = tuning_path.name
            payload["_artifact_tuning_key"] = str(tuning_key)
            artifacts[str(tuning_key)] = payload
    return artifacts


def _resolve_compatible_cached_tuning_artifact(
    *,
    manifest: Dict[str, Any],
    tuning_cache: Dict[str, Dict[str, Any]],
    tuning_key: str,
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    time_col: str,
    model_name: str,
    balance_mode: str,
    tuning_filename: str,
) -> Optional[Dict[str, Any]]:
    cached = tuning_cache.get(str(tuning_key))
    if cached is not None:
        return cached

    signature_cols = [col for col in [time_col, target_col] + list(feature_cols) if col in train_df.columns]
    expected_train_signature = _frame_signature(train_df, columns=signature_cols, include_index=True)
    matched_key: Optional[str] = None
    matched_payload: Optional[Dict[str, Any]] = None
    for cached_key, candidate in tuning_cache.items():
        if str(cached_key) == str(tuning_key):
            continue
        if str(candidate.get("model_name")) != str(model_name):
            continue
        if str(candidate.get("balance_mode")) != str(balance_mode):
            continue
        if str(candidate.get("train_signature")) != str(expected_train_signature):
            continue
        matched_key = str(cached_key)
        matched_payload = dict(candidate)
        break

    if matched_payload is None or matched_key is None:
        return None

    legacy_info = dict((manifest.get("tuning_index") or {}).get(matched_key) or {})
    legacy_filename = str(
        legacy_info.get("filename")
        or matched_payload.get("_artifact_filename")
        or tuning_filename
    )
    alias_payload = dict(matched_payload)
    alias_payload["tuning_key"] = str(tuning_key)
    alias_payload["_legacy_tuning_key"] = str(matched_key)
    alias_payload["_artifact_filename"] = legacy_filename
    tuning_cache[str(tuning_key)] = alias_payload
    manifest["tuning_index"][str(tuning_key)].update(
        {
            "filename": legacy_filename,
            "status": "completed",
            "study_id": alias_payload.get("study_id"),
            "matched_legacy_tuning_key": str(matched_key),
            "resolution": "train_signature_legacy_match",
        }
    )
    return alias_payload


def _resolve_or_create_tuning_artifact(
    *,
    paths: Dict[str, Path],
    manifest: Dict[str, Any],
    tuning_cache: Dict[str, Dict[str, Any]],
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    time_col: str,
    model_name: str,
    balance_mode: str,
    validation_size: float,
    folds: int,
    random_state: int,
    fast_mode: bool,
    resource_mode: str,
    grid_limit: Optional[int],
    custom_grid: Optional[Dict[str, Sequence[Any]]],
    tuning_context: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    persist_progress: bool = True,
    allow_create: bool = True,
) -> Dict[str, Any]:
    tuning_key = _build_global_tuning_key(
        model_name=model_name,
        balance_mode=balance_mode,
        feature_cols=feature_cols,
        target_col=target_col,
        time_col=time_col,
        canonical_train_df=train_df,
        validation_size=float(validation_size),
        folds=int(folds),
        random_state=int(random_state),
        fast_mode=bool(fast_mode),
        resource_mode=str(resource_mode),
        grid_limit=grid_limit,
        custom_grid=custom_grid,
    )
    tuning_filename = f"{tuning_key}.json"
    y_train = pd.to_numeric(train_df[target_col], errors="coerce").fillna(0).astype(int)
    signature_cols = [col for col in [time_col, target_col] + list(feature_cols) if col in train_df.columns]
    strategy_context = _to_json_safe(dict(tuning_context or {}))

    manifest["tuning_index"][tuning_key] = {
        **dict((manifest.get("tuning_index") or {}).get(tuning_key) or {}),
        "filename": tuning_filename,
        "status": str((manifest.get("tuning_index") or {}).get(tuning_key, {}).get("status", "pending")),
        "model_name": str(model_name),
        "balance_mode": str(balance_mode),
        "strategy_context": strategy_context,
    }

    cached = tuning_cache.get(tuning_key)
    if cached is not None:
        manifest["tuning_index"][tuning_key].update(
            {
                "filename": str(cached.get("_artifact_filename") or tuning_filename),
                "status": "completed",
                "study_id": cached.get("study_id"),
            }
        )
        return cached

    compatible_cached = _resolve_compatible_cached_tuning_artifact(
        manifest=manifest,
        tuning_cache=tuning_cache,
        tuning_key=tuning_key,
        train_df=train_df,
        feature_cols=feature_cols,
        target_col=target_col,
        time_col=time_col,
        model_name=model_name,
        balance_mode=balance_mode,
        tuning_filename=tuning_filename,
    )
    if compatible_cached is not None:
        return compatible_cached

    if not allow_create:
        raise RuntimeError(
            "Selected checkpoint is missing the persisted tuning artifact required for recompute-only mode. "
            f"Model {model_name} | Balance {balance_mode} | tuning_key {tuning_key}"
        )

    manifest["tuning_index"][tuning_key]["status"] = "running"
    if persist_progress:
        _persist_manifest(paths["manifest"], manifest)

    X_tune, y_tune = _prepare_xy_raw(train_df, feature_cols, target_col)
    tuning_study_id = f"{tuning_key}__window"
    try:
        best_params, search_df, tuning_artifact = tune_hyperparameters(
            X_tune,
            y_tune,
            model_name=model_name,
            folds=folds,
            random_state=random_state,
            fast_mode=fast_mode,
            resource_mode=resource_mode,
            grid_limit=grid_limit,
            custom_grid=custom_grid,
            balance_mode=balance_mode,
            study_id=tuning_study_id,
            progress_callback=progress_callback,
        )
    except Exception:
        manifest["tuning_index"][tuning_key]["status"] = "failed"
        if persist_progress:
            _persist_manifest(paths["manifest"], manifest)
        raise
    tuning_payload = {
        **dict(tuning_artifact),
        "tuning_key": tuning_key,
        "stage": "window_tuning",
        "strategy_context": strategy_context,
        "model_name": str(model_name),
        "balance_mode": str(balance_mode),
        "best_params": best_params,
        "n_train": int(len(train_df)),
        "positive_rows": int(y_train.sum()),
        "positive_rate": float(y_train.mean()) if len(y_train) else 0.0,
        "train_signature": _frame_signature(train_df, columns=signature_cols, include_index=True),
        "policy_signature": _recalibration_policy_signature(),
        "search_df": [] if search_df.empty else search_df.to_dict(orient="records"),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    if persist_progress:
        _persist_tuning_artifact(paths["tuning_dir"] / tuning_filename, tuning_payload)
    tuning_cache[tuning_key] = tuning_payload
    manifest["tuning_index"][tuning_key].update(
        {
            "filename": tuning_filename,
            "status": "completed",
            "study_id": tuning_payload.get("study_id"),
        }
    )
    if persist_progress:
        _persist_manifest(paths["manifest"], manifest)
    return tuning_payload


def _persist_drift_recalibration_json(
    *,
    run_manifest: Dict[str, Any],
    feature_selection_context: Dict[str, Any],
    studies: Sequence[Dict[str, Any]],
    yearly_results: pd.DataFrame,
    adaptive_results: pd.DataFrame,
    summary: pd.DataFrame,
    appendix_tables: Dict[str, pd.DataFrame],
    appendix_tables_mean: Dict[str, pd.DataFrame],
    average_roc: pd.DataFrame,
    roc_payload: Sequence[Dict[str, Any]],
    execution_log: pd.DataFrame,
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"drift_recalibration_optuna_{stamp}.json"
    run_manifest["optuna_json_path"] = str(out_path)
    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "run_manifest": run_manifest,
        "feature_selection_context": feature_selection_context,
        "studies": list(studies),
        "yearly_results": yearly_results,
        "adaptive_results": adaptive_results,
        "summary": summary,
        "appendix_tables": appendix_tables,
        "appendix_tables_mean": appendix_tables_mean,
        "average_roc": average_roc,
        "roc_payload": list(roc_payload),
        "execution_log": execution_log,
    }
    _atomic_write_json(out_path, payload)
    return out_path


def run_recalibration_experiments(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    target_col: str = "target",
    time_col: str = "interval_start",
    model_names: Optional[Sequence[str]] = None,
    strategies: Optional[Sequence[str]] = None,
    base_year: Optional[int] = None,
    validation_size: float = 0.2,
    folds: int = 3,
    random_state: int = 42,
    fast_mode: bool = True,
    resource_mode: str = DEFAULT_EXPERIMENT_RESOURCE_MODE,
    grid_limit: Optional[int] = 30,
    adwin_delta: float = 0.002,
    min_window: int = 45_000,
    min_retrain_size: Optional[int] = None,
    arf_variants: Optional[Sequence[str]] = None,
    kswin_variants: Optional[Sequence[str]] = None,
    kswin_top_k_features: int = 10,
    kswin_vote_threshold: int = 2,
    kswin_retrain_days: int = 90,
    kswin_min_retrain_rows: int = 100,
    custom_grids: Optional[Dict[str, Dict[str, Sequence[Any]]]] = None,
    repetition_seeds: Optional[Sequence[int]] = None,
    balance_modes: Optional[Sequence[str]] = None,
    feature_selection_context: Optional[Dict[str, Any]] = None,
    checkpoint_root: Optional[Path] = None,
    checkpoint_run_id_override: Optional[str] = None,
    auto_resume: bool = True,
    recompute_blocks_from_checkpoint: bool = False,
    persist_progress: bool = True,
    reuse_tuning_cache: bool = True,
    reuse_smote_cache: bool = True,
    continue_on_block_error: bool = False,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Main orchestrator for yearly, ADWIN-adaptive and ARF-adaptive strategies.
    """
    initial_progress_label = "Analizando configuracion de experimentos..."
    initial_progress_detail = "Construyendo bloques, ventanas de tuning y politica de checkpoint."
    if recompute_blocks_from_checkpoint:
        initial_progress_label = "Validando checkpoint para recompute-only..."
        initial_progress_detail = (
            "Reconstruyendo bloques y tuning_keys esperados para verificar el checkpoint; "
            "no se crearán nuevos tunings."
        )
    if progress_callback is not None:
        progress_callback(
            {
                "completed_units": 0.0,
                "total_units": 1,
                "label": initial_progress_label,
                "detail": initial_progress_detail,
            }
        )
    if model_names is None:
        model_names = MODEL_NAMES
    if strategies is None:
        strategies = EXPERIMENT_STRATEGIES
    if arf_variants is None:
        arf_variants = ARF_DEFAULT_VARIANTS
    if kswin_variants is None:
        kswin_variants = KSWIN_DEFAULT_VARIANTS
    if repetition_seeds is None:
        repetition_seeds = DEFAULT_REPETITION_SEEDS
    if balance_modes is None:
        balance_modes = DEFAULT_BATCH_BALANCE_MODES
    normalized_feature_selection_context = _normalize_feature_selection_context(
        feature_selection_context,
        feature_cols=feature_cols,
    )
    normalized_resource_mode = _normalize_experiment_resource_mode(resource_mode)
    resource_policy = _resolve_experiment_resource_policy(normalized_resource_mode)
    effective_base_year = _canonical_base_year(df, time_col=time_col, base_year=base_year)

    experiment_blocks = _build_experiment_blocks(
        model_names=list(model_names),
        strategies=list(strategies),
        arf_variants=list(arf_variants),
        kswin_variants=list(kswin_variants),
        balance_modes=list(balance_modes),
    )
    if not experiment_blocks:
        raise ValueError("No experiment blocks were generated. Check selected strategies/models.")

    progress_seeds = [int(seed) for seed in repetition_seeds]
    tuning_tasks = _batch_tuning_tasks(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        time_col=time_col,
        model_names=list(model_names),
        strategies=list(strategies),
        balance_modes=list(balance_modes),
        base_year=effective_base_year,
        validation_size=float(validation_size),
        folds=int(folds),
        random_state=int(random_state),
        fast_mode=bool(fast_mode),
        resource_mode=str(normalized_resource_mode),
        grid_limit=grid_limit,
        custom_grids=custom_grids,
    )
    total_block_units = len(experiment_blocks) * max(1, len(progress_seeds))
    total_units = max(1, len(tuning_tasks) + total_block_units)
    completed_units = 0.0
    live_status_path: Optional[Path] = None
    live_events_path: Optional[Path] = None
    last_live_event_signature: Dict[str, Optional[str]] = {"value": None}

    def emit_progress(
        *,
        label: str,
        detail: str = "",
        completed_override: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        completed_value = float(completed_units) if completed_override is None else float(completed_override)
        payload = {
            "completed_units": float(completed_value),
            "total_units": int(total_units),
            "label": str(label),
            "detail": str(detail),
        }
        if progress_callback is not None:
            progress_callback(payload)
        if not persist_progress or live_status_path is None:
            return
        ratio = max(0.0, min(float(completed_value) / float(max(1, int(total_units))), 1.0))
        context_payload = _to_json_safe(dict(context or {}))
        live_payload = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_id": str(run_manifest.get("run_id", "")) if isinstance(run_manifest, dict) else "",
            "status": str((manifest or {}).get("status", "running")),
            "completed_units": float(completed_value),
            "total_units": int(total_units),
            "progress_ratio": float(ratio),
            "label": str(label),
            "detail": str(detail),
            "context": context_payload,
        }
        _persist_recalibration_live_status(live_status_path, live_payload)
        if live_events_path is None:
            return
        signature = json.dumps(
            {
                "completed_units": round(float(completed_value), 6),
                "total_units": int(total_units),
                "label": str(label),
                "detail": str(detail),
                "context": context_payload,
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        if signature != last_live_event_signature["value"]:
            _append_recalibration_live_event(live_events_path, live_payload)
            last_live_event_signature["value"] = signature

    all_rows: List[pd.DataFrame] = []
    all_roc: List[Dict[str, Any]] = []
    adaptive_frames: List[pd.DataFrame] = []
    execution_log: List[Dict[str, Any]] = []
    studies: List[Dict[str, Any]] = []
    if progress_callback is not None:
        preparation_label = "Preparando experimentos..."
        preparation_detail = (
            f"Plan preliminar: {len(tuning_tasks)} tareas de tuning y "
            f"{total_block_units} bloques experimentales."
        )
        if recompute_blocks_from_checkpoint:
            preparation_label = "Preparando recompute desde checkpoint..."
            preparation_detail = (
                f"Plan reconstruido: {len(tuning_tasks)} tunings esperados y "
                f"{total_block_units} bloques. Solo se validarán artefactos persistidos."
            )
        progress_callback(
            {
                "completed_units": 0.0,
                "total_units": int(total_units),
                "label": preparation_label,
                "detail": preparation_detail,
            }
        )
    preflight = _estimate_recalibration_workload(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        time_col=time_col,
        model_names=list(model_names),
        strategies=list(strategies),
        arf_variants=list(arf_variants),
        kswin_variants=list(kswin_variants),
        balance_modes=list(balance_modes),
        repetition_seeds=list(repetition_seeds),
        base_year=effective_base_year,
        validation_size=float(validation_size),
        grid_limit=grid_limit,
        folds=folds,
        resource_mode=normalized_resource_mode,
        random_state=int(random_state),
        fast_mode=bool(fast_mode),
        custom_grids=custom_grids,
    )
    if progress_callback is not None:
        resolve_checkpoint_label = "Resolviendo checkpoint compatible..."
        resolve_checkpoint_detail = "Calculando identificador de corrida y validando artefactos persistidos."
        if recompute_blocks_from_checkpoint:
            resolve_checkpoint_label = (
                "Resolviendo checkpoint cargado..."
                if checkpoint_run_id_override
                else "Resolviendo checkpoint para recompute..."
            )
            resolve_checkpoint_detail = "Leyendo manifest persistido y validando tuning/cache requeridos."
        progress_callback(
            {
                "completed_units": 0.0,
                "total_units": int(total_units),
                "label": resolve_checkpoint_label,
                "detail": resolve_checkpoint_detail,
            }
        )
    available_years = _available_prediction_years(df, time_col)
    prediction_years = [
        int(year)
        for year in available_years
        if effective_base_year is not None and int(year) > int(effective_base_year)
    ]

    run_manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "models": list(model_names),
        "strategies": list(strategies),
        "arf_variants": list(arf_variants),
        "kswin_variants": list(kswin_variants),
        "balance_modes": [str(mode) for mode in balance_modes],
        "repetition_seeds": [int(seed) for seed in repetition_seeds],
        "base_year": int(effective_base_year) if effective_base_year is not None else None,
        "available_years": [int(year) for year in available_years],
        "prediction_years": prediction_years,
        "validation_size": float(validation_size),
        "folds": int(folds),
        "random_state_fallback": int(random_state),
        "fast_mode": bool(fast_mode),
        "resource_mode": str(normalized_resource_mode),
        "resource_mode_label": str(resource_policy["label"]),
        "resource_policy": _to_json_safe(resource_policy),
        "custom_grids": _to_json_safe(custom_grids or {}),
        "optuna_trials": int(grid_limit) if grid_limit is not None else None,
        "adwin_delta": float(adwin_delta),
        "min_window": int(min_window),
        "min_retrain_size": None if min_retrain_size is None else int(min_retrain_size),
        "kswin_top_k_features": int(kswin_top_k_features),
        "kswin_vote_threshold": int(kswin_vote_threshold),
        "kswin_retrain_days": int(kswin_retrain_days),
        "kswin_min_retrain_rows": int(kswin_min_retrain_rows),
        "continue_on_block_error": bool(continue_on_block_error),
        "feature_count": int(len(feature_cols)),
        "feature_cols": list(feature_cols),
        "target_col": str(target_col),
        "time_col": str(time_col),
        "input_rows": int(len(df)),
        "total_progress_units": int(total_units),
        "total_tuning_tasks": int(len(tuning_tasks)),
        "total_block_units": int(total_block_units),
        "feature_selection_context": normalized_feature_selection_context,
        "preflight": preflight,
    }
    computed_run_id = _build_recalibration_run_id(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        time_col=time_col,
        model_names=list(model_names),
        strategies=list(strategies),
        arf_variants=list(arf_variants),
        kswin_variants=list(kswin_variants),
        balance_modes=list(balance_modes),
        repetition_seeds=list(repetition_seeds),
        base_year=effective_base_year,
        validation_size=float(validation_size),
        folds=int(folds),
        random_state=int(random_state),
        fast_mode=bool(fast_mode),
        resource_mode=str(normalized_resource_mode),
        grid_limit=grid_limit,
        adwin_delta=float(adwin_delta),
        min_window=int(min_window),
        min_retrain_size=min_retrain_size,
        kswin_top_k_features=int(kswin_top_k_features),
        kswin_vote_threshold=int(kswin_vote_threshold),
        kswin_retrain_days=int(kswin_retrain_days),
        kswin_min_retrain_rows=int(kswin_min_retrain_rows),
        continue_on_block_error=bool(continue_on_block_error),
        custom_grids=custom_grids,
        feature_selection_context=normalized_feature_selection_context,
    )
    run_id = str(checkpoint_run_id_override or computed_run_id)
    run_dir = _recalibration_run_dir(run_id, checkpoint_root=checkpoint_root)
    paths = _recalibration_run_paths(run_dir)
    _ensure_recalibration_run_dirs(paths)
    live_status_path = paths["live_status"]
    live_events_path = paths["live_events"]
    run_manifest["run_id"] = str(run_id)
    run_manifest["computed_run_id"] = str(computed_run_id)
    run_manifest["checkpoint_run_id_override"] = None if checkpoint_run_id_override is None else str(checkpoint_run_id_override)
    run_manifest["checkpoint_run_dir"] = str(paths["run_dir"])
    run_manifest["checkpoint_manifest_path"] = str(paths["manifest"])

    manifest: Optional[Dict[str, Any]] = None
    existing_manifest = _load_manifest(paths["manifest"]) if auto_resume else None
    if existing_manifest is not None:
        manifest = _reconcile_recalibration_manifest(existing_manifest, paths=paths)
        manifest["run_manifest"] = _to_json_safe(run_manifest)
        manifest["feature_selection_context"] = _to_json_safe(normalized_feature_selection_context)
        existing_status = str(existing_manifest.get("status", "running"))
        if recompute_blocks_from_checkpoint:
            if persist_progress:
                _reset_recalibration_live_artifacts(paths)
            manifest = _reset_recalibration_blocks_for_rerun(
                manifest,
                paths=paths,
                checkpoint_status=existing_status,
            )
        else:
            manifest["resume"] = {
                "auto_resumed": True,
                "checkpoint_status": existing_status,
            }
            if existing_status != "completed":
                manifest["status"] = "running"
    if manifest is None:
        if persist_progress:
            _reset_recalibration_live_artifacts(paths)
        manifest = _initial_recalibration_manifest(
            run_id=run_id,
            run_manifest=run_manifest,
            feature_selection_context=normalized_feature_selection_context,
            experiment_blocks=experiment_blocks,
            tuning_tasks=tuning_tasks,
            preflight=preflight,
        )
    if persist_progress:
        _persist_manifest(paths["manifest"], manifest)

    if auto_resume and not recompute_blocks_from_checkpoint and str(manifest.get("status")) == "completed":
        emit_progress(
            label="Checkpoint ya completado. Cargando resultados persistidos...",
            detail=f"{total_units} / {total_units} unidades completadas",
            completed_override=total_units,
            context={
                "phase": "checkpoint_load",
                "checkpoint_status": "completed",
            },
        )
        outputs = _assemble_recalibration_outputs_from_checkpoints(
            manifest,
            paths=paths,
            feature_selection_context=normalized_feature_selection_context,
        )
        outputs["optuna_json_path"] = str(outputs["run_manifest"].get("optuna_json_path", ""))
        outputs["checkpoint_manifest"] = manifest
        outputs["checkpoint_manifest_path"] = str(paths["manifest"])
        outputs["checkpoint_run_dir"] = str(paths["run_dir"])
        outputs["auto_resumed"] = True
        outputs["preflight"] = preflight
        return outputs

    tuning_cache = _load_completed_tuning_cache(paths, manifest) if reuse_tuning_cache else {}
    if recompute_blocks_from_checkpoint:
        missing_tuning_tasks: List[Dict[str, Any]] = []
        for task in tuning_tasks:
            tuning_key = str(task.get("tuning_key", ""))
            if tuning_key in tuning_cache:
                continue
            train_df = _materialize_tuning_train_df(
                df,
                time_col=time_col,
                task=task,
            )
            compatible_cached = _resolve_compatible_cached_tuning_artifact(
                manifest=manifest,
                tuning_cache=tuning_cache,
                tuning_key=tuning_key,
                train_df=train_df,
                feature_cols=feature_cols,
                target_col=target_col,
                time_col=time_col,
                model_name=str(task.get("model_name", "")),
                balance_mode=str(task.get("balance_mode", "")),
                tuning_filename=f"{tuning_key}.json",
            )
            if compatible_cached is None:
                missing_tuning_tasks.append(dict(task))
        if missing_tuning_tasks:
            first_missing = missing_tuning_tasks[0]
            missing_error = RuntimeError(
                "Selected checkpoint is missing persisted tuning artifacts for the current training configuration. "
                f"First missing tuning: {first_missing.get('model_name')} | "
                f"{first_missing.get('balance_mode')} | "
                f"{first_missing.get('training_year_label') or first_missing.get('window_kind') or 'window'}"
            )
            manifest["status"] = "failed"
            manifest["failed_block_id"] = None
            manifest["last_error"] = {
                "phase": "checkpoint_validation",
                "mode": "recompute_blocks",
                "error": str(missing_error),
                "missing_tuning_key": str(first_missing.get("tuning_key", "")),
            }
            _append_execution_log(
                manifest.get("global_execution_log"),
                phase="checkpoint_validation_error",
                status="error",
                message="Recompute-only checkpoint validation detected missing persisted tuning artifacts.",
                error=str(missing_error),
                missing_tuning_key=str(first_missing.get("tuning_key", "")),
                model=str(first_missing.get("model_name", "")),
                balance_mode=str(first_missing.get("balance_mode", "")),
                window_kind=str(first_missing.get("window_kind", "")),
                training_years=first_missing.get("training_years"),
            )
            if persist_progress:
                _persist_manifest(paths["manifest"], manifest)
            emit_progress(
                label="Checkpoint incompleto para recompute-only.",
                detail=str(missing_error),
                completed_override=float(len(tuning_cache)),
                context={
                    "phase": "checkpoint_validation_error",
                    "checkpoint_mode": "recompute_blocks",
                    "missing_tuning_key": str(first_missing.get("tuning_key", "")),
                    "model": str(first_missing.get("model_name", "")),
                    "balance_mode": str(first_missing.get("balance_mode", "")),
                },
            )
            raise missing_error

    block_specs: List[Dict[str, Any]] = []
    for run_order, seed in enumerate(repetition_seeds, start=1):
        for block in experiment_blocks:
            block_id = _build_recalibration_block_id(block, run_seed=int(seed), run_order=int(run_order))
            block_specs.append(
                {
                    "block_id": block_id,
                    "block": dict(block),
                    "run_seed": int(seed),
                    "run_order": int(run_order),
                }
            )
            manifest["block_index"][block_id] = {
                **dict((manifest.get("block_index") or {}).get(block_id) or {}),
                "filename": f"{block_id}.json",
                "status": str((manifest.get("block_index") or {}).get(block_id, {}).get("status", "pending")),
                "strategy": str(block.get("strategy", "")),
                "model": str(block.get("model", "")),
                "detector_variant": str(block.get("detector_variant", "")),
                "balance_mode": str(block.get("balance_mode", BALANCE_MODE_NOT_APPLICABLE)),
                "run_seed": int(seed),
                "run_order": int(run_order),
            }

    completed_block_ids = {str(item) for item in manifest.get("completed_block_ids", [])}
    skipped_failed_block_ids = {str(item) for item in manifest.get("skipped_failed_block_ids", [])}
    terminal_block_ids = completed_block_ids | skipped_failed_block_ids
    completed_units = float(len(tuning_cache) + len(terminal_block_ids))
    pending_block_ids = [spec["block_id"] for spec in block_specs if spec["block_id"] not in terminal_block_ids]
    skipped_blocks_by_run: Dict[Tuple[int, int], int] = {}
    for skipped_block_id in skipped_failed_block_ids:
        skipped_info = dict((manifest.get("block_index") or {}).get(skipped_block_id) or {})
        skipped_key = (
            int(skipped_info.get("run_seed", 0)),
            int(skipped_info.get("run_order", 0)),
        )
        skipped_blocks_by_run[skipped_key] = skipped_blocks_by_run.get(skipped_key, 0) + 1
    pending_blocks_by_run: Dict[Tuple[int, int], int] = {}
    for spec in block_specs:
        if spec["block_id"] in terminal_block_ids:
            continue
        key = (int(spec["run_seed"]), int(spec["run_order"]))
        pending_blocks_by_run[key] = pending_blocks_by_run.get(key, 0) + 1
        skipped_blocks_by_run.setdefault(key, 0)
    _update_manifest_progress(
        manifest,
        completed_units=completed_units,
        pending_block_ids=pending_block_ids,
    )
    if persist_progress:
        _persist_manifest(paths["manifest"], manifest)

    resume_mode = str(manifest.get("resume", {}).get("checkpoint_mode", "auto"))
    if resume_mode == "recompute_blocks":
        checkpoint_label = "Retomando checkpoint y recalculando entrenamientos..."
    elif bool(manifest.get("resume", {}).get("auto_resumed")):
        checkpoint_label = "Reanudando checkpoint..."
    else:
        checkpoint_label = "Iniciando experimentos de recalibracion..."
    emit_progress(
        label=checkpoint_label,
        detail=f"{completed_units:.2f} / {total_units} unidades completadas",
        completed_override=completed_units,
        context={
            "phase": "checkpoint_start",
            "checkpoint_status": str(manifest.get("resume", {}).get("checkpoint_status", "fresh")),
            "auto_resumed": bool(manifest.get("resume", {}).get("auto_resumed")),
            "checkpoint_mode": resume_mode,
        },
    )

    for task_idx, task in enumerate(tuning_tasks, start=1):
        model_name = str(task["model_name"])
        balance_mode = str(task["balance_mode"])
        tuning_key = str(task["tuning_key"])
        if reuse_tuning_cache and tuning_key in tuning_cache:
            continue

        train_df = _materialize_tuning_train_df(
            df,
            time_col=time_col,
            task=task,
        )
        rss_before = _current_rss_mb()
        _append_execution_log(
            manifest.get("global_execution_log"),
            phase="window_tuning_start",
            status="started",
            message="Starting window-scoped tuning artifact creation.",
            tuning_key=tuning_key,
            model=model_name,
            balance_mode=balance_mode,
            task_index=int(task_idx),
            total_tasks=int(len(tuning_tasks)),
            run_id=run_id,
            training_years=task.get("training_years"),
            strategy_context=task.get("strategy_context"),
            rss_before_mb=rss_before,
        )
        try:
            emit_progress(
                label="Sintonizando hiperparametros por ventana...",
                detail=(
                    f"Modelo {model_name} | Balance {balance_mode} | "
                    f"ventana {task_idx} / {len(tuning_tasks)} | {str(task.get('training_year_label', '') or task.get('window_kind', 'window'))}"
                ),
                completed_override=completed_units,
                context={
                    "phase": "window_tuning",
                    "model": model_name,
                    "balance_mode": balance_mode,
                    "task_index": int(task_idx),
                    "total_tasks": int(len(tuning_tasks)),
                    "tuning_key": tuning_key,
                    "window_kind": str(task.get("window_kind", "")),
                    "training_years": task.get("training_years"),
                    "prediction_year": task.get("prediction_year"),
                },
            )
            tuning_artifact = _resolve_or_create_tuning_artifact(
                paths=paths,
                manifest=manifest,
                tuning_cache=tuning_cache,
                train_df=train_df,
                feature_cols=feature_cols,
                target_col=target_col,
                time_col=time_col,
                model_name=model_name,
                balance_mode=balance_mode,
                validation_size=float(validation_size),
                folds=int(folds),
                random_state=int(random_state),
                fast_mode=bool(fast_mode),
                resource_mode=str(normalized_resource_mode),
                grid_limit=grid_limit,
                custom_grid=(custom_grids or {}).get(model_name),
                tuning_context=dict(task.get("strategy_context") or {}),
                progress_callback=(
                    None
                    if progress_callback is None and not persist_progress
                    else lambda payload, _base=completed_units: emit_progress(
                        label=str(payload.get("label", "Sintonizando hiperparametros por ventana...")),
                        detail=f"Modelo {model_name} | Balance {balance_mode} | {str(payload.get('detail', '') or '')}".strip(" |"),
                        completed_override=_base + max(0.0, min(float(payload.get("ratio", 0.0)), 1.0)),
                        context={
                            "phase": "window_tuning",
                            "model": model_name,
                            "balance_mode": balance_mode,
                            "task_index": int(task_idx),
                            "total_tasks": int(len(tuning_tasks)),
                            "tuning_key": tuning_key,
                            "tuning_ratio": max(0.0, min(float(payload.get("ratio", 0.0)), 1.0)),
                            "window_kind": str(task.get("window_kind", "")),
                            "training_years": task.get("training_years"),
                            "prediction_year": task.get("prediction_year"),
                        },
                    )
                ),
                persist_progress=persist_progress,
                allow_create=not recompute_blocks_from_checkpoint,
            )
            rss_after = _cleanup_recalibration_memory()
            _update_memory_summary(
                manifest,
                phase="window_tuning_complete",
                rss_before_mb=rss_before,
                rss_after_mb=rss_after,
            )
            _append_execution_log(
                manifest.get("global_execution_log"),
                phase="window_tuning_complete",
                status="ok",
                message="Completed window-scoped tuning artifact creation.",
                tuning_key=tuning_key,
                model=model_name,
                balance_mode=balance_mode,
                study_id=tuning_artifact.get("study_id"),
                best_params=tuning_artifact.get("best_params"),
                best_value=tuning_artifact.get("best_value"),
                training_years=task.get("training_years"),
                prediction_year=task.get("prediction_year"),
                rss_after_mb=rss_after,
            )
            if str(model_name) == "NNet" and int(tuning_artifact.get("invalid_trial_count", 0)) > 0:
                total_non_converged_folds = int(
                    sum(
                        int(row.get("n_non_converged_folds", 0))
                        for row in (tuning_artifact.get("trials") or [])
                    )
                )
                _append_execution_log(
                    manifest.get("global_execution_log"),
                    phase="nnet_fold_non_converged",
                    status="warning",
                    message="NNet tuning produced non-converged cross-validation folds.",
                    tuning_key=tuning_key,
                    model=model_name,
                    balance_mode=balance_mode,
                    invalid_trial_count=int(tuning_artifact.get("invalid_trial_count", 0)),
                    non_converged_fold_count=total_non_converged_folds,
                    has_valid_trial=bool(tuning_artifact.get("has_valid_trial", True)),
                    study_id=tuning_artifact.get("study_id"),
                )
            if str(model_name) == "NNet" and not bool(tuning_artifact.get("has_valid_trial", True)):
                _append_execution_log(
                    manifest.get("global_execution_log"),
                    phase="nnet_trial_invalid",
                    status="skipped",
                    message="All persisted NNet tuning trials were invalid because none converged.",
                    tuning_key=tuning_key,
                    model=model_name,
                    balance_mode=balance_mode,
                    invalid_trial_count=int(tuning_artifact.get("invalid_trial_count", 0)),
                    study_id=tuning_artifact.get("study_id"),
                )
            completed_units += 1.0
            pending_block_ids = [spec["block_id"] for spec in block_specs if spec["block_id"] not in terminal_block_ids]
            _update_manifest_progress(
                manifest,
                completed_units=completed_units,
                pending_block_ids=pending_block_ids,
            )
            if persist_progress:
                _persist_manifest(paths["manifest"], manifest)
            emit_progress(
                label="Tuning por ventana completado.",
                detail=f"Modelo {model_name} | Balance {balance_mode} | {completed_units:.2f} / {total_units} unidades",
                completed_override=completed_units,
                context={
                    "phase": "window_tuning_complete",
                    "model": model_name,
                    "balance_mode": balance_mode,
                    "task_index": int(task_idx),
                    "total_tasks": int(len(tuning_tasks)),
                    "study_id": tuning_artifact.get("study_id"),
                    "best_value": tuning_artifact.get("best_value"),
                    "tuning_key": tuning_key,
                    "window_kind": str(task.get("window_kind", "")),
                    "training_years": task.get("training_years"),
                    "prediction_year": task.get("prediction_year"),
                },
            )
        except Exception as exc:
            rss_after = _cleanup_recalibration_memory()
            _update_memory_summary(
                manifest,
                phase="window_tuning_error",
                rss_before_mb=rss_before,
                rss_after_mb=rss_after,
            )
            manifest["status"] = "failed"
            manifest["failed_block_id"] = None
            manifest["last_error"] = {
                "phase": "window_tuning",
                "model": model_name,
                "balance_mode": balance_mode,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=10),
            }
            _append_execution_log(
                manifest.get("global_execution_log"),
                phase="window_tuning_error",
                status="error",
                message="Window-scoped tuning failed.",
                tuning_key=tuning_key,
                model=model_name,
                balance_mode=balance_mode,
                error=str(exc),
                rss_after_mb=rss_after,
            )
            if persist_progress:
                _persist_manifest(paths["manifest"], manifest)
            emit_progress(
                label="Error en tuning por ventana.",
                detail=f"Modelo {model_name} | Balance {balance_mode} | {exc}",
                completed_override=completed_units,
                context={
                    "phase": "window_tuning_error",
                    "model": model_name,
                    "balance_mode": balance_mode,
                    "task_index": int(task_idx),
                    "total_tasks": int(len(tuning_tasks)),
                    "tuning_key": tuning_key,
                },
            )
            raise

    for spec in block_specs:
        block_id = str(spec["block_id"])
        block = dict(spec["block"])
        run_seed = int(spec["run_seed"])
        run_order = int(spec["run_order"])
        if block_id in terminal_block_ids:
            continue
        run_key = (run_seed, run_order)
        if pending_blocks_by_run.get(run_key, 0) and pending_blocks_by_run.get(run_key, 0) == sum(
            1
            for item in block_specs
            if int(item["run_seed"]) == run_seed
            and int(item["run_order"]) == run_order
            and item["block_id"] not in terminal_block_ids
        ):
            _append_execution_log(
                manifest.get("global_execution_log"),
                phase="run_start",
                status="started",
                message="Starting sequential experiment repetition.",
                run_seed=run_seed,
                run_order=run_order,
                strategies=list(strategies),
                models=list(model_names),
                arf_variants=list(arf_variants),
                kswin_variants=list(kswin_variants),
                balance_modes=list(balance_modes),
            )
            if persist_progress:
                _persist_manifest(paths["manifest"], manifest)

        strategy = str(block["strategy"])
        model_name = str(block["model"])
        detector_variant = str(block.get("detector_variant") or "")
        balance_mode = str(block.get("balance_mode") or BALANCE_MODE_NOT_APPLICABLE)
        progress_model_label = f"{model_name} | {detector_variant}" if detector_variant else model_name
        block_detail_prefix = (
            f"Seed {run_seed} | Strategy {strategy} | Model {progress_model_label} | Balance {balance_mode}"
        )
        block_execution_log: List[Dict[str, Any]] = []
        smote_registry_block: Dict[str, Dict[str, Any]] = {}
        manifest["block_index"][block_id]["status"] = "running"
        if persist_progress:
            _persist_manifest(paths["manifest"], manifest)

        def block_progress(payload: Dict[str, Any]) -> None:
            block_ratio = max(0.0, min(float(payload.get("ratio", 0.0)), 1.0))
            inner_detail = str(payload.get("detail", "") or "")
            detail = block_detail_prefix
            if inner_detail:
                detail = f"{detail} | {inner_detail}"
            detail = f"{detail} | {completed_units + block_ratio:.2f} / {total_units} unidades"
            emit_progress(
                label=str(payload.get("label", "Ejecutando bloque de experimentos...")),
                detail=detail,
                completed_override=float(completed_units) + block_ratio,
                context={
                    "phase": "block_running",
                    "block_id": block_id,
                    "strategy": strategy,
                    "model": model_name,
                    "detector_variant": detector_variant,
                    "balance_mode": balance_mode,
                    "run_seed": run_seed,
                    "run_order": run_order,
                    "block_ratio": float(block_ratio),
                },
            )

        emit_progress(
            label="Ejecutando bloque de experimentos...",
            detail=f"{block_detail_prefix} | {completed_units:.2f} / {total_units} unidades completadas",
            completed_override=completed_units,
            context={
                "phase": "block_start",
                "block_id": block_id,
                "strategy": strategy,
                "model": model_name,
                "detector_variant": detector_variant,
                "balance_mode": balance_mode,
                "run_seed": run_seed,
                "run_order": run_order,
            },
        )

        tuning_refs: List[str] = []
        work_for_tuning = df.copy()
        work_for_tuning[time_col] = pd.to_datetime(work_for_tuning[time_col], errors="coerce")
        work_for_tuning = work_for_tuning.dropna(subset=[time_col]).reset_index(drop=True)

        def resolve_block_tuning_artifact(
            *,
            train_df: pd.DataFrame,
            strategy_context: Dict[str, Any],
            model_name: Optional[str] = None,
            balance_mode: Optional[str] = None,
        ) -> Dict[str, Any]:
            effective_model_name = str(model_name or block.get("model") or "")
            effective_balance_mode = str(balance_mode or block.get("balance_mode") or BALANCE_MODE_NOT_APPLICABLE)
            artifact = _resolve_or_create_tuning_artifact(
                paths=paths,
                manifest=manifest,
                tuning_cache=tuning_cache,
                train_df=train_df,
                feature_cols=feature_cols,
                target_col=target_col,
                time_col=time_col,
                model_name=effective_model_name,
                balance_mode=effective_balance_mode,
                validation_size=float(validation_size),
                folds=int(folds),
                random_state=int(random_state),
                fast_mode=bool(fast_mode),
                resource_mode=str(normalized_resource_mode),
                grid_limit=grid_limit,
                custom_grid=(custom_grids or {}).get(effective_model_name),
                tuning_context=strategy_context,
                persist_progress=persist_progress,
                allow_create=not recompute_blocks_from_checkpoint,
            )
            tuning_key = str(artifact.get("tuning_key", ""))
            if tuning_key and tuning_key not in tuning_refs:
                tuning_refs.append(tuning_key)
            return artifact

        base_train_block = pd.DataFrame()
        if (
            effective_base_year is not None
            and strategy in YEARLY_STRATEGIES + [ADAPTIVE_ADWIN_STRATEGY, ADAPTIVE_KSWIN_STRATEGY]
        ):
            base_train_block = work_for_tuning.loc[_as_year_mask(work_for_tuning, time_col, int(effective_base_year))].copy()

        rss_before = _current_rss_mb()
        _append_execution_log(
            block_execution_log,
            phase="block_start",
            status="started",
            message="Starting persisted experiment block.",
            block_id=block_id,
            strategy=strategy,
            model=progress_model_label,
            balance_mode=balance_mode,
            run_seed=run_seed,
            run_order=run_order,
            tuning_refs=tuning_refs,
            rss_before_mb=rss_before,
        )

        try:
            yearly_df = pd.DataFrame()
            adaptive_df = pd.DataFrame()
            roc_data: List[Dict[str, Any]] = []
            initial_tuning_artifact: Optional[Dict[str, Any]] = None
            if (
                str(model_name) == "NNet"
                and not base_train_block.empty
                and strategy in {"static", ADAPTIVE_ADWIN_STRATEGY, ADAPTIVE_KSWIN_STRATEGY}
            ):
                initial_tuning_artifact = resolve_block_tuning_artifact(
                    train_df=base_train_block,
                    strategy_context={
                        "stage": "block_precheck",
                        "strategy": strategy,
                        "window_kind": "base_year",
                        "training_year": str(int(effective_base_year)),
                        "run_seed": int(run_seed),
                        "run_order": int(run_order),
                        "detector_variant": detector_variant or None,
                    },
                )
            if (
                str(model_name) == "NNet"
                and initial_tuning_artifact is not None
                and not bool(initial_tuning_artifact.get("has_valid_trial", True))
            ):
                _append_execution_log(
                    block_execution_log,
                    phase="nnet_trial_invalid",
                    status="skipped",
                    message="Skipped experiment block because the persisted NNet tuning artifact has no converged trial.",
                    block_id=block_id,
                    strategy=strategy,
                    model=progress_model_label,
                    balance_mode=balance_mode,
                    run_seed=run_seed,
                    run_order=run_order,
                    tuning_refs=tuning_refs,
                    study_id=str((initial_tuning_artifact or {}).get("study_id", "")),
                    invalid_trial_count=int((initial_tuning_artifact or {}).get("invalid_trial_count", 0)),
                )
            elif strategy in YEARLY_STRATEGIES:
                yearly_df, roc_data = run_yearly_strategy(
                    df,
                    strategy=strategy,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    time_col=time_col,
                    model_names=[model_name],
                    base_year=effective_base_year,
                    validation_size=validation_size,
                    folds=folds,
                    random_state=random_state,
                    fast_mode=fast_mode,
                    resource_mode=normalized_resource_mode,
                    grid_limit=grid_limit,
                    custom_grids=custom_grids,
                    run_seed=run_seed,
                    run_order=run_order,
                    execution_log=block_execution_log,
                    balance_mode=balance_mode,
                    study_collector=None,
                    tuning_resolver=resolve_block_tuning_artifact,
                    run_id=run_id,
                    smote_cache_dir=paths["smote_dir"] if reuse_smote_cache else None,
                    smote_cache_registry=smote_registry_block,
                    progress_callback=block_progress,
                )
            elif strategy == ADAPTIVE_ADWIN_STRATEGY:
                adaptive_df, roc_data = run_adaptive_strategy(
                    df,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    time_col=time_col,
                    model_names=[model_name],
                    base_year=effective_base_year,
                    random_state=random_state,
                    validation_size=validation_size,
                    folds=folds,
                    fast_mode=fast_mode,
                    resource_mode=normalized_resource_mode,
                    grid_limit=grid_limit,
                    adwin_delta=adwin_delta,
                    min_window=min_window,
                    min_retrain_size=min_retrain_size,
                    custom_grids=custom_grids,
                    run_seed=run_seed,
                    run_order=run_order,
                    execution_log=block_execution_log,
                    balance_mode=balance_mode,
                    study_collector=None,
                    tuning_resolver=resolve_block_tuning_artifact,
                    run_id=run_id,
                    smote_cache_dir=paths["smote_dir"] if reuse_smote_cache else None,
                    smote_cache_registry=smote_registry_block,
                    progress_callback=block_progress,
                )
            elif strategy == ADAPTIVE_ARF_STRATEGY:
                adaptive_df, roc_data = run_arf_strategy(
                    df,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    time_col=time_col,
                    arf_variants=[model_name],
                    base_year=effective_base_year,
                    random_state=random_state,
                    validation_size=validation_size,
                    run_seed=run_seed,
                    run_order=run_order,
                    execution_log=block_execution_log,
                    progress_callback=block_progress,
                )
            elif strategy == ADAPTIVE_KSWIN_STRATEGY:
                adaptive_df, roc_data = run_kswin_strategy(
                    df,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    time_col=time_col,
                    model_names=[model_name],
                    kswin_variants=[detector_variant],
                    base_year=effective_base_year,
                    random_state=random_state,
                    validation_size=validation_size,
                    folds=folds,
                    fast_mode=fast_mode,
                    resource_mode=normalized_resource_mode,
                    grid_limit=grid_limit,
                    kswin_top_k_features=kswin_top_k_features,
                    kswin_vote_threshold=kswin_vote_threshold,
                    kswin_retrain_days=kswin_retrain_days,
                    kswin_min_retrain_rows=kswin_min_retrain_rows,
                    custom_grids=custom_grids,
                    run_seed=run_seed,
                    run_order=run_order,
                    execution_log=block_execution_log,
                    balance_mode=balance_mode,
                    study_collector=None,
                    tuning_resolver=resolve_block_tuning_artifact,
                    run_id=run_id,
                    smote_cache_dir=paths["smote_dir"] if reuse_smote_cache else None,
                    smote_cache_registry=smote_registry_block,
                    progress_callback=block_progress,
                )
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")

            block_errors = [
                entry
                for entry in block_execution_log
                if isinstance(entry, dict) and str(entry.get("status", "")).lower() == "error"
            ]
            if yearly_df.empty and adaptive_df.empty and block_errors:
                last_error = dict(block_errors[-1])
                error_text = str(last_error.get("error") or last_error.get("message") or "Internal block error.")
                error_phase = str(last_error.get("phase") or "block_internal_error")
                raise RuntimeError(f"{error_phase}: {error_text}")

            rss_after = _cleanup_recalibration_memory()
            _update_memory_summary(
                manifest,
                phase="block_complete",
                rss_before_mb=rss_before,
                rss_after_mb=rss_after,
            )
            _append_execution_log(
                block_execution_log,
                phase="block_complete",
                status="ok",
                message="Completed persisted experiment block.",
                block_id=block_id,
                strategy=strategy,
                model=progress_model_label,
                balance_mode=balance_mode,
                run_seed=run_seed,
                run_order=run_order,
                rss_after_mb=rss_after,
            )
            block_payload = _serialize_block_payload(
                block_id=block_id,
                block=block,
                run_seed=run_seed,
                run_order=run_order,
                yearly_rows=yearly_df,
                adaptive_rows=adaptive_df,
                roc_payload=roc_data,
                execution_log=block_execution_log,
                tuning_refs=tuning_refs,
                smote_refs=sorted(smote_registry_block.keys()),
            )
            if persist_progress:
                _persist_recalibration_block(paths["blocks_dir"] / f"{block_id}.json", block_payload)
            manifest["block_index"][block_id].update(
                {
                    "status": "completed",
                    "filename": f"{block_id}.json",
                    "tuning_refs": tuning_refs,
                    "smote_refs": sorted(smote_registry_block.keys()),
                }
            )
            for smote_key, smote_info in smote_registry_block.items():
                manifest["smote_index"][smote_key] = {
                    "filename": Path(str(smote_info.get("path", f"{smote_key}.duckdb"))).name,
                    "status": str(smote_info.get("status", "available")),
                    "path": str(smote_info.get("path", "")),
                }
            completed_block_ids.add(block_id)
            manifest["completed_block_ids"] = sorted(completed_block_ids)
            manifest["failed_block_id"] = None
            manifest["last_error"] = None
            completed_units += 1.0
            terminal_block_ids = completed_block_ids | skipped_failed_block_ids
            pending_block_ids = [item["block_id"] for item in block_specs if item["block_id"] not in terminal_block_ids]
            pending_blocks_by_run[run_key] = max(0, int(pending_blocks_by_run.get(run_key, 0)) - 1)
            if pending_blocks_by_run.get(run_key, 0) == 0:
                _append_execution_log(
                    manifest.get("global_execution_log"),
                    phase="run_complete",
                    status="warning" if int(skipped_blocks_by_run.get(run_key, 0)) > 0 else "ok",
                    message="Completed sequential experiment repetition.",
                    run_seed=run_seed,
                    run_order=run_order,
                    skipped_failed_blocks=int(skipped_blocks_by_run.get(run_key, 0)),
                )
            _update_manifest_progress(
                manifest,
                completed_units=completed_units,
                pending_block_ids=pending_block_ids,
            )
            if persist_progress:
                _persist_manifest(paths["manifest"], manifest)
            emit_progress(
                label="Bloque completado.",
                detail=f"{block_detail_prefix} | {completed_units:.2f} / {total_units} unidades",
                completed_override=completed_units,
                context={
                    "phase": "block_complete",
                    "block_id": block_id,
                    "strategy": strategy,
                    "model": model_name,
                    "detector_variant": detector_variant,
                    "balance_mode": balance_mode,
                    "run_seed": run_seed,
                    "run_order": run_order,
                    "yearly_rows": int(len(yearly_df)),
                    "adaptive_rows": int(len(adaptive_df)),
                },
            )

            del yearly_df, adaptive_df, roc_data, block_payload
            _cleanup_recalibration_memory()
        except Exception as exc:
            rss_after = _cleanup_recalibration_memory()
            _update_memory_summary(
                manifest,
                phase="block_error",
                rss_before_mb=rss_before,
                rss_after_mb=rss_after,
            )
            _append_execution_log(
                block_execution_log,
                phase="block_error",
                status="error",
                message="Experiment block failed.",
                block_id=block_id,
                strategy=strategy,
                model=progress_model_label,
                balance_mode=balance_mode,
                run_seed=run_seed,
                run_order=run_order,
                error=str(exc),
                rss_after_mb=rss_after,
            )
            manifest["status"] = "failed"
            manifest["failed_block_id"] = block_id
            manifest["last_error"] = {
                "phase": "block",
                "block_id": block_id,
                "strategy": strategy,
                "model": progress_model_label,
                "balance_mode": balance_mode,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=10),
            }
            manifest["block_index"][block_id].update(
                {
                    "status": "failed",
                    "error": str(exc),
                }
            )
            if persist_progress:
                _persist_manifest(paths["manifest"], manifest)
            emit_progress(
                label="Error en bloque de experimentos.",
                detail=f"{block_detail_prefix} | {exc}",
                completed_override=completed_units,
                context={
                    "phase": "block_error",
                    "block_id": block_id,
                    "strategy": strategy,
                    "model": model_name,
                    "detector_variant": detector_variant,
                    "balance_mode": balance_mode,
                    "run_seed": run_seed,
                    "run_order": run_order,
                },
            )
            if not continue_on_block_error:
                raise

            manifest["status"] = "running"
            manifest["failed_block_id"] = None
            manifest["last_error"] = None
            skipped_failed_block_ids.add(block_id)
            terminal_block_ids = completed_block_ids | skipped_failed_block_ids
            skipped_blocks_by_run[run_key] = int(skipped_blocks_by_run.get(run_key, 0)) + 1
            error_payload = {
                "phase": "block",
                "block_id": block_id,
                "strategy": strategy,
                "model": progress_model_label,
                "balance_mode": balance_mode,
                "run_seed": run_seed,
                "run_order": run_order,
                "error": str(exc),
            }
            manifest["nonfatal_block_errors"] = list(manifest.get("nonfatal_block_errors") or []) + [error_payload]
            manifest["skipped_failed_block_ids"] = sorted(skipped_failed_block_ids)
            manifest["block_index"][block_id].update(
                {
                    "status": "skipped_failed",
                    "error": str(exc),
                }
            )
            completed_units += 1.0
            pending_block_ids = [item["block_id"] for item in block_specs if item["block_id"] not in terminal_block_ids]
            pending_blocks_by_run[run_key] = max(0, int(pending_blocks_by_run.get(run_key, 0)) - 1)
            if pending_blocks_by_run.get(run_key, 0) == 0:
                _append_execution_log(
                    manifest.get("global_execution_log"),
                    phase="run_complete",
                    status="warning",
                    message="Completed sequential experiment repetition with skipped failed blocks.",
                    run_seed=run_seed,
                    run_order=run_order,
                    skipped_failed_blocks=int(skipped_blocks_by_run.get(run_key, 0)),
                )
            _update_manifest_progress(
                manifest,
                completed_units=completed_units,
                pending_block_ids=pending_block_ids,
            )
            if persist_progress:
                _persist_manifest(paths["manifest"], manifest)
            emit_progress(
                label="Bloque omitido por error; continuando corrida.",
                detail=f"{block_detail_prefix} | {exc}",
                completed_override=completed_units,
                context={
                    "phase": "block_skipped_failed",
                    "block_id": block_id,
                    "strategy": strategy,
                    "model": model_name,
                    "detector_variant": detector_variant,
                    "balance_mode": balance_mode,
                    "run_seed": run_seed,
                    "run_order": run_order,
                },
            )
            continue

    manifest["status"] = "completed"
    manifest["failed_block_id"] = None
    manifest["last_error"] = None
    manifest["resume"] = {
        "auto_resumed": bool(manifest.get("resume", {}).get("auto_resumed")),
        "checkpoint_status": "completed",
        "checkpoint_mode": str(manifest.get("resume", {}).get("checkpoint_mode", "auto")),
    }
    _update_manifest_progress(
        manifest,
        completed_units=float(total_units),
        pending_block_ids=[],
    )

    assembled = _assemble_recalibration_outputs_from_checkpoints(
        manifest,
        paths=paths,
        feature_selection_context=normalized_feature_selection_context,
    )
    studies = list(assembled.get("studies") or [])
    optuna_json_path = _persist_drift_recalibration_json(
        run_manifest=assembled["run_manifest"],
        feature_selection_context=normalized_feature_selection_context,
        studies=studies,
        yearly_results=assembled["yearly_results"],
        adaptive_results=assembled["adaptive_results"],
        summary=assembled["summary"],
        appendix_tables=assembled["appendix_tables"],
        appendix_tables_mean=assembled["appendix_tables_mean"],
        average_roc=assembled["average_roc"],
        roc_payload=assembled["roc_payload"],
        execution_log=assembled["execution_log"],
    )
    manifest["run_manifest"]["optuna_json_path"] = str(optuna_json_path)
    if persist_progress:
        _persist_manifest(paths["manifest"], manifest)

    emit_progress(
        label="Experimentos de recalibracion completados.",
        detail=f"{total_units} / {total_units} unidades completadas",
        completed_override=total_units,
        context={
            "phase": "run_complete",
            "checkpoint_status": "completed",
        },
    )
    assembled["run_manifest"] = dict(manifest.get("run_manifest") or {})
    assembled["checkpoint_manifest"] = manifest
    assembled["feature_selection_context"] = normalized_feature_selection_context
    assembled["optuna_json_path"] = str(optuna_json_path)
    assembled["checkpoint_manifest_path"] = str(paths["manifest"])
    assembled["checkpoint_run_dir"] = str(paths["run_dir"])
    assembled["auto_resumed"] = bool(manifest.get("resume", {}).get("auto_resumed"))
    assembled["preflight"] = preflight
    return assembled


def build_average_roc_curves(
    roc_payload: Sequence[Dict[str, Any]],
    *,
    n_points: int = 101,
) -> pd.DataFrame:
    """
    Builds Figure 7 style average ROC curves by strategy/model.
    """
    if not roc_payload:
        return pd.DataFrame(columns=["strategy", "model", "balance_mode", "fpr", "tpr", "label"])

    fpr_grid = np.linspace(0.0, 1.0, int(n_points))
    rows: List[Dict[str, Any]] = []

    keys = sorted({(r["strategy"], r["model"], r.get("balance_mode", BALANCE_MODE_NOT_APPLICABLE)) for r in roc_payload})
    for strategy, model, balance_mode in keys:
        curves: List[np.ndarray] = []
        for item in roc_payload:
            if (
                item["strategy"] != strategy
                or item["model"] != model
                or str(item.get("balance_mode", BALANCE_MODE_NOT_APPLICABLE)) != str(balance_mode)
            ):
                continue
            y_true = np.asarray(item["y_true"]).astype(int)
            scores = np.asarray(item["scores"]).astype(float)
            if np.unique(y_true).size < 2:
                continue
            try:
                fpr, tpr, _ = roc_curve(y_true, scores)
                interp_tpr = np.interp(fpr_grid, fpr, tpr)
                interp_tpr[0] = 0.0
                interp_tpr[-1] = 1.0
                curves.append(interp_tpr)
            except Exception:
                continue

        if not curves:
            continue
        mean_tpr = np.mean(np.vstack(curves), axis=0)
        label = f"{strategy} | {model} | {balance_mode}"
        for fpr_val, tpr_val in zip(fpr_grid, mean_tpr):
            rows.append(
                {
                    "strategy": strategy,
                    "model": model,
                    "balance_mode": str(balance_mode),
                    "fpr": float(fpr_val),
                    "tpr": float(tpr_val),
                    "label": label,
                }
            )

    return pd.DataFrame(rows)


YEARLY_EVOLUTION_METRICS = [
    "auc",
    "pr_auc",
    "f1",
    "sensitivity",
    "specificity",
    "sensitivity_before_calibration",
    "specificity_before_calibration",
    "sensitivity_after_calibration",
    "specificity_after_calibration",
    "error_rate",
    "threshold",
    "training_time_sec",
    "n_train",
    "n_test",
]
ADAPTIVE_EVOLUTION_METRICS = [
    "auc",
    "pr_auc",
    "f1",
    "sensitivity",
    "specificity",
    "sensitivity_before_calibration",
    "specificity_before_calibration",
    "sensitivity_after_calibration",
    "specificity_after_calibration",
    "error_rate",
    "threshold",
    "training_time_sec",
    "remaining_periods",
    "segment_rows",
    "n_internal_drifts",
    "n_internal_warnings",
    "W",
    "W0",
    "W1",
]


def _build_experiment_evolution_records(
    df: pd.DataFrame,
    *,
    source: str,
    metric_candidates: Sequence[str],
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "source",
                "strategy",
                "model",
                "balance_mode",
                "run_seed",
                "run_order",
                "training_year",
                "prediction_year",
                "drift",
                "drift_date",
                "event_time",
                "event_order",
                "event_label",
                "metric",
                "value",
            ]
        )

    work = df.copy()
    work["source"] = str(source)
    for col, default_value in [
        ("strategy", str(source)),
        ("model", ""),
        ("balance_mode", BALANCE_MODE_NOT_APPLICABLE),
        ("run_seed", np.nan),
        ("run_order", np.nan),
        ("training_year", np.nan),
        ("prediction_year", np.nan),
        ("drift", np.nan),
        ("drift_date", pd.NaT),
    ]:
        if col not in work.columns:
            work[col] = default_value

    prediction_year = pd.to_numeric(work["prediction_year"], errors="coerce")
    drift = pd.to_numeric(work["drift"], errors="coerce")
    drift_date = pd.to_datetime(work["drift_date"], errors="coerce")
    year_time = pd.to_datetime(
        prediction_year.map(lambda value: f"{int(value)}-01-01" if pd.notna(value) else None),
        errors="coerce",
    )
    work["event_time"] = drift_date.where(drift_date.notna(), year_time)
    work["event_order"] = prediction_year.where(prediction_year.notna(), drift)
    fallback_order = pd.Series(np.arange(1, len(work) + 1), index=work.index, dtype=float)
    work["event_order"] = work["event_order"].where(work["event_order"].notna(), fallback_order)
    event_label = drift_date.dt.strftime("%Y-%m-%d %H:%M").where(drift_date.notna(), None)
    event_label = event_label.where(event_label.notna(), prediction_year.map(lambda value: str(int(value)) if pd.notna(value) else None))
    event_label = event_label.where(event_label.notna(), drift.map(lambda value: f"drift_{int(value)}" if pd.notna(value) else None))
    work["event_label"] = event_label.where(event_label.notna(), fallback_order.map(lambda value: f"step_{int(value)}"))

    metric_cols = [str(col) for col in metric_candidates if str(col) in work.columns]
    if not metric_cols:
        return pd.DataFrame()

    id_cols = [
        "source",
        "strategy",
        "model",
        "balance_mode",
        "run_seed",
        "run_order",
        "training_year",
        "prediction_year",
        "drift",
        "drift_date",
        "event_time",
        "event_order",
        "event_label",
    ]
    long_df = work[id_cols + metric_cols].melt(
        id_vars=id_cols,
        value_vars=metric_cols,
        var_name="metric",
        value_name="value",
    )
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.loc[long_df["value"].notna()].copy()
    return long_df.sort_values(["metric", "event_order", "event_time", "strategy", "model"]).reset_index(drop=True)


def _build_tuning_trials_frame(studies: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for artifact in studies or []:
        if not isinstance(artifact, dict):
            continue
        model_name = str(artifact.get("model_name") or artifact.get("model") or "")
        balance_mode = str(artifact.get("balance_mode") or BALANCE_MODE_NOT_APPLICABLE)
        stage = str(artifact.get("stage") or "tuning")
        study_id = str(artifact.get("study_id") or "")
        for trial in artifact.get("trials") or []:
            rows.append(
                {
                    "study_id": study_id,
                    "stage": stage,
                    "model": model_name,
                    "balance_mode": balance_mode,
                    "trial_number": int(trial.get("trial_number", len(rows))),
                    "cv_auc": pd.to_numeric(trial.get("cv_auc"), errors="coerce"),
                    "state": str(trial.get("state", "")),
                    "converged": bool(trial.get("converged", True)),
                    "n_non_converged_folds": int(trial.get("n_non_converged_folds", 0)),
                    "n_valid_folds": int(trial.get("n_valid_folds", 0)),
                    "n_iter_final": pd.to_numeric(trial.get("n_iter_final"), errors="coerce"),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "study_id",
                "stage",
                "model",
                "balance_mode",
                "trial_number",
                "cv_auc",
                "state",
                "converged",
                "n_non_converged_folds",
                "n_valid_folds",
                "n_iter_final",
            ]
        )
    return pd.DataFrame(rows).sort_values(["model", "balance_mode", "stage", "trial_number"]).reset_index(drop=True)


def _build_execution_memory_trace(execution_log: pd.DataFrame) -> pd.DataFrame:
    if execution_log is None or execution_log.empty:
        return pd.DataFrame(columns=["order", "metric", "value", "phase", "status"])
    work = execution_log.copy()
    for col in ["order", "rss_before_mb", "rss_after_mb", "training_time_sec"]:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")
    rows: List[Dict[str, Any]] = []
    for metric in ["rss_before_mb", "rss_after_mb", "training_time_sec"]:
        metric_rows = work.loc[work[metric].notna(), ["order", "phase", "status", metric]].copy()
        if metric_rows.empty:
            continue
        metric_rows = metric_rows.rename(columns={metric: "value"})
        metric_rows["metric"] = metric
        rows.extend(metric_rows.to_dict(orient="records"))
    if not rows:
        return pd.DataFrame(columns=["order", "metric", "value", "phase", "status"])
    return pd.DataFrame(rows).sort_values(["metric", "order"]).reset_index(drop=True)


def _streamlit_arrow_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    work = df.copy()
    for col in work.columns:
        if pd.api.types.is_object_dtype(work[col]):
            work[col] = work[col].astype("string")
    return work


def summarize_results(
    yearly_results: pd.DataFrame,
    adaptive_results: pd.DataFrame,
) -> pd.DataFrame:
    """Builds compact summary table across strategies and models."""
    calibration_cols = [
        "sensitivity_before_calibration",
        "specificity_before_calibration",
        "sensitivity_after_calibration",
        "specificity_after_calibration",
    ]
    rows: List[pd.DataFrame] = []
    if yearly_results is not None and not yearly_results.empty:
        yearly_results = yearly_results.copy()
        for col in ["pr_auc", "f1", *calibration_cols]:
            if col not in yearly_results.columns:
                yearly_results[col] = np.nan
        y = (
            yearly_results.groupby(["strategy", "model", "balance_mode"], dropna=False)
            .agg(
                auc=("auc", "mean"),
                pr_auc=("pr_auc", "mean"),
                f1=("f1", "mean"),
                sensitivity=("sensitivity", "mean"),
                specificity=("specificity", "mean"),
                sensitivity_before_calibration=("sensitivity_before_calibration", "mean"),
                specificity_before_calibration=("specificity_before_calibration", "mean"),
                sensitivity_after_calibration=("sensitivity_after_calibration", "mean"),
                specificity_after_calibration=("specificity_after_calibration", "mean"),
                error_rate=("error_rate", "mean"),
                training_time_sec=("training_time_sec", "mean"),
                n_segments=("model", "size"),
                n_repetitions=(
                    "run_seed",
                    lambda s: int(s.dropna().astype(int).nunique()) if s.notna().any() else 1,
                ),
            )
            .reset_index()
        )
        rows.append(y)

    if adaptive_results is not None and not adaptive_results.empty:
        adaptive_results = adaptive_results.copy()
        for col in ["pr_auc", "f1", *calibration_cols]:
            if col not in adaptive_results.columns:
                adaptive_results[col] = np.nan
        a = (
            adaptive_results.groupby(["strategy", "model", "balance_mode"], dropna=False)
            .agg(
                auc=("auc", "mean"),
                pr_auc=("pr_auc", "mean"),
                f1=("f1", "mean"),
                sensitivity=("sensitivity", "mean"),
                specificity=("specificity", "mean"),
                sensitivity_before_calibration=("sensitivity_before_calibration", "mean"),
                specificity_before_calibration=("specificity_before_calibration", "mean"),
                sensitivity_after_calibration=("sensitivity_after_calibration", "mean"),
                specificity_after_calibration=("specificity_after_calibration", "mean"),
                error_rate=("error_rate", "mean"),
                training_time_sec=("training_time_sec", "mean"),
                n_segments=("model", "size"),
                n_repetitions=(
                    "run_seed",
                    lambda s: int(s.dropna().astype(int).nunique()) if s.notna().any() else 1,
                ),
            )
            .reset_index()
        )
        rows.append(a)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["strategy", "model", "balance_mode"]).reset_index(drop=True)


def format_appendix_tables(
    yearly_results: pd.DataFrame,
    adaptive_results: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Formats result outputs as paper Appendix A tables A.6-A.9."""
    calibration_cols = [
        "sensitivity_before_calibration",
        "specificity_before_calibration",
        "sensitivity_after_calibration",
        "specificity_after_calibration",
    ]
    tables: Dict[str, pd.DataFrame] = {
        "A.6": pd.DataFrame(),
        "A.7": pd.DataFrame(),
        "A.8": pd.DataFrame(),
        "A.9": pd.DataFrame(),
    }

    if yearly_results is not None and not yearly_results.empty:
        yearly_results = yearly_results.copy()
        for col in ["pr_auc", "f1", *calibration_cols]:
            if col not in yearly_results.columns:
                yearly_results[col] = np.nan
        common_cols = [
            "iteration",
            "training_year",
            "prediction_year",
            "model",
            "balance_mode",
            "auc",
            "pr_auc",
            "f1",
            "sensitivity",
            "specificity",
            "sensitivity_before_calibration",
            "specificity_before_calibration",
            "sensitivity_after_calibration",
            "specificity_after_calibration",
            "error_rate",
            "training_time_sec",
            "threshold",
            "n_train",
            "n_test",
            "run_seed",
            "run_order",
        ]
        static = yearly_results.loc[yearly_results["strategy"] == "static", common_cols]
        period = yearly_results.loc[yearly_results["strategy"] == "period_aligned", common_cols]
        cumulative = yearly_results.loc[yearly_results["strategy"] == "cumulative", common_cols]

        tables["A.6"] = static.reset_index(drop=True)
        tables["A.7"] = period.reset_index(drop=True)
        tables["A.8"] = cumulative.reset_index(drop=True)

    if adaptive_results is not None and not adaptive_results.empty:
        adaptive_results = adaptive_results.copy()
        for col in ["pr_auc", "f1", *calibration_cols]:
            if col not in adaptive_results.columns:
                adaptive_results[col] = np.nan
        cols = [
            "strategy",
            "drift",
            "drift_date",
            "prediction_year",
            "balance_mode",
            "segment_rows",
            "n_internal_drifts",
            "n_internal_warnings",
            "vote_count",
            "vote_threshold",
            "monitor_feature_count",
            "retrain_rows",
            "retrain_positive_rows",
            "base_model",
            "detector_variant",
            "detected_features",
            "monitored_features",
            "W",
            "W0",
            "W1",
            "remaining_periods",
            "model",
            "auc",
            "pr_auc",
            "f1",
            "sensitivity",
            "specificity",
            "sensitivity_before_calibration",
            "specificity_before_calibration",
            "sensitivity_after_calibration",
            "specificity_after_calibration",
            "error_rate",
            "training_time_sec",
            "threshold",
            "run_seed",
            "run_order",
        ]
        tables["A.9"] = adaptive_results[[c for c in cols if c in adaptive_results.columns]].reset_index(drop=True)

    return tables


def format_appendix_tables_mean(
    yearly_results: pd.DataFrame,
    adaptive_results: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Formats mean appendix tables across sequential seed repetitions."""
    calibration_cols = [
        "sensitivity_before_calibration",
        "specificity_before_calibration",
        "sensitivity_after_calibration",
        "specificity_after_calibration",
    ]
    tables: Dict[str, pd.DataFrame] = {
        "A.6": pd.DataFrame(),
        "A.7": pd.DataFrame(),
        "A.8": pd.DataFrame(),
        "A.9": pd.DataFrame(),
    }

    if yearly_results is not None and not yearly_results.empty:
        yearly_results = yearly_results.copy()
        for col in ["pr_auc", "f1", *calibration_cols]:
            if col not in yearly_results.columns:
                yearly_results[col] = np.nan
        group_cols = ["strategy", "iteration", "training_year", "prediction_year", "model", "balance_mode"]
        metric_cols = [
            "auc",
            "pr_auc",
            "f1",
            "sensitivity",
            "specificity",
            "sensitivity_before_calibration",
            "specificity_before_calibration",
            "sensitivity_after_calibration",
            "specificity_after_calibration",
            "error_rate",
            "training_time_sec",
            "threshold",
            "n_train",
            "n_test",
        ]
        agg_yearly = (
            yearly_results.groupby(group_cols, dropna=False)[metric_cols]
            .mean()
            .reset_index()
        )
        agg_yearly = agg_yearly.merge(_group_seed_metadata(yearly_results, group_cols), on=group_cols, how="left")

        common_cols = [
            "iteration",
            "training_year",
            "prediction_year",
            "model",
            "balance_mode",
            "auc",
            "pr_auc",
            "f1",
            "sensitivity",
            "specificity",
            "sensitivity_before_calibration",
            "specificity_before_calibration",
            "sensitivity_after_calibration",
            "specificity_after_calibration",
            "error_rate",
            "training_time_sec",
            "threshold",
            "n_train",
            "n_test",
            "n_repetitions",
            "seed_list",
            "run_orders",
        ]
        tables["A.6"] = agg_yearly.loc[agg_yearly["strategy"] == "static", common_cols].reset_index(drop=True)
        tables["A.7"] = agg_yearly.loc[agg_yearly["strategy"] == "period_aligned", common_cols].reset_index(drop=True)
        tables["A.8"] = agg_yearly.loc[agg_yearly["strategy"] == "cumulative", common_cols].reset_index(drop=True)

    if adaptive_results is not None and not adaptive_results.empty:
        adaptive_results = adaptive_results.copy()
        for col in ["pr_auc", "f1", *calibration_cols]:
            if col not in adaptive_results.columns:
                adaptive_results[col] = np.nan
        group_cols = ["strategy", "drift", "model", "balance_mode"]
        metric_cols = [
            "prediction_year",
            "segment_rows",
            "n_internal_drifts",
            "n_internal_warnings",
            "vote_count",
            "vote_threshold",
            "monitor_feature_count",
            "retrain_rows",
            "retrain_positive_rows",
            "W",
            "W0",
            "W1",
            "remaining_periods",
            "auc",
            "pr_auc",
            "f1",
            "sensitivity",
            "specificity",
            "sensitivity_before_calibration",
            "specificity_before_calibration",
            "sensitivity_after_calibration",
            "specificity_after_calibration",
            "error_rate",
            "training_time_sec",
            "threshold",
        ]
        metric_cols = [c for c in metric_cols if c in adaptive_results.columns]
        agg_adaptive = (
            adaptive_results.groupby(group_cols, dropna=False)[metric_cols]
            .mean()
            .reset_index()
        )
        drift_dates = (
            adaptive_results.groupby(group_cols, dropna=False)
            .agg(
                drift_date_min=("drift_date", "min"),
                drift_date_max=("drift_date", "max"),
                base_model=("base_model", "first") if "base_model" in adaptive_results.columns else ("model", "first"),
                detector_variant=("detector_variant", "first") if "detector_variant" in adaptive_results.columns else ("model", "first"),
                detected_features=("detected_features", "first") if "detected_features" in adaptive_results.columns else ("model", "first"),
                monitored_features=("monitored_features", "first") if "monitored_features" in adaptive_results.columns else ("model", "first"),
            )
            .reset_index()
        )
        agg_adaptive = agg_adaptive.merge(drift_dates, on=group_cols, how="left")
        agg_adaptive = agg_adaptive.merge(
            _group_seed_metadata(adaptive_results, group_cols),
            on=group_cols,
            how="left",
        )
        cols = [
            "strategy",
            "drift",
            "balance_mode",
            "drift_date_min",
            "drift_date_max",
            "prediction_year",
            "segment_rows",
            "n_internal_drifts",
            "n_internal_warnings",
            "vote_count",
            "vote_threshold",
            "monitor_feature_count",
            "retrain_rows",
            "retrain_positive_rows",
            "base_model",
            "detector_variant",
            "detected_features",
            "monitored_features",
            "W",
            "W0",
            "W1",
            "remaining_periods",
            "model",
            "auc",
            "pr_auc",
            "f1",
            "sensitivity",
            "specificity",
            "sensitivity_before_calibration",
            "specificity_before_calibration",
            "sensitivity_after_calibration",
            "specificity_after_calibration",
            "error_rate",
            "training_time_sec",
            "threshold",
            "n_repetitions",
            "seed_list",
            "run_orders",
        ]
        tables["A.9"] = agg_adaptive[[c for c in cols if c in agg_adaptive.columns]].reset_index(drop=True)

    return tables


def build_gate_layout_from_porticos(porticos_df: pd.DataFrame) -> pd.DataFrame:
    """
    Figure 1/2 support: extracts AC-07 to AC-10 gate layout from Porticos table.
    """
    if porticos_df is None or porticos_df.empty:
        return pd.DataFrame(columns=["portico", "km", "calzada", "eje"])

    work = porticos_df.copy()
    for col in ["portico", "km", "calzada", "eje"]:
        if col not in work.columns:
            return pd.DataFrame(columns=["portico", "km", "calzada", "eje"])

    keep_codes = {"AC-07", "AC-08", "AC-09", "AC-10"}
    work["portico"] = work["portico"].astype(str).str.strip().str.upper()
    work = work.loc[work["portico"].isin(keep_codes)].copy()
    work["km"] = pd.to_numeric(work["km"], errors="coerce")
    work = work.dropna(subset=["km"])\
        .sort_values("km")
    return work[["portico", "km", "calzada", "eje"]].reset_index(drop=True)


def load_csv_dataset(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Dataset not found: {p}")
    return pd.read_csv(p)


def load_duckdb_table(path: str, table_name: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
    p = Path(path)
    if duckdb is None:
        raise ImportError("duckdb is not installed.")
    if not p.exists():
        raise FileNotFoundError(f"DuckDB file not found: {p}")

    con = duckdb.connect(str(p), read_only=True)
    try:
        if table_name is None:
            tables = con.execute("SHOW TABLES").fetchall()
            if not tables:
                return pd.DataFrame()
            table_name = str(tables[0][0])
        query = f"SELECT * FROM {table_name}"
        if limit is not None and int(limit) > 0:
            query += f" LIMIT {int(limit)}"
        return con.execute(query).df()
    finally:
        con.close()


def _list_event_files() -> List[Path]:
    if not DATA_DIR.exists():
        return []
    out: List[Path] = []
    for path in DATA_DIR.glob("*.csv"):
        lower = path.name.lower()
        if lower.startswith("eventos") or lower.startswith("accidentes"):
            out.append(path)
    return sorted(out)


def _read_event_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=None, engine="python", encoding="latin-1")


def _normalize_portico_code(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().upper()
    if not text or text in {"NAN", "NONE", "NULL"}:
        return None
    try:
        num = float(text.replace(",", "."))
    except ValueError:
        return text
    if num.is_integer():
        return str(int(num))
    return str(num)


def _normalize_portico_series(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip().str.upper()
    invalid = (
        out.isna()
        | out.str.len().fillna(0).eq(0)
        | out.isin(["NAN", "NONE", "NULL"]).fillna(False)
    )
    numeric = pd.to_numeric(out.str.replace(",", ".", regex=False), errors="coerce")
    numeric_mask = numeric.notna()
    if numeric_mask.any():
        int_mask = numeric_mask & np.isclose(numeric, np.floor(numeric))
        float_mask = numeric_mask & ~int_mask
        if int_mask.any():
            out.loc[int_mask] = numeric.loc[int_mask].astype("Int64").astype("string")
        if float_mask.any():
            out.loc[float_mask] = numeric.loc[float_mask].astype("string")
    out.loc[invalid] = pd.NA
    return out


def _find_match_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    normalized: Dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "").replace(".", "")
        normalized[key] = str(col)
    for candidate in candidates:
        key = str(candidate).strip().lower().replace(" ", "").replace(".", "")
        if key in normalized:
            return normalized[key]
    return None


def _feature_columns_for_modeling(df: pd.DataFrame, target_col: str) -> List[str]:
    if df is None or df.empty:
        return []
    excluded = {
        target_col,
        "interval_start",
        "portico",
        "eje",
        "calzada",
        "portico_last",
        "portico_next",
        "portico_inicio",
        "portico_fin",
    }
    return [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]


def _date_defaults(summary: Dict[str, Any]) -> Tuple[date, date]:
    today = datetime.today().date()
    min_ts = summary.get("min_ts")
    max_ts = summary.get("max_ts")
    if isinstance(min_ts, pd.Timestamp) and isinstance(max_ts, pd.Timestamp):
        return min_ts.date(), max_ts.date()
    return today, today


def _build_flow_sample_mode_selector(*, key_prefix: str) -> str:
    return st.radio(
        "Muestreo",
        ["Todo", "Rango de fechas", "Porcentaje"],
        horizontal=True,
        key=f"{key_prefix}_sample_mode",
    )


def _build_flow_sample_inputs(
    summary: Dict[str, Any],
    mode: str,
    *,
    key_prefix: str,
) -> Tuple[FlowSampleSelection, bool, bool]:
    row_limit = None
    date_start = None
    date_end = None
    range_valid = True
    percent_mode = False

    if mode == "Rango de fechas":
        default_start, default_end = _date_defaults(summary)
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Fecha inicio",
                value=default_start,
                key=f"{key_prefix}_start_date",
            )
        with col2:
            end_date = st.date_input(
                "Fecha fin",
                value=default_end,
                key=f"{key_prefix}_end_date",
            )

        date_start = pd.Timestamp(datetime.combine(start_date, datetime.min.time()))
        date_end = pd.Timestamp(datetime.combine(end_date, datetime.max.time()))
        if date_end <= date_start:
            st.error("La fecha final debe ser posterior a la fecha de inicio.")
            range_valid = False

    elif mode == "Porcentaje":
        percent_mode = True
        row_count = int(summary.get("row_count", 0) or 0)
        if row_count <= 0:
            st.warning("No hay filas disponibles para muestrear.")
        else:
            percent = st.slider(
                "Porcentaje",
                min_value=1,
                max_value=100,
                value=10,
                key=f"{key_prefix}_percent",
            )
            row_limit = max(1, int(row_count * (percent / 100.0)))
            st.caption(f"Se consultaran aproximadamente {row_limit:,} filas.")

    sample = FlowSampleSelection(
        date_start=date_start,
        date_end=date_end,
        row_limit=row_limit,
    )
    return sample, percent_mode, range_valid


def _query_flujos_duckdb(
    db_path: str,
    *,
    table_name: str = "flujos_duckdb",
    row_limit: Optional[int] = None,
    date_start: Optional[pd.Timestamp] = None,
    date_end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    if duckdb is None:
        raise ImportError("duckdb is not installed.")

    con = duckdb.connect(db_path, read_only=True)
    try:
        query = f"""
            SELECT FECHA, VELOCIDAD, CATEGORIA, MATRICULA, PORTICO, CARRIL
            FROM {table_name}
        """
        params: List[Any] = []
        if date_start is not None and date_end is not None:
            query += " WHERE FECHA >= ? AND FECHA <= ?"
            params.extend([pd.Timestamp(date_start), pd.Timestamp(date_end)])
        query += " ORDER BY FECHA"
        if row_limit is not None and int(row_limit) > 0:
            query += " LIMIT ?"
            params.append(int(row_limit))
        return con.execute(query, params).df()
    finally:
        con.close()


def build_dataset_from_flujos_and_eventos(
    *,
    flow_db_path: str,
    flow_table_name: str = "flujos_duckdb",
    event_file_names: Sequence[str],
    interval_minutes: int = 5,
    flow_row_limit: Optional[int] = None,
    date_start: Optional[pd.Timestamp] = None,
    date_end: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Builds an interval-level dataset from raw flows + events following the same
    event processing logic used in Crash prediction (Eventos tab).
    """
    if not event_file_names:
        raise ValueError("No event files were selected.")

    porticos_df = load_porticos()

    event_frames: List[pd.DataFrame] = []
    resolved_event_paths: List[str] = []
    for name in event_file_names:
        path = DATA_DIR / str(name)
        if not path.exists():
            raise FileNotFoundError(f"Event file does not exist: {path}")
        event_frames.append(_read_event_csv(path))
        resolved_event_paths.append(str(path))

    raw_events = pd.concat(event_frames, ignore_index=True) if event_frames else pd.DataFrame()
    accidents_df, excluded_df = process_accidentes_df(
        raw_events, porticos_df, return_excluded=True
    )
    if accidents_df.empty:
        raise ValueError("No valid accidents remained after processing event files.")

    flows_df = _query_flujos_duckdb(
        flow_db_path,
        table_name=flow_table_name,
        row_limit=flow_row_limit,
        date_start=date_start,
        date_end=date_end,
    )
    if flows_df.empty:
        raise ValueError("Flow query returned zero rows.")

    flows_df["FECHA"] = pd.to_datetime(flows_df["FECHA"], errors="coerce")
    flows_df = flows_df.dropna(subset=["FECHA"])
    flows_df["VELOCIDAD"] = pd.to_numeric(flows_df["VELOCIDAD"], errors="coerce")
    flows_df["CATEGORIA"] = pd.to_numeric(flows_df["CATEGORIA"], errors="coerce")
    flows_df = flows_df.dropna(subset=["CATEGORIA"])
    flows_df["CATEGORIA"] = flows_df["CATEGORIA"].astype(int)

    features_df = _compute_drift_article_features(
        flows_df,
        interval_minutes=int(interval_minutes),
        allowed_porticos=DRIFT_FOCUS_PORTICOS,
    )
    if features_df.empty:
        raise ValueError("Feature engineering produced an empty dataset.")

    corridor_accidents = _filter_accidents_for_allowed_porticos(
        accidents_df,
        DRIFT_FOCUS_PORTICOS,
    )
    base_df = _add_interval_accident_target(
        features_df,
        corridor_accidents,
        interval_minutes=int(interval_minutes),
        interval_col="interval_start",
    )
    if base_df.empty:
        raise ValueError("Target merge produced an empty dataset.")

    meta = {
        "steps": [
            "Load and concatenate selected event files from Datos/.",
            "Process events with Porticos.csv using Crash prediction event logic (process_accidentes_df).",
            "Query raw flow records from flujos.duckdb (table flujos_duckdb).",
            "Compute article-aligned interval features for the selected corridor and consecutive segments.",
            "Filter accidents to the selected corridor and build interval-level target for t+1.",
        ],
        "flow_db_path": str(flow_db_path),
        "flow_table_name": str(flow_table_name),
        "event_files": resolved_event_paths,
        "interval_minutes": int(interval_minutes),
        "date_start": str(date_start) if date_start is not None else None,
        "date_end": str(date_end) if date_end is not None else None,
        "counts": {
            "raw_event_rows": int(len(raw_events)),
            "processed_accidents": int(len(accidents_df)),
            "total_accidents": int(len(corridor_accidents)),
            "excluded_events_without_portico": int(len(excluded_df)),
            "flow_rows_queried": int(len(flows_df)),
            "interval_rows_features": int(len(features_df)),
            "final_rows_with_target": int(len(base_df)),
            "positive_intervals_target": int(
                pd.to_numeric(base_df.get("target"), errors="coerce").fillna(0).astype(int).sum()
            ),
        },
    }
    return base_df, accidents_df, excluded_df, meta


def _list_feature_duckdb_files() -> List[Path]:
    if not RESULTS_DIR.exists():
        return []
    patterns = [
        "drift_flow_features_*.duckdb",
        "accident_flow_features_*.duckdb",
        "flow_features_*.duckdb",
    ]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(list(RESULTS_DIR.glob(pattern)))
    uniq = sorted(set(files), key=lambda p: p.stat().st_mtime, reverse=True)
    return uniq


def _load_feature_df_from_duckdb(path: Path) -> pd.DataFrame:
    if duckdb is None:
        raise ImportError("duckdb is not installed.")
    con = duckdb.connect(str(path), read_only=True)
    try:
        tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
        if not tables:
            return pd.DataFrame()
        priority = ["clean_features", "flow_features", "features", tables[0]]
        table_name = next((t for t in priority if t in tables), tables[0])
        return con.execute(f"SELECT * FROM {table_name}").df()
    finally:
        con.close()


def _load_feature_payload_from_duckdb(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if duckdb is None:
        raise ImportError("duckdb is not installed.")
    con = duckdb.connect(str(path), read_only=True)
    try:
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        raw_df = pd.DataFrame()
        clean_df = pd.DataFrame()
        if "raw_features" in tables:
            raw_df = con.execute("SELECT * FROM raw_features").df()
        elif "flow_features" in tables:
            raw_df = con.execute("SELECT * FROM flow_features").df()
        elif "features" in tables:
            raw_df = con.execute("SELECT * FROM features").df()

        if "clean_features" in tables:
            clean_df = con.execute("SELECT * FROM clean_features").df()
        elif not raw_df.empty:
            clean_df = raw_df.copy()

        return raw_df, clean_df
    finally:
        con.close()


def _normalize_feature_payload_frames(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    normalized_raw_df = raw_df.copy()
    normalized_clean_df = clean_df.copy()
    if normalized_clean_df.empty and not normalized_raw_df.empty:
        normalized_clean_df = normalized_raw_df.copy()

    for frame in (normalized_raw_df, normalized_clean_df):
        if "interval_start" in frame.columns:
            frame["interval_start"] = pd.to_datetime(frame["interval_start"], errors="coerce")
        if "portico" in frame.columns:
            frame["portico"] = _normalize_portico_series(frame["portico"])
        if "target" in frame.columns:
            frame["target"] = pd.to_numeric(frame["target"], errors="coerce").fillna(0).astype(int)
    return normalized_raw_df, normalized_clean_df


def _load_checkpoint_feature_bundle(
    *,
    run_manifest: Dict[str, Any],
    feature_selection_context: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    checkpoint_feature_selection_context = dict(
        feature_selection_context
        or run_manifest.get("feature_selection_context")
        or {}
    )
    resolved_export_path = _resolve_existing_feature_export_path(
        checkpoint_feature_selection_context.get("feature_export_path")
        or checkpoint_feature_selection_context.get("feature_export_name")
    )
    if resolved_export_path is None:
        return None

    raw_df, clean_df = _load_feature_payload_from_duckdb(resolved_export_path)
    raw_df, clean_df = _normalize_feature_payload_frames(raw_df, clean_df)
    selection_payload = _load_feature_selection_payload_from_duckdb(resolved_export_path)
    resolved_context = dict(checkpoint_feature_selection_context)
    resolved_context["feature_export_path"] = str(resolved_export_path)
    resolved_context.setdefault("feature_export_name", resolved_export_path.name)
    return {
        "raw_df": raw_df,
        "clean_df": clean_df,
        "selection_payload": selection_payload,
        "resolved_path": str(resolved_export_path),
        "feature_selection_context": resolved_context,
    }


def _corr_pairs_to_storage_df(corr_pairs: Optional[pd.Series]) -> pd.DataFrame:
    if not isinstance(corr_pairs, pd.Series) or corr_pairs.empty:
        return pd.DataFrame(columns=["feature_a", "feature_b", "abs_corr"])

    pairs = corr_pairs.reset_index()
    if len(pairs.columns) >= 3:
        pairs = pairs.iloc[:, :3].copy()
        pairs.columns = ["feature_a", "feature_b", "abs_corr"]
    else:
        pairs = pd.DataFrame(columns=["feature_a", "feature_b", "abs_corr"])
    pairs["feature_a"] = pairs["feature_a"].astype(str)
    pairs["feature_b"] = pairs["feature_b"].astype(str)
    pairs["abs_corr"] = pd.to_numeric(pairs["abs_corr"], errors="coerce").astype(float)
    return pairs


def _corr_pairs_from_storage_df(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    work = df.copy()
    required = {"feature_a", "feature_b", "abs_corr"}
    if not required.issubset(set(work.columns)):
        return pd.Series(dtype=float)
    work["feature_a"] = work["feature_a"].astype(str)
    work["feature_b"] = work["feature_b"].astype(str)
    work["abs_corr"] = pd.to_numeric(work["abs_corr"], errors="coerce").astype(float)
    work = work.dropna(subset=["abs_corr"])
    if work.empty:
        return pd.Series(dtype=float)
    out = pd.Series(
        work["abs_corr"].to_numpy(),
        index=pd.MultiIndex.from_frame(work[["feature_a", "feature_b"]]),
        dtype=float,
    )
    return out.sort_values(ascending=False)


def _build_feature_selection_payload(
    *,
    candidate_features: Optional[Sequence[str]],
    selected_features: Optional[Sequence[str]],
    corr_pairs: Optional[pd.Series],
    importance_df: Optional[pd.DataFrame],
    config: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    has_payload = any(
        [
            candidate_features,
            selected_features,
            isinstance(corr_pairs, pd.Series) and not corr_pairs.empty,
            isinstance(importance_df, pd.DataFrame) and not importance_df.empty,
            bool(config),
        ]
    )
    if not has_payload:
        return None

    return {
        "candidate_features": list(candidate_features or []),
        "selected_features": list(selected_features or []),
        "corr_pairs": corr_pairs.copy() if isinstance(corr_pairs, pd.Series) else pd.Series(dtype=float),
        "importance_df": importance_df.copy()
        if isinstance(importance_df, pd.DataFrame)
        else pd.DataFrame(columns=["feature", "importance", "rank"]),
        "config": dict(config or {}),
    }


def _write_feature_selection_payload_to_duckdb(
    path: Path,
    *,
    selection_payload: Optional[Dict[str, Any]],
) -> None:
    if duckdb is None:
        raise ImportError("duckdb is not installed.")

    path = Path(path)
    con = duckdb.connect(str(path))
    try:
        for table_name in [
            "feature_selection_config",
            "feature_selection_candidates",
            "feature_selection_selected",
            "feature_selection_corr_pairs",
            "feature_selection_importance",
        ]:
            con.execute(f"DROP TABLE IF EXISTS {table_name}")

        if not selection_payload:
            return

        config = dict(selection_payload.get("config") or {})
        config_rows = [
            {"key": str(key), "value": json.dumps(value, default=str, ensure_ascii=True)}
            for key, value in config.items()
        ]
        config_df = pd.DataFrame(config_rows, columns=["key", "value"])
        if not config_df.empty:
            con.register("feature_selection_config_view", config_df)
            con.execute("CREATE TABLE feature_selection_config AS SELECT * FROM feature_selection_config_view")

        candidate_features = list(selection_payload.get("candidate_features") or [])
        candidate_df = pd.DataFrame(
            {
                "feature": candidate_features,
                "candidate_rank": np.arange(1, len(candidate_features) + 1, dtype=int),
            }
        )
        if not candidate_df.empty:
            con.register("feature_selection_candidates_view", candidate_df)
            con.execute("CREATE TABLE feature_selection_candidates AS SELECT * FROM feature_selection_candidates_view")

        selected_features = list(selection_payload.get("selected_features") or [])
        selected_df = pd.DataFrame(
            {
                "feature": selected_features,
                "selected_rank": np.arange(1, len(selected_features) + 1, dtype=int),
            }
        )
        if not selected_df.empty:
            con.register("feature_selection_selected_view", selected_df)
            con.execute("CREATE TABLE feature_selection_selected AS SELECT * FROM feature_selection_selected_view")

        corr_pairs_df = _corr_pairs_to_storage_df(selection_payload.get("corr_pairs"))
        if not corr_pairs_df.empty:
            con.register("feature_selection_corr_pairs_view", corr_pairs_df)
            con.execute("CREATE TABLE feature_selection_corr_pairs AS SELECT * FROM feature_selection_corr_pairs_view")

        importance_df = selection_payload.get("importance_df")
        if isinstance(importance_df, pd.DataFrame) and not importance_df.empty:
            con.register("feature_selection_importance_view", importance_df)
            con.execute("CREATE TABLE feature_selection_importance AS SELECT * FROM feature_selection_importance_view")
    finally:
        con.close()


def _load_feature_selection_payload_from_duckdb(path: Path) -> Optional[Dict[str, Any]]:
    if duckdb is None:
        raise ImportError("duckdb is not installed.")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DuckDB file not found: {path}")

    con = duckdb.connect(str(path), read_only=True)
    try:
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        has_any = bool(
            {
                "feature_selection_config",
                "feature_selection_candidates",
                "feature_selection_selected",
                "feature_selection_corr_pairs",
                "feature_selection_importance",
            }
            & tables
        )
        if not has_any:
            return None

        config: Dict[str, Any] = {}
        if "feature_selection_config" in tables:
            cfg_df = con.execute("SELECT * FROM feature_selection_config").df()
            for row in cfg_df.itertuples(index=False):
                try:
                    config[str(row.key)] = json.loads(str(row.value))
                except Exception:
                    config[str(row.key)] = row.value

        candidate_features: List[str] = []
        if "feature_selection_candidates" in tables:
            candidate_order_col = "candidate_rank" if "candidate_rank" in con.execute(
                "DESCRIBE feature_selection_candidates"
            ).df()["column_name"].astype(str).tolist() else None
            order_clause = "candidate_rank" if candidate_order_col else "feature"
            candidate_df = con.execute(
                f"SELECT feature FROM feature_selection_candidates ORDER BY {order_clause}"
            ).df()
            candidate_features = candidate_df["feature"].astype(str).tolist()

        selected_features: List[str] = []
        if "feature_selection_selected" in tables:
            selected_df = con.execute(
                "SELECT feature FROM feature_selection_selected ORDER BY selected_rank"
            ).df()
            selected_features = selected_df["feature"].astype(str).tolist()

        corr_pairs = pd.Series(dtype=float)
        if "feature_selection_corr_pairs" in tables:
            corr_pairs = _corr_pairs_from_storage_df(
                con.execute("SELECT * FROM feature_selection_corr_pairs").df()
            )

        importance_df = pd.DataFrame(columns=["feature", "importance", "rank"])
        if "feature_selection_importance" in tables:
            importance_df = con.execute("SELECT * FROM feature_selection_importance").df()

        return {
            "candidate_features": candidate_features,
            "selected_features": selected_features,
            "corr_pairs": corr_pairs,
            "importance_df": importance_df,
            "config": config,
        }
    finally:
        con.close()


def _flow_db_summary(db_path: str, table_name: str) -> Dict[str, Any]:
    if duckdb is None:
        raise ImportError("duckdb is not installed.")
    con = duckdb.connect(db_path, read_only=True)
    try:
        row_count = int(con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])
        min_ts, max_ts = con.execute(
            f"SELECT MIN(FECHA), MAX(FECHA) FROM {table_name}"
        ).fetchone()
        return {
            "row_count": row_count,
            "min_ts": pd.Timestamp(min_ts) if min_ts is not None else None,
            "max_ts": pd.Timestamp(max_ts) if max_ts is not None else None,
        }
    finally:
        con.close()


def _count_flujos_rows(
    db_path: str,
    *,
    table_name: str,
    date_start: Optional[pd.Timestamp] = None,
    date_end: Optional[pd.Timestamp] = None,
    allowed_porticos: Optional[Sequence[str]] = None,
) -> int:
    if duckdb is None:
        raise ImportError("duckdb is not installed.")
    con = duckdb.connect(db_path, read_only=True)
    try:
        where_parts: List[str] = []
        params: List[Any] = []
        if date_start is not None and date_end is not None:
            where_parts.append("FECHA >= ? AND FECHA <= ?")
            params.extend([pd.Timestamp(date_start), pd.Timestamp(date_end)])
        if allowed_porticos:
            placeholders = ",".join(["?"] * len(allowed_porticos))
            where_parts.append(f"CAST(PORTICO AS VARCHAR) IN ({placeholders})")
            params.extend([str(p) for p in allowed_porticos])
        query = f"SELECT COUNT(*) FROM {table_name}"
        if where_parts:
            query += " WHERE " + " AND ".join(where_parts)
        return int(con.execute(query, params).fetchone()[0])
    finally:
        con.close()


def _query_flujos_duckdb_filtered(
    db_path: str,
    *,
    table_name: str,
    row_limit: Optional[int] = None,
    date_start: Optional[pd.Timestamp] = None,
    date_end: Optional[pd.Timestamp] = None,
    allowed_porticos: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if duckdb is None:
        raise ImportError("duckdb is not installed.")
    con = duckdb.connect(db_path, read_only=True)
    try:
        query = f"""
            SELECT FECHA, VELOCIDAD, CATEGORIA, MATRICULA, PORTICO, CARRIL
            FROM {table_name}
        """
        where_parts: List[str] = []
        params: List[Any] = []
        if date_start is not None and date_end is not None:
            where_parts.append("FECHA >= ? AND FECHA <= ?")
            params.extend([pd.Timestamp(date_start), pd.Timestamp(date_end)])
        if allowed_porticos:
            placeholders = ",".join(["?"] * len(allowed_porticos))
            where_parts.append(f"CAST(PORTICO AS VARCHAR) IN ({placeholders})")
            params.extend([str(p) for p in allowed_porticos])
        if where_parts:
            query += " WHERE " + " AND ".join(where_parts)
        query += " ORDER BY FECHA"
        if row_limit is not None and int(row_limit) > 0:
            query += " LIMIT ?"
            params.append(int(row_limit))
        return con.execute(query, params).df()
    finally:
        con.close()


def _build_crash_prediction_batch_ranges(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    mode: str,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, str]]:
    mode_normalized = str(mode).strip().lower()
    if mode_normalized not in {"month", "week"}:
        raise ValueError("mode must be 'month' or 'week'")
    if end_ts < start_ts:
        return []

    if mode_normalized == "month":
        range_start = pd.Timestamp(start_ts).to_period("M").start_time.normalize()
        range_end = (pd.Timestamp(end_ts) + pd.offsets.MonthBegin(1)).normalize()
        boundaries = pd.date_range(start=range_start, end=range_end, freq="MS")
        return [
            (boundaries[i], boundaries[i + 1], boundaries[i].strftime("%Y-%m"))
            for i in range(len(boundaries) - 1)
        ]

    range_start = pd.Timestamp(start_ts).normalize() - pd.Timedelta(days=pd.Timestamp(start_ts).weekday())
    range_end = pd.Timestamp(end_ts).normalize() + pd.Timedelta(days=7)
    boundaries = pd.date_range(start=range_start, end=range_end, freq="7D")
    ranges: List[Tuple[pd.Timestamp, pd.Timestamp, str]] = []
    for i in range(len(boundaries) - 1):
        current = boundaries[i]
        nxt = boundaries[i + 1]
        label = f"{current:%Y-%m-%d}_to_{(nxt - pd.Timedelta(days=1)):%Y-%m-%d}"
        ranges.append((current, nxt, label))
    return ranges


def _compute_batched_flow_features_to_duckdb(
    *,
    flow_db_path: str,
    flow_table_name: str,
    out_path: Path,
    batch_mode: str,
    interval_minutes: int,
    lanes: int,
    metrics: Sequence[str],
    categories: Sequence[str],
    date_start: Optional[pd.Timestamp] = None,
    date_end: Optional[pd.Timestamp] = None,
    allowed_porticos: Optional[Sequence[str]] = None,
    row_limit: Optional[int] = None,
    progress_bar: Optional[Any] = None,
) -> Dict[str, Any]:
    if duckdb is None:
        raise ImportError("duckdb is not installed.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _update_feature_progress(progress_bar, 0.02, "Creando base temporal DuckDB...")

    temp_db_path = out_path.with_name(f"{out_path.stem}__work_{time.time_ns()}.duckdb")
    if out_path.exists():
        out_path.unlink()
    if temp_db_path.exists():
        temp_db_path.unlink()

    con_temp = None
    con_feat = None
    ranges: List[Tuple[pd.Timestamp, pd.Timestamp, str]] = []
    input_rows = 0
    feature_rows = 0
    table_created = False

    try:
        con_temp = duckdb.connect(str(temp_db_path))
        flow_db_ref = str(flow_db_path).replace("'", "''")
        con_temp.execute(f"ATTACH '{flow_db_ref}' AS flow_db (READ_ONLY)")

        query = f"""
            CREATE TABLE work_flujos AS
            SELECT FECHA, VELOCIDAD, CATEGORIA, MATRICULA, PORTICO, CARRIL
            FROM flow_db.{flow_table_name}
            WHERE 1=1
        """
        params: List[Any] = []
        if date_start is not None:
            query += " AND FECHA >= ?"
            params.append(pd.Timestamp(date_start))
        if date_end is not None:
            query += " AND FECHA <= ?"
            params.append(pd.Timestamp(date_end))
        porticos_filter = [str(p) for p in allowed_porticos or [] if p is not None]
        if porticos_filter:
            placeholders = ",".join(["?"] * len(porticos_filter))
            query += f" AND CAST(PORTICO AS VARCHAR) IN ({placeholders})"
            params.extend(porticos_filter)
        query += " ORDER BY FECHA"
        if row_limit is not None and int(row_limit) > 0:
            query += " LIMIT ?"
            params.append(int(row_limit))
        con_temp.execute(query, params)
        _update_feature_progress(
            progress_bar,
            0.12,
            "Base temporal lista.",
            detail="Se materializo work_flujos para procesar por lotes.",
        )

        row_count = int(con_temp.execute("SELECT COUNT(*) FROM work_flujos").fetchone()[0])
        if row_count <= 0:
            return {
                "batch_ranges": [],
                "input_rows": 0,
                "feature_rows": 0,
                "output_path": str(out_path),
            }

        min_ts, max_ts = con_temp.execute("SELECT MIN(FECHA), MAX(FECHA) FROM work_flujos").fetchone()
        if min_ts is None or max_ts is None:
            return {
                "batch_ranges": [],
                "input_rows": 0,
                "feature_rows": 0,
                "output_path": str(out_path),
            }

        ranges = _build_crash_prediction_batch_ranges(
            pd.Timestamp(min_ts),
            pd.Timestamp(max_ts),
            batch_mode,
        )
        if not ranges:
            return {
                "batch_ranges": [],
                "input_rows": 0,
                "feature_rows": 0,
                "output_path": str(out_path),
            }
        _update_feature_progress(
            progress_bar,
            0.18,
            "Lotes construidos.",
            detail=f"Modo {batch_mode} | {len(ranges)} lotes.",
        )

        con_feat = duckdb.connect(str(out_path))
        for idx, (start_ts, end_exclusive, batch_label) in enumerate(ranges, start=1):
            batch_start_ratio = 0.18 + ((idx - 1) / max(len(ranges), 1)) * 0.62
            batch_end_ratio = 0.18 + (idx / max(len(ranges), 1)) * 0.62
            _update_feature_progress(
                progress_bar,
                batch_start_ratio,
                f"Lote {idx}/{len(ranges)}: cargando flujos...",
                detail=batch_label,
            )
            df_batch = con_temp.execute(
                """
                SELECT FECHA, VELOCIDAD, CATEGORIA, MATRICULA, PORTICO, CARRIL
                FROM work_flujos
                WHERE FECHA >= ? AND FECHA < ?
                ORDER BY FECHA
                """,
                [start_ts, end_exclusive],
            ).df()
            input_rows += int(len(df_batch))
            if df_batch.empty:
                _update_feature_progress(
                    progress_bar,
                    batch_end_ratio,
                    f"Lote {idx}/{len(ranges)} vacio.",
                    detail=batch_label,
                )
                continue

            _update_feature_progress(
                progress_bar,
                batch_start_ratio + (batch_end_ratio - batch_start_ratio) * 0.35,
                f"Lote {idx}/{len(ranges)}: limpiando y agregando features...",
                detail=f"{batch_label} | filas crudas: {len(df_batch):,}",
            )
            df_batch["FECHA"] = pd.to_datetime(df_batch["FECHA"], errors="coerce")
            df_batch["VELOCIDAD"] = pd.to_numeric(df_batch["VELOCIDAD"], errors="coerce")
            df_batch["CATEGORIA"] = pd.to_numeric(df_batch["CATEGORIA"], errors="coerce")
            df_batch = df_batch.dropna(subset=["FECHA", "CATEGORIA"])
            df_batch["CATEGORIA"] = df_batch["CATEGORIA"].astype(int)
            if df_batch.empty:
                _update_feature_progress(
                    progress_bar,
                    batch_end_ratio,
                    f"Lote {idx}/{len(ranges)} sin filas validas.",
                    detail=batch_label,
                )
                continue

            feat_batch = _compute_drift_article_features(
                df_batch,
                interval_minutes=int(interval_minutes),
                categories=list(categories),
                allowed_porticos=allowed_porticos,
            )
            if feat_batch.empty:
                _update_feature_progress(
                    progress_bar,
                    batch_end_ratio,
                    f"Lote {idx}/{len(ranges)} sin features.",
                    detail=batch_label,
                )
                continue

            _update_feature_progress(
                progress_bar,
                batch_start_ratio + (batch_end_ratio - batch_start_ratio) * 0.75,
                f"Lote {idx}/{len(ranges)}: escribiendo en DuckDB...",
                detail=f"{batch_label} | filas features: {len(feat_batch):,}",
            )
            feat_batch["interval_start"] = pd.to_datetime(feat_batch["interval_start"], errors="coerce")
            feat_batch = feat_batch.dropna(subset=["interval_start"])
            feat_batch = feat_batch.drop_duplicates(subset=["interval_start"]).sort_values("interval_start")
            if feat_batch.empty:
                _update_feature_progress(
                    progress_bar,
                    batch_end_ratio,
                    f"Lote {idx}/{len(ranges)} sin features validas.",
                    detail=batch_label,
                )
                continue

            con_feat.register("batch_view", feat_batch)
            if not table_created:
                con_feat.execute("CREATE TABLE flow_features AS SELECT * FROM batch_view")
                table_created = True
            else:
                con_feat.execute("INSERT INTO flow_features SELECT * FROM batch_view")
            feature_rows += int(len(feat_batch))
            _update_feature_progress(
                progress_bar,
                batch_end_ratio,
                f"Lote {idx}/{len(ranges)} completado.",
                detail=f"{batch_label} | acumulado features: {feature_rows:,}",
            )

        if not table_created and out_path.exists():
            out_path.unlink()

        _update_feature_progress(
            progress_bar,
            0.84,
            "Lectura por lotes completada.",
            detail=f"Flujos procesados: {input_rows:,} | features: {feature_rows:,}",
        )
        return {
            "batch_ranges": ranges,
            "input_rows": int(input_rows),
            "feature_rows": int(feature_rows),
            "output_path": str(out_path),
        }
    except Exception:
        if con_feat is not None:
            try:
                con_feat.close()
            except Exception:
                pass
            con_feat = None
        if con_temp is not None:
            try:
                con_temp.close()
            except Exception:
                pass
            con_temp = None
        if out_path.exists():
            out_path.unlink()
        raise
    finally:
        if con_feat is not None:
            con_feat.close()
        if con_temp is not None:
            con_temp.close()
        if temp_db_path.exists():
            temp_db_path.unlink()


def _build_batch_ranges(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    mode: str,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if end_ts <= start_ts:
        return []
    ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    current = pd.Timestamp(start_ts)
    mode = str(mode)
    while current < end_ts:
        if mode == "Diario":
            nxt = current + pd.Timedelta(days=1)
        elif mode == "Semanal":
            nxt = current + pd.Timedelta(days=7)
        else:
            nxt = current + pd.DateOffset(months=1)
        nxt = min(pd.Timestamp(nxt), pd.Timestamp(end_ts))
        ranges.append((current, nxt))
        current = nxt
    return ranges


def _write_feature_payload_to_duckdb(
    path: Path,
    *,
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    selection_payload: Optional[Dict[str, Any]] = None,
) -> None:
    if duckdb is None:
        raise ImportError("duckdb is not installed.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    try:
        con.register("raw_view", raw_df)
        con.execute("CREATE OR REPLACE TABLE raw_features AS SELECT * FROM raw_view")
        con.register("clean_view", clean_df)
        con.execute("CREATE OR REPLACE TABLE clean_features AS SELECT * FROM clean_view")
    finally:
        con.close()
    _write_feature_selection_payload_to_duckdb(path, selection_payload=selection_payload)


def _save_feature_duckdb(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    *,
    prefix: str = "drift_flow_features",
    path: Optional[Path] = None,
    selection_payload: Optional[Dict[str, Any]] = None,
) -> Path:
    if duckdb is None:
        raise ImportError("duckdb is not installed.")
    if path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"{prefix}_{stamp}.duckdb"
    else:
        out_path = Path(path)
    payload_to_store = selection_payload
    if payload_to_store:
        payload_to_store = dict(payload_to_store)
        payload_to_store["config"] = dict(payload_to_store.get("config") or {})
        payload_to_store["config"]["linked_duckdb_path"] = str(out_path.resolve())
        payload_to_store["config"]["linked_duckdb_name"] = out_path.name
    _write_feature_payload_to_duckdb(
        out_path,
        raw_df=raw_df,
        clean_df=clean_df,
        selection_payload=payload_to_store,
    )
    return out_path


def _clear_feature_selection_state(*, clear_selected_features: bool = False) -> None:
    st.session_state["drift_corr_pairs"] = None
    st.session_state["drift_importance_df"] = None
    st.session_state["drift_feature_selection_meta"] = None
    if clear_selected_features:
        st.session_state["drift_candidate_feature_cols"] = None
        st.session_state["drift_feature_cols"] = None


def _current_feature_selection_payload(*, linked_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    meta = dict(st.session_state.get("drift_feature_selection_meta") or {})
    corr_pairs = st.session_state.get("drift_corr_pairs")
    importance_df = st.session_state.get("drift_importance_df")

    has_selection_artifact = bool(meta) or (
        isinstance(corr_pairs, pd.Series)
        and not corr_pairs.empty
    ) or (
        isinstance(importance_df, pd.DataFrame)
        and not importance_df.empty
    )
    if not has_selection_artifact:
        return None

    raw_df = st.session_state.get("drift_raw_df")
    clean_df = st.session_state.get("drift_clean_df")
    if linked_path is not None:
        linked_path = Path(linked_path)
        meta["linked_duckdb_path"] = str(linked_path.resolve())
        meta["linked_duckdb_name"] = linked_path.name
    if isinstance(raw_df, pd.DataFrame):
        meta["raw_rows"] = int(len(raw_df))
    if isinstance(clean_df, pd.DataFrame):
        meta["clean_rows"] = int(len(clean_df))

    return _build_feature_selection_payload(
        candidate_features=st.session_state.get("drift_candidate_feature_cols"),
        selected_features=st.session_state.get("drift_feature_cols"),
        corr_pairs=corr_pairs,
        importance_df=importance_df,
        config=meta,
    )


def _current_feature_selection_context() -> Dict[str, Any]:
    meta = dict(st.session_state.get("drift_feature_selection_meta") or {})
    meta["selected_features"] = list(st.session_state.get("drift_feature_cols") or [])
    meta["candidate_features"] = list(st.session_state.get("drift_candidate_feature_cols") or [])
    meta["feature_count"] = int(len(meta["selected_features"]))
    export_path = st.session_state.get("drift_feature_export_path")
    if export_path:
        export = Path(str(export_path))
        meta["feature_export_path"] = str(export)
        meta["feature_export_name"] = export.name
    return meta


def _list_recalibration_checkpoints(
    *,
    checkpoint_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    root = _recalibration_checkpoint_root(checkpoint_root=checkpoint_root)
    if not root.exists():
        return []

    def _timestamp_sort_key(value: Any) -> Tuple[int, str]:
        raw = str(value or "").strip()
        if not raw:
            return (0, "")
        try:
            return (1, datetime.fromisoformat(raw).isoformat())
        except Exception:
            return (1, raw)

    entries: List[Dict[str, Any]] = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        manifest = _load_manifest(manifest_path)
        if manifest is None:
            continue
        run_dir = manifest_path.parent
        try:
            manifest = _reconcile_recalibration_manifest(
                manifest,
                paths=_recalibration_run_paths(run_dir),
            )
        except Exception:
            pass

        run_manifest = dict(manifest.get("run_manifest") or {})
        feature_selection_context = dict(
            manifest.get("feature_selection_context")
            or run_manifest.get("feature_selection_context")
            or {}
        )
        run_id = str(manifest.get("run_id") or run_manifest.get("run_id") or run_dir.name)
        status = str(manifest.get("status", "missing"))
        updated_at = manifest.get("updated_at")
        models = [str(item) for item in (run_manifest.get("models") or [])]
        strategies = [str(item) for item in (run_manifest.get("strategies") or [])]
        feature_count = pd.to_numeric(run_manifest.get("feature_count"), errors="coerce")
        entries.append(
            {
                "run_id": run_id,
                "status": status,
                "updated_at": updated_at,
                "run_dir": str(run_dir),
                "manifest_path": str(manifest_path),
                "run_manifest": run_manifest,
                "manifest": manifest,
                "feature_selection_context": feature_selection_context,
                "feature_count": None if pd.isna(feature_count) else int(feature_count),
                "label": (
                    f"{run_id} | {status} | {updated_at or 'desconocido'} | "
                    f"{len(models)} modelos | {len(strategies)} estrategias"
                ),
            }
        )

    entries.sort(
        key=lambda item: (
            _timestamp_sort_key(item.get("updated_at")),
            str(item.get("run_id", "")),
        ),
        reverse=True,
    )
    return entries


def _checkpoint_form_state_from_run_manifest(
    run_manifest: Dict[str, Any],
    *,
    available_columns: Sequence[str],
    feature_count: int,
) -> Dict[str, Any]:
    available_column_set = {str(col) for col in available_columns}
    bounded_feature_count = max(1, int(feature_count))
    bounded_vote_max = max(1, min(10, bounded_feature_count))
    feature_selection_context = dict(run_manifest.get("feature_selection_context") or {})

    normalized_models = [
        str(item) for item in (run_manifest.get("models") or MODEL_NAMES)
        if str(item) in MODEL_NAMES
    ] or list(MODEL_NAMES)
    normalized_strategies = [
        str(item) for item in (run_manifest.get("strategies") or EXPERIMENT_STRATEGIES)
        if str(item) in EXPERIMENT_STRATEGIES
    ] or list(EXPERIMENT_STRATEGIES)
    normalized_arf_variants = [
        str(item) for item in (run_manifest.get("arf_variants") or ARF_DEFAULT_VARIANTS)
        if str(item) in ARF_VARIANT_NAMES
    ] or list(ARF_DEFAULT_VARIANTS)
    normalized_kswin_variants = [
        str(item) for item in (run_manifest.get("kswin_variants") or KSWIN_DEFAULT_VARIANTS)
        if str(item) in KSWIN_VARIANT_NAMES
    ] or list(KSWIN_DEFAULT_VARIANTS)
    normalized_balance_modes = [
        str(item) for item in (run_manifest.get("balance_modes") or DEFAULT_BATCH_BALANCE_MODES)
        if str(item) in DEFAULT_BATCH_BALANCE_MODES
    ] or list(DEFAULT_BATCH_BALANCE_MODES)

    def _coerced_int(raw_value: Any, default: int) -> int:
        value = pd.to_numeric(raw_value, errors="coerce")
        return int(value) if pd.notna(value) else int(default)

    def _coerced_float(raw_value: Any, default: float) -> float:
        value = pd.to_numeric(raw_value, errors="coerce")
        return float(value) if pd.notna(value) else float(default)

    min_window = max(100, _coerced_int(run_manifest.get("min_window"), 45_000))
    min_retrain_size_raw = pd.to_numeric(run_manifest.get("min_retrain_size"), errors="coerce")
    min_retrain_size = (
        max(10, int(min_retrain_size_raw))
        if pd.notna(min_retrain_size_raw)
        else max(20, int(min_window) // 10)
    )

    repetition_seeds: List[int] = []
    for seed in (run_manifest.get("repetition_seeds") or DEFAULT_REPETITION_SEEDS):
        try:
            repetition_seeds.append(int(seed))
        except Exception:
            continue
    if not repetition_seeds:
        repetition_seeds = list(DEFAULT_REPETITION_SEEDS)

    updates: Dict[str, Any] = {
        "drift_exp_resource_mode": _normalize_experiment_resource_mode(
            str(run_manifest.get("resource_mode", DEFAULT_EXPERIMENT_RESOURCE_MODE))
        ),
        "drift_exp_models": normalized_models,
        "drift_exp_strategies": normalized_strategies,
        "drift_exp_fast_mode": bool(run_manifest.get("fast_mode", True)),
        "drift_exp_optuna_trials": max(1, _coerced_int(run_manifest.get("optuna_trials"), 30)),
        "drift_exp_folds": max(2, _coerced_int(run_manifest.get("folds"), 3)),
        "drift_exp_validation_size": _coerced_float(run_manifest.get("validation_size"), 0.2),
        "drift_exp_adwin_delta": _coerced_float(run_manifest.get("adwin_delta"), 0.002),
        "drift_exp_min_window": min_window,
        "drift_exp_min_retrain_size": min_retrain_size,
        "drift_exp_arf_variants": normalized_arf_variants,
        "drift_exp_kswin_variants": normalized_kswin_variants,
        "drift_exp_kswin_top_k_features": min(
            bounded_feature_count,
            max(1, _coerced_int(run_manifest.get("kswin_top_k_features"), 10)),
        ),
        "drift_exp_kswin_vote_threshold": min(
            bounded_vote_max,
            max(1, _coerced_int(run_manifest.get("kswin_vote_threshold"), 2)),
        ),
        "drift_exp_kswin_retrain_days": max(1, _coerced_int(run_manifest.get("kswin_retrain_days"), 90)),
        "drift_exp_kswin_min_retrain_rows": max(10, _coerced_int(run_manifest.get("kswin_min_retrain_rows"), 100)),
        "drift_exp_balance_modes": normalized_balance_modes,
        "drift_exp_seed_text": ", ".join(str(seed) for seed in repetition_seeds),
        "drift_exp_continue_on_block_error": bool(run_manifest.get("continue_on_block_error", False)),
    }

    target_col = str(run_manifest.get("target_col", "")).strip()
    if target_col in available_column_set:
        updates["drift_exp_target_col"] = target_col
    time_col = str(run_manifest.get("time_col", "")).strip()
    if time_col in available_column_set:
        updates["drift_exp_time_col"] = time_col

    selected_features = [
        str(item)
        for item in (feature_selection_context.get("selected_features") or [])
        if str(item) in available_column_set
    ]
    candidate_features = [
        str(item)
        for item in (feature_selection_context.get("candidate_features") or [])
        if str(item) in available_column_set
    ]
    if candidate_features:
        updates["drift_candidate_feature_cols"] = candidate_features
    if selected_features:
        updates["drift_feature_cols"] = selected_features
    resolved_export_path = _resolve_existing_feature_export_path(
        feature_selection_context.get("feature_export_path")
        or feature_selection_context.get("feature_export_name")
    )
    export_path = str(resolved_export_path or feature_selection_context.get("feature_export_path") or "").strip()
    if export_path:
        updates["drift_feature_export_path"] = export_path
    if feature_selection_context:
        normalized_context = dict(feature_selection_context)
        if export_path:
            normalized_context["feature_export_path"] = export_path
            normalized_context.setdefault("feature_export_name", _portable_path_name(export_path))
        updates["drift_feature_selection_meta"] = normalized_context
    return updates


def _execution_config_from_checkpoint_run_manifest(
    run_manifest: Dict[str, Any],
    *,
    available_columns: Sequence[str],
    current_config: Dict[str, Any],
    feature_selection_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config = dict(current_config)
    available_column_set = {str(col) for col in available_columns}

    checkpoint_feature_selection_context = dict(
        feature_selection_context
        or run_manifest.get("feature_selection_context")
        or config.get("feature_selection_context")
        or {}
    )
    resolved_export_path = _resolve_existing_feature_export_path(
        checkpoint_feature_selection_context.get("feature_export_path")
        or checkpoint_feature_selection_context.get("feature_export_name")
    )
    if resolved_export_path is not None:
        checkpoint_feature_selection_context["feature_export_path"] = str(resolved_export_path)
        checkpoint_feature_selection_context.setdefault("feature_export_name", resolved_export_path.name)
    if checkpoint_feature_selection_context:
        config["feature_selection_context"] = dict(checkpoint_feature_selection_context)
        selected_features = [
            str(item)
            for item in (checkpoint_feature_selection_context.get("selected_features") or [])
            if str(item) in available_column_set
        ]
        if selected_features:
            config["feature_cols"] = selected_features

    target_col = str(run_manifest.get("target_col") or "").strip()
    if target_col in available_column_set:
        config["target_col"] = target_col
    time_col = str(run_manifest.get("time_col") or "").strip()
    if time_col in available_column_set:
        config["time_col"] = time_col

    def _normalized_list(values: Any, *, allowed: Sequence[str], fallback: Sequence[str]) -> List[str]:
        normalized = [str(item) for item in (values or []) if str(item) in allowed]
        if normalized:
            return normalized
        return [str(item) for item in fallback if str(item) in allowed]

    config["model_names"] = _normalized_list(
        run_manifest.get("models"),
        allowed=MODEL_NAMES,
        fallback=config.get("model_names") or MODEL_NAMES,
    )
    config["strategies"] = _normalized_list(
        run_manifest.get("strategies"),
        allowed=EXPERIMENT_STRATEGIES,
        fallback=config.get("strategies") or EXPERIMENT_STRATEGIES,
    )
    config["arf_variants"] = _normalized_list(
        run_manifest.get("arf_variants"),
        allowed=ARF_VARIANT_NAMES,
        fallback=config.get("arf_variants") or ARF_DEFAULT_VARIANTS,
    )
    config["kswin_variants"] = _normalized_list(
        run_manifest.get("kswin_variants"),
        allowed=KSWIN_VARIANT_NAMES,
        fallback=config.get("kswin_variants") or KSWIN_DEFAULT_VARIANTS,
    )
    config["balance_modes"] = _normalized_list(
        run_manifest.get("balance_modes"),
        allowed=DEFAULT_BATCH_BALANCE_MODES,
        fallback=config.get("balance_modes") or DEFAULT_BATCH_BALANCE_MODES,
    )

    repetition_seeds: List[int] = []
    for seed in (run_manifest.get("repetition_seeds") or config.get("repetition_seeds") or DEFAULT_REPETITION_SEEDS):
        try:
            repetition_seeds.append(int(seed))
        except Exception:
            continue
    if repetition_seeds:
        config["repetition_seeds"] = tuple(repetition_seeds)

    def _apply_numeric(key: str, raw_value: Any, *, min_value: Optional[int] = None) -> None:
        value = pd.to_numeric(raw_value, errors="coerce")
        if pd.notna(value):
            numeric = int(value)
            if min_value is not None:
                numeric = max(int(min_value), numeric)
            config[key] = numeric

    validation_size = pd.to_numeric(run_manifest.get("validation_size"), errors="coerce")
    if pd.notna(validation_size):
        config["validation_size"] = float(validation_size)
    _apply_numeric("folds", run_manifest.get("folds"), min_value=2)
    config["fast_mode"] = bool(run_manifest.get("fast_mode", config.get("fast_mode", True)))
    config["resource_mode"] = _normalize_experiment_resource_mode(
        str(run_manifest.get("resource_mode", config.get("resource_mode", DEFAULT_EXPERIMENT_RESOURCE_MODE)))
    )
    _apply_numeric("grid_limit", run_manifest.get("optuna_trials"), min_value=1)

    adwin_delta = pd.to_numeric(run_manifest.get("adwin_delta"), errors="coerce")
    if pd.notna(adwin_delta):
        config["adwin_delta"] = float(adwin_delta)
    _apply_numeric("min_window", run_manifest.get("min_window"), min_value=100)
    min_retrain_size = pd.to_numeric(run_manifest.get("min_retrain_size"), errors="coerce")
    if pd.notna(min_retrain_size):
        config["min_retrain_size"] = max(10, int(min_retrain_size))

    feature_count = max(1, len(config.get("feature_cols") or []))
    vote_threshold_max = max(1, min(10, feature_count))
    _apply_numeric("kswin_retrain_days", run_manifest.get("kswin_retrain_days"), min_value=1)
    _apply_numeric("kswin_min_retrain_rows", run_manifest.get("kswin_min_retrain_rows"), min_value=10)

    top_k = pd.to_numeric(run_manifest.get("kswin_top_k_features"), errors="coerce")
    if pd.notna(top_k):
        config["kswin_top_k_features"] = min(feature_count, max(1, int(top_k)))
    vote_threshold = pd.to_numeric(run_manifest.get("kswin_vote_threshold"), errors="coerce")
    if pd.notna(vote_threshold):
        config["kswin_vote_threshold"] = min(vote_threshold_max, max(1, int(vote_threshold)))

    config["continue_on_block_error"] = bool(
        run_manifest.get("continue_on_block_error", config.get("continue_on_block_error", False))
    )

    base_year = pd.to_numeric(run_manifest.get("base_year"), errors="coerce")
    config["base_year"] = None if pd.isna(base_year) else int(base_year)
    random_state = pd.to_numeric(run_manifest.get("random_state_fallback"), errors="coerce")
    if pd.notna(random_state):
        config["random_state"] = int(random_state)

    checkpoint_custom_grids = run_manifest.get("custom_grids")
    if isinstance(checkpoint_custom_grids, dict):
        config["custom_grids"] = _to_json_safe(checkpoint_custom_grids)
    return config


def _build_tramo_selector_for_feature_tab() -> Optional[Tuple[str, str, str, str]]:
    try:
        porticos_df = load_porticos()
    except Exception:
        return None
    if porticos_df is None or porticos_df.empty:
        return None

    required = {"portico", "orden", "eje", "calzada"}
    if not required.issubset(set(porticos_df.columns)):
        return None

    porticos = porticos_df.copy()
    porticos["orden_num"] = pd.to_numeric(porticos["orden"], errors="coerce")
    porticos = porticos.dropna(subset=["orden_num"])
    if porticos.empty:
        return None

    options: List[Tuple[str, Optional[Tuple[str, str, str, str]]]] = [
        ("Porticos fijos: 11, 12, 14 y 15", DRIFT_FOCUS_TRAMO),
        ("Toda la autopista", None)
    ]

    for (eje, calzada), grp in porticos.groupby(["eje", "calzada"]):
        grp = grp.sort_values("orden_num")
        p_codes = grp["portico"].astype(str).str.strip().tolist()
        for i in range(len(p_codes) - 1):
            p_start = p_codes[i]
            p_end = p_codes[i + 1]
            label = f"{eje} | {calzada}: {p_start} -> {p_end}"
            options.append((label, (str(eje), str(calzada), str(p_start), str(p_end))))

    labels = [x[0] for x in options]
    selected = st.selectbox("Tramo", labels, index=0, key="drift_feature_tramo_selector")
    tramo_tuple = next((tpl for lbl, tpl in options if lbl == selected), None)
    return tramo_tuple


def _porticos_in_tramo(tramo: Optional[Tuple[str, str, str, str]]) -> Optional[List[str]]:
    if tramo is None:
        return None
    if tramo == DRIFT_FOCUS_TRAMO:
        return list(DRIFT_FOCUS_PORTICOS)
    eje, calzada, p_start, p_end = tramo
    try:
        porticos_df = load_porticos()
    except Exception:
        return None
    if porticos_df is None or porticos_df.empty:
        return None
    required = {"portico", "orden", "eje", "calzada"}
    if not required.issubset(set(porticos_df.columns)):
        return None
    porticos = porticos_df.copy()
    porticos["orden_num"] = pd.to_numeric(porticos["orden"], errors="coerce")
    porticos = porticos.dropna(subset=["orden_num"])
    grp = porticos[
        (porticos["eje"].astype(str) == str(eje))
        & (porticos["calzada"].astype(str) == str(calzada))
    ].sort_values("orden_num")
    if grp.empty:
        return None
    codes = grp["portico"].astype(str).str.strip().tolist()
    if p_start not in codes or p_end not in codes:
        return [p_start, p_end]
    i0 = codes.index(p_start)
    i1 = codes.index(p_end)
    lo, hi = sorted([i0, i1])
    return codes[lo : hi + 1]


def _describe_tramo_selection(tramo: Optional[Tuple[str, str, str, str]]) -> str:
    if tramo == DRIFT_FOCUS_TRAMO:
        return f"Porticos fijos: {', '.join(DRIFT_FOCUS_PORTICOS)}"
    if tramo is None:
        return "Toda la autopista"
    eje, calzada, p_start, p_end = tramo
    return f"{eje} | {calzada} | {p_start} -> {p_end}"


def generate_synthetic_article_dataset(
    *,
    years: Sequence[int] = (2018, 2019, 2020, 2021, 2022, 2023, 2024),
    rows_per_year: int = 900,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Synthetic dataset for tests and UI demos when real data is unavailable.
    Includes mild concept drift by changing coefficients over years.
    """
    rng = np.random.default_rng(int(random_state))

    rows: List[Dict[str, Any]] = []
    porticos = ["AC-07", "AC-08", "AC-09", "AC-10"]

    for year in years:
        start_ts = pd.Timestamp(f"{int(year)}-01-01 00:00:00")
        timestamps = pd.date_range(start_ts, periods=int(rows_per_year), freq="5min")

        drift_factor = (int(year) - int(years[0])) / max(1, len(years) - 1)
        c1 = 1.8 - 1.1 * drift_factor
        c2 = 0.9 + 0.8 * drift_factor
        base_bias = -3.2 + 0.4 * drift_factor

        x1 = rng.normal(0.0, 1.0, size=len(timestamps))
        x2 = rng.normal(0.0, 1.0, size=len(timestamps))
        x3 = rng.normal(0.0, 1.0, size=len(timestamps))
        flow = rng.gamma(shape=3.0 + 0.3 * drift_factor, scale=1.2, size=len(timestamps))
        speed = 80 + 8 * x1 - 5 * x2 + rng.normal(0, 3, size=len(timestamps))
        density = np.clip(flow / np.maximum(speed, 30), 0, None)

        logits = base_bias + c1 * x1 + c2 * x2 + 0.7 * x3 + 0.3 * density
        prob = 1.0 / (1.0 + np.exp(-logits))
        y = rng.binomial(1, np.clip(prob, 0.001, 0.999)).astype(int)

        # induce long no-accident run in 2020 for filtering tests / realism
        if int(year) == 2020 and len(y) >= 2100:
            y[1200:2100] = 0

        for i, ts in enumerate(timestamps):
            rows.append(
                {
                    "interval_start": ts,
                    "portico": porticos[i % len(porticos)],
                    "flow_light": float(flow[i] * 10 + 40),
                    "flow_heavy": float(flow[i] * 2 + 6),
                    "speed_light": float(speed[i]),
                    "speed_heavy": float(speed[i] - rng.normal(4, 1.5)),
                    "density_light": float(density[i] * 3.5),
                    "density_heavy": float(density[i] * 4.2),
                    "delta_speed_light": float(rng.normal(0, 1)),
                    "delta_density_light": float(rng.normal(0, 0.25)),
                    "x1": float(x1[i]),
                    "x2": float(x2[i]),
                    "x3": float(x3[i]),
                    "target": int(y[i]),
                }
            )

    return pd.DataFrame(rows)


def _init_state() -> None:
    st.session_state.setdefault("drift_raw_df", None)
    st.session_state.setdefault("drift_clean_df", None)
    st.session_state.setdefault("drift_stage1_info", None)
    st.session_state.setdefault("drift_zero_runs", None)
    st.session_state.setdefault("drift_prep_summary", None)
    st.session_state.setdefault("drift_feature_cols", None)
    st.session_state.setdefault("drift_candidate_feature_cols", None)
    st.session_state.setdefault("drift_importance_df", None)
    st.session_state.setdefault("drift_corr_pairs", None)
    st.session_state.setdefault("drift_feature_selection_meta", None)
    st.session_state.setdefault("drift_results", None)
    st.session_state.setdefault("drift_execution_log", None)
    st.session_state.setdefault("drift_run_manifest", None)
    st.session_state.setdefault("drift_events_df", None)
    st.session_state.setdefault("drift_events_excluded_df", None)
    st.session_state.setdefault("drift_pipeline_meta", None)
    st.session_state.setdefault("drift_event_files", [])
    st.session_state.setdefault("drift_feature_export_path", None)
    st.session_state.setdefault("drift_optuna_json_path", None)
    st.session_state.setdefault("drift_checkpoint_manifest_path", None)
    st.session_state.setdefault("drift_exp_loaded_checkpoint_run_id", None)
    st.session_state.setdefault("drift_exp_loaded_checkpoint_manifest_path", None)
    st.session_state.setdefault("drift_exp_loaded_checkpoint_feature_selection_context", None)


def _render_blueprint_tab() -> None:
    st.subheader("Paper Replication Blueprint")
    st.caption(PAPER_TITLE)

    blueprint = article_replication_blueprint()
    st.markdown("**Analyses to replicate**")
    st.dataframe(blueprint["analyses"], width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Figures**")
        st.dataframe(blueprint["figures"], width="stretch")
    with c2:
        st.markdown("**Tables**")
        st.dataframe(blueprint["tables"], width="stretch")

    st.markdown("**Related work (Table 1 replica)**")
    st.dataframe(build_related_work_table(), width="stretch")

    st.markdown("**Migration and review plan**")
    st.dataframe(build_python_migration_review_plan(), width="stretch")


def _render_events_tab() -> None:
    st.subheader("Eventos (accidentes)")
    st.markdown("Selecciona uno o varios archivos de eventos desde `Datos/`.")

    event_files = _list_event_files()
    if not event_files:
        st.warning("No se encontraron archivos de eventos en la carpeta Datos.")
        return

    selected_names = st.multiselect(
        "Archivos de eventos disponibles",
        [path.name for path in event_files],
        default=st.session_state.get("drift_event_files") or [path.name for path in event_files],
        key="drift_event_file_select",
    )

    if st.button("Procesar eventos", key="drift_process_events"):
        if not selected_names:
            st.warning("Seleccione al menos un archivo de eventos.")
            return
        try:
            porticos_df = load_porticos()
        except Exception as exc:
            st.error(f"No se pudieron cargar los pórticos: {exc}")
            return

        frames: List[pd.DataFrame] = []
        for name in selected_names:
            path = DATA_DIR / name
            try:
                frames.append(_read_event_csv(path))
            except Exception as exc:
                st.error(f"No se pudo leer {name}: {exc}")
                return

        raw_df = pd.concat(frames, ignore_index=True)
        try:
            accidents_df, excluded_df = process_accidentes_df(
                raw_df, porticos_df, return_excluded=True
            )
        except Exception as exc:
            st.error(f"No se pudieron procesar los eventos: {exc}")
            return

        st.session_state["drift_events_df"] = accidents_df
        st.session_state["drift_events_excluded_df"] = excluded_df
        st.session_state["drift_event_files"] = selected_names
        st.success(
            f"Eventos procesados: {len(accidents_df):,} | "
            f"Excluidos sin pórtico: {len(excluded_df):,}"
        )

    accidents_df = st.session_state.get("drift_events_df")
    if not isinstance(accidents_df, pd.DataFrame) or accidents_df.empty:
        st.info("No hay eventos procesados en memoria.")
        return

    preview_cols = [
        col
        for col in [
            "accidente_time",
            "ultimo_portico",
            "proximo_portico",
            "duracion_accidente",
            "severidad",
            "Km.",
            "Eje",
            "Calzada",
        ]
        if col in accidents_df.columns
    ]

    st.caption(f"Archivos: {', '.join(st.session_state.get('drift_event_files', []))}")
    st.dataframe(accidents_df[preview_cols].head(120), width="stretch")

    excluded_df = st.session_state.get("drift_events_excluded_df")
    if isinstance(excluded_df, pd.DataFrame) and not excluded_df.empty:
        st.markdown("**Eventos excluidos por asignacion de pórtico**")
        st.dataframe(excluded_df.head(80), width="stretch")

    st.markdown("**Accidentes por tramo (replica de Crash prediction)**")
    km_col = _find_match_column(accidents_df, ["Km.", "Km", "Kilometro"])
    eje_col = _find_match_column(accidents_df, ["Eje"])
    calzada_col = _find_match_column(accidents_df, ["Calzada"])
    if not km_col or not eje_col or not calzada_col:
        st.info("No se encontraron columnas Km/Eje/Calzada para construir tramos.")
        return

    try:
        porticos_df = load_porticos()
    except Exception as exc:
        st.warning(f"No se pudieron cargar pórticos para construir tramos: {exc}")
        return

    acc_seg = accidents_df[[eje_col, calzada_col, km_col]].copy()
    acc_seg = acc_seg.rename(columns={eje_col: "eje", calzada_col: "calzada", km_col: "km_acc"})
    acc_seg["km_acc"] = pd.to_numeric(
        acc_seg["km_acc"].astype(str).str.replace(",", "."),
        errors="coerce",
    )
    acc_seg = acc_seg.dropna(subset=["km_acc", "eje", "calzada"])

    keys: List[Dict[str, Any]] = []
    for row in acc_seg.itertuples(index=False):
        try:
            cand = find_candidate_porticos(
                acc_km=row.km_acc,
                porticos_df=porticos_df,
                eje=row.eje,
                calzada=row.calzada,
            )
        except Exception:
            continue
        posterior = cand.get("posterior")
        cercano = cand.get("cercano")
        if posterior is None or cercano is None:
            continue
        keys.append(
            {
                "Eje": posterior["eje"],
                "Calzada": posterior["calzada"],
                "portico_inicio": _normalize_portico_code(posterior["portico"]),
                "portico_fin": _normalize_portico_code(cercano["portico"]),
            }
        )

    if not keys:
        st.info("No se pudo asignar accidentes a tramos.")
        return

    tramo_df = (
        pd.DataFrame(keys)
        .groupby(["Eje", "Calzada", "portico_inicio", "portico_fin"], dropna=False)
        .size()
        .reset_index(name="accidentes")
        .sort_values(["Eje", "Calzada", "portico_inicio", "portico_fin"])
        .reset_index(drop=True)
    )
    st.dataframe(tramo_df, width="stretch")


def _render_feature_engineering_output() -> None:
    raw_df = st.session_state.get("drift_raw_df")
    clean_df = st.session_state.get("drift_clean_df")
    summary = st.session_state.get("drift_prep_summary")
    pipeline_meta = st.session_state.get("drift_pipeline_meta")
    prep_config = pipeline_meta.get("prep_config") if isinstance(pipeline_meta, dict) else {}

    if isinstance(pipeline_meta, dict):
        steps = pipeline_meta.get("steps") or []
        if steps:
            st.markdown("**Pasos ejecutados**")
            for idx, step in enumerate(steps, start=1):
                st.write(f"{idx}. {step}")

        counts = pipeline_meta.get("counts") or {}
        if counts:
            final_accidents = int(
                counts.get("final_positive_rows", _count_positive_target_rows(clean_df))
            )
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Eventos procesados", f"{int(counts.get('processed_accidents', 0)):,}")
            c2.metric("Accidentes finales", f"{final_accidents:,}")
            c3.metric("Filas flujo consultadas", f"{int(counts.get('flow_rows_queried', 0)):,}")
            c4.metric("Filas base (target)", f"{int(counts.get('base_rows_with_target', 0)):,}")
            c5.metric("Filas finales", f"{int(counts.get('final_rows', 0)):,}")

    if isinstance(summary, DataPreparationSummary):
        st.markdown("**Resumen Stage 1 / Stage 2**")
        if isinstance(prep_config, dict) and prep_config.get("reconstructed_from_payload"):
            st.caption("Resumen reconstruido desde `raw_features` y `clean_features` del archivo cargado.")
        stage_df = pd.DataFrame(
            [
                {
                    "etapa": "Entrada",
                    "filas": summary.initial_rows,
                    "variables": summary.initial_features,
                },
                {
                    "etapa": "Post Stage 1",
                    "filas": summary.rows_after_missing_drop,
                    "variables": summary.remaining_features_after_stage1,
                },
                {
                    "etapa": "Post Stage 2",
                    "filas": summary.final_rows,
                    "variables": summary.remaining_features_after_stage1,
                },
            ]
        )
        st.dataframe(stage_df, width="stretch")
        st.markdown("**Detalle de remociones por etapa**")
        st.dataframe(
            _build_preparation_detail_table(summary, prep_config=prep_config),
            width="stretch",
        )

    if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
        st.markdown("**Preview base (raw_features)**")
        st.dataframe(raw_df.head(60), width="stretch")
    if isinstance(clean_df, pd.DataFrame) and not clean_df.empty:
        st.markdown("**Preview limpia (clean_features)**")
        st.dataframe(clean_df.head(60), width="stretch")

    export_path = st.session_state.get("drift_feature_export_path")
    if export_path:
        st.success(f"Archivo DuckDB generado: {export_path}")


def _render_feature_engineering_tab() -> None:
    st.subheader("Feature engineering")
    st.caption(f"DuckDB fijo: {FLOW_DB_PATH} | tabla: {FLOW_TABLE_NAME}")
    st.markdown(
        "**Flujo de esta pestaña:**\n"
        "1. Cargar eventos procesados (tab Eventos).\n"
        "2. Consultar `Datos/flujos.duckdb` con muestreo y filtro de tramo.\n"
        "3. Generar variables por intervalo para pórticos, segmentos y lanes (por lote o en una pasada).\n"
        "4. Construir `target` con accidentes.\n"
        "5. Aplicar Stage 1 y Stage 2 con selectores.\n"
        "6. Exportar resultado en DuckDB (`raw_features` + `clean_features`)."
    )

    has_memory = isinstance(st.session_state.get("drift_clean_df"), pd.DataFrame) and not st.session_state.get(
        "drift_clean_df"
    ).empty
    source_options = ["Cargar existentes", "Calcular nuevas", "En memoria"]
    if "drift_feature_source" not in st.session_state:
        st.session_state["drift_feature_source"] = "En memoria" if has_memory else "Calcular nuevas"
    source = st.radio(
        "Fuente",
        source_options,
        horizontal=True,
        key="drift_feature_source",
    )

    if source == "En memoria":
        if not has_memory:
            st.info("No hay dataset en memoria.")
            return
        _render_feature_engineering_output()
        export_prefix = st.text_input(
            "Prefijo exportación (sin .duckdb)",
            value="drift_flow_features_manual",
            key="drift_export_prefix_memory",
        ).strip()
        if st.button("Exportar DuckDB desde memoria", key="drift_export_memory"):
            try:
                out_path = _save_feature_duckdb(
                    st.session_state["drift_raw_df"],
                    st.session_state["drift_clean_df"],
                    prefix=export_prefix or "drift_flow_features_manual",
                    selection_payload=_current_feature_selection_payload(),
                )
            except Exception as exc:
                st.error(f"No se pudo exportar: {exc}")
            else:
                st.session_state["drift_feature_export_path"] = str(out_path)
                st.success(f"Exportado en {out_path}")
        return

    if source == "Cargar existentes":
        files = _list_feature_duckdb_files()
        if not files:
            st.warning("No se encontraron archivos de features en Resultados.")
            return
        names = [path.name for path in files]
        selected = st.selectbox("Archivo DuckDB", options=names, key="drift_load_feature_file")
        if st.button("Cargar archivo", key="drift_load_existing_features"):
            try:
                raw_df, clean_df = _load_feature_payload_from_duckdb(RESULTS_DIR / selected)
                selection_payload = _load_feature_selection_payload_from_duckdb(RESULTS_DIR / selected)
            except Exception as exc:
                st.error(f"No se pudo cargar el archivo: {exc}")
                return
            if raw_df.empty and clean_df.empty:
                st.warning("El archivo no contiene tablas de features esperadas.")
                return
            if clean_df.empty:
                clean_df = raw_df.copy()
            for df in [raw_df, clean_df]:
                if "interval_start" in df.columns:
                    df["interval_start"] = pd.to_datetime(df["interval_start"], errors="coerce")
                if "portico" in df.columns:
                    df["portico"] = _normalize_portico_series(df["portico"])
                if "target" in df.columns:
                    df["target"] = pd.to_numeric(df["target"], errors="coerce").fillna(0).astype(int)

            default_feature_cols = _feature_columns_for_modeling(clean_df, target_col="target")
            prep_summary, stage1_info, zero_runs = _rebuild_preparation_artifacts_from_payload(
                raw_df,
                clean_df,
                missing_threshold=0.01,
                target_col="target",
                time_col="interval_start",
                min_zero_days=7,
            )
            _clear_feature_selection_state(clear_selected_features=True)
            candidate_feature_cols = list(default_feature_cols)
            selected_feature_cols = list(default_feature_cols)
            if selection_payload:
                stored_candidates = [
                    col for col in selection_payload.get("candidate_features", []) if col in clean_df.columns
                ]
                stored_selected = [
                    col for col in selection_payload.get("selected_features", []) if col in clean_df.columns
                ]
                candidate_feature_cols = stored_candidates or list(default_feature_cols)
                selected_feature_cols = stored_selected or candidate_feature_cols
                st.session_state["drift_corr_pairs"] = selection_payload.get("corr_pairs")
                st.session_state["drift_importance_df"] = selection_payload.get("importance_df")
                st.session_state["drift_feature_selection_meta"] = selection_payload.get("config") or None

            st.session_state["drift_raw_df"] = raw_df
            st.session_state["drift_clean_df"] = clean_df
            st.session_state["drift_candidate_feature_cols"] = candidate_feature_cols
            st.session_state["drift_feature_cols"] = selected_feature_cols
            st.session_state["drift_stage1_info"] = stage1_info
            st.session_state["drift_zero_runs"] = zero_runs
            st.session_state["drift_prep_summary"] = prep_summary
            st.session_state["drift_pipeline_meta"] = {
                "steps": [
                    f"Archivo existente cargado: {selected}",
                    "Resultados de Feature Selection restaurados desde DuckDB."
                    if selection_payload
                    else "El archivo no incluye resultados persistidos de Feature Selection.",
                ],
                "prep_config": {
                    "missing_threshold": 0.01,
                    "min_zero_days": 7,
                    "reconstructed_from_payload": True,
                },
                "counts": {
                    "processed_accidents": int(
                        len(st.session_state.get("drift_events_df"))
                    )
                    if isinstance(st.session_state.get("drift_events_df"), pd.DataFrame)
                    else 0,
                    "final_positive_rows": _count_positive_target_rows(clean_df),
                    "flow_rows_queried": int(len(raw_df)),
                    "base_rows_with_target": int(len(raw_df)),
                    "final_rows": int(len(clean_df)),
                },
            }
            st.session_state["drift_feature_export_path"] = str(RESULTS_DIR / selected)
            st.success(f"Archivo cargado: {selected}")
        _render_feature_engineering_output()
        return

    accidents_df = st.session_state.get("drift_events_df")
    if not isinstance(accidents_df, pd.DataFrame) or accidents_df.empty:
        st.info("Primero procesa eventos en la pestaña Eventos.")
        return
    if duckdb is None:
        st.error("duckdb no esta instalado. Ejecuta `pip install duckdb`.")
        return
    if not FLOW_DB_PATH.exists():
        st.error(f"No existe la base de flujos: {FLOW_DB_PATH}")
        return

    try:
        summary = _flow_db_summary(str(FLOW_DB_PATH), FLOW_TABLE_NAME)
    except Exception as exc:
        st.error(f"No se pudo leer {FLOW_DB_PATH}: {exc}")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Filas", f"{int(summary.get('row_count', 0)):,}")
    c2.metric(
        "Fecha min",
        summary["min_ts"].strftime("%Y-%m-%d %H:%M") if isinstance(summary.get("min_ts"), pd.Timestamp) else "-",
    )
    c3.metric(
        "Fecha max",
        summary["max_ts"].strftime("%Y-%m-%d %H:%M") if isinstance(summary.get("max_ts"), pd.Timestamp) else "-",
    )

    sample_mode = _build_flow_sample_mode_selector(key_prefix="drift_flow")
    sample, percent_mode, range_valid = _build_flow_sample_inputs(summary, sample_mode, key_prefix="drift_flow")

    use_batches = st.checkbox("Procesamiento por lote", value=True, key="drift_use_batches")
    if percent_mode and use_batches:
        st.warning("Muestreo por porcentaje no es compatible con lote; se ejecutará en una sola consulta.")
        use_batches = False
    batch_mode = st.selectbox(
        "Modo de lote",
        ["Mensual", "Semanal"],
        index=0,
        disabled=not use_batches,
        key="drift_batch_mode",
    )

    tramo_tuple = _build_tramo_selector_for_feature_tab()
    allowed_porticos = _porticos_in_tramo(tramo_tuple)
    article_categories = list(DRIFT_ARTICLE_CATEGORIES)
    st.caption(
        "Variables article-aligned calculadas automaticamente: "
        "Vel, Sd, Flow, Den y deltas por portico/categoria; "
        "Ft_Motorcycle/Ft_Heavy; "
        "Vel, Sd, Flow, Den, DeltaFlow, DeltaVel, DeltaDen y CL por segmentos consecutivos; "
        "ademas de bloques globales y por lane para segmentos."
    )
    st.caption(
        f"Categorias fijas del articulo: {', '.join(article_categories)} | "
        f"Porticos del corredor: {', '.join(DRIFT_FOCUS_PORTICOS)} | "
        f"Orden operacional de segmentos: {' -> '.join(_resolve_drift_article_portico_order(allowed_porticos))}"
    )
    with st.expander("Variables y formulas del feature engineering", expanded=False):
        st.caption(
            "Nota metodologica: la Table 4 resume las familias principales, pero el propio articulo y la Figure 6 "
            "muestran que parte del set predictivo se calcula tambien a nivel de lane."
        )
        st.dataframe(feature_engineering_reference_table(), width="stretch")
    interval_minutes = st.number_input("Intervalo (minutos)", min_value=1, max_value=60, value=5, step=1)

    stage1_mode = st.selectbox(
        "Stage 1: missing data >1% + drop intervalos incompletos",
        ["Aplicar", "Omitir"],
        index=0,
        key="drift_stage1_selector",
    )
    missing_threshold = st.number_input("Umbral Stage 1", min_value=0.0, max_value=1.0, value=0.01, step=0.001)
    stage2_mode = st.selectbox(
        "Stage 2: filtrar ventanas multi-dia sin accidentes",
        ["Aplicar", "Omitir"],
        index=0,
        key="drift_stage2_selector",
    )
    min_zero_days = st.number_input(
        "Dias minimos sin accidentes (Stage 2)",
        min_value=1,
        max_value=60,
        value=7,
        step=1,
    )

    if st.button("Calcular nuevas variables", key="drift_run_feature_engineering", disabled=not range_valid):
        flow_date_start = sample.date_start
        flow_date_end = sample.date_end
        if flow_date_start is None and isinstance(summary.get("min_ts"), pd.Timestamp):
            flow_date_start = summary["min_ts"]
        if flow_date_end is None and isinstance(summary.get("max_ts"), pd.Timestamp):
            flow_date_end = summary["max_ts"]
        if flow_date_start is None or flow_date_end is None:
            st.error("No se pudo inferir el rango de fechas para consultar flujos.")
            return

        flow_row_limit = int(sample.row_limit) if sample.row_limit is not None else None
        queried_rows = 0
        progress = _FeatureEngineeringProgress()
        out_path: Optional[Path] = None
        batch_ranges: List[Any] = []

        if use_batches:
            batch_mode_code = "month" if str(batch_mode) == "Mensual" else "week"
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = RESULTS_DIR / f"drift_flow_features_{stamp}.duckdb"
            try:
                batch_meta = _compute_batched_flow_features_to_duckdb(
                    flow_db_path=str(FLOW_DB_PATH),
                    flow_table_name=FLOW_TABLE_NAME,
                    out_path=out_path,
                    batch_mode=batch_mode_code,
                    interval_minutes=int(interval_minutes),
                    lanes=1,
                    metrics=[],
                    categories=article_categories,
                    date_start=pd.Timestamp(flow_date_start),
                    date_end=pd.Timestamp(flow_date_end),
                    allowed_porticos=allowed_porticos,
                    row_limit=flow_row_limit,
                    progress_bar=progress,
                )
            except Exception as exc:
                progress.set_progress(1.0, "Error en procesamiento por lotes.", detail=str(exc))
                st.error(f"No se pudo ejecutar el procesamiento por lotes en DuckDB: {exc}")
                return

            batch_ranges = list(batch_meta.get("batch_ranges") or [])
            queried_rows = int(batch_meta.get("input_rows", 0) or 0)
            if not out_path.exists():
                progress.set_progress(1.0, "No se generaron features en lotes.")
                st.warning("No se generaron variables con la configuracion actual.")
                return
            try:
                progress.set_progress(
                    0.88,
                    "Cargando features batcheadas desde DuckDB...",
                    detail=f"{len(batch_ranges)} lotes procesados.",
                )
                features_df = _load_feature_df_from_duckdb(out_path)
            except Exception as exc:
                progress.set_progress(1.0, "Error cargando salida batcheada.", detail=str(exc))
                st.error(f"No se pudo cargar el resultado por lotes desde DuckDB: {exc}")
                return
        else:
            batch_ranges = [
                (
                    pd.Timestamp(flow_date_start),
                    pd.Timestamp(flow_date_end) + pd.Timedelta(nanoseconds=1),
                )
            ]
            progress.set_progress(
                0.10,
                "Consultando flujos...",
                detail=f"Rango {pd.Timestamp(flow_date_start)} a {pd.Timestamp(flow_date_end)}",
            )
            df_full = _query_flujos_duckdb_filtered(
                str(FLOW_DB_PATH),
                table_name=FLOW_TABLE_NAME,
                row_limit=flow_row_limit,
                date_start=pd.Timestamp(flow_date_start),
                date_end=pd.Timestamp(flow_date_end),
                allowed_porticos=allowed_porticos,
            )
            queried_rows = int(len(df_full))
            if df_full.empty:
                progress.set_progress(1.0, "Consulta completada sin filas.")
                st.warning("No se generaron variables con la configuracion actual.")
                return

            progress.set_progress(
                0.30,
                "Limpiando registros crudos...",
                detail=f"Filas consultadas: {queried_rows:,}",
            )
            df_full["FECHA"] = pd.to_datetime(df_full["FECHA"], errors="coerce")
            df_full["VELOCIDAD"] = pd.to_numeric(df_full["VELOCIDAD"], errors="coerce")
            df_full["CATEGORIA"] = pd.to_numeric(df_full["CATEGORIA"], errors="coerce")
            df_full = df_full.dropna(subset=["FECHA", "CATEGORIA"])
            df_full["CATEGORIA"] = df_full["CATEGORIA"].astype(int)
            if df_full.empty:
                progress.set_progress(1.0, "Limpieza completada sin filas validas.")
                st.warning("No se generaron variables con la configuracion actual.")
                return

            progress.set_progress(
                0.55,
                "Calculando features article-aligned...",
                detail=f"Filas validas: {len(df_full):,}",
            )
            features_df = _compute_drift_article_features(
                df_full,
                interval_minutes=int(interval_minutes),
                categories=article_categories,
                allowed_porticos=allowed_porticos,
            )

        if features_df.empty:
            progress.set_progress(1.0, "No se generaron variables con la configuracion actual.")
            st.warning("No se generaron variables con la configuracion actual.")
            return

        progress.set_progress(
            0.90,
            "Normalizando features y construyendo target...",
            detail=f"Filas features: {len(features_df):,}",
        )
        features_df["interval_start"] = pd.to_datetime(features_df["interval_start"], errors="coerce")
        features_df = features_df.dropna(subset=["interval_start"]).drop_duplicates(
            subset=["interval_start"]
        ).sort_values("interval_start").reset_index(drop=True)

        corridor_accidents = _filter_accidents_for_allowed_porticos(accidents_df, allowed_porticos)
        base_df = _add_interval_accident_target(
            features_df,
            corridor_accidents,
            interval_minutes=int(interval_minutes),
            interval_col="interval_start",
            accident_time_col="accidente_time",
        )
        if base_df.empty:
            progress.set_progress(1.0, "No se pudo construir base con target.")
            st.warning("No se pudo generar base con target.")
            return

        base_df["target"] = pd.to_numeric(base_df["target"], errors="coerce").fillna(0).astype(int)
        feature_cols = _feature_columns_for_modeling(base_df, target_col="target")

        progress.set_progress(
            0.94,
            "Ejecutando Stage 1 / Stage 2...",
            detail=f"Base con target: {len(base_df):,} filas",
        )
        clean_df, prep_summary, stage1_info, zero_runs, prep_steps = run_configurable_preparation_pipeline(
            base_df,
            feature_cols=feature_cols,
            apply_stage1=(stage1_mode == "Aplicar"),
            missing_threshold=float(missing_threshold),
            apply_stage2=(stage2_mode == "Aplicar"),
            min_zero_days=int(min_zero_days),
            target_col="target",
            time_col="interval_start",
            interval_minutes=int(interval_minutes),
        )

        if clean_df.empty:
            progress.set_progress(1.0, "Preparacion completada sin dataset util.")
            st.warning("Las etapas de preparacion dejaron el dataset vacio.")
            return

        progress.set_progress(
            0.98,
            "Exportando resultado a DuckDB...",
            detail=f"Filas limpias: {len(clean_df):,}",
        )
        _clear_feature_selection_state()
        try:
            out_path = _save_feature_duckdb(
                base_df,
                clean_df,
                prefix="drift_flow_features",
                path=out_path,
                selection_payload=None,
            )
        except Exception as exc:
            progress.set_progress(1.0, "Error exportando DuckDB.", detail=str(exc))
            st.error(f"No se pudo exportar el resultado a DuckDB: {exc}")
            return

        flow_count = _count_flujos_rows(
            str(FLOW_DB_PATH),
            table_name=FLOW_TABLE_NAME,
            date_start=flow_date_start,
            date_end=flow_date_end,
            allowed_porticos=allowed_porticos,
        )

        steps = [
            "Eventos procesados desde la pestaña Eventos (Crash prediction replica).",
            f"Consulta base de flujos fija: {FLOW_DB_PATH} / {FLOW_TABLE_NAME}.",
            f"Muestreo aplicado: {sample_mode}.",
            "Variables generadas segun Section 4.1 + Table 4 + Figure 6, incluyendo bloques por lane en segmentos.",
            f"Categorias fijas del articulo: {', '.join(article_categories)}.",
            f"Porticos del corredor: {', '.join(DRIFT_FOCUS_PORTICOS)}.",
            f"Orden operacional de segmentos: {' -> '.join(_resolve_drift_article_portico_order(allowed_porticos))}.",
        ]
        if tramo_tuple:
            steps.append(f"Filtro tramo: {_describe_tramo_selection(tramo_tuple)}.")
        if use_batches:
            steps.append(
                f"Procesamiento por lote DuckDB estilo Crash prediction: {batch_mode} ({len(batch_ranges)} lotes)."
            )
        else:
            steps.append("Procesamiento sin lotes (consulta unica).")
        steps.extend(prep_steps)
        steps.append(f"Exportacion DuckDB: {out_path}.")

        st.session_state["drift_raw_df"] = base_df
        st.session_state["drift_clean_df"] = clean_df
        st.session_state["drift_stage1_info"] = stage1_info
        st.session_state["drift_zero_runs"] = zero_runs
        st.session_state["drift_prep_summary"] = prep_summary
        st.session_state["drift_candidate_feature_cols"] = stage1_info.get("remaining_features", feature_cols)
        st.session_state["drift_feature_cols"] = stage1_info.get("remaining_features", feature_cols)
        st.session_state["drift_feature_export_path"] = str(out_path)
        st.session_state["drift_pipeline_meta"] = {
            "steps": steps,
            "prep_config": {
                "stage1_applied": bool(stage1_mode == "Aplicar"),
                "missing_threshold": float(missing_threshold),
                "stage2_applied": bool(stage2_mode == "Aplicar"),
                "min_zero_days": int(min_zero_days),
                "interval_minutes": int(interval_minutes),
            },
            "counts": {
                "processed_accidents": int(len(accidents_df)),
                "final_positive_rows": _count_positive_target_rows(clean_df),
                "flow_rows_available_in_filter": int(flow_count),
                "flow_rows_queried": int(queried_rows),
                "base_rows_with_target": int(len(base_df)),
                "positive_intervals_target": int(base_df["target"].sum()),
                "final_rows": int(len(clean_df)),
            },
        }
        progress.set_progress(
            1.0,
            "Feature engineering completado.",
            detail=f"Base: {len(base_df):,} | Limpio: {len(clean_df):,} | DuckDB: {out_path.name}",
        )
        st.success(
            f"Pipeline completado. Base={len(base_df):,} filas | "
            f"Limpio={len(clean_df):,} filas | Exportado={out_path.name}"
        )

    _render_feature_engineering_output()


def _render_feature_selection_tab() -> None:
    st.subheader("Feature Selection")

    clean_df = st.session_state.get("drift_clean_df")
    feature_cols = st.session_state.get("drift_feature_cols")

    if not isinstance(clean_df, pd.DataFrame) or clean_df.empty:
        st.info("Ejecute Feature engineering primero.")
        return
    if not feature_cols:
        st.info("No feature columns selected.")
        return

    target_col = "target" if "target" in clean_df.columns else st.selectbox("Target column", options=list(clean_df.columns))

    corr_threshold = st.number_input("Correlation threshold", min_value=0.5, max_value=0.999, value=0.95, step=0.01)
    top_n = st.number_input("Top-N features", min_value=5, max_value=100, value=20, step=1)

    if st.button("Run correlation + Random Forest selection", key="drift_run_feat_sel"):
        kept, dropped, corr = drop_highly_correlated_features(
            clean_df,
            feature_cols=feature_cols,
            threshold=float(corr_threshold),
        )
        corr_pairs = compute_abs_correlations(clean_df, feature_cols)
        importance_df = rank_features_random_forest(
            clean_df,
            kept,
            target_col=target_col,
            top_n=int(top_n),
        )
        selected_for_experiments = (
            importance_df["feature"].astype(str).tolist() if not importance_df.empty else list(kept)
        )

        st.session_state["drift_candidate_feature_cols"] = kept
        st.session_state["drift_feature_cols"] = selected_for_experiments
        st.session_state["drift_corr_pairs"] = corr_pairs
        st.session_state["drift_importance_df"] = importance_df
        st.session_state["drift_feature_selection_meta"] = {
            "method": "correlation_plus_random_forest",
            "target_col": str(target_col),
            "corr_threshold": float(corr_threshold),
            "top_n": int(top_n),
            "candidate_feature_count": int(len(kept)),
            "selected_feature_count": int(len(selected_for_experiments)),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }

        export_path = st.session_state.get("drift_feature_export_path")
        persisted_msg = ""
        if export_path and Path(str(export_path)).exists():
            try:
                _write_feature_selection_payload_to_duckdb(
                    Path(str(export_path)),
                    selection_payload=_current_feature_selection_payload(linked_path=Path(str(export_path))),
                )
                persisted_msg = f" Persistido en {Path(str(export_path)).name}."
            except Exception as exc:
                persisted_msg = f" No se pudo persistir en DuckDB: {exc}"

        st.success(
            f"Dropped {len(dropped)} highly correlated features. "
            f"Using top {len(selected_for_experiments)} features for experiments."
            f"{persisted_msg}"
        )

    corr_pairs = st.session_state.get("drift_corr_pairs")
    importance_df = st.session_state.get("drift_importance_df")
    selection_meta = st.session_state.get("drift_feature_selection_meta") or {}

    export_path = st.session_state.get("drift_feature_export_path")
    if export_path:
        linked_name = Path(str(export_path)).name
        if selection_meta:
            st.caption(f"Resultado de Feature Selection vinculado a `{linked_name}`.")
        else:
            st.caption(
                f"Si ejecutas Feature Selection, el resultado se persistirá dentro de `{linked_name}`."
            )

    if isinstance(corr_pairs, pd.Series) and not corr_pairs.empty:
        st.markdown("**Figure 5 replica: density of |rho|**")
        hist, edges = np.histogram(corr_pairs.values, bins=60, range=(0.0, 1.0), density=True)
        centers = (edges[:-1] + edges[1:]) / 2.0
        try:
            import plotly.graph_objects as go

            q1, median, q3 = np.percentile(corr_pairs.values, [25, 50, 75])
            y_max = float(hist.max())

            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(
                x=centers, y=hist, mode="lines",
                line=dict(color="#1f77b4", width=2), name="Density",
            ))
            # Vertical reference lines with staggered y-positions to avoid overlap
            vline_annotations = [
                (q1, "Q1", "gray", 0.70),
                (median, "Median", "gray", 0.85),
                (q3, "Q3", "gray", 1.00),
                (0.95, "|ρ|=0.95", "red", 1.00),
            ]
            for val, label, color, y_frac in vline_annotations:
                fig5.add_shape(
                    type="line", x0=val, x1=val, y0=0, y1=y_max * 1.08,
                    line=dict(color=color, width=1.5, dash="dot"),
                )
                fig5.add_annotation(
                    x=val, y=y_max * 1.08 * y_frac,
                    text=f"<b>{label}={val:.2f}</b>" if label != "|ρ|=0.95" else f"<b>{label}</b>",
                    showarrow=True, arrowhead=0, arrowcolor=color,
                    ax=30, ay=-20,
                    font=dict(size=13, color=color),
                    bordercolor=color, borderwidth=1, borderpad=3,
                    bgcolor="rgba(255,255,255,0.85)",
                )
            fig5.update_layout(
                title=dict(text="Density of |ρ|", font=dict(size=18)),
                xaxis=dict(title=dict(text="|ρ|", font=dict(size=15)), tickfont=dict(size=13)),
                yaxis=dict(title=dict(text="Density", font=dict(size=15)), tickfont=dict(size=13)),
                template="plotly_white",
                showlegend=False,
                height=480,
                margin=dict(l=60, r=40, t=50, b=50),
            )
            st.plotly_chart(fig5, width="stretch")
        except ImportError:
            density_df = pd.DataFrame({"|ρ|": centers, "Density": hist})
            st.line_chart(density_df, x="|ρ|", y="Density", width="stretch")

    if isinstance(importance_df, pd.DataFrame) and not importance_df.empty:
        st.markdown("**Figure 6 replica: top feature importances**")
        try:
            import plotly.graph_objects as go

            plot_df = importance_df.sort_values("importance", ascending=True).copy()
            # Normalize importance to 0-100 scale (average across classes)
            max_imp = plot_df["importance"].max()
            if max_imp > 0:
                plot_df["importance_scaled"] = plot_df["importance"] / max_imp * 100
            else:
                plot_df["importance_scaled"] = plot_df["importance"]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=plot_df["importance_scaled"],
                y=plot_df["feature"],
                mode="markers",
                marker=dict(size=10, color="#5B1A18"),
            ))
            fig.update_layout(
                title="Top 20 Variables by Importance",
                xaxis_title="Importance (Average across classes)",
                yaxis_title="Variable",
                template="plotly_white",
                height=max(400, len(plot_df) * 28),
                margin=dict(l=10, r=10, t=40, b=40),
                xaxis=dict(showgrid=True, gridcolor="#E5E5E5"),
                yaxis=dict(showgrid=True, gridcolor="#E5E5E5"),
            )
            st.plotly_chart(fig, width="stretch")
        except ImportError:
            st.bar_chart(importance_df.set_index("feature")["importance"], width="stretch")
        st.dataframe(importance_df, width="stretch")
        st.caption(
            "Experiments use the ordered top-N features shown above, after the correlation filter."
        )


def _render_experiments_tab() -> None:
    st.subheader("Recalibration Experiments")

    clean_df = st.session_state.get("drift_clean_df")
    feature_cols = st.session_state.get("drift_feature_cols")

    if not isinstance(clean_df, pd.DataFrame) or clean_df.empty:
        st.info("Ejecute Feature engineering primero.")
        return
    if not feature_cols:
        st.info("Run feature selection first.")
        return

    st.caption(
        f"Feature set locked for experiments: {len(feature_cols)} variables after correlation filtering + RF ranking."
    )
    st.caption(
        "ARF corre como estrategia online inmediata separada de los modelos batch. "
        "Los modelos seleccionados abajo aplican a static / period_aligned / cumulative / adaptive_adwin / adaptive_kswin."
    )

    feature_count = max(1, len(feature_cols))
    base_year: Optional[int] = None
    random_state = 42
    custom_grids: Optional[Dict[str, Dict[str, Sequence[Any]]]] = None
    st.session_state["drift_experiment_feature_count"] = feature_count
    if _normalize_experiment_resource_mode(st.session_state.get("drift_exp_resource_mode")) != str(
        st.session_state.get("drift_exp_resource_mode", "")
    ):
        st.session_state["drift_exp_resource_mode"] = DEFAULT_EXPERIMENT_RESOURCE_MODE
    if "drift_experiment_preset" not in st.session_state:
        st.session_state["drift_experiment_preset"] = DEFAULT_EXPERIMENT_PRESET
    elif str(st.session_state["drift_experiment_preset"]) not in EXPERIMENT_PRESET_CONFIGS:
        st.session_state["drift_experiment_preset"] = DEFAULT_EXPERIMENT_PRESET
    if st.session_state.get("drift_experiment_preset_applied") is None:
        _apply_experiment_preset(
            str(st.session_state["drift_experiment_preset"]),
            feature_count=feature_count,
        )
        st.session_state["drift_experiment_preset_applied"] = str(
            st.session_state["drift_experiment_preset"]
        )
    else:
        _sync_experiment_control_bounds(feature_count)

    available_checkpoints = _list_recalibration_checkpoints()
    checkpoint_map = {
        str(item["run_id"]): item
        for item in available_checkpoints
    }
    checkpoint_selector_options: List[Optional[str]] = [None] + list(checkpoint_map.keys())
    if (
        st.session_state.get("drift_exp_checkpoint_selector") not in checkpoint_map
        and st.session_state.get("drift_exp_checkpoint_selector") is not None
    ):
        st.session_state["drift_exp_checkpoint_selector"] = None

    st.markdown("**Checkpoint Autofill**")
    if available_checkpoints:
        selected_checkpoint_run_id = st.selectbox(
            "Available training checkpoints",
            options=checkpoint_selector_options,
            key="drift_exp_checkpoint_selector",
            format_func=lambda run_id: (
                "Seleccionar checkpoint..."
                if run_id is None
                else str(checkpoint_map.get(str(run_id), {}).get("label", run_id))
            ),
            help="Carga automaticamente la configuracion del experimento desde el run_manifest de un checkpoint persistido.",
        )
        selected_checkpoint = (
            checkpoint_map.get(str(selected_checkpoint_run_id))
            if selected_checkpoint_run_id is not None
            else None
        )
        if isinstance(selected_checkpoint, dict):
            checkpoint_feature_count = selected_checkpoint.get("feature_count")
            st.caption(
                f"Checkpoint `{selected_checkpoint['run_id']}` | estado {selected_checkpoint['status']} | "
                f"actualizado {selected_checkpoint.get('updated_at') or 'desconocido'}"
            )
            if checkpoint_feature_count is not None and int(checkpoint_feature_count) != int(feature_count):
                st.warning(
                    f"El checkpoint fue ejecutado con {int(checkpoint_feature_count)} features y el dataset actual tiene "
                    f"{int(feature_count)} features seleccionadas. Se cargará la configuracion y, si las columnas siguen "
                    "disponibles, tambien se restaurará el feature set del checkpoint."
                )
        if st.button("Load checkpoint configuration", key="drift_exp_load_checkpoint_configuration"):
            if not isinstance(selected_checkpoint, dict):
                st.warning("Seleccione un checkpoint para cargar su configuracion.")
            else:
                checkpoint_run_manifest = dict(selected_checkpoint.get("run_manifest") or {})
                checkpoint_feature_selection_context = dict(
                    selected_checkpoint.get("feature_selection_context") or {}
                )
                if checkpoint_feature_selection_context:
                    checkpoint_run_manifest["feature_selection_context"] = checkpoint_feature_selection_context
                for state_key, state_value in _checkpoint_form_state_from_run_manifest(
                    checkpoint_run_manifest,
                    available_columns=list(clean_df.columns),
                    feature_count=feature_count,
                ).items():
                    st.session_state[state_key] = state_value
                st.session_state["drift_exp_loaded_checkpoint_run_id"] = str(selected_checkpoint["run_id"])
                st.session_state["drift_exp_loaded_checkpoint_manifest_path"] = str(
                    selected_checkpoint["manifest_path"]
                )
                st.session_state["drift_exp_loaded_checkpoint_feature_selection_context"] = (
                    checkpoint_feature_selection_context or None
                )
                _sync_experiment_control_bounds(feature_count)
                st.rerun()
    else:
        st.caption("No hay checkpoints persistidos disponibles para autocompletar la configuracion.")

    loaded_checkpoint_run_id = st.session_state.get("drift_exp_loaded_checkpoint_run_id")
    loaded_checkpoint_manifest_path = st.session_state.get("drift_exp_loaded_checkpoint_manifest_path")
    loaded_checkpoint_preview: Optional[Dict[str, Any]] = None
    if loaded_checkpoint_run_id:
        loaded_checkpoint_preview = _preview_recalibration_checkpoint_run(str(loaded_checkpoint_run_id))
    if loaded_checkpoint_run_id:
        st.caption(
            f"Configuracion cargada desde checkpoint `{loaded_checkpoint_run_id}`"
            + (
                f" | manifest {loaded_checkpoint_manifest_path}"
                if loaded_checkpoint_manifest_path
                else ""
            )
        )

    preset_name = st.selectbox(
        "Configuration preset",
        options=list(EXPERIMENT_PRESET_CONFIGS.keys()),
        key="drift_experiment_preset",
        on_change=_on_experiment_preset_change,
        help="Aplica una configuracion base al formulario. Luego puedes ajustar cualquier campo manualmente.",
    )
    preset_meta = EXPERIMENT_PRESET_CONFIGS.get(str(preset_name), {})
    preset_description = str(preset_meta.get("description", "")).strip()
    if preset_description:
        st.caption(preset_description)

    resource_mode = st.selectbox(
        "Modo de recursos",
        options=list(EXPERIMENT_RESOURCE_MODE_CONFIGS.keys()),
        key="drift_exp_resource_mode",
        format_func=_experiment_resource_mode_label,
        help=(
            "Controla la politica interna de recursos del experimento. "
            "`Modo restringido` replica los caps actuales; `Modo full memoria` "
            "relaja los limites internos de CPU para aprovechar mas memoria."
        ),
    )
    resource_policy = _resolve_experiment_resource_policy(resource_mode)
    st.caption(str(resource_policy["description"]))
    st.caption(
        "Perfil efectivo: "
        f"RF {int((resource_policy.get('estimator_n_jobs') or {}).get('Random Forest', 1))} hilos | "
        f"XGBoost {int((resource_policy.get('estimator_n_jobs') or {}).get('XGBoost', 1))} hilos | "
        f"CV paralela {', '.join(resource_policy.get('parallel_cv_models') or ['ninguna'])}"
    )

    target_col_default = "target" if "target" in clean_df.columns else clean_df.columns[-1]
    time_col_default = "interval_start" if "interval_start" in clean_df.columns else clean_df.columns[0]
    if st.session_state.get("drift_exp_target_col") not in clean_df.columns:
        st.session_state["drift_exp_target_col"] = target_col_default
    if st.session_state.get("drift_exp_time_col") not in clean_df.columns:
        st.session_state["drift_exp_time_col"] = time_col_default

    target_col = st.selectbox(
        "Target column",
        options=list(clean_df.columns),
        key="drift_exp_target_col",
    )
    time_col = st.selectbox(
        "Time column",
        options=list(clean_df.columns),
        key="drift_exp_time_col",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_models = st.multiselect(
            "Models",
            MODEL_NAMES,
            default=MODEL_NAMES,
            key="drift_exp_models",
        )
        selected_strategies = st.multiselect(
            "Strategies",
            EXPERIMENT_STRATEGIES,
            default=EXPERIMENT_STRATEGIES,
            key="drift_exp_strategies",
        )
    with col2:
        fast_mode = st.checkbox(
            "Fast hyperparameter mode",
            value=True,
            key="drift_exp_fast_mode",
            help=(
                "Usa una grilla reducida de hiperparametros para los modelos batch y, si aplica, "
                "tambien una grilla reducida para SMOTE. Sirve para acelerar la sintonia inicial "
                "o comparaciones rapidas, pero explora menos combinaciones que el modo completo. "
                "No cambia las metricas ni la metodologia de evaluacion; solo restringe el espacio "
                "de busqueda. La cantidad de intentos la controla `Optuna trials`."
            ),
        )
        grid_limit = st.number_input(
            "Optuna trials",
            min_value=1,
            max_value=2000,
            value=30,
            step=1,
            key="drift_exp_optuna_trials",
        )
        folds = st.number_input(
            "CV folds",
            min_value=2,
            max_value=10,
            value=3,
            step=1,
            key="drift_exp_folds",
        )
    with col3:
        validation_size = st.number_input(
            "Validation size",
            min_value=0.05,
            max_value=0.5,
            value=0.2,
            step=0.05,
            key="drift_exp_validation_size",
        )
        adwin_delta = st.number_input(
            "ADWIN delta",
            min_value=0.0001,
            max_value=0.1,
            value=0.002,
            step=0.0005,
            format="%.4f",
            key="drift_exp_adwin_delta",
        )
        min_window = st.number_input(
            "ADWIN min window",
            min_value=100,
            max_value=200000,
            value=45000,
            step=100,
            key="drift_exp_min_window",
        )
        min_retrain_size = st.number_input(
            "ADWIN min retrain rows",
            min_value=10,
            max_value=200000,
            value=max(20, int(min_window) // 10),
            step=10,
            key="drift_exp_min_retrain_size",
            help="Tamano minimo de W1 para permitir reentrenamiento despues de drift. Se separa del `min window` del detector.",
        )

    arf_variants = st.multiselect(
        "ARF variants",
        ARF_VARIANT_NAMES,
        default=list(ARF_DEFAULT_VARIANTS),
        key="drift_exp_arf_variants",
        help="Solo se usan cuando la estrategia `adaptive_arf` esta seleccionada.",
    )
    if ADAPTIVE_ARF_STRATEGY in selected_strategies:
        variant_desc = [
            f"`{name}`: {ARF_VARIANT_CONFIGS[name]['description']}"
            for name in arf_variants
            if name in ARF_VARIANT_CONFIGS
        ]
        if variant_desc:
            st.caption(" | ".join(variant_desc))

    kswin_variants = st.multiselect(
        "KSWIN variants",
        KSWIN_VARIANT_NAMES,
        default=list(KSWIN_DEFAULT_VARIANTS),
        key="drift_exp_kswin_variants",
        help="Solo se usan cuando la estrategia `adaptive_kswin` esta seleccionada.",
    )
    kswin_col1, kswin_col2, kswin_col3, kswin_col4 = st.columns(4)
    with kswin_col1:
        kswin_top_k_features = st.number_input(
            "KSWIN top-k features",
            min_value=1,
            max_value=max(1, len(feature_cols)),
            value=min(3, max(1, len(feature_cols))),
            step=1,
            key="drift_exp_kswin_top_k_features",
        )
    with kswin_col2:
        kswin_vote_threshold = st.number_input(
            "KSWIN vote threshold",
            min_value=1,
            max_value=max(1, min(10, len(feature_cols))),
            value=min(3, max(1, len(feature_cols))),
            step=1,
            key="drift_exp_kswin_vote_threshold",
        )
    with kswin_col3:
        kswin_retrain_days = st.number_input(
            "KSWIN retrain horizon (days)",
            min_value=1,
            max_value=365,
            value=30,
            step=1,
            key="drift_exp_kswin_retrain_days",
        )
    with kswin_col4:
        kswin_min_retrain_rows = st.number_input(
            "KSWIN min retrain rows",
            min_value=10,
            max_value=200000,
            value=50,
            step=10,
            key="drift_exp_kswin_min_retrain_rows",
        )
    if ADAPTIVE_KSWIN_STRATEGY in selected_strategies:
        variant_desc = [
            f"`{name}`: {KSWIN_VARIANT_CONFIGS[name]['description']}"
            for name in kswin_variants
            if name in KSWIN_VARIANT_CONFIGS
        ]
        if variant_desc:
            st.caption(" | ".join(variant_desc))

    balance_modes = st.multiselect(
        "Balance modes",
        list(DEFAULT_BATCH_BALANCE_MODES),
        default=list(DEFAULT_BATCH_BALANCE_MODES),
        key="drift_exp_balance_modes",
        help="Se comparan solo en estrategias batch y adaptativas batch; `adaptive_arf` queda como `not_applicable`.",
    )

    continue_on_block_error = st.checkbox(
        "Skip failed blocks and continue",
        value=False,
        key="drift_exp_continue_on_block_error",
        help=(
            "Si un bloque batch/adaptativo falla, se marca como `skipped_failed` en el manifest y la corrida sigue "
            "con el resto. El tuning global sigue siendo fail-fast."
        ),
    )

    seed_text = st.text_input(
        "Sequential repetition seeds",
        value=", ".join(str(seed) for seed in DEFAULT_REPETITION_SEEDS),
        key="drift_exp_seed_text",
        help="The article reports mean metrics across repeated runs. Seeds are executed sequentially and logged.",
    )

    preflight_payload: Optional[Dict[str, Any]] = None
    preview_seeds: Optional[List[int]] = None
    preflight_from_loaded_checkpoint = False
    if bool((loaded_checkpoint_preview or {}).get("checkpoint_available")):
        preflight_payload = dict(
            ((loaded_checkpoint_preview or {}).get("run_manifest") or {}).get("preflight")
            or ((loaded_checkpoint_preview or {}).get("manifest") or {}).get("preflight")
            or {}
        )
        preflight_from_loaded_checkpoint = bool(preflight_payload)
    if not preflight_from_loaded_checkpoint:
        try:
            preview_seeds = parse_repetition_seeds(seed_text)
            preflight_payload = _estimate_recalibration_workload(
                df=clean_df,
                feature_cols=feature_cols,
                target_col=target_col,
                time_col=time_col,
                model_names=selected_models,
                strategies=selected_strategies,
                arf_variants=arf_variants,
                kswin_variants=kswin_variants,
                balance_modes=balance_modes,
                repetition_seeds=preview_seeds,
                base_year=base_year,
                validation_size=float(validation_size),
                grid_limit=int(grid_limit),
                folds=int(folds),
                resource_mode=resource_mode,
                random_state=int(random_state),
                fast_mode=bool(fast_mode),
                custom_grids=custom_grids,
            )
        except Exception:
            preflight_payload = None

    if isinstance(preflight_payload, dict):
        st.caption(
            ("Preflight (checkpoint cargado): " if preflight_from_loaded_checkpoint else "Preflight: ")
            + (
            f"{int(preflight_payload.get('n_tuning_tasks', 0))} tunings por ventana conocidos | "
            f"{int(preflight_payload.get('n_blocks', 0))} bloques | "
            f"{int(preflight_payload.get('estimated_batch_fits', 0))} fits batch estimados"
            )
        )
        if bool(preflight_payload.get("high_risk_memory")):
            st.warning(
                f"Configuracion de alto riesgo de memoria detectada para {_experiment_resource_mode_label(resource_mode)}: "
                "Random Forest + SMOTE + folds/trials altos. "
                "El pipeline usara checkpoints y resume automatico, pero conviene bajar trials/folds si no necesitas la corrida completa."
            )

    checkpoint_preview: Optional[Dict[str, Any]] = None
    with st.expander("Checkpoint transfer", expanded=False):
        uploaded_checkpoint_zip = st.file_uploader(
            "Import training checkpoint ZIP",
            type=["zip"],
            key="drift_training_checkpoint_import_zip",
            help="Importa un run dir persistido para luego cargarlo, reanudarlo o recalcular entrenamientos.",
        )
        overwrite_imported_checkpoint = st.checkbox(
            "Overwrite imported checkpoint if the same run_id already exists",
            key="drift_training_checkpoint_overwrite",
            value=False,
        )
        if uploaded_checkpoint_zip is not None and st.button("Import checkpoint ZIP", key="drift_training_checkpoint_import_button"):
            try:
                import_result = _import_recalibration_checkpoint_archive(
                    uploaded_checkpoint_zip.getvalue(),
                    overwrite=bool(overwrite_imported_checkpoint),
                )
                st.success(
                    f"Checkpoint importado: run `{import_result['run_id']}` | estado {import_result['status']}"
                )
                st.cache_data.clear()
                st.rerun()
            except Exception as exc:
                st.error(f"No se pudo importar el checkpoint: {exc}")

    if isinstance(preview_seeds, list) and not bool((loaded_checkpoint_preview or {}).get("checkpoint_available")):
        checkpoint_preview = _preview_recalibration_checkpoint(
            clean_df,
            feature_cols=feature_cols,
            target_col=target_col,
            time_col=time_col,
            model_names=selected_models,
            strategies=selected_strategies,
            validation_size=float(validation_size),
            folds=int(folds),
            fast_mode=bool(fast_mode),
            resource_mode=resource_mode,
            grid_limit=int(grid_limit),
            adwin_delta=float(adwin_delta),
            min_window=int(min_window),
            min_retrain_size=int(min_retrain_size),
            arf_variants=arf_variants,
            kswin_variants=kswin_variants,
            kswin_top_k_features=int(kswin_top_k_features),
            kswin_vote_threshold=int(kswin_vote_threshold),
            kswin_retrain_days=int(kswin_retrain_days),
            kswin_min_retrain_rows=int(kswin_min_retrain_rows),
            continue_on_block_error=bool(continue_on_block_error),
            repetition_seeds=preview_seeds,
            balance_modes=balance_modes,
            feature_selection_context=_current_feature_selection_context(),
        )

    execution_mode: Optional[str] = None
    execution_checkpoint_run_id_override: Optional[str] = None
    if isinstance(checkpoint_preview, dict) and bool(checkpoint_preview.get("checkpoint_available")):
        checkpoint_status = str(checkpoint_preview.get("status", "running"))
        status_label = "completado" if checkpoint_status == "completed" else "recuperable"
        can_recompute_trainings = bool(checkpoint_preview.get("can_recompute_trainings"))
        st.markdown("**Compatible checkpoint**")
        st.caption(
            f"Run `{checkpoint_preview.get('run_id', '')}` | estado {checkpoint_status} ({status_label}) | "
            f"actualizado {checkpoint_preview.get('updated_at') or 'desconocido'} | "
            f"tunings {int(checkpoint_preview.get('completed_tuning_tasks', 0))}/{int(checkpoint_preview.get('total_tuning_tasks', 0))} | "
            f"bloques {int(checkpoint_preview.get('completed_blocks', 0))}/{int(checkpoint_preview.get('total_blocks', 0))} | "
            f"bloques omitidos {int(checkpoint_preview.get('skipped_failed_blocks', 0))} | "
            f"SMOTE cache {int(checkpoint_preview.get('smote_artifacts', 0))}"
        )
        st.caption(f"Manifest: {checkpoint_preview.get('manifest_path', '')}")
        try:
            checkpoint_zip = _build_recalibration_checkpoint_archive(Path(str(checkpoint_preview.get("run_dir", ""))))
            st.download_button(
                label="Export checkpoint ZIP",
                data=checkpoint_zip,
                file_name=f"drift_checkpoint_{checkpoint_preview.get('run_id', 'run')}.zip",
                mime="application/zip",
                key="drift_training_checkpoint_export_button",
            )
        except Exception as exc:
            st.caption(f"No se pudo preparar export del checkpoint: {exc}")
        if bool(checkpoint_preview.get("can_load_completed")):
            if can_recompute_trainings:
                st.info(
                    "Existe una corrida compatible ya completada. Puedes cargar ese checkpoint, "
                    "recalcular entrenamientos reutilizando tuning/cache persistidos o iniciar una corrida nueva."
                )
            else:
                st.info("Existe una corrida compatible ya completada. Puedes cargar ese checkpoint o iniciar una corrida nueva.")
        else:
            if can_recompute_trainings:
                st.info(
                    "Existe un checkpoint compatible recuperable. Puedes reanudarlo, reiniciar los bloques "
                    "reutilizando tuning/cache o forzar una corrida limpia."
                )
            else:
                st.info("Existe un checkpoint compatible recuperable. Puedes reanudarlo desde aqui o forzar una corrida limpia.")
        action_cols = st.columns(3 if can_recompute_trainings else 2)
        action_col_1 = action_cols[0]
        action_col_2 = action_cols[1]
        with action_col_1:
            resume_label = "Load completed checkpoint" if bool(checkpoint_preview.get("can_load_completed")) else "Resume compatible checkpoint"
            if st.button(resume_label, key="drift_resume_experiments"):
                execution_mode = "resume"
                execution_checkpoint_run_id_override = str(checkpoint_preview.get("run_id") or "")
        if can_recompute_trainings:
            with action_col_2:
                if st.button("Recompute trainings from checkpoint", key="drift_recompute_from_checkpoint"):
                    execution_mode = "retrain"
                    execution_checkpoint_run_id_override = str(checkpoint_preview.get("run_id") or "")
            action_col_fresh = action_cols[2]
        else:
            action_col_fresh = action_col_2
        with action_col_fresh:
            if st.button("Start fresh ignoring checkpoint", key="drift_run_experiments_fresh"):
                execution_mode = "fresh"
    elif st.button("Run configured experiment set", key="drift_run_experiments"):
        execution_mode = "resume"

    if loaded_checkpoint_run_id:
        compatibility_run_id = str((checkpoint_preview or {}).get("run_id", "")) if isinstance(checkpoint_preview, dict) else ""
        if (
            bool(loaded_checkpoint_preview.get("checkpoint_available"))
            and str(loaded_checkpoint_preview.get("run_id", "")) != compatibility_run_id
        ):
            loaded_checkpoint_status = str(loaded_checkpoint_preview.get("status", "running"))
            loaded_status_label = "completado" if loaded_checkpoint_status == "completed" else "recuperable"
            loaded_can_recompute = bool(loaded_checkpoint_preview.get("can_recompute_trainings"))
            st.markdown("**Loaded Checkpoint Actions**")
            st.caption(
                f"Run `{loaded_checkpoint_preview.get('run_id', '')}` | estado {loaded_checkpoint_status} ({loaded_status_label}) | "
                f"actualizado {loaded_checkpoint_preview.get('updated_at') or 'desconocido'} | "
                f"tunings {int(loaded_checkpoint_preview.get('completed_tuning_tasks', 0))}/{int(loaded_checkpoint_preview.get('total_tuning_tasks', 0))} | "
                f"bloques {int(loaded_checkpoint_preview.get('completed_blocks', 0))}/{int(loaded_checkpoint_preview.get('total_blocks', 0))} | "
                f"SMOTE cache {int(loaded_checkpoint_preview.get('smote_artifacts', 0))}"
            )
            st.info(
                "Estas acciones usan directamente el checkpoint cargado desde el selector, "
                "aunque el preview compatible del formulario actual no coincida exactamente."
            )
            loaded_action_cols = st.columns(3 if loaded_can_recompute else 2)
            with loaded_action_cols[0]:
                loaded_resume_label = (
                    "Load loaded checkpoint"
                    if bool(loaded_checkpoint_preview.get("can_load_completed"))
                    else "Resume loaded checkpoint"
                )
                if st.button(loaded_resume_label, key="drift_resume_loaded_checkpoint"):
                    execution_mode = "resume"
                    execution_checkpoint_run_id_override = str(loaded_checkpoint_preview.get("run_id") or "")
            if loaded_can_recompute:
                with loaded_action_cols[1]:
                    if st.button("Recompute trainings from loaded checkpoint", key="drift_recompute_loaded_checkpoint"):
                        execution_mode = "retrain"
                        execution_checkpoint_run_id_override = str(loaded_checkpoint_preview.get("run_id") or "")
                loaded_fresh_col = loaded_action_cols[2]
            else:
                loaded_fresh_col = loaded_action_cols[1]
            with loaded_fresh_col:
                if st.button("Start fresh with loaded configuration", key="drift_run_loaded_checkpoint_fresh"):
                    execution_mode = "fresh"

    if execution_mode is not None:
        requires_batch_models = any(
            strategy in YEARLY_STRATEGIES + [ADAPTIVE_ADWIN_STRATEGY, ADAPTIVE_KSWIN_STRATEGY]
            for strategy in selected_strategies
        )
        if requires_batch_models and not selected_models:
            st.error("Seleccione al menos un modelo para las estrategias batch, `adaptive_adwin` o `adaptive_kswin`.")
            return
        if ADAPTIVE_ARF_STRATEGY in selected_strategies and not arf_variants:
            st.error("Seleccione al menos una variante ARF para ejecutar `adaptive_arf`.")
            return
        if ADAPTIVE_KSWIN_STRATEGY in selected_strategies and not kswin_variants:
            st.error("Seleccione al menos una variante KSWIN para ejecutar `adaptive_kswin`.")
            return
        try:
            repetition_seeds = parse_repetition_seeds(seed_text)
        except ValueError as exc:
            st.error(f"Invalid seed list: {exc}")
            return
        st.session_state["drift_results"] = None
        st.session_state["drift_execution_log"] = None
        st.session_state["drift_run_manifest"] = None
        st.session_state["drift_optuna_json_path"] = None
        st.session_state["drift_checkpoint_manifest_path"] = None
        progress = _ExperimentProgress()
        execution_config: Dict[str, Any] = {
            "feature_cols": list(feature_cols),
            "target_col": target_col,
            "time_col": time_col,
            "model_names": list(selected_models),
            "strategies": list(selected_strategies),
            "validation_size": float(validation_size),
            "folds": int(folds),
            "base_year": base_year,
            "random_state": int(random_state),
            "fast_mode": bool(fast_mode),
            "resource_mode": resource_mode,
            "grid_limit": int(grid_limit),
            "adwin_delta": float(adwin_delta),
            "min_window": int(min_window),
            "min_retrain_size": int(min_retrain_size),
            "arf_variants": list(arf_variants),
            "kswin_variants": list(kswin_variants),
            "kswin_top_k_features": int(kswin_top_k_features),
            "kswin_vote_threshold": int(kswin_vote_threshold),
            "kswin_retrain_days": int(kswin_retrain_days),
            "kswin_min_retrain_rows": int(kswin_min_retrain_rows),
            "repetition_seeds": tuple(repetition_seeds),
            "balance_modes": list(balance_modes),
            "feature_selection_context": _current_feature_selection_context(),
            "custom_grids": custom_grids,
            "continue_on_block_error": bool(continue_on_block_error),
        }
        loaded_checkpoint_feature_selection_context = dict(
            st.session_state.get("drift_exp_loaded_checkpoint_feature_selection_context") or {}
        )
        loaded_checkpoint_run_manifest = dict(
            (loaded_checkpoint_preview or {}).get("run_manifest")
            or {}
        )
        execution_df = clean_df
        try:
            if (
                execution_checkpoint_run_id_override
                and loaded_checkpoint_run_id
                and str(execution_checkpoint_run_id_override) == str(loaded_checkpoint_run_id)
            ):
                checkpoint_feature_bundle = _load_checkpoint_feature_bundle(
                    run_manifest=loaded_checkpoint_run_manifest,
                    feature_selection_context=loaded_checkpoint_feature_selection_context,
                )
                if checkpoint_feature_bundle is None:
                    raise RuntimeError(
                        "Could not resolve the feature DuckDB referenced by the loaded checkpoint. "
                        "Load or import that feature export before resuming/recomputing."
                    )
                execution_df = checkpoint_feature_bundle["clean_df"]
                loaded_checkpoint_feature_selection_context = dict(
                    checkpoint_feature_bundle.get("feature_selection_context") or loaded_checkpoint_feature_selection_context
                )
                st.session_state["drift_feature_export_path"] = str(checkpoint_feature_bundle["resolved_path"])
                execution_config = _execution_config_from_checkpoint_run_manifest(
                    loaded_checkpoint_run_manifest,
                    available_columns=list(execution_df.columns),
                    current_config=execution_config,
                    feature_selection_context=loaded_checkpoint_feature_selection_context,
                )
            results = run_recalibration_experiments(
                execution_df,
                feature_cols=execution_config["feature_cols"],
                target_col=execution_config["target_col"],
                time_col=execution_config["time_col"],
                model_names=execution_config["model_names"],
                strategies=execution_config["strategies"],
                validation_size=execution_config["validation_size"],
                folds=execution_config["folds"],
                base_year=execution_config["base_year"],
                random_state=execution_config["random_state"],
                fast_mode=execution_config["fast_mode"],
                resource_mode=execution_config["resource_mode"],
                grid_limit=execution_config["grid_limit"],
                adwin_delta=execution_config["adwin_delta"],
                min_window=execution_config["min_window"],
                min_retrain_size=execution_config["min_retrain_size"],
                arf_variants=execution_config["arf_variants"],
                kswin_variants=execution_config["kswin_variants"],
                kswin_top_k_features=execution_config["kswin_top_k_features"],
                kswin_vote_threshold=execution_config["kswin_vote_threshold"],
                kswin_retrain_days=execution_config["kswin_retrain_days"],
                kswin_min_retrain_rows=execution_config["kswin_min_retrain_rows"],
                repetition_seeds=execution_config["repetition_seeds"],
                balance_modes=execution_config["balance_modes"],
                feature_selection_context=execution_config["feature_selection_context"],
                custom_grids=execution_config["custom_grids"],
                checkpoint_run_id_override=execution_checkpoint_run_id_override,
                auto_resume=bool(execution_mode != "fresh"),
                recompute_blocks_from_checkpoint=bool(execution_mode == "retrain"),
                continue_on_block_error=execution_config["continue_on_block_error"],
                progress_callback=progress.update,
            )
        except Exception as exc:
            progress.update(
                {
                    "completed_units": 0,
                    "total_units": 1,
                    "label": "Error ejecutando experimentos.",
                    "detail": str(exc),
                }
            )
            st.error(f"No se pudieron ejecutar los experimentos: {exc}")
            return
        st.session_state["drift_results"] = results
        st.session_state["drift_execution_log"] = results.get("execution_log")
        st.session_state["drift_run_manifest"] = results.get("run_manifest")
        st.session_state["drift_optuna_json_path"] = results.get("optuna_json_path")
        st.session_state["drift_checkpoint_manifest_path"] = results.get("checkpoint_manifest_path")

    results = st.session_state.get("drift_results")
    if not isinstance(results, dict):
        return

    summary = results.get("summary")
    yearly = results.get("yearly_results")
    adaptive = results.get("adaptive_results")
    avg_roc = results.get("average_roc")
    roc_payload = results.get("roc_payload")
    appendix = results.get("appendix_tables")
    appendix_mean = results.get("appendix_tables_mean")
    execution_log = results.get("execution_log")
    run_manifest = results.get("run_manifest")
    optuna_json_path = results.get("optuna_json_path")
    checkpoint_manifest = results.get("checkpoint_manifest")
    checkpoint_manifest_path = results.get("checkpoint_manifest_path")
    auto_resumed = bool(results.get("auto_resumed"))
    preflight = results.get("preflight")
    studies = list(results.get("studies") or [])

    if isinstance(run_manifest, dict):
        st.markdown("**Run manifest**")
        st.json(run_manifest, expanded=False)
    if isinstance(checkpoint_manifest, dict):
        progress = checkpoint_manifest.get("progress") or {}
        st.markdown("**Checkpoint state**")
        st.caption(
            f"{'Resume automatico' if auto_resumed else 'Corrida nueva'} | "
            f"tunings {int(progress.get('completed_tuning_tasks', 0))}/{int(progress.get('total_tuning_tasks', 0))} | "
            f"bloques {int(progress.get('completed_blocks', 0))}/{int(progress.get('total_blocks', 0))} | "
            f"SMOTE cache {len(checkpoint_manifest.get('smote_index') or {})} artefactos"
        )
        if checkpoint_manifest_path:
            st.caption(f"Checkpoint manifest: {checkpoint_manifest_path}")
    if optuna_json_path:
        st.success(f"Optuna JSON consolidado: {optuna_json_path}")
    if isinstance(preflight, dict) and bool(preflight.get("high_risk_memory")):
        st.warning(
            "La configuracion ejecutada fue marcada como de alto riesgo de memoria "
            f"en el preflight ({str(preflight.get('resource_mode_label') or _experiment_resource_mode_label(run_manifest.get('resource_mode') if isinstance(run_manifest, dict) else None))})."
        )

    yearly_evolution = _build_experiment_evolution_records(
        yearly if isinstance(yearly, pd.DataFrame) else pd.DataFrame(),
        source="yearly",
        metric_candidates=YEARLY_EVOLUTION_METRICS,
    )
    adaptive_evolution = _build_experiment_evolution_records(
        adaptive if isinstance(adaptive, pd.DataFrame) else pd.DataFrame(),
        source="adaptive",
        metric_candidates=ADAPTIVE_EVOLUTION_METRICS,
    )
    tuning_trials = _build_tuning_trials_frame(studies)
    memory_trace = _build_execution_memory_trace(
        execution_log if isinstance(execution_log, pd.DataFrame) else pd.DataFrame()
    )

    record_col_1, record_col_2, record_col_3, record_col_4, record_col_5 = st.columns(5)
    record_col_1.metric("Yearly rows", int(len(yearly)) if isinstance(yearly, pd.DataFrame) else 0)
    record_col_2.metric("Adaptive rows", int(len(adaptive)) if isinstance(adaptive, pd.DataFrame) else 0)
    record_col_3.metric("Execution log rows", int(len(execution_log)) if isinstance(execution_log, pd.DataFrame) else 0)
    record_col_4.metric("ROC payloads", int(len(roc_payload)) if isinstance(roc_payload, list) else 0)
    record_col_5.metric("Tuning trials", int(len(tuning_trials)))

    summary_tab, evolution_tab, records_tab = st.tabs(["Summary", "Online evolution", "Records"])

    with summary_tab:
        st.markdown("**Strategy summary**")
        if isinstance(summary, pd.DataFrame) and not summary.empty:
            st.dataframe(_streamlit_arrow_safe_df(summary), width="stretch")

        st.markdown("**Appendix tables (A.6-A.9) replica**")
        if isinstance(appendix, dict):
            for key in ["A.6", "A.7", "A.8", "A.9"]:
                df_tab = appendix.get(key)
                st.caption(f"Table {key}")
                if isinstance(df_tab, pd.DataFrame) and not df_tab.empty:
                    st.dataframe(_streamlit_arrow_safe_df(df_tab), width="stretch")
                else:
                    st.info(f"Table {key} has no rows for current run.")

        if isinstance(appendix_mean, dict):
            st.markdown("**Appendix mean tables across sequential seeds**")
            for key in ["A.6", "A.7", "A.8", "A.9"]:
                df_tab = appendix_mean.get(key)
                st.caption(f"Mean Table {key}")
                if isinstance(df_tab, pd.DataFrame) and not df_tab.empty:
                    st.dataframe(_streamlit_arrow_safe_df(df_tab), width="stretch")
                else:
                    st.info(f"Mean Table {key} has no rows for current run.")

        st.markdown("**Figure 7 replica: average ROC curves**")
        if isinstance(avg_roc, pd.DataFrame) and not avg_roc.empty:
            pivot = avg_roc.pivot_table(index="fpr", columns="label", values="tpr", aggfunc="mean")
            st.line_chart(pivot, width="stretch")
        else:
            st.info("No ROC curves available.")

    with evolution_tab:
        st.markdown("**Yearly metric evolution**")
        if not yearly_evolution.empty:
            yearly_metric = st.selectbox(
                "Yearly metric",
                options=sorted(yearly_evolution["metric"].astype(str).unique().tolist()),
                index=0,
                key="drift_yearly_metric",
            )
            yearly_models = sorted(yearly_evolution["model"].astype(str).unique().tolist())
            yearly_strategies = sorted(yearly_evolution["strategy"].astype(str).unique().tolist())
            yearly_balances = sorted(yearly_evolution["balance_mode"].astype(str).unique().tolist())
            yearly_filter_col_1, yearly_filter_col_2, yearly_filter_col_3 = st.columns(3)
            selected_yearly_strategies = yearly_filter_col_1.multiselect(
                "Yearly strategies",
                yearly_strategies,
                default=yearly_strategies,
                key="drift_yearly_strategies",
            )
            selected_yearly_models = yearly_filter_col_2.multiselect(
                "Yearly models",
                yearly_models,
                default=yearly_models,
                key="drift_yearly_models",
            )
            selected_yearly_balances = yearly_filter_col_3.multiselect(
                "Yearly balances",
                yearly_balances,
                default=yearly_balances,
                key="drift_yearly_balances",
            )
            avg_yearly_seeds = st.checkbox("Average yearly seeds", value=True, key="drift_yearly_avg_seeds")
            yearly_plot = yearly_evolution.loc[
                yearly_evolution["metric"].eq(yearly_metric)
                & yearly_evolution["strategy"].astype(str).isin(selected_yearly_strategies)
                & yearly_evolution["model"].astype(str).isin(selected_yearly_models)
                & yearly_evolution["balance_mode"].astype(str).isin(selected_yearly_balances)
            ].copy()
            if avg_yearly_seeds:
                yearly_plot["series_label"] = (
                    yearly_plot["strategy"].astype(str)
                    + " | "
                    + yearly_plot["model"].astype(str)
                    + " | "
                    + yearly_plot["balance_mode"].astype(str)
                )
                yearly_plot = yearly_plot.groupby(
                    ["prediction_year", "series_label"],
                    dropna=False,
                    as_index=False,
                )["value"].mean()
            else:
                yearly_plot["series_label"] = (
                    yearly_plot["strategy"].astype(str)
                    + " | "
                    + yearly_plot["model"].astype(str)
                    + " | "
                    + yearly_plot["balance_mode"].astype(str)
                    + " | seed "
                    + yearly_plot["run_seed"].fillna(-1).astype(int).astype(str)
                )
            if not yearly_plot.empty:
                yearly_pivot = yearly_plot.pivot_table(
                    index="prediction_year",
                    columns="series_label",
                    values="value",
                    aggfunc="mean",
                ).sort_index()
                st.line_chart(yearly_pivot, width="stretch")
                st.dataframe(_streamlit_arrow_safe_df(yearly_plot), width="stretch")
            else:
                st.info("No yearly rows match the selected filters.")
        else:
            st.info("No yearly evolution rows available.")

        st.markdown("**Adaptive metric evolution**")
        if not adaptive_evolution.empty:
            adaptive_metric = st.selectbox(
                "Adaptive metric",
                options=sorted(adaptive_evolution["metric"].astype(str).unique().tolist()),
                index=0,
                key="drift_adaptive_metric",
            )
            adaptive_models = sorted(adaptive_evolution["model"].astype(str).unique().tolist())
            adaptive_strategies = sorted(adaptive_evolution["strategy"].astype(str).unique().tolist())
            adaptive_balances = sorted(adaptive_evolution["balance_mode"].astype(str).unique().tolist())
            adaptive_filter_col_1, adaptive_filter_col_2, adaptive_filter_col_3 = st.columns(3)
            selected_adaptive_strategies = adaptive_filter_col_1.multiselect(
                "Adaptive strategies",
                adaptive_strategies,
                default=adaptive_strategies,
                key="drift_adaptive_strategies",
            )
            selected_adaptive_models = adaptive_filter_col_2.multiselect(
                "Adaptive models",
                adaptive_models,
                default=adaptive_models,
                key="drift_adaptive_models",
            )
            selected_adaptive_balances = adaptive_filter_col_3.multiselect(
                "Adaptive balances",
                adaptive_balances,
                default=adaptive_balances,
                key="drift_adaptive_balances",
            )
            avg_adaptive_seeds = st.checkbox("Average adaptive seeds", value=True, key="drift_adaptive_avg_seeds")
            adaptive_plot = adaptive_evolution.loc[
                adaptive_evolution["metric"].eq(adaptive_metric)
                & adaptive_evolution["strategy"].astype(str).isin(selected_adaptive_strategies)
                & adaptive_evolution["model"].astype(str).isin(selected_adaptive_models)
                & adaptive_evolution["balance_mode"].astype(str).isin(selected_adaptive_balances)
            ].copy()
            adaptive_index_col = "event_time" if adaptive_plot["event_time"].notna().all() else "event_order"
            if avg_adaptive_seeds:
                adaptive_plot["series_label"] = (
                    adaptive_plot["strategy"].astype(str)
                    + " | "
                    + adaptive_plot["model"].astype(str)
                    + " | "
                    + adaptive_plot["balance_mode"].astype(str)
                )
                adaptive_plot = adaptive_plot.groupby(
                    [adaptive_index_col, "event_label", "series_label"],
                    dropna=False,
                    as_index=False,
                )["value"].mean()
            else:
                adaptive_plot["series_label"] = (
                    adaptive_plot["strategy"].astype(str)
                    + " | "
                    + adaptive_plot["model"].astype(str)
                    + " | "
                    + adaptive_plot["balance_mode"].astype(str)
                    + " | seed "
                    + adaptive_plot["run_seed"].fillna(-1).astype(int).astype(str)
                )
            if not adaptive_plot.empty:
                adaptive_pivot = adaptive_plot.pivot_table(
                    index=adaptive_index_col,
                    columns="series_label",
                    values="value",
                    aggfunc="mean",
                ).sort_index()
                st.line_chart(adaptive_pivot, width="stretch")
                st.dataframe(_streamlit_arrow_safe_df(adaptive_plot), width="stretch")
            else:
                st.info("No adaptive rows match the selected filters.")
        else:
            st.info("No adaptive evolution rows available.")

        st.markdown("**Tuning evolution (Optuna trials)**")
        if not tuning_trials.empty:
            tuning_filter_col_1, tuning_filter_col_2 = st.columns(2)
            selected_tuning_models = tuning_filter_col_1.multiselect(
                "Tuning models",
                sorted(tuning_trials["model"].astype(str).unique().tolist()),
                default=sorted(tuning_trials["model"].astype(str).unique().tolist()),
                key="drift_tuning_models",
            )
            selected_tuning_balances = tuning_filter_col_2.multiselect(
                "Tuning balances",
                sorted(tuning_trials["balance_mode"].astype(str).unique().tolist()),
                default=sorted(tuning_trials["balance_mode"].astype(str).unique().tolist()),
                key="drift_tuning_balances",
            )
            tuning_plot = tuning_trials.loc[
                tuning_trials["model"].astype(str).isin(selected_tuning_models)
                & tuning_trials["balance_mode"].astype(str).isin(selected_tuning_balances)
            ].copy()
            tuning_plot["series_label"] = (
                tuning_plot["model"].astype(str)
                + " | "
                + tuning_plot["balance_mode"].astype(str)
                + " | "
                + tuning_plot["stage"].astype(str)
            )
            tuning_pivot = tuning_plot.pivot_table(
                index="trial_number",
                columns="series_label",
                values="cv_auc",
                aggfunc="max",
            ).sort_index()
            st.line_chart(tuning_pivot, width="stretch")
            st.dataframe(_streamlit_arrow_safe_df(tuning_plot), width="stretch")
        else:
            st.info("No tuning trials available.")

        st.markdown("**Execution resource trace**")
        if not memory_trace.empty:
            trace_metric = st.selectbox(
                "Trace metric",
                options=sorted(memory_trace["metric"].astype(str).unique().tolist()),
                index=0,
                key="drift_trace_metric",
            )
            trace_plot = memory_trace.loc[memory_trace["metric"].eq(trace_metric)].copy()
            trace_plot["series_label"] = trace_plot["phase"].astype(str) + " | " + trace_plot["status"].astype(str)
            trace_pivot = trace_plot.pivot_table(
                index="order",
                columns="series_label",
                values="value",
                aggfunc="last",
            ).sort_index()
            st.line_chart(trace_pivot, width="stretch")
            st.dataframe(_streamlit_arrow_safe_df(trace_plot), width="stretch")
        else:
            st.info("No execution resource traces available.")

    with records_tab:
        if isinstance(yearly, pd.DataFrame) and not yearly.empty:
            st.markdown("**Yearly strategy results**")
            st.dataframe(_streamlit_arrow_safe_df(yearly), width="stretch")
        if isinstance(adaptive, pd.DataFrame) and not adaptive.empty:
            st.markdown("**Adaptive strategy segments (ADWIN + ARF + KSWIN)**")
            st.caption(
                "ADWIN reporta segmentos delimitados por drift detectado. "
                "ARF reporta segmentos anuales online con su conteo interno de drifts y warnings. "
                "KSWIN reporta segmentos cerrados por drift en covariables, con voto de features y ventana de reentrenamiento fija."
            )
            st.dataframe(_streamlit_arrow_safe_df(adaptive), width="stretch")
        if isinstance(execution_log, pd.DataFrame) and not execution_log.empty:
            st.markdown("**Execution log**")
            st.dataframe(_streamlit_arrow_safe_df(execution_log), width="stretch")
        if not tuning_trials.empty:
            st.markdown("**Tuning trials**")
            st.dataframe(_streamlit_arrow_safe_df(tuning_trials), width="stretch")


def _render_coverage_tab() -> None:
    st.subheader("Coverage 100% Check")
    matrix = build_article_coverage_matrix()
    coverage = article_coverage_percentage(matrix)

    c1, c2 = st.columns(2)
    c1.metric("Coverage", f"{coverage:.1f}%")
    c2.metric("Missing items", int((~matrix["implemented"]).sum()))

    if coverage >= 100.0:
        st.success("Coverage verification passed: 100% of article items mapped.")
    else:
        st.warning("Coverage below 100%. Review missing items.")

    st.dataframe(matrix, width="stretch")


def main(set_page_config: bool = False, show_exit_button: bool = False) -> None:
    if set_page_config:
        st.set_page_config(page_title="Drift detection", layout="wide")

    _init_state()

    st.title("Drift detection")
    st.caption(
        "Replication workspace for the crash prediction recalibration paper "
        "(static, period-aligned, cumulative, adaptive ADWIN) "
        "with extended online ARF and KSWIN comparisons."
    )

    tab_blueprint, tab_events, tab_feature_eng, tab_selection, tab_experiments, tab_coverage = st.tabs(
        [
            "Blueprint",
            "Eventos",
            "Feature engineering",
            "Feature Selection",
            "Experiments",
            "Coverage",
        ]
    )

    with tab_blueprint:
        _render_blueprint_tab()

    with tab_events:
        _render_events_tab()

    with tab_feature_eng:
        _render_feature_engineering_tab()

    with tab_selection:
        _render_feature_selection_tab()

    with tab_experiments:
        _render_experiments_tab()

    with tab_coverage:
        _render_coverage_tab()

    if show_exit_button and st.sidebar.button("Cerrar app"):
        st.sidebar.write("Cerrando...")


if __name__ == "__main__":
    main(set_page_config=True)
