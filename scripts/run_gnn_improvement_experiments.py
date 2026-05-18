#!/usr/bin/env python3
"""Run fresh GNN training-process experiments for rare accident prediction.

The current protocol treats test performance as the decision point: every
trained checkpoint is evaluated on ``test_mask`` with Platt calibration and a
validation-selected FAR threshold. Pilot mode remains available only as a cheap
debug pass; full mode is the scientific comparison mode.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import gnn_main  # noqa: E402


DEFAULT_GRAPH = ROOT / "Resultados" / "highway_graph_stream_build_15052026_1521.pt"
DEFAULT_HPARAMS = (
    ROOT
    / "Resultados"
    / "optuna_hyperparams_network_20260515_172309_533b8548afefd789_Base.csv"
)
DEFAULT_OUT_DIR = ROOT / "Resultados" / "experimentos" / "gnn_accident_training_fresh"
DEFAULT_BASELINE_MODEL = (
    ROOT
    / "Resultados"
    / "gat_model_BEST_GNN_gat_edge_mlp_gru_20260515_172330_533b8548.pt"
)
BASELINE_METRICS = {
    "auc": 0.7551,
    "auprc": 0.0060,
    "mcc": 0.0309,
    "far": 0.1376,
    "brier": 0.0960,
}


@dataclass(frozen=True)
class Experiment:
    name: str
    hypothesis: str | None = None
    rationale: str | None = None
    expected_metric_movement: str | None = None
    failure_mode: str | None = None
    success_criterion: str | None = None
    cost: str | None = None
    train_sampler_mode: str = "neighbor"
    positive_fraction: float | None = None
    hard_window: int | None = None
    hard_per_positive: int | None = None
    disable_hard_undersampling: bool = True
    loss_type: str | None = None
    focal_alpha: float | None = None
    focal_gamma: float | None = None
    loss_weight_mode: str | None = None
    checkpoint_metric: str | None = None
    eval_neighbors_mode: str | None = None
    eval_num_neighbors: Any | None = None
    ranking_loss_mode: str | None = None
    ranking_loss_weight: float | None = None
    ranking_loss_margin: float | None = None
    ranking_loss_max_pairs: int | None = None
    objective_metric: str | None = None
    threshold_beta: float | None = None
    num_neighbors: Any | None = None
    lr: float | None = None
    weight_decay: float | None = None
    batch_size: int | None = None
    dropout: float | None = None
    grad_clip: float | None = None
    accumulation_steps: int | None = None
    lr_scheduler: str | None = None
    max_epochs: int | None = None
    early_stop_patience: int | None = None
    seed_offset: int = 0
    eval_baseline_only: bool = False
    initialize_from_baseline: bool = False


def _base_hypothesis(
    *,
    statement: str,
    rationale: str,
    expected: str,
    failure: str,
    success: str = (
        "test_auprc supera la base por encima de ruido numerico, "
        "test_auc y test_mcc no caen, y test_far/test_brier no empeoran."
    ),
    cost: str = "1 entrenamiento completo; usar modo pilot solo para depurar.",
) -> dict[str, str]:
    return {
        "hypothesis": statement,
        "rationale": rationale,
        "expected_metric_movement": expected,
        "failure_mode": failure,
        "success_criterion": success,
        "cost": cost,
    }


def _fresh_analysis_experiments() -> list[Experiment]:
    temporal_spatial_asym = {"temporal": [25, 15], "spatial": [3, 1]}
    compact_asym = {"temporal": [15, 8], "spatial": [2, 1]}
    temporal_focused = {"temporal": [30, 15], "spatial": [1, 1]}
    return [
        Experiment(
            "fresh_00_base_recipe_test_eval",
            **_base_hypothesis(
                statement=(
                    "Reproducir la receta base y medirla con el mismo protocolo "
                    "test_mask + Platt + FAR que usara toda comparacion nueva."
                ),
                rationale=(
                    "Las corridas previas solo guardaron validacion; con 23 positivos "
                    "en val, el primer control debe cerrar la brecha entre seleccion "
                    "por validacion y rendimiento real en los 30 positivos de test."
                ),
                expected="Metricas cercanas a la base declarada; sirve como control.",
                failure="Si no reproduce la escala base, el protocolo de evaluacion o el checkpoint no son comparables.",
            ),
            checkpoint_metric="val_auprc",
        ),
        Experiment(
            "fresh_01_asym_neighbors_base_loss",
            **_base_hypothesis(
                statement=(
                    "Reducir fanout espacial y mantener contexto temporal deberia "
                    "bajar falsas alarmas sin perder ranking temporal."
                ),
                rationale=(
                    "El grafo tiene mas aristas espaciales que temporales y solo cuatro "
                    "porticos; muestrear 25 vecinos espaciales por capa puede mezclar "
                    "muchos negativos parecidos y diluir los pocos positivos."
                ),
                expected="test_far y Brier bajan; AUPRC se mantiene o sube.",
                failure="Si los accidentes dependen de interaccion espacial amplia, AUPRC y AUC caeran.",
            ),
            num_neighbors=temporal_spatial_asym,
            eval_num_neighbors=temporal_spatial_asym,
            checkpoint_metric="val_auprc",
        ),
        Experiment(
            "fresh_02_compact_neighbors_high_alpha",
            **_base_hypothesis(
                statement=(
                    "Un vecindario compacto con focal alpha alto y gamma bajo "
                    "puede mejorar recall de positivos raros sin la agresividad "
                    "de pairwise ni oversampling fuerte."
                ),
                rationale=(
                    "La prevalencia train es ~0.1%; focal alpha=0.75 probablemente "
                    "subpondera positivos. Bajar gamma evita ignorar ejemplos positivos "
                    "ya parcialmente aprendidos."
                ),
                expected="AUPRC y MCC suben con FAR no mayor a la base.",
                failure="Alpha demasiado alto puede inflar probabilidades y subir FAR tras calibracion.",
            ),
            loss_type="FocalLoss",
            focal_alpha=0.995,
            focal_gamma=0.75,
            num_neighbors=compact_asym,
            eval_num_neighbors=compact_asym,
            checkpoint_metric="val_auprc",
        ),
        Experiment(
            "fresh_03_compact_neighbors_monitor_mcc",
            **_base_hypothesis(
                statement=(
                    "Seleccionar checkpoint por MCC puede evitar checkpoints con "
                    "AUPRC alta pero umbral operativo debil."
                ),
                rationale=(
                    "MCC usa la matriz de confusion y castiga falsas alarmas bajo "
                    "desbalance extremo, mientras AUPRC puede mejorar por ranking "
                    "sin dar un punto operativo estable."
                ),
                expected="MCC y FAR test mejoran; AUPRC puede subir menos.",
                failure="Con solo 23 positivos en val, MCC puede ser mas ruidoso que AUPRC.",
            ),
            loss_type="FocalLoss",
            focal_alpha=0.99,
            focal_gamma=0.50,
            num_neighbors=compact_asym,
            eval_num_neighbors=compact_asym,
            checkpoint_metric="val_mcc",
        ),
        Experiment(
            "fresh_04_ce_tempered_asym_neighbors",
            **_base_hypothesis(
                statement=(
                    "CrossEntropy con pesos sqrt puede calibrar mejor que focal "
                    "cuando la metrica final incluye Brier y FAR."
                ),
                rationale=(
                    "La CE ponderada del pipeline usa pesos templados, menos extremos "
                    "que focal alpha cercano a 1; podria reducir sobreconfianza."
                ),
                expected="Brier y FAR bajan; AUPRC no cae mas que el margen permitido.",
                failure="Si CE no concentra suficiente gradiente en positivos, recall y AUPRC caeran.",
            ),
            loss_type="CrossEntropy",
            num_neighbors=temporal_spatial_asym,
            eval_num_neighbors=temporal_spatial_asym,
            checkpoint_metric="val_auprc",
        ),
        Experiment(
            "fresh_05_distance_weighted_focal_asym",
            **_base_hypothesis(
                statement=(
                    "Ponderar positivos por distancia deberia favorecer eventos "
                    "fisicamente cercanos y reducir alertas espurias."
                ),
                rationale=(
                    "Los positivos de snapshot pueden incluir nodos con relacion menos "
                    "directa al accidente. La ponderacion por distancia prioriza senal "
                    "local sin cambiar la mascara de entrenamiento."
                ),
                expected="FAR y Brier bajan con AUPRC estable.",
                failure="Si la etiqueta ya codifica bien proximidad temporal/espacial, el peso puede quitar senal util.",
            ),
            loss_type="FocalLoss",
            focal_alpha=0.975,
            focal_gamma=1.00,
            loss_weight_mode="distance",
            num_neighbors=temporal_spatial_asym,
            eval_num_neighbors=temporal_spatial_asym,
            checkpoint_metric="val_auprc",
        ),
        Experiment(
            "fresh_06_posaware003_focal_asym",
            **_base_hypothesis(
                statement=(
                    "Un sampler positive-aware cerca de la prevalencia real ampliada "
                    "3x deberia estabilizar batches sin distorsionar como 5%-10%."
                ),
                rationale=(
                    "Las corridas previas probaron oversampling mucho mas fuerte. "
                    "Con 0.3% objetivo se aumenta exposicion positiva, pero se conserva "
                    "una distribucion de negativos parecida al problema real."
                ),
                expected="AUPRC sube con FAR controlado.",
                failure="Si 0.3% aun es demasiado bajo por batch, la ganancia sera nula.",
            ),
            train_sampler_mode="positive_aware",
            positive_fraction=0.003,
            hard_window=60,
            hard_per_positive=0,
            loss_type="FocalLoss",
            focal_alpha=0.99,
            focal_gamma=0.75,
            num_neighbors=temporal_spatial_asym,
            eval_num_neighbors=temporal_spatial_asym,
            checkpoint_metric="val_auprc",
        ),
        Experiment(
            "fresh_07_posaware005_hard1_focal_asym",
            **_base_hypothesis(
                statement=(
                    "Agregar un negativo duro temporal por positivo puede mejorar "
                    "discriminacion local sin el salto de oversampling fuerte previo."
                ),
                rationale=(
                    "El error operativo mas caro son falsas alarmas cercanas a eventos; "
                    "hard negatives cercanos fuerzan separacion local."
                ),
                expected="MCC sube y FAR baja; AUPRC se mantiene.",
                failure="Si los hard negatives son demasiado parecidos a precursores reales, el modelo perdera recall.",
            ),
            train_sampler_mode="positive_aware",
            positive_fraction=0.005,
            hard_window=45,
            hard_per_positive=1,
            loss_type="FocalLoss",
            focal_alpha=0.99,
            focal_gamma=0.75,
            num_neighbors=temporal_spatial_asym,
            eval_num_neighbors=temporal_spatial_asym,
            checkpoint_metric="val_mcc",
        ),
        Experiment(
            "fresh_08_pairwise003_asym",
            **_base_hypothesis(
                statement=(
                    "Una perdida pairwise pequena puede mejorar ranking de positivos "
                    "sin dominar la calibracion."
                ),
                rationale=(
                    "La version previa uso pesos mayores y/o oversampling fuerte. "
                    "Aqui el pairwise es auxiliar, con margen pequeno, para no romper Brier/FAR."
                ),
                expected="AUPRC sube; Brier y FAR permanecen cerca de la base.",
                failure="Si el pairwise ordena ruido de validacion, subira val_auprc pero no test_auprc.",
            ),
            loss_type="FocalLoss",
            focal_alpha=0.99,
            focal_gamma=0.75,
            num_neighbors=temporal_spatial_asym,
            eval_num_neighbors=temporal_spatial_asym,
            ranking_loss_mode="pairwise_softplus",
            ranking_loss_weight=0.03,
            ranking_loss_margin=0.05,
            ranking_loss_max_pairs=2048,
            checkpoint_metric="val_auprc",
        ),
        Experiment(
            "fresh_09_posaware003_pairwise003_recall100",
            **_base_hypothesis(
                statement=(
                    "Combinar oversampling moderado y pairwise debil con checkpoint "
                    "Recall@100 puede mejorar captura operativa de pocos positivos."
                ),
                rationale=(
                    "En test hay solo 30 positivos; Recall@100 mide si el modelo sube "
                    "eventos reales al ranking superior antes de fijar umbral FAR."
                ),
                expected="Recall top-k y AUPRC suben; FAR no supera la base tras calibracion.",
                failure="Recall@100 en validacion puede ser demasiado discreto con 23 positivos.",
            ),
            train_sampler_mode="positive_aware",
            positive_fraction=0.003,
            hard_window=60,
            hard_per_positive=0,
            loss_type="FocalLoss",
            focal_alpha=0.99,
            focal_gamma=0.75,
            num_neighbors=temporal_spatial_asym,
            eval_num_neighbors=temporal_spatial_asym,
            ranking_loss_mode="pairwise_softplus",
            ranking_loss_weight=0.03,
            ranking_loss_margin=0.05,
            ranking_loss_max_pairs=2048,
            checkpoint_metric="val_recall_at_100",
        ),
        Experiment(
            "fresh_10_temporal_focused_low_lr",
            **_base_hypothesis(
                statement=(
                    "Entrenar con menor LR y fanout temporal mas fuerte puede mejorar "
                    "estabilidad de ranking en el regimen de pocos positivos."
                ),
                rationale=(
                    "Con batches muy desbalanceados, cambios bruscos de logits afectan "
                    "mucho calibracion y MCC. Menor LR prueba si el problema es varianza "
                    "de optimizacion, no arquitectura."
                ),
                expected="Brier y MCC mejoran; AUPRC no cae.",
                failure="Si el modelo queda subentrenado en los epochs disponibles, todas las metricas bajaran.",
            ),
            loss_type="FocalLoss",
            focal_alpha=0.99,
            focal_gamma=0.75,
            num_neighbors=temporal_focused,
            eval_num_neighbors=temporal_focused,
            lr=1.5e-4,
            weight_decay=2e-4,
            checkpoint_metric="val_auprc",
            early_stop_patience=6,
        ),
    ]


def _finetune_experiments() -> list[Experiment]:
    temporal_spatial_asym = {"temporal": [25, 15], "spatial": [3, 1]}
    compact_asym = {"temporal": [15, 8], "spatial": [2, 1]}
    temporal_focused = {"temporal": [30, 15], "spatial": [2, 1]}
    return [
        Experiment(
            "finetune_00_baseline_reference_eval",
            **_base_hypothesis(
                statement=(
                    "Evaluar directamente el checkpoint base con el mismo protocolo "
                    "test_mask + Platt + FAR antes de tocar pesos."
                ),
                rationale=(
                    "El control fresco no reprodujo la base historica; este paso "
                    "verifica si la comparacion declarada coincide con el evaluador "
                    "actual y evita optimizar contra un objetivo mal anclado."
                ),
                expected="Metricas iguales o muy cercanas a la base declarada.",
                failure=(
                    "Si difiere mucho, el problema principal es comparabilidad de "
                    "evaluacion/checkpoint antes que entrenamiento."
                ),
                cost="Evaluacion sin entrenamiento.",
            ),
            eval_baseline_only=True,
        ),
        Experiment(
            "finetune_01_low_lr_preserve_ranking",
            **_base_hypothesis(
                statement=(
                    "Continuar desde el checkpoint base con LR bajo deberia mejorar "
                    "ligeramente el ranking sin olvidar la solucion ya validada."
                ),
                rationale=(
                    "El entrenamiento fresco bajo test_auprc y MCC; partir de la base "
                    "reduce varianza y limita el cambio a una adaptacion fina sobre "
                    "los mismos splits y grafo."
                ),
                expected="AUPRC y MCC suben o se mantienen, con FAR no mayor a la base.",
                failure="Si el LR aun es alto, el modelo pierde calibracion y sube Brier.",
                cost="Fine-tuning corto; 6 epochs maximos.",
            ),
            initialize_from_baseline=True,
            lr=3e-5,
            weight_decay=2e-4,
            max_epochs=6,
            early_stop_patience=3,
            checkpoint_metric="val_auprc",
        ),
        Experiment(
            "finetune_02_calibration_ce_low_lr",
            **_base_hypothesis(
                statement=(
                    "Fine-tuning con CrossEntropy ponderada y LR bajo puede corregir "
                    "Brier sin destruir el ranking aprendido por la base."
                ),
                rationale=(
                    "El primer control fresco fallo especialmente en Brier. CE es menos "
                    "sobreconfiada que focal en probabilidades, y el checkpoint base ya "
                    "contiene separacion util."
                ),
                expected="Brier baja y MCC no cae; AUPRC se mantiene o sube.",
                failure="CE puede bajar recall positivo si domina la clase negativa.",
                cost="Fine-tuning corto; 6 epochs maximos.",
            ),
            initialize_from_baseline=True,
            loss_type="CrossEntropy",
            lr=2e-5,
            weight_decay=2e-4,
            max_epochs=6,
            early_stop_patience=3,
            checkpoint_metric="val_loss",
        ),
        Experiment(
            "finetune_03_high_alpha_soft_focal",
            **_base_hypothesis(
                statement=(
                    "Focal mas sensible a positivos, pero con gamma bajo, puede "
                    "recuperar positivos de test sin inflar el umbral operativo."
                ),
                rationale=(
                    "La prevalencia de accidentes es ~0.1%; un ajuste pequeno desde "
                    "la base permite reforzar positivos con menor riesgo que entrenar "
                    "desde cero con alpha alto."
                ),
                expected="AUPRC y recall suben, con FAR controlado por calibracion val.",
                failure="Si alpha sobrecorrige, Brier y FAR empeoran.",
                cost="Fine-tuning corto; 6 epochs maximos.",
            ),
            initialize_from_baseline=True,
            loss_type="FocalLoss",
            focal_alpha=0.99,
            focal_gamma=0.50,
            lr=2e-5,
            weight_decay=2e-4,
            max_epochs=6,
            early_stop_patience=3,
            checkpoint_metric="val_auprc",
        ),
        Experiment(
            "finetune_04_tiny_pairwise_rank",
            **_base_hypothesis(
                statement=(
                    "Una perdida pairwise muy pequena desde la base puede mejorar "
                    "el orden relativo de positivos sin cambiar mucho calibracion."
                ),
                rationale=(
                    "La base ya tiene AUC aceptable pero AUPRC bajo; el ajuste debe "
                    "mover pocos logits de positivos hacia arriba, no rehacer la frontera."
                ),
                expected="AUPRC sube y MCC no baja.",
                failure="Pairwise puede optimizar ruido de validacion y no transferir a test.",
                cost="Fine-tuning corto; 6 epochs maximos.",
            ),
            initialize_from_baseline=True,
            loss_type="FocalLoss",
            focal_alpha=0.99,
            focal_gamma=0.75,
            ranking_loss_mode="pairwise_softplus",
            ranking_loss_weight=0.01,
            ranking_loss_margin=0.03,
            ranking_loss_max_pairs=1024,
            lr=1.5e-5,
            weight_decay=2e-4,
            max_epochs=6,
            early_stop_patience=3,
            checkpoint_metric="val_auprc",
        ),
        Experiment(
            "finetune_05_compact_neighbors_mcc",
            **_base_hypothesis(
                statement=(
                    "Fine-tuning con vecindario compacto y seleccion por MCC puede "
                    "conservar ranking mientras reduce falsas alarmas operativas."
                ),
                rationale=(
                    "El control fresco redujo FAR pero perdio MCC/AUPRC; desde la base, "
                    "un vecindario menos ruidoso deberia ajustar la frontera sin perder "
                    "la solucion aprendida."
                ),
                expected="MCC sube y FAR no supera la base; Brier no empeora.",
                failure="Validacion tiene pocos positivos y MCC puede ser inestable.",
                cost="Fine-tuning corto; 6 epochs maximos.",
            ),
            initialize_from_baseline=True,
            num_neighbors=compact_asym,
            eval_num_neighbors=compact_asym,
            loss_type="FocalLoss",
            focal_alpha=0.99,
            focal_gamma=0.75,
            lr=2e-5,
            weight_decay=2e-4,
            max_epochs=6,
            early_stop_patience=3,
            checkpoint_metric="val_mcc",
        ),
        Experiment(
            "finetune_06_temporal_focused_rank",
            **_base_hypothesis(
                statement=(
                    "Preservar mas contexto temporal que espacial durante fine-tuning "
                    "puede mejorar el ranking de accidentes raros."
                ),
                rationale=(
                    "Las etiquetas vienen de snapshots y la senal previa al accidente "
                    "deberia propagarse temporalmente; reducir ruido espacial evita "
                    "arrastrar muchos negativos parecidos."
                ),
                expected="AUC/AUPRC suben sin aumentar FAR.",
                failure="Si la base depende de contexto espacial amplio, la adaptacion cae.",
                cost="Fine-tuning corto; 6 epochs maximos.",
            ),
            initialize_from_baseline=True,
            num_neighbors=temporal_focused,
            eval_num_neighbors=temporal_focused,
            loss_type="FocalLoss",
            focal_alpha=0.99,
            focal_gamma=0.75,
            ranking_loss_mode="pairwise_softplus",
            ranking_loss_weight=0.005,
            ranking_loss_margin=0.03,
            ranking_loss_max_pairs=1024,
            lr=1e-5,
            weight_decay=2e-4,
            max_epochs=8,
            early_stop_patience=4,
            checkpoint_metric="val_auprc",
        ),
    ]


def _ablation_experiments() -> list[Experiment]:
    temporal_only = {"temporal": [25, 15], "spatial": [0, 0]}
    spatial_only = {"temporal": [0, 0], "spatial": [3, 1]}
    exhaustive_eval = {"temporal": [50, 25], "spatial": [6, 3]}
    return [
        Experiment(
            "fresh_ablate_00_val_loss_checkpoint",
            **_base_hypothesis(
                statement="Probar seleccion por val_loss como control anti-overfit de AUPRC.",
                rationale="La validacion tiene 23 positivos; val_auprc puede moverse por pocos nodos.",
                expected="Brier baja; AUPRC podria bajar moderadamente.",
                failure="Val_loss puede favorecer la clase negativa y perder recall.",
            ),
            checkpoint_metric="val_loss",
        ),
        Experiment(
            "fresh_ablate_01_temporal_only",
            **_base_hypothesis(
                statement="Aislar aristas temporales para medir si espacial agrega ruido.",
                rationale="La senal de accidente puede ser principalmente pre-evento temporal.",
                expected="Si espacial mete ruido, FAR baja y AUPRC no cae.",
                failure="Si hay propagacion espacial real, AUC/AUPRC caeran.",
            ),
            num_neighbors=temporal_only,
            eval_num_neighbors=temporal_only,
            checkpoint_metric="val_auprc",
        ),
        Experiment(
            "fresh_ablate_02_spatial_only",
            **_base_hypothesis(
                statement="Aislar aristas espaciales como prueba negativa de temporalidad.",
                rationale="Si temporalidad domina, esta configuracion deberia fallar claramente.",
                expected="AUPRC y AUC bajan; confirma que no conviene invertir en spatial-only.",
                failure="Si mejora, la hipotesis temporal principal esta incompleta.",
            ),
            num_neighbors=spatial_only,
            eval_num_neighbors=spatial_only,
            checkpoint_metric="val_auprc",
        ),
        Experiment(
            "fresh_ablate_03_train_asym_eval_wide",
            **_base_hypothesis(
                statement="Evaluar con vecindario mas amplio que el entrenamiento puede recuperar contexto sin ensuciar gradiente.",
                rationale="El entrenamiento compacto reduce ruido; inferencia mas amplia puede sumar evidencia temporal/espacial.",
                expected="AUC/AUPRC suben sin afectar demasiado Brier.",
                failure="Si el modelo no aprendio a usar ese contexto, la evaluacion amplia solo agrega ruido.",
            ),
            num_neighbors={"temporal": [25, 15], "spatial": [3, 1]},
            eval_num_neighbors=exhaustive_eval,
            eval_neighbors_mode="custom",
            checkpoint_metric="val_auprc",
        ),
    ]


def _seed_sensitivity_experiments() -> list[Experiment]:
    asym = {"temporal": [25, 15], "spatial": [3, 1]}
    compact = {"temporal": [15, 8], "spatial": [2, 1]}
    templates = [
        Experiment(
            "seed_asym_focal",
            **_base_hypothesis(
                statement="Medir sensibilidad a semilla del candidato focal/asimetrico.",
                rationale="Con 132 positivos train, una sola semilla puede ser anecdotica.",
                expected="La media supera la base o identifica alta varianza.",
                failure="Si la varianza domina, no hay mejora robusta aunque una corrida gane.",
                cost="3 entrenamientos completos.",
            ),
            loss_type="FocalLoss",
            focal_alpha=0.995,
            focal_gamma=0.75,
            num_neighbors=compact,
            eval_num_neighbors=compact,
            checkpoint_metric="val_auprc",
        ),
        Experiment(
            "seed_posaware_pairwise",
            **_base_hypothesis(
                statement="Medir sensibilidad a semilla del candidato posaware+pairwise moderado.",
                rationale="El sampler cambia exposicion de positivos y puede depender mucho de la semilla.",
                expected="AUPRC test mejora en al menos dos de tres semillas.",
                failure="Si solo una semilla gana, no se integra.",
                cost="3 entrenamientos completos.",
            ),
            train_sampler_mode="positive_aware",
            positive_fraction=0.003,
            hard_window=60,
            hard_per_positive=0,
            loss_type="FocalLoss",
            focal_alpha=0.99,
            focal_gamma=0.75,
            num_neighbors=asym,
            eval_num_neighbors=asym,
            ranking_loss_mode="pairwise_softplus",
            ranking_loss_weight=0.03,
            ranking_loss_margin=0.05,
            ranking_loss_max_pairs=2048,
            checkpoint_metric="val_auprc",
        ),
    ]
    out: list[Experiment] = []
    for seed_offset in (0, 101, 202):
        for exp in templates:
            out.append(
                Experiment(
                    **{
                        **asdict(exp),
                        "name": f"{exp.name}_s{seed_offset}",
                        "seed_offset": seed_offset,
                    }
                )
            )
    return out


def _quick_experiments() -> list[Experiment]:
    asym = {"temporal": [25, 15], "spatial": [3, 1]}
    return [
        Experiment(
            "quick_base_test_eval",
            **_base_hypothesis(
                statement="Smoke test del protocolo base.",
                rationale="Verifica entrenamiento, sidecar y evaluacion test obligatoria.",
                expected="Debe producir fila con post_eval_status=ok.",
                failure="Cualquier error aqui bloquea las suites largas.",
                cost="1 corrida corta en modo pilot.",
            ),
            checkpoint_metric="val_auprc",
        ),
        Experiment(
            "quick_asym_neighbors",
            **_base_hypothesis(
                statement="Smoke test del formato dict de vecinos por relacion.",
                rationale="Las claves temporal/spatial deben mapear a edge types heterogeneos.",
                expected="Entrena y evalua sin error de NeighborLoader.",
                failure="Error de resolucion de num_neighbors.",
                cost="1 corrida corta en modo pilot.",
            ),
            num_neighbors=asym,
            eval_num_neighbors=asym,
            checkpoint_metric="val_auprc",
        ),
        Experiment(
            "quick_posaware003_pairwise",
            **_base_hypothesis(
                statement="Smoke test del sampler moderado y pairwise debil.",
                rationale="Valida que los nuevos knobs se escriben y llegan al entrenamiento.",
                expected="Sidecar registra sampler_impl y ranking_loss_mode.",
                failure="Error de sampler o perdida auxiliar.",
                cost="1 corrida corta en modo pilot.",
            ),
            train_sampler_mode="positive_aware",
            positive_fraction=0.003,
            hard_window=60,
            hard_per_positive=0,
            num_neighbors=asym,
            eval_num_neighbors=asym,
            ranking_loss_mode="pairwise_softplus",
            ranking_loss_weight=0.03,
            ranking_loss_margin=0.05,
            ranking_loss_max_pairs=2048,
            checkpoint_metric="val_auprc",
        ),
    ]


def _experiments(suite: str) -> list[Experiment]:
    key = str(suite or "analysis").strip().lower()
    if key == "analysis":
        return _fresh_analysis_experiments()
    if key == "finetune":
        return _finetune_experiments()
    if key == "ablation":
        return _ablation_experiments()
    if key == "seed":
        return _seed_sensitivity_experiments()
    if key == "quick":
        return _quick_experiments()
    if key == "all":
        return (
            _finetune_experiments()
            + _fresh_analysis_experiments()
            + _ablation_experiments()
            + _seed_sensitivity_experiments()
        )
    raise ValueError(f"Suite desconocida: {suite!r}")


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _mask_stats(data: Any) -> dict[str, int]:
    y = data["pm"].y.detach().cpu().long()
    out: dict[str, int] = {}
    for split in ("train", "val", "test"):
        mask = getattr(data["pm"], f"{split}_mask").detach().cpu().bool()
        out[f"{split}_n"] = int(mask.sum().item())
        out[f"{split}_pos"] = int((y[mask] == 1).sum().item())
        out[f"{split}_neg"] = int((y[mask] == 0).sum().item())
    return out


def _make_pilot_train_mask(data: Any, neg_fraction: float, seed: int) -> None:
    train_mask = data["pm"].train_mask.detach().cpu().bool()
    y = data["pm"].y.detach().cpu().long()
    pos_idx = torch.where(train_mask & (y == 1))[0]
    neg_idx = torch.where(train_mask & (y == 0))[0]
    keep_neg = int(max(1, round(float(neg_fraction) * int(neg_idx.numel()))))
    gen = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(int(neg_idx.numel()), generator=gen)[:keep_neg]
    new_train = torch.zeros_like(train_mask)
    new_train[pos_idx] = True
    new_train[neg_idx[perm]] = True
    data["pm"].train_mask = new_train.to(data["pm"].train_mask.device)


def _experiment_seed(args: argparse.Namespace, exp: Experiment) -> int:
    return int(args.seed) + int(exp.seed_offset or 0)


def _prepare_loaded_obj(
    base_obj: dict[str, Any],
    exp: Experiment,
    args: argparse.Namespace,
) -> dict[str, Any]:
    obj = copy.deepcopy(base_obj)
    data = obj["data"]
    if args.mode == "pilot":
        _make_pilot_train_mask(data, args.pilot_neg_fraction, _experiment_seed(args, exp))
    return obj


def _write_hparams(base_hparams: pd.DataFrame, exp: Experiment, path: Path, args: argparse.Namespace) -> None:
    row = base_hparams.iloc[0].copy()
    row["value"] = 0.0
    row["hparams_source"] = f"gnn_fresh_training_{args.suite}_{args.mode}"
    row["train_sampler_mode"] = exp.train_sampler_mode
    row["disable_hard_undersampling"] = bool(exp.disable_hard_undersampling)
    row["deterministic_sampling"] = True
    row["sampling_seed"] = _experiment_seed(args, exp)
    row["checkpoint_metric"] = exp.checkpoint_metric or "val_auprc"
    if exp.eval_neighbors_mode is not None:
        row["eval_neighbors_mode"] = str(exp.eval_neighbors_mode)
    elif exp.eval_num_neighbors is not None:
        row["eval_neighbors_mode"] = "custom"
    if exp.eval_num_neighbors is not None:
        row["eval_num_neighbors"] = json.dumps(exp.eval_num_neighbors)
    if exp.objective_metric is not None:
        row["objective_metric"] = str(exp.objective_metric)
    if exp.threshold_beta is not None:
        row["threshold_beta"] = float(exp.threshold_beta)
    if exp.loss_type is not None:
        row["loss_type"] = str(exp.loss_type)
    if exp.positive_fraction is not None:
        row["positive_sampler_target_fraction"] = float(exp.positive_fraction)
    if exp.hard_window is not None:
        row["positive_sampler_hard_window_minutes"] = int(exp.hard_window)
    if exp.hard_per_positive is not None:
        row["positive_sampler_hard_negatives_per_positive"] = int(exp.hard_per_positive)
    if exp.focal_alpha is not None:
        row["focal_alpha"] = float(exp.focal_alpha)
    if exp.focal_gamma is not None:
        row["focal_gamma"] = float(exp.focal_gamma)
    if exp.loss_weight_mode is not None:
        row["loss_weight_mode"] = str(exp.loss_weight_mode)
    if exp.ranking_loss_mode is not None:
        row["ranking_loss_mode"] = str(exp.ranking_loss_mode)
    if exp.ranking_loss_weight is not None:
        row["ranking_loss_weight"] = float(exp.ranking_loss_weight)
    if exp.ranking_loss_margin is not None:
        row["ranking_loss_margin"] = float(exp.ranking_loss_margin)
    if exp.ranking_loss_max_pairs is not None:
        row["ranking_loss_max_pairs"] = int(exp.ranking_loss_max_pairs)
    if exp.num_neighbors is not None:
        row["num_neighbors"] = json.dumps(exp.num_neighbors)
    if exp.lr is not None:
        row["lr"] = float(exp.lr)
    if exp.weight_decay is not None:
        row["weight_decay"] = float(exp.weight_decay)
    if exp.batch_size is not None:
        row["batch_size"] = int(exp.batch_size)
    if exp.dropout is not None:
        row["dropout"] = float(exp.dropout)
    if exp.grad_clip is not None:
        row["grad_clip"] = float(exp.grad_clip)
    if exp.accumulation_steps is not None:
        row["accumulation_steps"] = int(exp.accumulation_steps)
    if exp.lr_scheduler is not None:
        row["lr_scheduler"] = str(exp.lr_scheduler)
    pd.DataFrame([row]).to_csv(path, index=False)


def _find_new_model(start_time: float) -> Path | None:
    candidates = []
    for path in (ROOT / "Resultados").glob("gat_model_BEST_GNN_*.pt"):
        try:
            if path.stat().st_mtime >= start_time - 2.0:
                candidates.append(path)
        except OSError:
            pass
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _read_sidecar(model_path: Path | None) -> dict[str, Any]:
    if model_path is None:
        return {}
    sidecar = model_path.with_name(model_path.stem + "_hparams.json")
    if not sidecar.exists():
        return {}
    try:
        return json.loads(sidecar.read_text())
    except Exception:
        return {}


def _checkpoint_model_state(path: Path) -> dict[str, Any]:
    obj = _torch_load(path)
    if isinstance(obj, dict) and "model_state" in obj:
        state = obj["model_state"]
    elif isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
    elif hasattr(obj, "state_dict"):
        state = obj.state_dict()
    elif isinstance(obj, dict):
        state = obj
    else:
        raise TypeError(f"Checkpoint no compatible para warm-start: {path}")
    if not isinstance(state, dict):
        raise TypeError(f"El checkpoint no contiene state_dict: {path}")
    return state


def _baseline_resume_state_path(
    *,
    baseline_model: Path,
    out_dir: Path,
    exp: Experiment,
    monitor_metric: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{exp.name}_resume_from_baseline.pt"
    state = _checkpoint_model_state(baseline_model)
    monitor_mode = "min" if str(monitor_metric) == "val_loss" else "max"
    payload = {
        "model_state": state,
        "epoch": 0,
        "best_val_loss": float("inf"),
        "best_val_f1": 0.0,
        "best_val_auprc": 0.0,
        "best_val_auc": None,
        "best_val_mcc": None,
        "best_val_far": None,
        "best_val_f05": None,
        "best_val_tau": None,
        "best_val_accuracy": None,
        "best_val_objective_score": float("-inf"),
        "best_epoch": 0,
        "best_monitor_value": float("inf") if monitor_mode == "min" else float("-inf"),
        "monitor_metric": str(monitor_metric),
        "monitor_mode": monitor_mode,
        "patience_counter": 0,
        "source_checkpoint": str(baseline_model.resolve()),
        "warm_start_experiment": exp.name,
    }
    torch.save(payload, path)
    return path


def _append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = _read_result_rows(path)
    rows.append({k: _json_for_csv(v) for k, v in row.items()})
    _write_result_rows(path, rows)


def _expanded_result_header() -> list[str]:
    return sorted(
        {
            "best_epoch",
            "best_val_auc",
            "best_val_auprc",
            "best_val_f05",
            "best_val_f1",
            "best_val_far",
            "best_val_tau",
            "accumulation_steps",
            "batch_size",
            "cost",
            "checkpoint_metric",
            "disable_hard_undersampling",
            "elapsed_seconds",
            "eval_neighbors_mode",
            "eval_neighbors_mode_used",
            "eval_num_neighbors",
            "eval_baseline_only",
            "expected_metric_movement",
            "failure_mode",
            "focal_alpha",
            "focal_gamma",
            "grad_clip",
            "dropout",
            "hard_per_positive",
            "hard_window",
            "hparams_path",
            "hypothesis",
            "initialize_from_baseline",
            "lr",
            "lr_scheduler",
            "loss_type",
            "loss_type_used",
            "loss_weight_mode",
            "loss_weight_mode_used",
            "max_epochs",
            "max_epochs_used",
            "metrics_history_path",
            "mode",
            "model_hparams_path",
            "model_path",
            "monitor_metric",
            "name",
            "num_neighbors",
            "objective_metric",
            "positive_fraction",
            "positive_sampler_stats",
            "post_eval_error",
            "post_eval_status",
            "rationale",
            "ranking_loss_margin",
            "ranking_loss_max_pairs",
            "ranking_loss_mode",
            "ranking_loss_mode_used",
            "ranking_loss_weight",
            "ranking_loss_weight_used",
            "seed",
            "seed_offset",
            "started_at",
            "status",
            "success_criterion",
            "suite",
            "threshold_beta",
            "early_stop_patience",
            "early_stop_patience_used",
            "best_val_precision_at_k",
            "best_val_recall_at_k",
            "beats_baseline",
            "baseline_auc",
            "baseline_auprc",
            "baseline_brier",
            "baseline_far",
            "baseline_mcc",
            "eval_calibration_count",
            "eval_calibration_mask_source",
            "eval_calibration_method",
            "eval_far_target",
            "eval_mask",
            "eval_threshold",
            "eval_threshold_count",
            "eval_threshold_mask_source",
            "test_n",
            "test_auc",
            "test_auprc",
            "test_brier",
            "test_confusion_matrix",
            "test_delta_auc",
            "test_delta_auprc",
            "test_delta_brier",
            "test_delta_far",
            "test_delta_mcc",
            "test_f1",
            "test_far",
            "test_mcc",
            "test_neg",
            "test_pos",
            "test_precision",
            "test_recall",
            "test_specificity",
            "train_n",
            "train_neg",
            "train_pos",
            "train_sampler_impl",
            "train_sampler_mode",
            "val_n",
            "val_neg",
            "val_pos",
            "val_threshold_far",
            "val_threshold_sens",
            "weight_decay",
            "warm_start_checkpoint",
        }
    )


def _read_result_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return []
        expanded_header = _expanded_result_header()
        rows: list[dict[str, Any]] = []
        for raw in reader:
            if not raw:
                continue
            if len(raw) == len(header):
                mapped = dict(zip(header, raw))
            elif len(raw) == len(expanded_header):
                mapped = dict(zip(expanded_header, raw))
            else:
                mapped = dict(zip(header, raw[: len(header)]))
                if len(raw) > len(header):
                    mapped["_extra"] = json.dumps(raw[len(header) :], ensure_ascii=True)
            rows.append(mapped)
    return rows


def _write_result_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for item in rows for key in item.keys()})
    if not fields:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _repair_results_file(path: Path) -> None:
    if path.exists():
        _write_result_rows(path, _read_result_rows(path))


def _json_for_csv(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return value


def _completed_names(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()
    try:
        df = pd.read_csv(results_path)
    except Exception:
        return set()
    if "status" not in df.columns or "name" not in df.columns:
        return set()
    return set(df.loc[df["status"].astype(str) == "ok", "name"].astype(str).tolist())


def _metric(sidecar: dict[str, Any], key: str) -> Any:
    value = sidecar.get(key)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _mps_diagnostics() -> dict[str, Any]:
    import platform

    backend = getattr(getattr(torch, "backends", None), "mps", None)
    built = False
    available = False
    alloc_ok = False
    alloc_error = ""
    if backend is not None:
        try:
            built = bool(backend.is_built())
        except Exception as exc:
            alloc_error = f"is_built failed: {exc!r}"
        try:
            available = bool(backend.is_available())
        except Exception as exc:
            alloc_error = f"is_available failed: {exc!r}"
    if available:
        try:
            _ = torch.ones(1, device="mps")
            alloc_ok = True
        except Exception as exc:
            alloc_error = repr(exc)
    elif built and not alloc_error:
        try:
            _ = torch.ones(1, device="mps")
            alloc_ok = True
            available = True
        except Exception as exc:
            alloc_error = repr(exc)
    return {
        "torch": torch.__version__,
        "mps_built": built,
        "mps_available": available,
        "mps_allocation_ok": alloc_ok,
        "mps_allocation_error": alloc_error,
        "macos": platform.mac_ver()[0] or platform.platform(),
        "machine": platform.machine(),
    }


def _resolve_device(args: argparse.Namespace) -> torch.device:
    requested = str(getattr(args, "device", "mps") or "mps").lower()
    mps_diag = _mps_diagnostics()
    mps_built = bool(mps_diag["mps_built"])
    mps_available = bool(mps_diag["mps_available"])
    cuda_available = bool(torch.cuda.is_available())

    if requested == "mps":
        if mps_built:
            return torch.device("mps")
        if not bool(getattr(args, "allow_device_fallback", False)):
            raise RuntimeError(
                "MPS fue solicitado pero este PyTorch no fue construido con MPS. "
                f"Diagnostico: torch={mps_diag['torch']}, "
                f"mps_built={mps_built}, mps_available={mps_available}, "
                f"mps_allocation_error={mps_diag['mps_allocation_error']}, "
                f"macOS={mps_diag['macos']}, machine={mps_diag['machine']}. "
                "Ejecuta con la .venv del proyecto o reinstala PyTorch con soporte MPS."
            )
        return torch.device("cuda" if cuda_available else "cpu")
    if requested == "cuda":
        if cuda_available:
            return torch.device("cuda")
        if not bool(getattr(args, "allow_device_fallback", False)):
            raise RuntimeError(
                "CUDA no esta disponible. Usa --device mps o --allow-device-fallback."
            )
        return torch.device("mps" if mps_available else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if mps_available:
        return torch.device("mps")
    if cuda_available:
        return torch.device("cuda")
    return torch.device("cpu")


def _configure_gnn_device(args: argparse.Namespace) -> torch.device:
    device = _resolve_device(args)

    def _fixed_device() -> torch.device:
        return device

    # gnn_main importa get_auto_device por valor; replicamos la seleccion que usa
    # Streamlit para que el launcher no vuelva a caer a CPU despues de pedir MPS.
    gnn_main.get_auto_device = _fixed_device
    return device


def _coerce_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return value


def _eval_num_neighbors(exp: Experiment, sidecar: dict[str, Any]) -> Any:
    if exp.eval_num_neighbors is not None:
        return exp.eval_num_neighbors
    sidecar_eval = _coerce_json(sidecar.get("eval_num_neighbors"))
    if sidecar_eval is not None:
        return sidecar_eval
    if exp.num_neighbors is not None:
        return exp.num_neighbors
    return _coerce_json(sidecar.get("num_neighbors"))


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _test_eval_fields(payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    calibration = payload.get("calibration") or {}
    metrics = payload.get("metrics") or {}
    report = payload.get("report") or {}
    pos_report = report.get("Accidente (1)", {}) if isinstance(report, dict) else {}
    out = {
        "post_eval_status": "ok",
        "eval_mask": payload.get("test_mask"),
        "eval_threshold": payload.get("threshold"),
        "eval_far_target": float(args.far_target),
        "eval_calibration_method": "platt_scaling"
        if calibration.get("platt_scaling")
        else "softmax_raw",
        "eval_calibration_mask_source": payload.get("calibration_mask_source"),
        "eval_threshold_mask_source": payload.get("threshold_mask_source"),
        "eval_calibration_count": calibration.get("calibration_count"),
        "eval_threshold_count": calibration.get("threshold_count"),
        "test_auc": payload.get("auc"),
        "test_auprc": payload.get("auprc"),
        "test_mcc": payload.get("mcc"),
        "test_far": payload.get("false_alarm_ratio"),
        "test_brier": payload.get("brier_score"),
        "test_precision": metrics.get("precision", pos_report.get("precision")),
        "test_recall": metrics.get("recall", pos_report.get("recall")),
        "test_f1": metrics.get("f1", pos_report.get("f1-score")),
        "test_specificity": metrics.get("specificity"),
        "test_confusion_matrix": payload.get("confusion_matrix"),
        "val_threshold_far": calibration.get("far"),
        "val_threshold_sens": calibration.get("sens"),
        "baseline_auc": float(args.baseline_auc),
        "baseline_auprc": float(args.baseline_auprc),
        "baseline_mcc": float(args.baseline_mcc),
        "baseline_far": float(args.baseline_far),
        "baseline_brier": float(args.baseline_brier),
    }
    for metric in ("auc", "auprc", "mcc", "far", "brier"):
        value = _safe_float(out.get(f"test_{metric}"))
        baseline = _safe_float(out.get(f"baseline_{metric}"))
        out[f"test_delta_{metric}"] = (
            float(value - baseline)
            if value is not None and baseline is not None
            else None
        )
    out["beats_baseline"] = _beats_baseline(out, args)
    return out


def _evaluate_checkpoint_on_test(
    *,
    loaded_obj: dict[str, Any],
    model_path: Path | None,
    exp: Experiment,
    sidecar: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    if model_path is None:
        raise FileNotFoundError("No se encontro checkpoint nuevo para evaluar en test_mask.")
    from src import graph_builder_app

    payload = graph_builder_app._evaluate_gnn_model_far_target(
        model_path=str(model_path),
        graph_data=loaded_obj["data"],
        sequence_index=loaded_obj.get("sequence_index"),
        device=device,
        far_target=float(args.far_target),
        batch_size=int(args.eval_batch_size),
        num_neighbors=_eval_num_neighbors(exp, sidecar),
        seed=int(args.seed),
        pm_index=loaded_obj.get("pm_index"),
    )
    return _test_eval_fields(payload, args)


def _beats_baseline(fields: dict[str, Any], args: argparse.Namespace) -> bool:
    auc = _safe_float(fields.get("test_auc"))
    auprc = _safe_float(fields.get("test_auprc"))
    mcc = _safe_float(fields.get("test_mcc"))
    far = _safe_float(fields.get("test_far"))
    brier = _safe_float(fields.get("test_brier"))
    if None in {auc, auprc, mcc, far, brier}:
        return False
    min_auprc_gain = float(getattr(args, "min_auprc_gain", 1e-6) or 0.0)
    min_mcc_gain = float(getattr(args, "min_mcc_gain", 0.0) or 0.0)
    max_auc_drop = float(getattr(args, "max_auc_drop", 0.0) or 0.0)
    max_far_increase = float(getattr(args, "max_far_increase", 0.0) or 0.0)
    max_brier_increase = float(getattr(args, "max_brier_increase", 0.0) or 0.0)
    return bool(
        auprc >= float(args.baseline_auprc) + min_auprc_gain
        and mcc >= float(args.baseline_mcc) + min_mcc_gain
        and auc >= float(args.baseline_auc) - max_auc_drop
        and far <= float(args.baseline_far) + max_far_increase
        and brier <= float(args.baseline_brier) + max_brier_increase
    )


def _summarize_existing_results(out_dir: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"files": [], "best_by_val_auprc": None}
    frames = []
    for path in sorted(out_dir.glob("*_results.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        df["_results_file"] = str(path)
        frames.append(df)
        summary["files"].append({"path": str(path), "rows": int(len(df))})
    if not frames:
        return summary
    all_rows = pd.concat(frames, ignore_index=True, sort=False)
    if "status" in all_rows.columns:
        ok = all_rows.loc[all_rows["status"].astype(str) == "ok"].copy()
    else:
        ok = all_rows.copy()
    summary["ok_rows"] = int(len(ok))
    for metric in ("best_val_auprc", "test_auprc", "test_mcc"):
        if metric not in ok.columns:
            continue
        numeric = pd.to_numeric(ok[metric], errors="coerce")
        if numeric.notna().any():
            idx = int(numeric.idxmax())
            row = ok.loc[idx]
            summary[f"best_by_{metric}"] = {
                "name": str(row.get("name", "")),
                metric: float(numeric.loc[idx]),
                "model_path": str(row.get("model_path", "")),
            }
    return summary


def run(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir).resolve()
    run_id = time.strftime("%Y%m%d_%H%M%S")
    results_path = out_dir / f"{args.mode}_results.csv"
    hparams_dir = out_dir / "hparams"
    history_dir = out_dir / "histories"
    experiments = _experiments(args.suite)
    if args.only:
        wanted = {name.strip() for name in args.only.split(",") if name.strip()}
        experiments = [exp for exp in experiments if exp.name in wanted]
    if bool(getattr(args, "list_experiments", False)):
        print(json.dumps([asdict(exp) for exp in experiments], indent=2, ensure_ascii=True))
        return results_path

    hparams_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    _repair_results_file(results_path)
    device = _configure_gnn_device(args)

    base_obj = _torch_load(Path(args.graph))
    if not isinstance(base_obj, dict) or "data" not in base_obj:
        raise ValueError(f"Grafo invalido: {args.graph}")
    base_hparams = pd.read_csv(args.hparams)
    baseline_model = Path(args.baseline_model).resolve()
    if not baseline_model.exists():
        raise FileNotFoundError(f"No existe baseline_model: {baseline_model}")
    completed = _completed_names(results_path) if args.resume else set()

    manifest = {
        "run_id": run_id,
        "mode": args.mode,
        "suite": args.suite,
        "graph": str(Path(args.graph).resolve()),
        "hparams": str(Path(args.hparams).resolve()),
        "max_epochs": int(args.max_epochs),
        "early_stop_patience": int(args.early_stop_patience),
        "pilot_neg_fraction": float(args.pilot_neg_fraction),
        "device": str(device),
        "mps_diagnostics": _mps_diagnostics(),
        "baseline_model": str(Path(args.baseline_model).resolve()),
        "baseline_metrics": {
            "auc": float(args.baseline_auc),
            "auprc": float(args.baseline_auprc),
            "mcc": float(args.baseline_mcc),
            "far": float(args.baseline_far),
            "brier": float(args.baseline_brier),
        },
        "far_target": float(args.far_target),
        "success_thresholds": {
            "min_auprc_gain": float(args.min_auprc_gain),
            "min_mcc_gain": float(args.min_mcc_gain),
            "max_auc_drop": float(args.max_auc_drop),
            "max_far_increase": float(args.max_far_increase),
            "max_brier_increase": float(args.max_brier_increase),
        },
        "test_eval_protocol": "test_mask_with_val_platt_and_val_far_target",
        "previous_results_summary": _summarize_existing_results(out_dir),
        "experiments": [asdict(exp) for exp in experiments],
    }
    (out_dir / f"{args.mode}_manifest_{run_id}.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True)
    )

    for idx, exp in enumerate(experiments, start=1):
        if exp.name in completed:
            print(f"[{idx}/{len(experiments)}] skip completed {exp.name}", flush=True)
            continue
        print(f"[{idx}/{len(experiments)}] running {exp.name}", flush=True)
        exp_seed = _experiment_seed(args, exp)
        max_epochs_used = int(exp.max_epochs if exp.max_epochs is not None else args.max_epochs)
        patience_used = int(
            exp.early_stop_patience
            if exp.early_stop_patience is not None
            else args.early_stop_patience
        )
        eval_neighbors_mode_used = (
            exp.eval_neighbors_mode
            if exp.eval_neighbors_mode is not None
            else ("custom" if exp.eval_num_neighbors is not None else "same")
        )
        monitor_metric = exp.checkpoint_metric or "val_auprc"
        hp_path = hparams_dir / f"{args.mode}_{exp.name}.csv"
        _write_hparams(base_hparams, exp, hp_path, args)
        loaded_obj = _prepare_loaded_obj(base_obj, exp, args)
        stats = _mask_stats(loaded_obj["data"])
        history_path = history_dir / f"{args.mode}_{exp.name}_metrics_history.jsonl"
        start = time.time()
        row: dict[str, Any] = {
            "name": exp.name,
            "mode": args.mode,
            "suite": args.suite,
            "status": "ok",
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hparams_path": str(hp_path),
            "metrics_history_path": str(history_path),
            "seed": exp_seed,
            "max_epochs_used": max_epochs_used,
            "early_stop_patience_used": patience_used,
            "eval_neighbors_mode_used": eval_neighbors_mode_used,
            **asdict(exp),
            **stats,
        }
        try:
            if bool(exp.eval_baseline_only):
                sidecar = _read_sidecar(baseline_model)
                post_eval = _evaluate_checkpoint_on_test(
                    loaded_obj=loaded_obj,
                    model_path=baseline_model,
                    exp=exp,
                    sidecar=sidecar,
                    args=args,
                    device=device,
                )
                row.update(
                    {
                        "elapsed_seconds": round(time.time() - start, 3),
                        "model_path": str(baseline_model),
                        "model_hparams_path": str(
                            baseline_model.with_name(baseline_model.stem + "_hparams.json")
                        ),
                        "best_epoch": _metric(sidecar, "best_epoch"),
                        "best_val_auprc": _metric(sidecar, "best_val_auprc"),
                        "best_val_auc": _metric(sidecar, "best_val_auc"),
                        "best_val_f1": _metric(sidecar, "best_val_f1"),
                        "best_val_f05": _metric(sidecar, "best_val_f05"),
                        "best_val_far": _metric(sidecar, "best_val_far"),
                        "best_val_tau": _metric(sidecar, "best_val_tau"),
                        "monitor_metric": _metric(sidecar, "monitor_metric"),
                        "loss_type_used": _metric(sidecar, "loss_type"),
                        "loss_weight_mode_used": _metric(sidecar, "loss_weight_mode"),
                        "ranking_loss_mode_used": _metric(sidecar, "ranking_loss_mode"),
                        "ranking_loss_weight_used": _metric(sidecar, "ranking_loss_weight"),
                        "best_val_recall_at_k": _metric(sidecar, "best_val_recall_at_k"),
                        "best_val_precision_at_k": _metric(sidecar, "best_val_precision_at_k"),
                        "train_sampler_impl": _metric(sidecar, "sampler_impl"),
                        "positive_sampler_stats": _metric(sidecar, "positive_sampler_stats"),
                    }
                )
                row.update(post_eval)
                _append_row(results_path, row)
                if bool(row.get("beats_baseline")) and bool(args.stop_on_improvement):
                    print(
                        f"[{idx}/{len(experiments)}] baseline superado por {exp.name}; "
                        "deteniendo ejecucion.",
                        flush=True,
                    )
                    break
                continue

            resume_state_path = None
            if bool(exp.initialize_from_baseline):
                resume_state_path = _baseline_resume_state_path(
                    baseline_model=baseline_model,
                    out_dir=out_dir / "resume_checkpoints",
                    exp=exp,
                    monitor_metric=monitor_metric,
                )
                row["warm_start_checkpoint"] = str(resume_state_path)

            gnn_main.run_gat_training(
                loaded_obj,
                force_use_graphsmote=False,
                purpose=f"gnn_fresh_training_{args.suite}_{args.mode}:{exp.name}",
                early_stop=True,
                early_stop_patience=patience_used,
                early_stop_min_delta=float(args.early_stop_min_delta),
                max_epochs=max_epochs_used,
                resume_state_path=str(resume_state_path) if resume_state_path else None,
                accumulation_steps=(
                    int(exp.accumulation_steps)
                    if exp.accumulation_steps is not None
                    else None
                ),
                train_sampler_mode=exp.train_sampler_mode,
                deterministic_sampling=True,
                sampling_seed=exp_seed,
                disable_hard_undersampling=bool(exp.disable_hard_undersampling),
                positive_sampler_target_fraction=exp.positive_fraction,
                positive_sampler_hard_window_minutes=exp.hard_window,
                positive_sampler_hard_negatives_per_positive=exp.hard_per_positive,
                eval_neighbors_mode=eval_neighbors_mode_used,
                eval_num_neighbors=exp.eval_num_neighbors,
                checkpoint_metric=monitor_metric,
                ranking_loss_mode=exp.ranking_loss_mode,
                ranking_loss_weight=exp.ranking_loss_weight,
                ranking_loss_margin=exp.ranking_loss_margin,
                ranking_loss_max_pairs=exp.ranking_loss_max_pairs,
                metrics_history_path=str(history_path),
                test_eval_interval_epochs=0,
                hparams_path=str(hp_path),
                hparams_index=None,
                reuse_hparams=True,
                allow_hpo_search=False,
            )
            model_path = _find_new_model(start)
            sidecar = _read_sidecar(model_path)
            post_eval: dict[str, Any] = {}
            try:
                post_eval = _evaluate_checkpoint_on_test(
                    loaded_obj=loaded_obj,
                    model_path=model_path,
                    exp=exp,
                    sidecar=sidecar,
                    args=args,
                    device=device,
                )
            except Exception as eval_exc:
                post_eval = {
                    "post_eval_status": "error",
                    "post_eval_error": repr(eval_exc),
                    "beats_baseline": False,
                }
                if not bool(args.allow_missing_test_eval):
                    row["status"] = "error"
                    row["error"] = (
                        "Fallo la evaluacion obligatoria en test_mask con "
                        f"Platt/FAR target: {eval_exc!r}"
                    )
            row.update(
                {
                    "elapsed_seconds": round(time.time() - start, 3),
                    "model_path": str(model_path) if model_path else "",
                    "model_hparams_path": str(model_path.with_name(model_path.stem + "_hparams.json"))
                    if model_path
                    else "",
                    "best_epoch": _metric(sidecar, "best_epoch"),
                    "best_val_auprc": _metric(sidecar, "best_val_auprc"),
                    "best_val_auc": _metric(sidecar, "best_val_auc"),
                    "best_val_f1": _metric(sidecar, "best_val_f1"),
                    "best_val_f05": _metric(sidecar, "best_val_f05"),
                    "best_val_far": _metric(sidecar, "best_val_far"),
                    "best_val_tau": _metric(sidecar, "best_val_tau"),
                    "monitor_metric": _metric(sidecar, "monitor_metric"),
                    "loss_type_used": _metric(sidecar, "loss_type"),
                    "loss_weight_mode_used": _metric(sidecar, "loss_weight_mode"),
                    "ranking_loss_mode_used": _metric(sidecar, "ranking_loss_mode"),
                    "ranking_loss_weight_used": _metric(sidecar, "ranking_loss_weight"),
                    "best_val_recall_at_k": _metric(sidecar, "best_val_recall_at_k"),
                    "best_val_precision_at_k": _metric(sidecar, "best_val_precision_at_k"),
                    "train_sampler_impl": _metric(sidecar, "sampler_impl"),
                    "positive_sampler_stats": _metric(sidecar, "positive_sampler_stats"),
                }
            )
            row.update(post_eval)
        except Exception as exc:
            row.update(
                {
                    "status": "error",
                    "elapsed_seconds": round(time.time() - start, 3),
                    "error": repr(exc),
                }
            )
            print(f"[{idx}/{len(experiments)}] ERROR {exp.name}: {exc!r}", flush=True)
        _append_row(results_path, row)
        if bool(row.get("beats_baseline")) and bool(args.stop_on_improvement):
            print(
                f"[{idx}/{len(experiments)}] baseline superado por {exp.name}; "
                "deteniendo ejecucion.",
                flush=True,
            )
            break
    return results_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default=str(DEFAULT_GRAPH))
    parser.add_argument("--hparams", default=str(DEFAULT_HPARAMS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--mode", choices=["pilot", "full"], default="full")
    parser.add_argument(
        "--suite",
        choices=[
            "analysis",
            "finetune",
            "ablation",
            "seed",
            "quick",
            "all",
        ],
        default="finetune",
    )
    parser.add_argument("--pilot-neg-fraction", type=float, default=0.15)
    parser.add_argument("--max-epochs", type=int, default=24)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=19091985)
    parser.add_argument("--only", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list-experiments", action="store_true")
    parser.add_argument("--device", choices=["mps", "cuda", "cpu", "auto"], default="mps")
    parser.add_argument("--allow-device-fallback", action="store_true")
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--far-target", type=float, default=BASELINE_METRICS["far"])
    parser.add_argument("--baseline-model", default=str(DEFAULT_BASELINE_MODEL))
    parser.add_argument("--baseline-auc", type=float, default=BASELINE_METRICS["auc"])
    parser.add_argument("--baseline-auprc", type=float, default=BASELINE_METRICS["auprc"])
    parser.add_argument("--baseline-mcc", type=float, default=BASELINE_METRICS["mcc"])
    parser.add_argument("--baseline-far", type=float, default=BASELINE_METRICS["far"])
    parser.add_argument("--baseline-brier", type=float, default=BASELINE_METRICS["brier"])
    parser.add_argument("--min-auprc-gain", type=float, default=1e-6)
    parser.add_argument("--min-mcc-gain", type=float, default=0.0)
    parser.add_argument("--max-auc-drop", type=float, default=0.0)
    parser.add_argument("--max-far-increase", type=float, default=0.0)
    parser.add_argument("--max-brier-increase", type=float, default=0.0)
    parser.add_argument("--allow-missing-test-eval", action="store_true")
    parser.add_argument("--stop-on-improvement", dest="stop_on_improvement", action="store_true")
    parser.add_argument("--no-stop-on-improvement", dest="stop_on_improvement", action="store_false")
    parser.set_defaults(stop_on_improvement=True)
    return parser.parse_args()


if __name__ == "__main__":
    output = run(parse_args())
    print(f"results_path={output}", flush=True)
