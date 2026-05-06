try:
    import torch  # type: ignore
except Exception:
    torch = None
import numpy as np # type: ignore
import os

def get_auto_device():
    """Retorna el mejor dispositivo disponible: MPS > CUDA > CPU."""
    if torch is not None:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return "cpu"

# --------------------------------------------------------------------------- #
# CONFIGURACIÓN GLOBAL Y CONSTANTES
# --------------------------------------------------------------------------- #

# --- XAI --- Pesos de atención #
XAI = 1 # 1 = Activado, 0 = Desactivado

# Archivos y directorios
RESULTADOS_DIR = "Resultados"

# Export legacy duplicate CSVs (backward-compat) for GAT report
# 1 = exportar también con prefijo antiguo 'gnn_anomaly_*' (duplica archivos)
# 0 = solo prefijo nuevo 'gat_report_*' (recomendado)
EXPORT_LEGACY_GAT_CSV = 0

# Semilla para reproducibilidad
SEED = 19091985
if torch is not None:
    torch.manual_seed(SEED)
np.random.seed(SEED)

# --- MODO DEBUG ---
# Activa/desactiva mensajes de depuración en todo el proyecto.
# 1 = Activado, 0 = Desactivado
DEBUG = 0

# Precisión de punto flotante para tensores
FLOAT = "float32"  # Opciones: 16, 32, 64

# Granularidad temporal para agregar datos de flujo en minutos
# Nota: originalmente este proyecto fue calibrado para DT_MINUTES = 5.
# Para trabajar a resolución de 1 minuto, ajusta a 1 y revisa
# el bloque "Flujo" de más abajo (incluye auto‑ajuste opcional).
DT_MINUTES = 1
ACC_NEIGHBORHOODS = 0 # Número de vecindarios afectados por el accidente

# Bins fijos para entropía de velocidades: rango [0, 120] km/h en 10 intervalos
# -> 11 bordes equiespaciados
SPEED_BINS = np.linspace(0, 120, 11)

# Velocidad mínima para media armónica (seguridad numérica)
HARMONIC_V_MIN = 1.0  # km/h

# --------------------------------------------------------------------------- #
# Flujo: parámetros para métricas robustas y temporales
# --------------------------------------------------------------------------- #
#
# Idea clave: algunos parámetros dependen de la granularidad temporal (DT_MINUTES).
# Para conservar horizontes temporales equivalentes al pasar de DT=5 → DT=1,
# se incluye un modo de auto‑ajuste a partir de una configuración base (DT_REF=5).
#
# ▶ Opción rápida (recomendada): AUTOSCALE_FLOW_PARAMS = 1
#    - Mantiene aprox. 1 hora de historial en rolling
#    - Mantiene EWMAs con horizontes ~45 min y ~15 min
#    - Reescala umbrales por conteos proporcionalmente al tamaño de ventana
#
# ▶ Opción manual: AUTOSCALE_FLOW_PARAMS = 0
#    - Define explícitamente FLOW_ROLL_W, FLOW_EWMA_ALPHAS y FLOW_EXCEED_THRESHOLDS
#
DT_REF = 5  # minutos (configuración base para la que se eligieron los valores)
AUTOSCALE_FLOW_PARAMS = 0  # 1=activar auto‑ajuste desde DT_REF; 0=usar manual

# Sub‑ventanas por cada ventana DT_MINUTES
# - Escalado automático: 3 sub‑bins por minuto → sub‑bins de 20 segundos
#   Ej.: DT=1 → 3 sub‑bins (20s); DT=5 → 15 sub‑bins (20s).
#   (features.py soporta sub‑minuto mediante 'ts_second')
FLOW_SUBBINS_PER_MIN = 3
FLOW_SUBBINS_M = max(1, int(FLOW_SUBBINS_PER_MIN * DT_MINUTES))

# Porcentajes robustos (no dependen de DT)
FLOW_TRIM_P = 0.1      # media recortada (0–0.5)
FLOW_WINSOR_P = 0.1    # media winsorizada (0–0.5)

# Valores base pensados para DT_REF=5
_FLOW_ROLL_W_BASE = 12           # ≈ 12×5min ≈ 60 min de historial
_FLOW_EWMA_ALPHAS_BASE = [0.2, 0.5]  # ≈ 45 min y ≈ 15 min a DT=5
_FLOW_EXCEED_THRESHOLDS_BASE = []    # p.ej., [800, 1000] (conteos por DT)

def _alpha_rescaled_from_minutes(alpha_base: float, dt_base: float, dt_new: float) -> float:
    """Reescala alpha de un EWMA para conservar el horizonte temporal aproximado.
    Aproximación vía span: alpha ≈ 2/(span+1), con span en número de ventanas.
    """
    try:
        span_base = max(1.0, 2.0/float(alpha_base) - 1.0)
        minutes_horizon = span_base * float(dt_base)
        span_new = max(1.0, minutes_horizon / float(dt_new))
        return float(2.0/(span_new + 1.0))
    except Exception:
        return float(alpha_base)

if AUTOSCALE_FLOW_PARAMS:
    # Tamaño de ventana rolling (en número de ventanas DT_MINUTES)
    # - Mantiene ~60 minutos de historial independientemente de DT
    FLOW_ROLL_W = max(1, int(round(_FLOW_ROLL_W_BASE * (DT_REF / float(DT_MINUTES)))))
    # Alphas para EWMA ajustadas para conservar horizontes aproximados de 45 y 15 min
    FLOW_EWMA_ALPHAS = [
        _alpha_rescaled_from_minutes(a, DT_REF, DT_MINUTES) for a in _FLOW_EWMA_ALPHAS_BASE
    ]
    # Umbrales operativos: reescala conteos por tamaño de ventana (lineal en minutos)
    #   Ej.: si thr_base=800 veh/5min → thr_new ≈ 800 * (1/5) = 160 veh/1min
    FLOW_EXCEED_THRESHOLDS = [
        float(th) * float(DT_MINUTES) / float(DT_REF) for th in _FLOW_EXCEED_THRESHOLDS_BASE
    ]
    # Suavizado de velocidad: mantener horizonte temporal equivalente
    # Base DT=5: 5 ventanas → ~25 min. A DT=1 → 25 ventanas.
    SMOOTH_V_W_BASE = 5
    SMOOTH_V_W = max(1, int(round(SMOOTH_V_W_BASE * (DT_REF / float(DT_MINUTES)))))
else:
    # MODO MANUAL — ajusta explícitamente para tu DT
    # - Para DT=1 min, sugerencias comunes:
    #   * FLOW_ROLL_W = 60       # ~1h de historial
    #   * FLOW_EWMA_ALPHAS = [0.044, 0.125]  # ~45m y ~15m
    #   * FLOW_EXCEED_THRESHOLDS = [160, 200] si antes eran [800, 1000] a DT=5
    FLOW_ROLL_W = 12
    FLOW_EWMA_ALPHAS = [0.2, 0.5]
    FLOW_EXCEED_THRESHOLDS = []  # p.ej., [800, 1000]
    # Suavizado de velocidad manual (en nº de ventanas de DT_MINUTES)
    # Sugerencia para DT=1: SMOOTH_V_W=25 (≈25 min, equivalente a 5×5min)
    SMOOTH_V_W = 5

# Cuantiles para run-length alto/bajo (invariantes a DT)
FLOW_RUN_HIGH_Q = 0.9
FLOW_RUN_LOW_Q = 0.1

# --- Otras opciones útiles (documentación) ---
# - Opción A (horizontes largos): 60 min rolling, EWMA ≈ [45m, 15m]
#     • Con DT=1 → FLOW_ROLL_W=60; FLOW_EWMA_ALPHAS≈[0.044, 0.125]
# - Opción B (más reactivo): 30 min rolling, EWMA ≈ [30m, 10m]
#     • Con DT=1 → FLOW_ROLL_W=30; FLOW_EWMA_ALPHAS≈[0.0645, 0.1818]
# - Umbrales de excedencia (si se usan): thr_new ≈ thr_base × (DT_new/DT_base)

# Tramos horarios para filtrar los datos. Formato: (HH:MM, HH:MM)
TIME_SLOTS_TO_FILTER = [
    ("06:00", "10:00"),
    ("18:00", "22:00")
]

# Parámetros de entrenamiento
MAX_EPOCHS = 3000
EARLY_STOPPING_PATIENCE = 5 # 30 Número de evaluaciones sin mejora
EARLY_STOPPING_MIN_DELTA = 0.0
# Gradiente acumulado (para reducir memoria efectiva de batch)
ACCUMULATION_STEPS = 2

# optimize
TIMEOUT = 172800 #48HR
N_TRIALS = 30  #100
NUM_EPOCHS_OPTUNA = 10 #50 epoch por cada prueba
#EARLY_STOPPING_EPOCH_OPTUNA = 10

# NeighborLoader
BATCH_SIZE = 512
NUM_NEIGHBORS = {
    ('pm', 'spatial', 'pm'): [10, 5, 5],
    ('pm', 'temporal', 'pm'): [10, 5, 5],
} #2 niveles (Recordar desvanecimiento de la información con la profundidad.)

TRAIN_RATIO = 70
VAL_RATIO = 15

# Selección de umbral (PR)
F_BETA_THRESHOLD = 0.5  # Beta para F_beta al elegir el umbral de decisión

# GraphSMOTE
GRAPHSMOTE_MODE = "offline"     # 'offline' | 'online' | 'none'
TARGET_POS_RATIO = 0.1          # proporción objetivo de positivos en TRAIN
GRAPHSMOTE_K =  5               # k-NN para síntesis
PRETRAIN_EDGE_EPOCHS = 5        # épocas para pre-entrenar generador de aristas
SMOTE_EVERY_N_EPOCHS = 5        # refresco en modo online
GS_SEED = SEED                  # semilla reproducible para SMOTE
SAVE_AUG_GRAPH_PATH = os.path.join(RESULTADOS_DIR, "graph_aug.pt")
GRAPHSMOTE_CONECT = 0           # 1 = método actual (Top-K por edge_gen sobre TRAIN reales);
                                # 0 = conectar a vecinos (1-hop) de nodos reales con clase=1

# Para loaders de embeddings 
EMB_NUM_NEIGHBORS = NUM_NEIGHBORS          #  reducir si hay OOM
EMB_BATCH_SIZE = BATCH_SIZE                # idem

# ImGAGN
AUTO_IMGAGN_PRETRAIN = 0  # 1 = ejecutar ImGAGN antes de cada entrenamiento GAT si no hay artefacto previo

# Calibración automática de probabilidades (Platt scaling sobre validación)
AUTOCALIBRATE_PROBS = 1  # 1 = habilitar

# ImGAGN edges
# 1 = bidireccionales (real <-> sintético), 0 = solo dirigidas real -> sintético
BIDIRECCTION = 0

# Decoders z->x
# Épocas por defecto para entrenar los decodificadores de embeddings a features.
# Usado desde main.py; early stopping puede finalizar antes si no hay mejora.
DECODER_EPOCHS = 300

# --- Snapshots to sequences ---
SEQ_LENGTH = 6             # L snapshots por secuencia (ej. 6*5min = 30min historial)
GUARD_BAND_MINUTES = 10    # γ: separación entre features y target en minutos
HORIZON_MINUTES = 20       # H: ventana futura a evaluar (minutos)
INCLUDE_DOWNSTREAM_IN_LABEL = 1  # 1=considerar accidente en primer downstream como positivo

# --- Variantes GNN temporales ---
# Ejemplos:
#   "gat_snapshot", "gat_gru", "gat_transformer", "gat_attnpool",
#   "gat_edge_mlp", "gat_edge_mlp_gru", "gnn_edge_aware", "gnn_edge_aware_gru"
GNN_VARIANT = "gat_gru"

# Estrategia de caché para heads temporales:
# - "global_epoch": priming por época con embeddings globales (recomendado para comparabilidad)
# - "incremental": caché incremental por batch (más liviano)
GNN_TEMPORAL_CACHE_STRATEGY = "global_epoch"

"""
Nodos
-----
* **pm** (pórtico-minuto) 

Aristas (todas dirigidas)
------------------------
* **spatial**: pm  P_i,t  → pm P_{i+1},t (atrib.:[dist_km], Δ features)
* **temporal**: pm  P_i,t → pm P_i,{t+1} (atrib.: Δ features)
"""

# Guardado de modelos GAT (duplicados por alias)
# 1 = guardar también alias por variante y alias global (crea duplicados .pt)
# 0 = solo guardar archivos únicos con timestamp/hash (recomendado)
SAVE_GAT_ALIASES = 0
