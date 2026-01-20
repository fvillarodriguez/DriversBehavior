import os, glob
import re
import pandas as pd
from datetime import datetime
import numpy as np
import torch
from pathlib import Path
import json
from torch_geometric.data import HeteroData
from src.graphsmote import _safe_clean_edges

from src.utils import (
    load_porticos,
    load_flujos,
    load_accidentes,
    buscar_columna,
    reconstruct_portico_sequence,
)
from src.snapshot_pipeline import (
    SnapshotConfig,
    SnapshotFeatureBuilder,
    flatten_snapshot_features,
    build_static_topology,
)
from src.snapshot_sequences import (
    SequenceConfig,
    build_sequence_index,
)
from src.config import (
    FLOAT,
    DT_MINUTES,
    ACC_NEIGHBORHOODS,
    RESULTADOS_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    DEBUG,
    SEQ_LENGTH,
    GUARD_BAND_MINUTES,
    HORIZON_MINUTES,
    INCLUDE_DOWNSTREAM_IN_LABEL,
)

class PMIndex:
    """Contenedor para mapeos de nodos PM."""
    def __init__(self, p_map, r_map):
        self._map = p_map
        self._rev = r_map
    def __len__(self):
        return len(self._rev)

from typing import Optional, List, Tuple


def _compute_time_cutoffs(
    accident_ts: List[int],
    train_ratio: int,
    val_ratio: int,
) -> Tuple[float, float]:
    """
    Derive train/validation cutoffs (inclusive) using sorted accident timestamps.

    Returns:
        (train_ts_cutoff, val_ts_cutoff) in minutes since epoch. `float('inf')`
        indicates that the split is not applied (e.g., insufficient incidents).
    """
    if not accident_ts:
        return float("inf"), float("inf")

    sorted_ts = sorted(set(accident_ts))
    n_ts = len(sorted_ts)
    if n_ts < 3:
        return float("inf"), float("inf")

    raw_train_end_idx = int((train_ratio / 100) * n_ts)
    raw_val_end_idx = int(((train_ratio + val_ratio) / 100) * n_ts)

    train_end_idx = max(0, min(raw_train_end_idx, n_ts - 3))
    val_end_idx = max(train_end_idx + 1, min(raw_val_end_idx, n_ts - 2))

    train_cutoff = float(sorted_ts[train_end_idx])
    val_cutoff = float(sorted_ts[val_end_idx])
    return train_cutoff, val_cutoff


def _save_normalization_stats(
    feature_names: List[str],
    means: np.ndarray,
    stds: np.ndarray,
    train_cutoff: float,
    val_cutoff: float,
) -> str:
    """Persist normalization statistics to JSON and return the file path."""
    Path(RESULTADOS_DIR).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTADOS_DIR, f"normalization_stats_{ts}.json")
    payload = {
        "generated_at": datetime.now().isoformat(),
        "train_ts_cutoff": None if not np.isfinite(train_cutoff) else float(train_cutoff),
        "val_ts_cutoff": None if not np.isfinite(val_cutoff) else float(val_cutoff),
        "features": {
            name: {
                "mean": float(mu),
                "std": float(sigma),
            }
            for name, mu, sigma in zip(feature_names, means, stds)
        },
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return out_path


def build_graph() -> Optional[str]:
    # 1) Cargar pórticos y accidentes
    print("\n▶️ Cargando datos base…\n")
    print("\n▶️ 1.- Pórticos")
    df_port = load_porticos()
    if df_port is None:
        print("❌ Carga de pórticos cancelada o fallida. Abortando construcción de grafo.")
        return None

    # Filtrar pórticos según selección: Punto Negro (si hay reporte) o Manual
    from src.utils import seleccionar_tramo_y_porticos
    porticos_a_incluir = None
    preselected_time_choice = None  # '1' todos, '2' mañana, '3' tarde

    # Buscar reporte más reciente de puntos negros
    pn_reports = sorted(glob.glob(os.path.join(RESULTADOS_DIR, "analisis_puntos_negros_*.csv")))
    use_pn = False
    if pn_reports:
        print("\n▶️ Se encontró un reporte de Puntos Negros.")
        print("  [1] Usar Punto Negro (tramo y franja del último reporte)")
        print("  [2] Selección manual de tramo y franja horaria")
        sel = input("Elija una opción [1/2]: ").strip() or '1'
        use_pn = (sel == '1')

    if use_pn:
        # Cargar último CSV y tomar primer registro (ya ordenado por peligrosidad)
        pn_path = pn_reports[-1]
        try:
            df_pn = pd.read_csv(pn_path, sep=';', decimal=',')
            if df_pn.empty:
                raise ValueError("Reporte vacío")
            top = df_pn.iloc[0]
            tramo_str = str(top['tramo'])
            franja = str(top['franja_horaria']) if 'franja_horaria' in df_pn.columns else None

            # Determinar time_choice por franja
            if franja and '06-10' in franja:
                preselected_time_choice = '2'
            elif franja and '18-22' in franja:
                preselected_time_choice = '3'
            else:
                preselected_time_choice = '1'

            # Construir lista de pórticos del tramo (± vecinos) con lógica centralizada
            p1_raw, p2_raw = [s.strip() for s in tramo_str.split('->')]
            porticos_a_incluir = reconstruct_portico_sequence(
                df_porticos=df_port,
                p1=p1_raw,
                p2=p2_raw,
                max_anterior=4,
                max_posterior=0,
                prefer_family=True,
                deduplicate_slice=True,
            )

            print("\n✅ Usando Punto Negro del último reporte:")
            print(f"   - Tramo: {tramo_str}")
            print(f"   - Franja horaria: {franja or 'Sin filtro'}")
            print(f"   - Pórticos incluidos: {porticos_a_incluir}")
        except Exception as e:
            print(f"⚠️ No se pudo usar el reporte de Puntos Negros ({e}). Se solicitará selección manual.")
            porticos_a_incluir = None

    if porticos_a_incluir is None:
        porticos_a_incluir = seleccionar_tramo_y_porticos(df_port)
        if porticos_a_incluir is None:
            print("❌ Operación cancelada. No se seleccionaron pórticos para el grafo.")
            return None

    df_port = df_port[df_port['portico'].isin(porticos_a_incluir)].copy()
    print(f"\n✅ El grafo se construirá con los siguientes pórticos: {porticos_a_incluir}")

    print("\n▶️ 2.- Accidentes")
    df_acc = load_accidentes()
    if df_acc is None:
        print("❌ Carga de accidentes cancelada o fallida. Abortando construcción de grafo.")
        return None
    # NEW: Filter accidents based on selected porticos
    df_acc = df_acc[df_acc['ultimo_portico'].isin(porticos_a_incluir)].copy()
    print(f"✅ Se considerarán {len(df_acc)} accidentes en los pórticos seleccionados.")

    # 2) Elegir modo de obtención de features
    choice = input("¿Desea \n[1] Construir features desde cero, \n[2] Cargar features desde un CSV o \n[3] Cargar features usando una lista de features seleccionadas? \n Elija una opción [1/2/3]: ").strip()
    
    df_pm = None
    snapshot_topology = None
    snap_config = SnapshotConfig()

    if choice == '2':
        all_csv_files = glob.glob(os.path.join(RESULTADOS_DIR, "features_*.csv"))
        # Excluir únicamente listas de nombres; permitir cualquier features_*.csv completo
        csv_files = sorted([
            f for f in all_csv_files
            if not os.path.basename(f).startswith("features_selected_names_")
        ])
        if not csv_files:
            print("No se encontraron archivos CSV de features en 'Resultados'. Generando desde cero.")
            choice = "1"
        elif len(csv_files) == 1:
            csv_path = csv_files[0]
            df_pm = pd.read_csv(csv_path)
            print(f"Cargado automáticamente el único archivo de features encontrado -> {os.path.basename(csv_path)}")
        else:
            print("Archivos de features completos disponibles:\n")
            for i, f in enumerate(csv_files, 1):
                print(f"  [{i}] {os.path.basename(f)}")
            try:
                sel = int(input("Seleccione el número de archivo CSV: ").strip()) - 1
                if sel < 0 or sel >= len(csv_files):
                    raise ValueError
                csv_path = csv_files[sel]
                df_pm = pd.read_csv(csv_path)
                print(f"Cargado features desde -> {csv_path}")
            except (ValueError, IndexError):
                print("Selección inválida. Generando desde cero.")
                choice = "1"

    elif choice == '3':
        # Cargar nombres de features seleccionadas
        selected_names_files = sorted(glob.glob(os.path.join(RESULTADOS_DIR, "features_selected_names_*.csv")))
        if not selected_names_files:
            print("No se encontraron archivos con nombres de features seleccionadas. Por favor, ejecute primero la Selección de Features.")
            return None

        selected_features_path = None
        if len(selected_names_files) == 1:
            selected_features_path = selected_names_files[0]
            print(f"Cargando automáticamente el único archivo de nombres de features -> {os.path.basename(selected_features_path)}")
        else:
            print("Archivos con nombres de features seleccionadas disponibles:")
            for i, f in enumerate(selected_names_files, 1):
                print(f"  [{i}] {os.path.basename(f)}")
            try:
                sel_names = int(input("Seleccione el archivo de nombres: ").strip()) - 1
                if not (0 <= sel_names < len(selected_names_files)):
                    raise ValueError
                selected_features_path = selected_names_files[sel_names]
            except (ValueError, IndexError):
                print("Selección inválida.")
                return None

        if selected_features_path:
            selected_feature_names = pd.read_csv(selected_features_path)['feature_name'].tolist()

        # Cargar el dataset de features completo
        features_files = sorted(glob.glob(os.path.join(RESULTADOS_DIR, "features_*.csv")))
        # Excluir únicamente listas de nombres; permitir cualquier features_*.csv completo
        all_features_files = sorted([
            f for f in features_files
            if not os.path.basename(f).startswith("features_selected_names_")
        ])
        if not all_features_files:
            print("No se encontraron archivos CSV de features completos para filtrar.")
            return None

        if len(all_features_files) == 1:
            full_features_path = all_features_files[0]
            df_full = pd.read_csv(full_features_path)
            print(f"Cargado automáticamente el único archivo de features completo -> {os.path.basename(full_features_path)}")
        else:
            print("\nArchivos de features COMPLETOS disponibles:")
            for i, f in enumerate(all_features_files, 1):
                try:
                    ts = datetime.fromtimestamp(os.path.getmtime(f)).strftime('%Y-%m-%d %H:%M')
                    size_mb = os.path.getsize(f) / (1024*1024)
                    meta = f" ({ts}, {size_mb:.1f} MB)"
                except Exception:
                    meta = ""
                print(f"  [{i}] {os.path.basename(f)}{meta}")
            try:
                sel_full = input("Seleccione el archivo de features completo [número]: ").strip()
                sel_idx = int(sel_full) - 1
                if not (0 <= sel_idx < len(all_features_files)):
                    raise ValueError
                full_features_path = all_features_files[sel_idx]
                df_full = pd.read_csv(full_features_path)
                print(f"Cargado dataset completo desde -> {full_features_path}")
            except (ValueError, IndexError):
                print("❌ Selección inválida.")
                return None

        # Filtrar el DataFrame con manejo robusto de incompatibilidades
        portico_col = buscar_columna(df_full, "portico")
        essential_cols = [portico_col, 'ts_min']
        present = set(df_full.columns)
        chosen = [c for c in selected_feature_names if c in present]
        missing = [c for c in selected_feature_names if c not in present]

        if missing:
            print("\n⚠️ Algunas features seleccionadas no existen en el CSV elegido:")
            for name in (missing[:20]):
                print(f"   - {name}")
            if len(missing) > 20:
                print(f"   … y {len(missing)-20} más")
            print(f"➡️ Se usarán {len(chosen)} features disponibles de {len(selected_feature_names)} seleccionadas.")

        if len(chosen) == 0:
            # Ofrecer alternativas si ninguna feature coincide
            print("\n❌ Ninguna de las features seleccionadas está presente en este archivo.")
            print("   ¿Qué desea hacer?")
            print("   [1] Elegir otro archivo de features completo")
            print("   [2] Usar TODAS las columnas de features disponibles en este CSV")
            print("   [0] Cancelar")
            opt = input("Elija una opción [1/2/0]: ").strip()
            if opt == '1':
                print("\nArchivos de features COMPLETOS disponibles:")
                for i, f in enumerate(all_features_files, 1):
                    try:
                        ts = datetime.fromtimestamp(os.path.getmtime(f)).strftime('%Y-%m-%d %H:%M')
                        size_mb = os.path.getsize(f) / (1024*1024)
                        meta = f" ({ts}, {size_mb:.1f} MB)"
                    except Exception:
                        meta = ""
                    print(f"  [{i}] {os.path.basename(f)}{meta}")
                try:
                    sel_full2 = int(input("Seleccione el archivo de features completo: ").strip()) - 1
                    if not (0 <= sel_full2 < len(all_features_files)):
                        raise ValueError
                    full_features_path = all_features_files[sel_full2]
                    df_full = pd.read_csv(full_features_path)
                    present = set(df_full.columns)
                    chosen = [c for c in selected_feature_names if c in present]
                    missing = [c for c in selected_feature_names if c not in present]
                    if not chosen:
                        print("❌ El archivo elegido tampoco contiene ninguna de las features seleccionadas. Abortando.")
                        return None
                    print(f"✅ Archivo cargado. Se usarán {len(chosen)} features disponibles.")
                except (ValueError, IndexError):
                    print("❌ Selección inválida. Abortando.")
                    return None
            elif opt == '2':
                chosen = [c for c in df_full.columns if c not in essential_cols]
                print(f"✅ Usando todas las {len(chosen)} columnas de features presentes en el CSV.")
            else:
                print("Operación cancelada por el usuario.")
                return None

        final_cols = essential_cols + chosen
        df_pm = df_full[final_cols].copy()
        print(f"\nDataset filtrado a {len(chosen)} features (de {len(selected_feature_names)} seleccionadas).")

    # 3) Si no se cargó desde CSV, generar desde cero
    if choice == "1":
        print("\n▶️ Flujos")
        df_flows = load_flujos()
        if df_flows is None:
            print("❌ No se pudieron cargar los flujos.")
            return None
        print("▶️ Generando snapshots con topología fija (Δ=5 min, W=30 min).")
        builder = SnapshotFeatureBuilder(snap_config)
        snapshot_features, snapshot_topology = builder.build(df_flows, df_port)
        df_pm = flatten_snapshot_features(snapshot_features)
        df_pm = df_pm.sort_values(["portico", "ts_min"]).reset_index(drop=True)

    if snapshot_topology is None:
        snapshot_topology = build_static_topology(df_port, snap_config)

    # --- NEW: Time-based filtering ---
    print("\n▶️ Selección de tramo horario...")
    if preselected_time_choice is None:
        print("  [1] Sin filtro horario (todos los datos)")
        print("  [2] Horario punta mañana (6:00 - 10:00)")
        print("  [3] Horario punta tarde (18:00 - 22:00)")
        time_choice = input("Elija una opción de tramo horario [1/2/3]: ").strip()
    else:
        time_choice = preselected_time_choice
        label = {
            '1': 'Sin filtro horario (todos los datos)',
            '2': 'Horario punta mañana (6:00 - 10:00)',
            '3': 'Horario punta tarde (18:00 - 22:00)'
        }.get(time_choice, 'Sin filtro horario (todos los datos)')
        print(f"  ↳ Usando franja preseleccionada por Punto Negro: {label}")

    if time_choice in ['2', '3']:
        # Convert ts_min (minutes since epoch) to datetime objects to extract the hour
        timestamps = pd.to_datetime(df_pm['ts_min'], unit='m', origin='unix')
        hours = timestamps.dt.hour

        if time_choice == '2':
            # Morning rush hour (6:00 to 9:59)
            time_mask = (hours >= 6) & (hours < 10)
            print(f"✅ Filtrando datos para el horario punta de la mañana (6:00 - 10:00).")
        elif time_choice == '3':
            # Evening rush hour (18:00 to 21:59)
            time_mask = (hours >= 18) & (hours < 22)
            print(f"✅ Filtrando datos para el horario punta de la tarde (18:00 - 22:00).")
        
        initial_rows = len(df_pm)
        df_pm = df_pm[time_mask].copy()
        
        if df_pm.empty:
            print(f"❌ Después de filtrar por horario, no quedan datos (de {initial_rows} iniciales). Abortando.")
            return None
        else:
            print(f"ℹ️ Se conservan {len(df_pm)} de {initial_rows} nodos después del filtro horario.")

    else:
        print("✅ Usando todos los datos sin filtro horario.")

    # --- 3a) Preprocesamiento y Normalización ---
    # 3a.1) Identificar las columnas de features
    portico_col = buscar_columna(df_pm, "portico")
    feature_cols = [
        c for c in df_pm.columns
        if c not in [portico_col, "ts_min"]
    ]

    # Filter df_pm based on selected porticos
    df_pm = df_pm[df_pm[portico_col].isin(porticos_a_incluir)].copy()
    if df_pm.empty:
        print("❌ No se encontraron datos de flujo (PM nodes) para los pórticos seleccionados. Abortando construcción de grafo.")
        return None
    # IMPORTANT: Reset index after filtering to ensure node_idx is contiguous
    df_pm = df_pm.reset_index(drop=True)
    df_pm['ts_min'] = df_pm['ts_min'].astype(int)
    df_pm['node_idx'] = df_pm.index

    # --- Determinar DT efectivo para alinear accidentes y snapshots ---
    try:
        diffs = []
        for _, g in df_pm.groupby(portico_col):
            ts_sorted = np.sort(g['ts_min'].unique())
            d = np.diff(ts_sorted)
            d = d[d > 0]
            if d.size:
                diffs.extend(d.tolist())
        if diffs:
            vals, counts = np.unique(np.array(diffs, dtype=int), return_counts=True)
            dt_feat_minutes = int(vals[np.argmax(counts)])
        else:
            dt_feat_minutes = int(DT_MINUTES)
    except Exception:
        dt_feat_minutes = int(DT_MINUTES)
    if dt_feat_minutes != int(DT_MINUTES):
        print(f"ℹ️ Detectado DT en features distinto a config: {dt_feat_minutes} min (config={DT_MINUTES}). Alineando construcción a {dt_feat_minutes} min.")
    else:
        print(f"ℹ️ DT efectivo de features: {dt_feat_minutes} min.")

    # Mapear accidentes al DT de los snapshots
    df_acc = df_acc.copy()
    df_acc['ts_min'] = (
        df_acc["accidente_time"]
        .dt.floor(f"{int(dt_feat_minutes)}min")
        .astype("int64") // 60_000_000_000
    )

    # Alinear tipos de pórtico antes del merge
    try:
        pm_dtype = df_pm[portico_col].dtype
        if np.issubdtype(pm_dtype, np.number):
            df_acc['ultimo_portico'] = pd.to_numeric(df_acc['ultimo_portico'], errors='coerce').astype('Int64')
            df_pm[portico_col] = pd.to_numeric(df_pm[portico_col], errors='coerce').astype('Int64')
        else:
            df_acc['ultimo_portico'] = df_acc['ultimo_portico'].astype(str)
            df_pm[portico_col] = df_pm[portico_col].astype(str)
    except Exception:
        pass

    affected_pms = pd.merge(
        df_acc,
        df_pm[[portico_col, 'ts_min', 'node_idx']],
        left_on=['ultimo_portico', 'ts_min'],
        right_on=[portico_col, 'ts_min'],
        how='inner'
    )
    accident_ts_unique = affected_pms['ts_min'].unique().tolist() if not affected_pms.empty else []
    train_ts_cutoff, val_ts_cutoff = _compute_time_cutoffs(
        accident_ts_unique,
        TRAIN_RATIO,
        VAL_RATIO,
    )
    if not np.isfinite(train_ts_cutoff):
        print("ℹ️ División temporal: todos los nodos se asignarán a entrenamiento (insuficientes incidentes).")

    sequence_config = SequenceConfig(
        sequence_length=SEQ_LENGTH,
        guard_band_minutes=GUARD_BAND_MINUTES,
        horizon_minutes=HORIZON_MINUTES,
        include_downstream=bool(INCLUDE_DOWNSTREAM_IN_LABEL),
    )
    seq_index, labels_df = build_sequence_index(
        df_pm,
        df_acc[['ultimo_portico', 'ts_min']].copy(),
        snapshot_topology,
        sequence_config,
    )
    labels_df = labels_df.sort_values('row_id')
    df_pm['future_label'] = labels_df['label'].to_numpy(dtype=int)
    future_positives = int(df_pm['future_label'].sum())
    print(f"ℹ️ Total de nodos con label futuro=1 (γ={sequence_config.guard_band_minutes}m, H={sequence_config.horizon_minutes}m): {future_positives}")

    sequence_mask_np = np.zeros(len(df_pm), dtype=bool)
    if seq_index.target_rows.size:
        sequence_mask_np[seq_index.target_rows] = True
    sequence_mask = torch.from_numpy(sequence_mask_np)

    # Preparar snapshot de variables crudas clave para aristas espaciales
    def _ensure_or_compute_globals(df: pd.DataFrame):
        # flujo_total
        if 'flujo_total' in df.columns:
            q = df['flujo_total']
        else:
            q_cols = [f'flujo_{c}' for c in [1,2,3] if f'flujo_{c}' in df.columns]
            q = df[q_cols].sum(axis=1) if q_cols else pd.Series(np.zeros(len(df)))
        # densidad_total
        if 'densidad_total' in df.columns:
            k = df['densidad_total']
        else:
            k_cols = [f'densidad_{c}' for c in [1,2,3] if f'densidad_{c}' in df.columns]
            k = df[k_cols].sum(axis=1) if k_cols else pd.Series(np.zeros(len(df)))
        # v_armonica
        if 'v_armonica' in df.columns:
            v_h = df['v_armonica']
        else:
            v_h = q / (k.replace(0, np.nan))
            v_h = v_h.fillna(0.0)
        # prop_pesados
        if 'prop_pesados' in df.columns:
            pi = df['prop_pesados']
        else:
            f2 = df['flujo_2'] if 'flujo_2' in df.columns else 0.0
            pi = f2 / (q + 1e-12)
        return q.astype(float), k.astype(float), v_h.astype(float), pi.astype(float)

    q_raw, k_raw, vhar_raw, pi_raw = _ensure_or_compute_globals(df_pm)
    raw_globals = pd.DataFrame({
        portico_col: df_pm[portico_col].values,
        'ts_min': df_pm['ts_min'].values,
        'q_total_raw': q_raw.values,
        'k_total_raw': k_raw.values,
        'v_armonica_raw': vhar_raw.values,
        'prop_pesados_raw': pi_raw.values,
    })

    # Check for NaNs and Infs before normalization
    if DEBUG:
        print("DEBUG: Checking for NaNs and Infs in features before normalization...")
        print(df_pm[feature_cols].isnull().sum())
        print(np.isinf(df_pm[feature_cols]).sum())

    # 3a.2) Calcular mu y sigma para la normalización Z-score
    if np.isfinite(train_ts_cutoff):
        train_stats_mask = df_pm['ts_min'] <= train_ts_cutoff
    else:
        train_stats_mask = np.ones(len(df_pm), dtype=bool)
    if not np.any(train_stats_mask):
        train_stats_mask = np.ones(len(df_pm), dtype=bool)
    train_stats_df = df_pm.loc[train_stats_mask, feature_cols]
    mu = train_stats_df.mean().values
    sigmasigma = train_stats_df.std(ddof=0).fillna(0.0).values + 1e-9  # Evitar división por cero
    norm_stats_path = _save_normalization_stats(feature_cols, mu, sigmasigma, train_ts_cutoff, val_ts_cutoff)
    print(f"ℹ️ Estadísticas de normalización guardadas en: {norm_stats_path}")
    normalization_metadata = {
        "feature_names": feature_cols,
        "mean": mu.tolist(),
        "std": sigmasigma.tolist(),
        "train_ts_cutoff": None if not np.isfinite(train_ts_cutoff) else float(train_ts_cutoff),
        "val_ts_cutoff": None if not np.isfinite(val_ts_cutoff) else float(val_ts_cutoff),
        "stats_path": norm_stats_path,
    }

    # 3a.3) Normalizar las features
    feat_mat = (df_pm[feature_cols].values - mu) / sigmasigma
    if np.isnan(feat_mat).any():
        print("⚠️  Advertencia: Se encontraron NaNs en las features normalizadas. Se reemplazarán por 0.")
        feat_mat = np.nan_to_num(feat_mat)
    df_pm[feature_cols] = feat_mat
    
    # 2a) Preguntar por entrenamiento con severidad
    use_severity = False

    if df_pm.empty:
        print("\n❌ Error: No se encontraron datos de flujo en las ventanas de tiempo previas a los accidentes.")
        return None

    # --- 4) Vectorización: Creación de Nodos y Aristas --- #
    print("▶️ Vectorizando la construcción de nodos y aristas...")

    # 4.1 - Mapeo de (pórtico, ts_min) a un índice numérico único
    # Asegurar tipos consistentes para 'ts_min'
    try:
        df_pm['ts_min'] = df_pm['ts_min'].astype(int)
    except Exception:
        pass
    pm_map = df_pm.set_index([portico_col, 'ts_min'])['node_idx'].to_dict()

    # 4.2 - Features de Nodos PM
    float_type = getattr(torch, FLOAT)
    pm_feats = torch.tensor(df_pm[feature_cols].values, dtype=float_type)
    pm_y = torch.tensor(df_pm['future_label'].to_numpy(), dtype=torch.long)

    # 4.3 - Marcar nodos PM afectados por accidentes
    # Detectar el DT efectivo del archivo de features cargado (puede diferir de DT_MINUTES)
    try:
        # Diferencia modal positiva por pórtico; fallback al mínimo global positivo
        diffs = []
        for _, g in df_pm.groupby(portico_col):
            ts_sorted = np.sort(g['ts_min'].unique())
            d = np.diff(ts_sorted)
            d = d[d > 0]
            if d.size:
                diffs.extend(d.tolist())
        if diffs:
            # Elegir el paso más común
            vals, counts = np.unique(np.array(diffs, dtype=int), return_counts=True)
            dt_feat_minutes = int(vals[np.argmax(counts)])
        else:
            dt_feat_minutes = int(DT_MINUTES)
    except Exception:
        dt_feat_minutes = int(DT_MINUTES)
    if dt_feat_minutes != int(DT_MINUTES):
        print(f"ℹ️ Detectado DT en features distinto a config: {dt_feat_minutes} min (config={DT_MINUTES}). Alineando construcción a {dt_feat_minutes} min.")
    else:
        print(f"ℹ️ DT efectivo de features: {dt_feat_minutes} min.")

    # Marcar nodos PM que son accidentes y asignar etiquetas 'y'
    is_accident_pm = torch.zeros(len(df_pm), dtype=torch.bool)
    if not affected_pms.empty:
        affected_indices = affected_pms['node_idx'].unique()
        is_accident_pm[affected_indices] = True
    print(f"    · Accidentes totales={len(df_acc)}, mapeados a nodos PM={affected_pms['node_idx'].nunique() if not affected_pms.empty else 0}")


    # --- 5) Selección y Construcción de Aristas ---
    #print("\n▶️ Selección de tipos de aristas a construir...")
    build_temporal = 's' #in input("  - ¿Construir aristas 'temporales'? [s/n]: ").strip().lower()
    build_spatial = 's' #in input("  - ¿Construir aristas 'espaciales'? [s/n]: ").strip().lower()
    build_st_fwd = False #in input("  - ¿Construir aristas 'st_fwd' (espacio-temporales)? [s/n]: ").strip().lower()
    build_spatial_back = False #in input("  - ¿Construir aristas 'spatial_back' (inversas)? [s/n]: ").strip().lower()

    # Inicializar listas de aristas
    temporal_src, temporal_dst, temporal_attr = [], [], []
    spatial_src, spatial_dst, spatial_attr = [], [], []
    stfwd_src, stfwd_dst, stfwd_attr = [], [], []
    spatial_back_src, spatial_back_dst, spatial_back_attr = [], [], []

    if build_temporal:
        print("  - Calculando aristas 'temporales' (vectorizado)...")
        df_pm['next_ts_min'] = df_pm['ts_min'] + int(dt_feat_minutes)
        temporal_edges = pd.merge(
            df_pm[[portico_col, 'ts_min', 'node_idx', 'next_ts_min']],
            df_pm[[portico_col, 'ts_min', 'node_idx']],
            left_on=[portico_col, 'next_ts_min'],
            right_on=[portico_col, 'ts_min'],
            suffixes=('_src', '_dst')
        )
        temporal_src = temporal_edges['node_idx_src'].tolist()
        temporal_dst = temporal_edges['node_idx_dst'].tolist()
        print(f"    · Aristas temporales: {len(temporal_src)}")
        delta_norm = feat_mat[temporal_dst] - feat_mat[temporal_src]
        # Alinear dimensión con aristas espaciales si corresponde
        EXTRA_SPATIAL = 6  # debe coincidir con el usado en espaciales
        temporal_attr = np.concatenate([np.array(delta_norm, dtype=np.float32), np.zeros((len(temporal_src), EXTRA_SPATIAL), dtype=np.float32)], axis=1)

    # Para las aristas espaciales y st_fwd, necesitamos las secuencias de pórticos
    if build_spatial or build_st_fwd:
        print("  - Generando secuencias de pórticos para aristas espaciales/st_fwd...")
        portico_sequences = []
        df_port_sorted = df_port.sort_values(by=["calzada", "eje", "orden"], ascending=[False, True, True])
        for (calzada, eje), group in df_port_sorted.groupby(["calzada", "eje"]):
            sorted_porticos = group["portico"].tolist()
            kms = group['km'].astype(float).tolist() if 'km' in group.columns else [np.nan]*len(sorted_porticos)
            if len(sorted_porticos) > 1:
                for i in range(len(sorted_porticos) - 1):
                    dkm = abs(float(kms[i+1]) - float(kms[i])) if np.isfinite(kms[i]) and np.isfinite(kms[i+1]) else np.nan
                    portico_sequences.append((sorted_porticos[i], sorted_porticos[i+1], dkm))

        df_seq = pd.DataFrame(portico_sequences, columns=['portico_src', 'portico_dst', 'dist_km'])
        
        if len(df_seq) == 0:
            print("[ADVERTENCIA] No se generaron secuencias de pórticos. Las aristas espaciales y st_fwd no se crearán.")
        else:
            print(f"[INFO] Se generaron {len(df_seq)} secuencias de pórticos.")

            # Dimensión extra de features de arista espacial que agregamos (además del delta de nodos)
            EXTRA_SPATIAL = 6  # grad_q, grad_k, grad_v, shock_speed, shock_indicator, dist_km

            if build_spatial:
                print("  - Calculando aristas 'espaciales' (vectorizado)...")
                spatial_edges = pd.merge(
                    df_pm.rename(columns={portico_col: 'portico_src', 'node_idx': 'node_idx_src'}),
                    df_seq,
                    on='portico_src'
                )
                spatial_edges = pd.merge(
                    spatial_edges,
                    df_pm.rename(columns={portico_col: 'portico_dst', 'node_idx': 'node_idx_dst'}),
                    left_on=['ts_min', 'portico_dst'],
                    right_on=['ts_min', 'portico_dst'],
                    how='inner' # Usar inner join para asegurar que ambos nodos existan en el mismo ts
                )
                spatial_src = spatial_edges['node_idx_src'].tolist()
                spatial_dst = spatial_edges['node_idx_dst'].tolist()
                print(f"    · Aristas espaciales: {len(spatial_src)}")
                delta_norm_spatial = feat_mat[spatial_dst] - feat_mat[spatial_src]
                # Unir valores crudos para gradientes y shock
                src_vals = raw_globals.rename(columns={
                    portico_col: 'portico_src',
                    'q_total_raw': 'q_src',
                    'k_total_raw': 'k_src',
                    'v_armonica_raw': 'v_src',
                    'prop_pesados_raw': 'pi_src'
                })
                dst_vals = raw_globals.rename(columns={
                    portico_col: 'portico_dst',
                    'q_total_raw': 'q_dst',
                    'k_total_raw': 'k_dst',
                    'v_armonica_raw': 'v_dst',
                    'prop_pesados_raw': 'pi_dst'
                })
                se = spatial_edges.merge(src_vals, on=['portico_src', 'ts_min'], how='left').merge(dst_vals, on=['portico_dst', 'ts_min'], how='left')
                dkm = se['dist_km'].to_numpy(dtype=float)
                dq = (se['q_dst'] - se['q_src']).to_numpy(dtype=float)
                dk = (se['k_dst'] - se['k_src']).to_numpy(dtype=float)
                dv = (se['v_dst'] - se['v_src']).to_numpy(dtype=float)
                grad_q = dq / (dkm + 1e-9)
                grad_k = dk / (dkm + 1e-9)
                grad_v = dv / (dkm + 1e-9)
                shock_speed = dq / (dk + 1e-9)
                shock_indicator = np.maximum(0.0, dk) * np.maximum(0.0, -dq)
                extra = np.stack([grad_q, grad_k, grad_v, shock_speed, shock_indicator, np.nan_to_num(dkm, nan=0.0)], axis=1)
                spatial_attr = np.concatenate([np.array(delta_norm_spatial, dtype=np.float32), extra.astype(np.float32)], axis=1)

                if build_spatial_back:
                    # Invertir para `spatial_back`
                    print("  - Calculando aristas 'spatial_back' (inversas)...")
                    spatial_back_src = spatial_dst
                    spatial_back_dst = spatial_src
                    spatial_back_attr = -spatial_attr

            if build_st_fwd:
                print("  - Calculando aristas 'st_fwd' (vectorizado)...")
                if 'next_ts_min' not in df_pm.columns:
                     df_pm['next_ts_min'] = df_pm['ts_min'] + int(dt_feat_minutes)
                stfwd_edges = pd.merge(
                    df_pm.rename(columns={portico_col: 'portico_src', 'node_idx': 'node_idx_src'}),
                    df_seq,
                    on='portico_src'
                )
                stfwd_edges = pd.merge(
                    stfwd_edges,
                    df_pm.rename(columns={portico_col: 'portico_dst', 'node_idx': 'node_idx_dst'}),
                    left_on=['next_ts_min', 'portico_dst'],
                    right_on=['ts_min', 'portico_dst']
                )
                stfwd_src = stfwd_edges['node_idx_src'].tolist()
                stfwd_dst = stfwd_edges['node_idx_dst'].tolist()
                print(f"    · Aristas st_fwd: {len(stfwd_src)}")
                delta_norm_stfwd = feat_mat[stfwd_dst] - feat_mat[stfwd_src]
                stfwd_attr = np.concatenate([np.array(delta_norm_stfwd, dtype=np.float32), np.zeros((len(stfwd_src), EXTRA_SPATIAL), dtype=np.float32)], axis=1)

    # --- 7) Ensamblar HeteroData y guardar ------------------------------------- #
    data = HeteroData()
    data["pm"].x = pm_feats
    data["pm"].y = pm_y
    data["pm"].is_accident_pm = is_accident_pm

    def _add(et, src, dst, attr):
        if not src: return
        data[("pm", et, "pm")].edge_index = torch.tensor([src, dst], dtype=torch.long)
        data[("pm", et, "pm")].edge_attr  = torch.tensor(attr, dtype=float_type)

    if build_spatial:
        _add("spatial", spatial_src, spatial_dst, spatial_attr)
    if build_temporal:
        _add("temporal", temporal_src, temporal_dst, temporal_attr)
    if build_spatial_back:
        _add("spatial_back", spatial_back_src, spatial_back_dst, spatial_back_attr)
    if build_st_fwd:
        _add("st_fwd",  stfwd_src,    stfwd_dst,    stfwd_attr)


    # --- NEW: Limpieza de Aristas ---
    print("\n▶️ Limpiando y validando aristas...")
    for store in data.edge_stores:
        num_src = data[store._key[0]].num_nodes
        num_dst = data[store._key[2]].num_nodes
        _safe_clean_edges(store, num_src, num_dst)

    # Crear el objeto pm_index para guardarlo
    pm_rev_map = {v: k for k, v in pm_map.items()}
    pm_index = PMIndex(pm_map, pm_rev_map)

    # --- 8) Crear y guardar máscaras de datos (División temporal basada en PM is_accident_pm) ---
    print("▶️ Creando máscaras de entrenamiento/validación/prueba (división temporal fija)...")
    num_pm_nodes = data['pm'].num_nodes
    if num_pm_nodes > 0:
        ts_array = df_pm['ts_min'].to_numpy()
        if np.isfinite(train_ts_cutoff):
            train_mask_np = ts_array <= train_ts_cutoff
            if np.isfinite(val_ts_cutoff):
                val_mask_np = (ts_array > train_ts_cutoff) & (ts_array <= val_ts_cutoff)
                test_mask_np = ts_array > val_ts_cutoff
            else:
                val_mask_np = np.zeros_like(ts_array, dtype=bool)
                test_mask_np = np.zeros_like(ts_array, dtype=bool)
        else:
            train_mask_np = np.ones_like(ts_array, dtype=bool)
            val_mask_np = np.zeros_like(ts_array, dtype=bool)
            test_mask_np = np.zeros_like(ts_array, dtype=bool)

        train_mask = torch.from_numpy(train_mask_np.astype(np.bool_))
        val_mask = torch.from_numpy(val_mask_np.astype(np.bool_))
        test_mask = torch.from_numpy(test_mask_np.astype(np.bool_))

        sequence_mask = sequence_mask.to(train_mask.device)
        train_mask = train_mask & sequence_mask
        val_mask = val_mask & sequence_mask
        test_mask = test_mask & sequence_mask

        data['pm'].train_mask = train_mask
        data['pm'].val_mask = val_mask
        data['pm'].test_mask = test_mask

        pos_train = int((data['pm'].is_accident_pm & train_mask).sum())
        pos_val = int((data['pm'].is_accident_pm & val_mask).sum())
        pos_test = int((data['pm'].is_accident_pm & test_mask).sum())

        if np.isfinite(train_ts_cutoff):
            try:
                train_str = datetime.fromtimestamp(int(train_ts_cutoff) * 60).strftime('%Y-%m-%d %H:%M')
            except Exception:
                train_str = str(train_ts_cutoff)
        else:
            train_str = '∞'
        if np.isfinite(val_ts_cutoff):
            try:
                val_str = datetime.fromtimestamp(int(val_ts_cutoff) * 60).strftime('%Y-%m-%d %H:%M')
            except Exception:
                val_str = str(val_ts_cutoff)
        else:
            val_str = '∞'

        print(f"ℹ️ Corte temporal train: <= {train_str}")
        print(f"ℹ️ Corte temporal val: <= {val_str}")
        print(f"  - Nodos 'pm' divididos por tiempo en: {train_mask.sum()} (train), {val_mask.sum()} (val), {test_mask.sum()} (test)")
        print(f"  - Positivos directos (accidentes) por split: train={pos_train}, val={pos_val}, test={pos_test}")
        future_pos_train = int((pm_y.bool() & train_mask).sum())
        future_pos_val = int((pm_y.bool() & val_mask).sum())
        future_pos_test = int((pm_y.bool() & test_mask).sum())
        print(f"  - Positivos futuros por split: train={future_pos_train}, val={future_pos_val}, test={future_pos_test}")
        print(f"  - Nodos con secuencia válida (mask): {sequence_mask.sum()}")
    else:
        data['pm'].train_mask = torch.zeros(0, dtype=torch.bool)
        data['pm'].val_mask = torch.zeros(0, dtype=torch.bool)
        data['pm'].test_mask = torch.zeros(0, dtype=torch.bool)
        sequence_mask = torch.zeros(0, dtype=torch.bool)

    data['pm'].sequence_mask = sequence_mask

    # --- Mark nodes neighborhoods for binary classification ---
    print(f"▶️ Marcando nodos PM a {ACC_NEIGHBORHOODS} vecindarios de accidentes (para clasificación binaria)...")
    
    # Collect all directly triggered PM nodes
    current_influenced_nodes = set(affected_pms['node_idx'].unique())
    all_influenced_nodes = set(current_influenced_nodes) # Start with direct triggers

    # Get edge indices for temporal and spatial relations (reversed for backward traversal)
    # Note: edge_index is (src, dst). For predecessors, we look for dst in the current set.
    temporal_edges = data[('pm', 'temporal', 'pm')].edge_index if ('pm', 'temporal', 'pm') in data.edge_types else None
    spatial_edges = data[('pm', 'spatial', 'pm')].edge_index if ('pm', 'spatial', 'pm') in data.edge_types else None

    for step in range(ACC_NEIGHBORHOODS): # Steps backward
        newly_found_nodes = set()
        
        # Traverse temporal predecessors
        if temporal_edges is not None:
            for i in range(temporal_edges.shape[1]):
                src_node, dst_node = temporal_edges[0, i].item(), temporal_edges[1, i].item()
                if dst_node in current_influenced_nodes and src_node not in all_influenced_nodes:
                    newly_found_nodes.add(src_node)
        
        # Traverse spatial predecessors
        if spatial_edges is not None:
            for i in range(spatial_edges.shape[1]):
                src_node, dst_node = spatial_edges[0, i].item(), spatial_edges[1, i].item()
                if dst_node in current_influenced_nodes and src_node not in all_influenced_nodes:
                    newly_found_nodes.add(src_node)
        
        if not newly_found_nodes: # No new nodes found, stop traversal
            break
        
        current_influenced_nodes = newly_found_nodes # For the next step, consider only newly found nodes
        all_influenced_nodes.update(newly_found_nodes)

    # Mark all nodes in the final set as 1 in pm_y
    # Ensure original trigger nodes (if any) retain their '1' label if already set.
    # This will only change 0s to 1s for newly influenced nodes.
    accident_scope = torch.zeros(len(df_pm), dtype=torch.bool)
    for node_idx in all_influenced_nodes:
        accident_scope[node_idx] = True
    data["pm"].accident_scope = accident_scope
    print(f"  - Se marcaron {len(all_influenced_nodes)} nodos PM como influenciados por accidentes (atributo accident_scope).")

    # Etiqueta de período temporal MM-YYYY_to_MM-YYYY basada en ts_min de df_pm
    try:
        ts_series = pd.to_datetime(df_pm['ts_min'], unit='m')
        if len(ts_series) > 0:
            start_dt = pd.to_datetime(ts_series.min())
            end_dt = pd.to_datetime(ts_series.max())
            period_tag = f"{start_dt.strftime('%m-%Y')}_to_{end_dt.strftime('%m-%Y')}"
        else:
            period_tag = "unknown_period"
    except Exception:
        period_tag = "unknown_period"

    timestamp = datetime.now().strftime('%d%m%Y_%H%M')
    filename = f"highway_graph_{period_tag}_{timestamp}.pt"
    output_path = os.path.join(RESULTADOS_DIR, filename)

    # --- Post-build Invariants ---
    print("\n▶️ Verificando invariantes post-construcción...")
    assert data['pm'].x.shape[0] == len(data['pm'].train_mask), "Invariante fallido: pm.x y train_mask tienen diferente número de nodos."
    assert data['pm'].x.shape[0] == len(data['pm'].val_mask), "Invariante fallido: pm.x y val_mask tienen diferente número de nodos."
    assert data['pm'].x.shape[0] == len(data['pm'].test_mask), "Invariante fallido: pm.x y test_mask tienen diferente número de nodos."

    for edge_type in data.edge_types:
        if hasattr(data[edge_type], 'edge_attr'):
            assert data[edge_type].edge_index.shape[1] == data[edge_type].edge_attr.shape[0], f"Invariante fallido para {edge_type}: edge_index y edge_attr tienen diferente número de aristas."
    print("✅ Invariantes verificados con éxito.")

    torch.save({
        "data": data,
        "pm_index": pm_index,
        "mu": mu,
        "sigma": sigmasigma,
        "feature_cols": feature_cols,
        "use_severity": use_severity,
        "normalization": normalization_metadata,
        "train_ts_cutoff": train_ts_cutoff,
        "val_ts_cutoff": val_ts_cutoff,
        "normalization": normalization_metadata,
        "train_ts_cutoff": train_ts_cutoff,
        "val_ts_cutoff": val_ts_cutoff,
        "sequence_index": seq_index,
        "sequence_config": sequence_config,
        "snapshot_topology": {
            "edge_index": snapshot_topology.edge_index if snapshot_topology else None,
            "node_order": snapshot_topology.node_order if snapshot_topology else None,
        } if snapshot_topology else None,
    }, output_path)
    print(f"✅ Grafo guardado en -> {filename}{data}")
    return output_path
