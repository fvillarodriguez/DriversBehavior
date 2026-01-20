#!/usr/bin/env python
"""
utils.py
========
Funciones auxiliares para la carga y procesamiento de datos de pórticos, flujos y accidentes.
"""
from typing import Optional, Union, List, Tuple
import pandas as pd
import math
import os, glob
import io
import time
import numpy as np
import re
import fnmatch
from tqdm import tqdm
from pathlib import Path
try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:
    tk = None
    filedialog = None
import networkx as nx
from datetime import datetime
try:
    import matplotlib
    import matplotlib.pyplot as plt
except Exception:
    matplotlib = None
    plt = None
import torch
try:
    import seaborn as sns
except Exception:
    sns = None
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
except Exception:
    lowess = None
from sklearn.ensemble import RandomForestClassifier
# --------------------------------------------------------------------------- #
# CONFIGURACIÓN Y CONSTANTES
# --------------------------------------------------------------------------- #
from src.config import (
    SEED,
    RESULTADOS_DIR,
    DT_MINUTES,
    TRAIN_RATIO,
    VAL_RATIO,
    DEBUG
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, RFE, SequentialFeatureSelector
from torch_geometric.data import HeteroData


# ===================================================================
# CACHE GLOBAL DE PÓRTICOS
# ===================================================================
_porticos_df_cache = None

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def _require_matplotlib() -> None:
    if matplotlib is None or plt is None:
        raise ImportError(
            "matplotlib no esta disponible. Instala con `pip install matplotlib`."
        )


def _require_seaborn() -> None:
    if sns is None:
        raise ImportError(
            "seaborn no esta disponible. Instala con `pip install seaborn`."
        )


def _require_statsmodels() -> None:
    if lowess is None:
        raise ImportError(
            "statsmodels no esta disponible. Instala con `pip install statsmodels`."
        )

def _count_lines(path):
    """Cuenta las líneas de un archivo de forma eficiente."""
    with open(path, 'rb') as f:
        buf_gen = iter(lambda: f.read(1024 * 1024), b'')
        return sum(buf.count(b'\n') for buf in buf_gen)

# Carpeta para leer csv a procesar
DATA_FOLDER = "Datos"
os.makedirs(DATA_FOLDER, exist_ok=True)

# Carpeta para exportar resultados
DATA_FOLDER = "Resultados"
os.makedirs(DATA_FOLDER, exist_ok=True)

# Marcadores que consideraremos como valores faltantes
MISSING_MARKERS = ["NULL", "N/A", "NA", "NaN", "", " "]

def choose_file(start_folder):
    """
    Permite navegar por carpetas para seleccionar un archivo CSV.

    Args:
        start_folder (str): Carpeta inicial donde comenzar la búsqueda.
    Returns:
        str o None: Ruta del archivo seleccionado o None si se cancela.
    """
    start_folder = os.path.abspath(start_folder)
    current_folder = start_folder

    while True:
        print(f"\n📁 Carpeta actual: {current_folder}")
        items = os.listdir(current_folder)
        dirs = sorted([f for f in items if os.path.isdir(os.path.join(current_folder, f))])
        files = sorted([f for f in items if os.path.isfile(os.path.join(current_folder, f)) and f.lower().endswith(".csv")])
        opciones = dirs + files

        if current_folder == start_folder:
            print("  [0] 🔙 Volver/Salir")
            offset = 1
        else:
            print("  [0] 🔙 Subir un nivel")
            offset = 1

        for i, name in enumerate(opciones):
            icon = "📂" if name in dirs else "📄"
            print(f"  [{i+offset}] {icon} {name}")

        choice = input("Seleccione una opción: ").strip()
        if choice == "0":
            if current_folder == start_folder:
                print("Cancelado.")
                return None
            else:
                current_folder = os.path.dirname(current_folder)
                continue

        try:
            idx = int(choice) - offset
            if 0 <= idx < len(opciones):
                selected = opciones[idx]
                selected_path = os.path.join(current_folder, selected)
                if os.path.isdir(selected_path):
                    current_folder = selected_path
                else:
                    print(f"Archivo seleccionado: {selected_path}\n")
                    return selected_path
            else:
                print("Selección fuera de rango.")
        except ValueError:
            print("Entrada inválida.")

def choose_file_filtered(start_folder, patterns):
    """
    Igual a choose_file, pero solo muestra archivos CSV cuyo nombre coincida con
    uno o varios patrones (glob) dados, p.ej. 'FLUJO*.csv' o 'Eventos*.csv'.
    La comparación es case-insensitive y permite navegar subcarpetas.

    Args:
        start_folder (str): Carpeta inicial.
        patterns (str | list[str]): Patron(es) glob para el nombre del archivo (no la ruta).
    Returns:
        str | None: Ruta del archivo seleccionado o None si se cancela.
    """
    if isinstance(patterns, str):
        patterns = [patterns]
    patterns_l = [p.lower() for p in patterns]

    start_folder = os.path.abspath(start_folder)
    current_folder = start_folder

    while True:
        print(f"\n📁 Carpeta actual: {current_folder}")
        items = os.listdir(current_folder)
        dirs = sorted([f for f in items if os.path.isdir(os.path.join(current_folder, f))])
        # Filtrar solo CSV que matcheen algún patrón
        files = []
        for f in items:
            fp = os.path.join(current_folder, f)
            if os.path.isfile(fp) and f.lower().endswith('.csv'):
                name_l = f.lower()
                if any(fnmatch.fnmatch(name_l, p) for p in patterns_l):
                    files.append(f)
        files = sorted(files)

        opciones = dirs + files

        if current_folder == start_folder:
            print("  [0] 🔙 Volver/Salir")
            offset = 1
        else:
            print("  [0] 🔙 Subir un nivel")
            offset = 1

        if not files and not dirs:
            print("  (No hay carpetas ni archivos que coincidan con el patrón.)")

        print(f"  ↳ Mostrando solo archivos: {', '.join(patterns)}")
        for i, name in enumerate(opciones):
            icon = "📂" if name in dirs else "📄"
            print(f"  [{i+offset}] {icon} {name}")

        choice = input("Seleccione una opción: ").strip()
        if choice == "0":
            if current_folder == start_folder:
                print("Cancelado.")
                return None
            else:
                current_folder = os.path.dirname(current_folder)
                continue

        try:
            idx = int(choice) - offset
            if 0 <= idx < len(opciones):
                selected = opciones[idx]
                selected_path = os.path.join(current_folder, selected)
                if os.path.isdir(selected_path):
                    current_folder = selected_path
                else:
                    print(f"Archivo seleccionado: {selected_path}\n")
                    return selected_path
            else:
                print("Selección fuera de rango.")
        except ValueError:
            print("Entrada inválida.")

def read_csv_with_progress(path_csv, chunksize=100000, sep=None):
    """
    Lee un CSV en chunks mostrando una barra de progreso. Intenta leer con UTF-8
    y si falla, reintenta con 'latin-1'.
    Si el separador `sep` no se especifica, intenta autodetectarlo.
    """
    
    def _read_chunks_with_progress(path, encoding, separator, chunk_size):
        total_size = os.path.getsize(path)
        chunks = []
        with open(path, 'rb') as f:
            f_text = io.TextIOWrapper(f, encoding=encoding, errors='strict')
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=f'Cargando CSV ({encoding})') as pbar:
                reader = pd.read_csv(f_text, sep=separator, chunksize=chunk_size, low_memory=False)
                last_position = f.tell()
                for chunk in reader:
                    chunks.append(chunk)
                    current_position = f.tell()
                    pbar.update(current_position - last_position)
                    last_position = current_position
        return pd.concat(chunks, ignore_index=True)

    # --- Separator detection ---
    detected_sep = sep
    if detected_sep is None:
        try:
            with open(path_csv, 'r', encoding='utf-8') as f:
                sample = f.read(4096)
        except UnicodeDecodeError:
            with open(path_csv, 'r', encoding='latin-1') as f:
                sample = f.read(4096)
        
        import csv
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=',;')
            detected_sep = dialect.delimiter
            print(f"Separador detectado: '{detected_sep}'")
        except (csv.Error, UnicodeDecodeError):
            detected_sep = ','
            print(f"No se pudo detectar el separador, usando '{detected_sep}' por defecto.")

    # --- Read with encoding fallback ---
    try:
        print("Intentando leer con codificación 'utf-8'...")
        return _read_chunks_with_progress(path_csv, 'utf-8', detected_sep, chunksize)
    except UnicodeDecodeError:
        print("⚠️  Lectura con UTF-8 fallida. Reintentando con 'latin-1'...")
        return _read_chunks_with_progress(path_csv, 'latin-1', detected_sep, chunksize)

def select_flujos_csv_path():
    """
    Selecciona el archivo de flujos (FLUJO*.csv) desde la carpeta 'Datos'.
    Si hay exactamente uno en el nivel superior, lo elige automáticamente.

    Returns:
        str | None: ruta del archivo o None si se cancela.
    """
    base_dir = "Datos"
    patterns = ["flujo*.csv"]
    try:
        top_files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
    except FileNotFoundError:
        top_files = []
    matches_top = [
        os.path.join(base_dir, f)
        for f in top_files
        if f.lower().endswith('.csv') and any(fnmatch.fnmatch(f.lower(), p) for p in patterns)
    ]
    if len(matches_top) == 1:
        file_path = matches_top[0]
        print(f"🔎 Detectado único archivo de flujos en '{base_dir}'. Cargando automáticamente → {file_path}")
        return file_path
    else:
        return choose_file_filtered(base_dir, patterns)

def buscar_columna(df, nombre_esperado):
    """
    Busca una columna que se parezca a 'nombre_esperado'. Si no se encuentra, solicita al usuario seleccionar.

    Args:
        df (pd.DataFrame): DataFrame donde buscar.
        nombre_esperado (str): Nombre aproximado de la columna.
    Returns:
        str: Nombre de la columna encontrada.
    """
    columnas = list(df.columns)
    if nombre_esperado in columnas:
        return nombre_esperado
    columnas_norm = {col.strip().lower().replace(" ", ""): col for col in columnas}
    clave = nombre_esperado.strip().lower().replace(" ", "")
    if clave in columnas_norm:
        return columnas_norm[clave]
    print(f"No se encontró la columna '{nombre_esperado}'. Columnas disponibles:")
    for i, col in enumerate(columnas):
        print(f"  [{i}] {col}")
    idx = input(f"Ingrese el número correspondiente a la columna que represente '{nombre_esperado}': ").strip()
    try:
        idx = int(idx)
        return columnas[idx]
    except Exception:
        raise ValueError("Selección inválida. Proceso abortado.")

def limpiar_pantalla():
    os.system('cls' if os.name == 'nt' else 'clear')

# =============================================================================
# FUNCIONES DE CARGA DE DATOS
# =============================================================================
def find_candidate_porticos(acc_km, porticos_df, eje, calzada):
    """
    Dado el km del accidente, el DataFrame de pórticos y los valores de eje y calzada,
    retorna un diccionario con:
      - "posterior": el pórtico inmediatamente anterior en el orden de viaje (último por el que pasó)
      - "cercano": el pórtico inmediatamente siguiente (el siguiente más cercano) 
    
    Se filtra usando las columnas "Eje" y "Calzada" (comparando en mayúsculas)
    y se ordena por "orden". Se convierte "km" a float.
    """
    try:
        accident_km = float(str(acc_km).replace(",", "."))
    except Exception as e:
        raise ValueError("Valor de km del accidente inválido.") from e

    df = porticos_df.copy()
    df["Eje_norm"] = df["eje"].astype(str).str.strip().str.upper()
    df["Calzada_norm"] = df["calzada"].astype(str).str.strip().str.upper()
    eje_filtro = str(eje).strip().upper()
    calzada_filtro = str(calzada).strip().upper()
    df_filtrado = df[(df["Eje_norm"] == eje_filtro) & (df["Calzada_norm"] == calzada_filtro)].copy()
    
    if df_filtrado.empty:
        return {"posterior": None, "cercano": None}
    
    df_filtrado["km_num"] = df_filtrado["km"].astype(str).str.replace(",", ".").astype(float)
    df_filtrado = df_filtrado.sort_values(by="orden", ascending=True).reset_index(drop=True)
    
    if len(df_filtrado) >= 2:
        first_km = df_filtrado.loc[0, "km_num"]
        second_km = df_filtrado.loc[1, "km_num"]
        direction = "asc" if first_km <= second_km else "desc"
    else:
        return {"posterior": None, "cercano": df_filtrado.iloc[0].to_dict()}
    
    posterior_candidate = None
    cercano_candidate = None
    found = False
    for i in range(len(df_filtrado) - 1):
        km_current = df_filtrado.loc[i, "km_num"]
        km_next = df_filtrado.loc[i+1, "km_num"]
        if direction == "asc":
            if km_current <= accident_km <= km_next:
                posterior_candidate = df_filtrado.loc[i].to_dict()
                cercano_candidate = df_filtrado.loc[i+1].to_dict()
                found = True
                break
        else:
            if km_current >= accident_km >= km_next:
                posterior_candidate = df_filtrado.loc[i].to_dict()
                cercano_candidate = df_filtrado.loc[i+1].to_dict()
                found = True
                break
    if not found:
        if direction == "asc":
            if accident_km < df_filtrado.loc[0, "km_num"]:
                posterior_candidate = None
                cercano_candidate = df_filtrado.loc[0].to_dict()
            elif accident_km > df_filtrado.loc[len(df_filtrado)-1, "km_num"]:
                posterior_candidate = df_filtrado.loc[len(df_filtrado)-1].to_dict()
                cercano_candidate = None
        else:
            if accident_km > df_filtrado.loc[0, "km_num"]:
                posterior_candidate = None
                cercano_candidate = df_filtrado.loc[0].to_dict()
            elif accident_km < df_filtrado.loc[len(df_filtrado)-1, "km_num"]:
                posterior_candidate = df_filtrado.loc[len(df_filtrado)-1].to_dict()
                cercano_candidate = None
    return {"posterior": posterior_candidate, "cercano": cercano_candidate}

def load_porticos():
    """
    Carga el CSV de pórticos con las columnas:
      cod_portico, Km, Calzada, Orden, Eje
    Se convierte 'Km' a float y 'Orden' a numérico.

    Returns:
        pd.DataFrame: DataFrame con columnas renombradas a: portico, km, calzada, orden, eje.
    """

    global _porticos_df_cache

    # Intento 1: carga automática si existe 'Porticos.csv' en la carpeta 'Datos'
    auto_dir = "Datos"
    auto_path = None
    try:
        if os.path.isdir(auto_dir):
            for fname in os.listdir(auto_dir):
                if fname.lower() == "porticos.csv":
                    auto_path = os.path.join(auto_dir, fname)
                    break
    except Exception:
        auto_path = None

    if auto_path is not None and os.path.isfile(auto_path):
        print(f"🔎 Detectado 'Porticos.csv' en '{auto_dir}'. Cargando automáticamente → {auto_path}")
        file_path = auto_path
    else:
        # Intento 2: selección manual
        file_path = choose_file("Datos")
        if file_path is None:
            return None

    print("Cargando información de pórticos...")
    df = read_csv_with_progress(file_path, sep=';')
    
    cod_portico_col = buscar_columna(df, "cod_portico")
    km_col = buscar_columna(df, "Km")
    calzada_col = buscar_columna(df, "Calzada")
    orden_col = buscar_columna(df, "Orden")
    eje_col = buscar_columna(df, "Eje")
    autopista_col = buscar_columna(df, "Autopista")
    
    df[km_col] = df[km_col].astype(str).str.replace(",", ".")
    df[km_col] = pd.to_numeric(df[km_col], errors="coerce")
    df[orden_col] = pd.to_numeric(df[orden_col], errors="coerce")
    
    df_porticos = df[[cod_portico_col, km_col, calzada_col, orden_col, eje_col, autopista_col]].rename(
        columns={
            cod_portico_col: "portico",
            km_col: "km",
            calzada_col: "calzada",
            orden_col: "orden",
            eje_col: "eje",
            autopista_col: "autopista"
        }
    )
    _porticos_df_cache = df_porticos
    return df_porticos

def load_flujos():
    """
    Carga el CSV de flujos con columnas:
      FECHA, VELOCIDAD, CATEGORIA, MATRICULA, PORTICO, CARRIL.
    Permite cargar una muestra del archivo para agilizar el procesamiento.

    Returns:
        pd.DataFrame: DataFrame de flujos filtrado y procesado.
    """
    # Auto-selección SI hay exactamente un archivo FLUJO*.csv en la carpeta raíz 'Datos' (no recursivo)
    base_dir = "Datos"
    patterns = ["flujo*.csv"]
    try:
        top_files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
    except FileNotFoundError:
        top_files = []
    matches_top = [
        os.path.join(base_dir, f)
        for f in top_files
        if f.lower().endswith('.csv') and any(fnmatch.fnmatch(f.lower(), p) for p in patterns)
    ]

    if len(matches_top) == 1:
        file_path = matches_top[0]
        print(f"🔎 Detectado único archivo de flujos en '{base_dir}'. Cargando automáticamente → {file_path}")
    else:
        file_path = choose_file_filtered(base_dir, patterns)  # Mostrar solo FLUJO*.csv (case-insensitive)
        if file_path is None:
            return None

    # Preguntar al usuario si quiere una muestra
    sample_frac = 1.0
    use_sample = input("¿Desea cargar una muestra del archivo en lugar del archivo completo? (s/n): ").strip().lower()
    if use_sample in ('s', 'si', 'y', 'yes'):
        while True:
            try:
                percentage = float(input("Ingrese el porcentaje a cargar (1-100): ").strip())
                if 1 <= percentage <= 100:
                    sample_frac = percentage / 100.0
                    break
                else:
                    print("Por favor, ingrese un número entre 1 y 100.")
            except ValueError:
                print("Entrada inválida. Por favor, ingrese un número.")

    print("Cargando flujos...")
    df = read_csv_with_progress(file_path)
    
    if sample_frac < 1.0:
        print(f"\nTomando una muestra aleatoria del {sample_frac*100:.2f}% de los datos...")
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        print(f"La muestra contiene {len(df):,} filas.")

    print("\n🔄 Aplicando transformaciones...")
    with tqdm(total=4, desc="Procesando flujos") as pbar:
        pbar.set_description("Paso 1/4: Parseando fechas")
        try:
            df["FECHA"] = pd.to_datetime(df["FECHA"], dayfirst=True, errors="coerce", format='mixed')
        except TypeError:
            df["FECHA"] = pd.to_datetime(df["FECHA"], dayfirst=True, errors="coerce")
        df["FECHA"] = df["FECHA"].dt.tz_localize(None)
        pbar.update(1)

        pbar.set_description("Paso 2/4: Limpiando y convirtiendo categorías")
        df["CATEGORIA"] = pd.to_numeric(df["CATEGORIA"], errors="coerce").fillna(0).astype(int)
        df = df[df["CATEGORIA"] != 0]
        pbar.update(1)

        pbar.set_description("Paso 3/4: Mapeando categorías a nuevos valores")
        df["CATEGORIA"] = df["CATEGORIA"].replace({2: 2, 3: 2, 4: 3})
        pbar.update(1)

        pbar.set_description("Paso 4/4: Finalizando")
        cat_map = {1: "Light", 2: "Heavy", 3: "Motorcycles"}
        print("\n📊 Distribución de categorías (después del mapeo):")
        print(df["CATEGORIA"].map(cat_map).value_counts())
        pbar.update(1)
    
    return df

def load_accidentes(file_path: Optional[str] = None):
    """
    Carga el CSV de eventos (accidentes) y filtra para mantener solo los accidentes
    en vía expresa. Procesa columnas de fecha, hora, km, calzada, y clasifica severidad.
    Además, asigna el último pórtico ("ultimo_portico") usando find_candidate_porticos
    y excluye los accidentes sin pórtico asignado.
    
    Args:
        file_path (Optional[str]): Ruta directa al archivo. Si es None, intenta auto-detectar o pide input.

    Returns:
        pd.DataFrame: DataFrame de accidentes procesado.
    """
    global _porticos_df_cache

    if file_path is None:
        # Auto-selección SI hay exactamente un archivo que calza el patrón en la carpeta raíz 'Datos' (no recursivo)
        base_dir = "Datos"
        patterns = ["eventos*.csv"]
        try:
            top_files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
        except FileNotFoundError:
            top_files = []
        matches_top = [
            os.path.join(base_dir, f)
            for f in top_files
            if f.lower().endswith('.csv') and any(fnmatch.fnmatch(f.lower(), p) for p in patterns)
        ]

        if len(matches_top) == 1:
            file_path = matches_top[0]
            print(f"🔎 Detectado único archivo de eventos en '{base_dir}'. Cargando automáticamente → {file_path}")
        else:
            file_path = choose_file_filtered(base_dir, patterns)  # Mostrar solo Eventos*.csv (case-insensitive)
            if file_path is None:
                return None
    
    print(f"Cargando eventos (accidentes) desde: {file_path}...")
    df = read_csv_with_progress(path_csv=file_path, chunksize=100000)

    # Filtrar solo accidentes en vía expresa
    tipo_col = buscar_columna(df, "Tipo")
    via_col  = buscar_columna(df, "Via")
    allowed_via = ["expresa", "via expresa"]
    df = df[(df[tipo_col] == "Accidente") & (df[via_col].str.lower().isin(allowed_via))].copy()

    # Normalizar calzada
    calzada_col = buscar_columna(df, "Calzada")
    df[calzada_col] = (
        df[calzada_col]
        .str.strip()
        .str.lower()
        .map({"oriente": "Oriente", "poniente": "Poniente", "ambas": "Ambas"})
    )

    # Limpiar subtipo
    subtipo_col = buscar_columna(df, "SubTipo")
    df[subtipo_col] = df[subtipo_col].str.replace(r"^(AC-0[1-6] -\s*)", "", regex=True)

    # Parsear tiempos de inicio y fin
    fecha_inicio_col = buscar_columna(df, "Fechas Inicio")
    hora_inicio_col  = buscar_columna(df, "Hora Inicio")
    s_start = df[fecha_inicio_col].astype(str).str.strip() + " " + df[hora_inicio_col].astype(str).str.strip()
    # Evita el warning de 'infer format' usando formato mixto por elemento (Pandas >= 2.0)
    try:
        dt = pd.to_datetime(s_start, errors="coerce", dayfirst=True, format='mixed')
    except TypeError:
        dt = pd.to_datetime(s_start, errors="coerce", dayfirst=True)
    try:
        dt = dt.dt.tz_localize(None)
    except TypeError:
        dt = dt.dt.tz_convert(None)
    df["accidente_time"] = dt

    fecha_fin_col = buscar_columna(df, "Fecha Fin")
    hora_fin_col  = buscar_columna(df, "Hora Fin")
    s_end = df[fecha_fin_col].astype(str).str.strip() + " " + df[hora_fin_col].astype(str).str.strip()
    try:
        end_dt = pd.to_datetime(s_end, errors="coerce", dayfirst=True, format='mixed')
    except TypeError:
        end_dt = pd.to_datetime(s_end, errors="coerce", dayfirst=True)
    try:
        end_dt = end_dt.dt.tz_localize(None)
    except TypeError:
        end_dt = end_dt.dt.tz_convert(None)
    df["duracion_accidente"] = ((end_dt - dt).dt.total_seconds() / 60).round()

    # Procesar km
    km_col = buscar_columna(df, "Km.")
    df[km_col] = pd.to_numeric(df[km_col].astype(str).str.replace(",", "."), errors="coerce")

    # Clasificar severidad
    print("Clasificando severidad...")
    cols_leve = ["1", "2", "3", "4", "5", "6", "8", "9"]
    cols_grave = ["7"]
    
    if all(c in df.columns for c in cols_leve) and all(c in df.columns for c in cols_grave):
        tmp_leve  = df[cols_leve].apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)
        tmp_grave = df[cols_grave].apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)
        df["severidad"] = 0
        df.loc[(tmp_grave > 0).any(axis=1), "severidad"] = 1
    else:
        print("⚠️  Advertencia: No se encontraron todas las columnas para clasificar la severidad. Se asignará 'Na'.")
        df["severidad"] = "Na"

    # Asignar último pórtico y excluir sin candidato
    print("Asignando último pórtico a cada accidente...")

    # Verificar cache de pórticos
    if _porticos_df_cache is None:
        print("⚠️ No se han cargado los pórticos. Llamando a load_porticos()...")
        _porticos_df_cache = load_porticos()
        if _porticos_df_cache is None:
            print("❌ No se pudo cargar pórticos. Abortando load_accidentes().")
            return None

    porticos_df = _porticos_df_cache
    eje_col      = buscar_columna(df, "Eje")
    calz_col_acc = calzada_col  # ya obtenido más arriba

    def _get_ultimo(row):
        cand = find_candidate_porticos(
            acc_km    = row[km_col],
            porticos_df = porticos_df,
            eje       = row[eje_col],
            calzada   = row[calz_col_acc]
        )
        return cand["posterior"]["portico"] if cand["posterior"] is not None else None

    df["ultimo_portico"] = df.apply(_get_ultimo, axis=1)

    # Informar y excluir accidentes sin pórtico asignado
    excluded_accidents = df[df["ultimo_portico"].isna()]
    if not excluded_accidents.empty:
        print("\n⚠️  Excluyendo los siguientes accidentes por no tener un pórtico anterior claro:")
        for _, row in excluded_accidents.iterrows():
            print(f"  - Accidente en Eje: {row[eje_col]}, Calzada: {row[calz_col_acc]}, Km: {row[km_col]}, Fecha: {row['accidente_time']}")
    
    df = df[df["ultimo_portico"].notna()].copy()

    #  Preguntar al usuario si desea exportar a CSV
    respuesta = input("¿Desea exportar la tabla accidentes procesada? (s/n): ").strip().lower()
    if respuesta in ('s', 'si', 'y', 'yes'):
        # Asegura que la carpeta exista
        os.makedirs('Resultados', exist_ok=True)
        timestamp = datetime.now().strftime('%d%m%Y_%H%M')
        filename = f"accidentes_{timestamp}.csv"
        output_path = os.path.join('Resultados', filename)
        df.to_csv(output_path, index=False)
        print(f"Archivo exportado en → {output_path}")
    return df

# =============================================================================
# FUNCIONES CSV
# =============================================================================
def elegir_directorio():
    # Inicia la ventana de selección de directorio y la oculta
    if tk is None:
        print("⚠️  Tkinter no disponible. No se puede abrir selector de carpeta.")
        return None
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title="Elige la carpeta que contiene los CSVs")
    return directory

def resumen_csvs(lista_df, nombres_archivos):
    print("📊 Resumen de archivos leídos:")
    
    # Número de filas por archivo
    for nombre, df in zip(nombres_archivos, lista_df):
        print(f"📁 '{nombre}': {len(df)} filas")

    # Columnas comunes entre todos los archivos
    columnas_comunes = set.intersection(*(set(df.columns) for df in lista_df))
    columnas_comunes = sorted(list(columnas_comunes))  # Conversión a lista ordenada
    print(f"🧩 Columnas comunes entre los archivos: {columnas_comunes}")

    # Concatenar los DataFrames utilizando solo las columnas comunes
    df_unido = pd.concat([df[columnas_comunes] for df in lista_df], ignore_index=True)

    # Valores nulos por columna
    print("🕳️  Valores nulos por columna:")
    print(df_unido.isnull().sum())

    # Estadísticas descriptivas
    print("📈 Estadísticas descriptivas:")
    print(df_unido.describe(include='all'))

    return df_unido

def unir_csvs_en_directorio():
    # Permite al usuario elegir la carpeta que contiene los CSVs
    dir_path = elegir_directorio()
    if not dir_path:
        print("❌ No se seleccionó ninguna carpeta.")
        return

    archivos_csv = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith('.csv')]
    if not archivos_csv:
        print("❌ No se encontraron archivos .csv en la carpeta seleccionada.")
        return

    dfs = []
    nombres_validos = []

    with tqdm(total=len(archivos_csv), desc="Uniendo CSVs") as pbar:
        for archivo in archivos_csv:
            pbar.set_description(f"Procesando {os.path.basename(archivo)}")
            df = read_csv_with_progress(archivo)
            if df is not None:
                dfs.append(df)
                nombres_validos.append(os.path.basename(archivo))
            pbar.update(1)

    if not dfs:
        print("❌ Ningún archivo válido fue cargado.")
        return

    df_final = resumen_csvs(dfs, nombres_validos)

    nombre_salida = input("💾 Ingresa el nombre del archivo de salida (sin .csv): ").strip()
    nombre_salida = nombre_salida if nombre_salida else "salida"
    output_path = f"{nombre_salida}.csv"

    # Guardar con barra de progreso
    with tqdm(total=len(df_final), desc=f"Guardando '{output_path}'") as pbar:
        chunksize = 100000  # Escribir en chunks de 100,000 filas
        for i in range(0, len(df_final), chunksize):
            chunk = df_final.iloc[i:i+chunksize]
            chunk.to_csv(output_path, mode='a' if i > 0 else 'w', header=i==0, index=False, encoding='utf-8')
            pbar.update(len(chunk))

    print(f"✅ Archivo guardado como '{output_path}'")

def generar_estadisticas_descriptivas():
    """    Selecciona un archivo CSV de las carpetas 'Datos' o 'Resultados',
    lo carga y muestra un informe de estadísticas descriptivas completo.
    """
    print("\n--- Estadísticas Descriptivas ---")
    # 1. Elegir la carpeta
    while True:
        choice = input("¿Desde qué carpeta desea cargar el CSV? \n[1] Datos \n[2] Resultados \n[0] Cancelar \nElige una opción: ").strip()
        if choice == '1':
            start_folder = "Datos"
            break
        elif choice == '2':
            start_folder = "Resultados"
            break
        elif choice == '0':
            print("Operación cancelada.")
            return
        else:
            print("❌ Opción no válida.")

    # 2. Elegir el archivo
    file_path = choose_file(start_folder)
    if not file_path:
        return  # Cancelado desde choose_file

    print(f"▶️  Generando estadísticas para: {os.path.basename(file_path)}")
    # 3. Cargar el archivo
    df = read_csv_with_progress(file_path)
    if df is None:
        print("❌ No se pudo cargar el archivo.")
        return

    # --- Detección y eliminación de outliers ---
    respuesta_outliers = input("¿Desea detectar y eliminar outliers en alguna columna? (s/n): ").strip().lower()
    if respuesta_outliers in ('s', 'si', 'y', 'yes'):
        col_name = buscar_columna(df, "velocidad") # Sugerir 'velocidad' por defecto
        if col_name and col_name in df.columns and pd.api.types.is_numeric_dtype(df[col_name]):
            Q1 = df[col_name].quantile(0.25)
            Q3 = df[col_name].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[col_name] < lower_bound) | (df[col_name] > upper_bound)]

            if not outliers.empty:
                print(f"\nSe han detectado {len(outliers)} outliers en la columna '{col_name}'.")
                print(f"Límites para la detección de outliers: Inferior={lower_bound:.2f}, Superior={upper_bound:.2f}")

                outliers_upper = outliers[outliers[col_name] > upper_bound]
                outliers_lower = outliers[outliers[col_name] < lower_bound]

                print(f"De los outliers, {len(outliers_upper)} superan el límite superior y {len(outliers_lower)} están por debajo del límite inferior.")
                if not outliers_lower.empty:
                    print(f"Valor mínimo de outliers inferiores: {outliers_lower[col_name].min():.2f}")
                if not outliers_upper.empty:
                    print(f"Valor máximo de outliers superiores: {outliers_upper[col_name].max():.2f}")

                df = df[(df[col_name] >= lower_bound) & (df[col_name] <= upper_bound)]
                print(f"\nSe han eliminado {len(outliers)} outliers. El DataFrame ahora tiene {len(df)} filas.\n")
            else:
                print(f"\nNo se encontraron outliers en la columna '{col_name}'.\n")
        else:
            print(f"La columna '{col_name}' no es numérica o no existe.")

    # 4. Generar y mostrar el informe
    print("\n" + "="*50)
    print(" 📊 INFORME DE ESTADÍSTICAS DESCRIPTIVAS")
    print("="*50)
    # Información general
    print("\n[ 1. Información General ]")
    buffer = io.StringIO()
    df.info(buf=buffer)
    print(buffer.getvalue())
    # Valores Nulos
    print("\n[ 2. Conteo de Valores Nulos por Columna ]")
    null_counts = df.isnull().sum()
    if null_counts.sum() == 0:
        print("✅ No hay valores nulos en el dataset.")
    else:
        print(null_counts[null_counts > 0])
    # Tipos de datos
    print("\n[ 3. Tipos de Datos por Columna ]")
    print(df.dtypes)
    # Estadísticas Descriptivas
    print("\n[ 4. Estadísticas Descriptivas ]")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.describe(include='all'))
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    # Primeras 5 filas
    print("\n[ 5. Primeras 5 Filas (Head) ]")
    print(df.head())
    print("\n" + "="*50)
    print("✅ Informe completado.")
    print("="*50 + "\n")

def estandarizar_csvs_FLUJOS(carpeta_destino: str = "Estandarizados") -> None:
    """
    Recorre todos los CSV de una carpeta seleccionada por el usuario, los estandariza y
    los guarda en `carpeta_destino` con el mismo nombre.
    """
    print("Seleccione la carpeta que contiene los CSVs a estandarizar.")
    carpeta_fuentes = elegir_directorio()
    if not carpeta_fuentes:
        print("❌ No se seleccionó ninguna carpeta.")
        return

    src_path = Path(carpeta_fuentes)
    dst_path = Path(carpeta_destino)
    dst_path.mkdir(parents=True, exist_ok=True)

    csv_files = list(src_path.glob("*.csv"))
    if not csv_files:
        print(f"⚠️  No se encontraron archivos CSV en {src_path.resolve()}")
        return

    for csv_file in csv_files:
        print(f"\n🛠️  Procesando {csv_file.name}")
        try:
            # 1. Cargar con barra de progreso
            df = read_csv_with_progress(csv_file)
            total_registros = len(df)

            # 2. Eliminar columnas no deseadas
            df.drop(columns=[c for c in ("MATRICULA", "CARRIL") if c in df.columns],
                    inplace=True, errors="ignore")

            # 3. Normalizar valores faltantes
            df.replace(MISSING_MARKERS, pd.NA, inplace=True)

            # 4. Parsear la fecha (dos formatos admitidos) evitando warnings (dayfirst + ISO)
            try:
                parsed = pd.to_datetime(df["FECHA"], dayfirst=True, utc=True, errors="coerce", format='mixed')
            except TypeError:
                parsed = pd.to_datetime(df["FECHA"], dayfirst=True, utc=True, errors="coerce")
            df["FECHA"] = parsed.dt.tz_localize(None)

            # 5. Identificar filas a eliminar
            mask_drop = df.isna().any(axis=1)
            filas_a_eliminar = mask_drop.sum()

            # 6. Barra de progreso para eliminación fila a fila
            if filas_a_eliminar:
                with tqdm(total=filas_a_eliminar, desc='Eliminando filas', unit='fila') as pbar_del:
                    # Eliminamos de forma vectorizada pero actualizamos la barra manualmente
                    df = df.loc[~mask_drop].copy()
                    pbar_del.update(filas_a_eliminar)
            else:
                print("✅ No hay filas con valores faltantes.")

            # 7. Ordenar por fecha descendente
            df.sort_values("FECHA", ascending=False, inplace=True)

            # 8. Formatear la fecha al estándar requerido
            df["FECHA"] = df["FECHA"].dt.strftime("%d-%m-%Y %H:%M:%S")

            # 9. Guardar con el mismo nombre en la carpeta destino
            output_file_path = dst_path / csv_file.name
            print(f"💾 Guardando archivo estandarizado en '{output_file_path}' con codificación UTF-8...")
            df.to_csv(output_file_path, index=False, encoding="utf-8")

            # 10. Resumen final del archivo
            registros_finales = len(df)
            print(f"🔍 Resumen {csv_file.name}: "
                  f"{total_registros} leídos • "
                  f"{filas_a_eliminar} eliminados • "
                  f"{registros_finales} guardados")

        except Exception as e:
            print(f"❌ Error procesando {csv_file.name}: {e}")

def visualizar_estructura_porticos():
    """
    Carga el archivo de pórticos y muestra:
      - La secuencia por (calzada, eje) como antes.
      - Una vista unificada por 'familia de eje' (p.ej. RUTA 5),
        combinando NORTE/SUR dentro de la misma calzada y marcando con '|'
        cuando hay un cambio de eje. Esta vista refleja el nuevo criterio
        usado para armar tramos, que permite retroceder a ejes "anteriores"
        dentro de la misma autopista.
    """
    print("\n--- Visualizador de Estructura de Pórticos ---")

    # Usar el caché si está disponible, si no, cargar desde archivo
    global _porticos_df_cache
    if _porticos_df_cache is None:
        print("No hay un archivo de pórticos en caché. Por favor, cárguelo primero.")
        # Opcional: llamar a load_porticos() directamente si se quiere forzar la carga
        _porticos_df_cache = load_porticos()
        if _porticos_df_cache is None:
            print("❌ No se pudo cargar el archivo de pórticos.")
            return
    
    df_porticos = _porticos_df_cache
    
    if not all(col in df_porticos.columns for col in ['calzada', 'orden', 'portico', 'km']):
        print("❌ El DataFrame de pórticos no contiene las columnas esperadas ('calzada', 'orden', 'portico', 'km').")
        return

    print("\nEstructura de Pórticos por Calzada y Autopista (eje exacto):")
    
    # Ordenar el DataFrame de pórticos según la lógica especificada
    df_porticos_sorted = df_porticos.sort_values(
        by=["calzada", "eje", "orden"],
        ascending=[False, True, True]
    )

    # Agrupar por 'calzada' y 'eje' (autopista) para mostrar las secuencias
    # y recolectar tramos (p1 -> p2) para una demostración con la nueva función.
    tramos_demo = []  # [(display_str, calzada, eje, p1, p2)]
    for (calzada, eje), group in df_porticos_sorted.groupby(['calzada', 'eje']):
        print(f"\n🛣️  Calzada: {calzada} | Autopista: {eje}")
        
        # El grupo ya está ordenado por 'orden' gracias al sort_values anterior
        sorted_group = group
        
        if sorted_group.empty:
            print("   No hay pórticos definidos para esta combinación.")
            continue
        
        # Crear la representación de la secuencia
        secuencia_str = " -> ".join([
            f"{row['portico']} [km {row['km']}]"
            for _, row in sorted_group.iterrows()
        ])
        
        print(f"   {secuencia_str}")

        # Recolectar tramos consecutivos del grupo (para demo de reconstrucción)
        ports = sorted_group['portico'].tolist()
        for i in range(len(ports) - 1):
            p1, p2 = ports[i], ports[i+1]
            tramos_demo.append((f"Calzada {calzada} - Eje {eje}: {p1} -> {p2}", calzada, eje, p1, p2))
        
    print("\n"+"="*50+"\n")

    # ------------------------------------------------------------------
    # Vista unificada por familia de eje (NOR/SUR combinados)
    # ------------------------------------------------------------------
    def _eje_family(eje: str) -> str:
        eje = str(eje or "").strip()
        eje = re.sub(r"\b(SUR|NORTE)\b", "", eje, flags=re.IGNORECASE)
        eje = re.sub(r"\s+", " ", eje).strip()
        return eje.upper()

    df_porticos_fam = df_porticos.copy()
    df_porticos_fam["eje_fam"] = df_porticos_fam["eje"].apply(_eje_family)
    df_porticos_fam = df_porticos_fam.sort_values(by=["calzada", "eje_fam", "orden"])

    print("Estructura Unificada por Familia de Eje (con '|' cuando cambia NORTE/SUR):")
    for (calzada, eje_fam), group in df_porticos_fam.groupby(["calzada", "eje_fam"]):
        print(f"\n🛣️  Calzada: {calzada} | Autopista (familia): {eje_fam}")

        g = group.sort_values("orden")
        tokens = []
        last_port = None
        last_eje = None
        for _, row in g.iterrows():
            cur_eje = row["eje"]
            cur_port = row["portico"]
            # Insertar marca '|' si cambia NORTE/SUR (o el eje exacto)
            if last_eje is not None and cur_eje != last_eje:
                tokens.append("|")
            # Evitar duplicados consecutivos de pórtico
            if cur_port != last_port:
                tokens.append(f"{cur_port} [km {row['km']}]")
            last_port = cur_port
            last_eje = cur_eje

        if tokens:
            print("   " + " -> ".join(tokens))
        else:
            print("   (Sin pórticos en esta combinación)")

    print("\n"+"="*50+"\n")

    # ------------------------------------------------------------------
    # Demostración usando la función centralizada de reconstrucción
    # ------------------------------------------------------------------
    if tramos_demo:
        try:
            use_demo = input("¿Desea reconstruir una secuencia usando la lógica centralizada? (s/n): ").strip().lower()
        except Exception:
            use_demo = 'n'
        if use_demo in ('s', 'si', 'y', 'yes'):
            print("\nSeleccione un tramo para reconstruir la secuencia (ventana ± vecinos):")
            for i, (disp, _, _, _, _) in enumerate(tramos_demo):
                print(f"  [{i}] {disp}")
            try:
                sel = int(input("Número de tramo: ").strip())
                assert 0 <= sel < len(tramos_demo)
                _, calz, eje, p1, p2 = tramos_demo[sel]
            except Exception:
                print("Selección inválida. Omitiendo demostración.")
                return

            try:
                seq = reconstruct_portico_sequence(
                    df_porticos=df_porticos,
                    p1=p1,
                    p2=p2,
                    max_anterior=4,
                    max_posterior=0,
                    prefer_family=True,
                    deduplicate_slice=True,
                )
                print("\nSecuencia reconstruida (lógica centralizada):")
                print("   " + " -> ".join(str(x) for x in seq))
            except Exception as e:
                print(f"❌ No se pudo reconstruir la secuencia: {e}")

# -------------------------------------------------------------------------- #
# UTILIDADES GRAFOS
# -------------------------------------------------------------------------- #
def _eje_family(eje: str) -> str:
    """Normaliza el nombre del eje a una 'familia' (p.ej., 'RUTA 5', 'G VELASQUEZ').
    Elimina los sufijos NORTE/SUR y espacios duplicados para poder encadenar
    secuencias que cruzan de un eje a otro dentro de la misma autopista.
    """
    eje = str(eje or "").strip()
    eje = re.sub(r"\b(SUR|NORTE)\b", "", eje, flags=re.IGNORECASE)
    eje = re.sub(r"\s+", " ", eje).strip()
    return eje.upper()

def normalize_portico_code(v) -> str:
    """Normaliza un identificador de pórtico a string estable para comparaciones.
    - 18, '18', '18.0' -> '18'
    - 18.5 -> '18.5'
    - Otros valores -> str(v).strip()
    """
    try:
        f = float(str(v).strip())
        if abs(f - int(f)) < 1e-9:
            return str(int(f))
        return str(f)
    except Exception:
        return str(v).strip()

def reconstruct_portico_sequence(
    df_porticos: pd.DataFrame,
    p1,
    p2,
    max_anterior: int = 4,
    max_posterior: int = 0,
    prefer_family: bool = True,
    deduplicate_slice: bool = True,
) -> list:
    """
    Reconstruye la lista de pórticos a incluir alrededor del par consecutivo (p1 -> p2),
    estandarizando la misma lógica usada en varios módulos:

    1) Localiza el par en alguna secuencia (calzada, eje, orden). Si no aparece como par,
       usa la secuencia donde aparece p1.
    2) Intenta usar la secuencia por 'familia de eje' (mismo calzada) para poder cruzar de NORTE/SUR.
       Si ahí existe el par consecutivo p1->p2, recorta (max_anterior, max_posterior).
       Opcionalmente, deduplica dentro del recorte conservando la última aparición.
    3) Si no fue posible, hace el recorte sobre la secuencia local donde fue hallado p1.

    Devuelve la lista de pórticos en su dtype original del DataFrame.
    Lanza ValueError si no se pudo ubicar p1 en ninguna secuencia.
    """
    required_cols = {"portico", "eje", "calzada", "orden"}
    if not required_cols.issubset(df_porticos.columns):
        raise ValueError(f"df_porticos debe contener columnas {sorted(required_cols)}")

    p1n = normalize_portico_code(p1)
    p2n = normalize_portico_code(p2)

    df_sorted = df_porticos.sort_values(by=["calzada", "eje", "orden"])  # base local

    # 1) Buscar par local p1->p2; si no, caer al eje que contenga p1
    pair_ctx = None  # (calzada, eje, i_local, seq_local_raw, seq_local_norm)
    for (calzada, eje), group in df_sorted.groupby(["calzada", "eje"]):
        seq_raw = group["portico"].tolist()
        seq_norm = [normalize_portico_code(x) for x in seq_raw]
        for j in range(len(seq_norm) - 1):
            if seq_norm[j] == p1n and seq_norm[j + 1] == p2n:
                pair_ctx = (calzada, eje, j, seq_raw, seq_norm)
                break
        if pair_ctx is not None:
            break

    if pair_ctx is None:
        for (calzada, eje), group in df_sorted.groupby(["calzada", "eje"]):
            seq_raw = group["portico"].tolist()
            seq_norm = [normalize_portico_code(x) for x in seq_raw]
            if p1n in seq_norm:
                pair_ctx = (calzada, eje, seq_norm.index(p1n), seq_raw, seq_norm)
                break

    if pair_ctx is None:
        raise ValueError("No se encontró p1 en las secuencias de pórticos cargadas")

    sel_calzada, sel_eje, i_local, seq_local_raw, _seq_local_norm = pair_ctx

    # 2) Intentar familia de eje (mismo calzada) si se prefiere
    if prefer_family:
        df_tmp = df_porticos.copy()
        df_tmp["eje_fam"] = df_tmp["eje"].apply(_eje_family)
        fam_key = (sel_calzada, _eje_family(sel_eje))
        fam_groups = (
            df_tmp.sort_values(["calzada", "eje_fam", "orden"]).groupby(["calzada", "eje_fam"])
        )
        if fam_key in fam_groups.groups:
            seq_fam_raw = fam_groups.get_group(fam_key)["portico"].tolist()
            seq_fam_norm = [normalize_portico_code(x) for x in seq_fam_raw]
            j = None
            for k in range(len(seq_fam_norm) - 1):
                if seq_fam_norm[k] == p1n and seq_fam_norm[k + 1] == p2n:
                    j = k
            if j is not None:
                start_idx = max(0, j - max_anterior)
                end_idx = min(len(seq_fam_raw), (j + 1) + max_posterior + 1)
                slice_seq = seq_fam_raw[start_idx:end_idx]
                if deduplicate_slice:
                    seen = set()
                    out = []
                    for x in reversed(slice_seq):
                        if x not in seen:
                            out.append(x)
                            seen.add(x)
                    return list(reversed(out))
                else:
                    return slice_seq

    # 3) Fallback: usar secuencia local donde se encontró p1
    start_idx = max(0, i_local - max_anterior)
    end_idx = min(len(seq_local_raw), (i_local + 1) + max_posterior + 1)
    return seq_local_raw[start_idx:end_idx]

def seleccionar_tramo_y_porticos(df_porticos, max_anterior=4, max_posterior=0):
    """
    Permite al usuario seleccionar un tramo y devuelve una lista de pórticos
    a su alrededor (hasta `max_anterior` y `max_posterior`).

    Args:
        df_porticos (pd.DataFrame): DataFrame de pórticos ya cargado.
        max_anterior (int): Número máximo de pórticos anteriores a incluir.
        max_posterior (int): Número máximo de pórticos posteriores a incluir.

    Returns:
        list or None: Lista de códigos de pórticos a incluir en el grafo,
                      o None si se cancela la operación.
    """
    if df_porticos is None or df_porticos.empty:
        print("❌ El DataFrame de pórticos está vacío o no se ha cargado.")
        return None

    print("\n--- Selección de Tramo para Modelar en el Grafo ---")

    # 1. Generar secuencias y tramos
    df_sorted = df_porticos.sort_values(by=["calzada", "eje", "orden"])  # mantiene lógica de visualización

    tramos = []
    secuencias_completas = {}

    for (calzada, eje), group in df_sorted.groupby(['calzada', 'eje']):
        porticos_ordenados = group['portico'].tolist()
        kms_ordenados = group['km'].tolist()
        secuencias_completas[(calzada, eje)] = porticos_ordenados
        
        for i in range(len(porticos_ordenados) - 1):
            p1 = porticos_ordenados[i]
            p2 = porticos_ordenados[i+1]
            km1 = kms_ordenados[i]
            km2 = kms_ordenados[i+1]
            tramos.append({
                "calzada": calzada,
                "eje": eje,
                "tramo_str": f"Calzada {calzada} - Eje {eje}: {p1} (km {km1}) -> {p2} (km {km2})",
                "p1_idx": i,
                "p2_idx": i + 1
            })

    if not tramos:
        print("❌ No se pudieron generar tramos a partir de los pórticos.")
        return None

    # 2. Mostrar tramos y solicitar selección
    print("Seleccione el tramo de interés:")
    for i, tramo in enumerate(tramos):
        print(f"  [{i}] {tramo['tramo_str']}")
    
    try:
        choice_str = input("\nIngrese el número del tramo (o presione Enter para cancelar): ").strip()
        if not choice_str:
            print("Operación cancelada.")
            return None
        choice = int(choice_str)
        if not (0 <= choice < len(tramos)):
            raise ValueError("Selección fuera de rango.")
    except ValueError as e:
        print(f"❌ Selección inválida: {e}")
        return None

    # 3. Identificar pórticos a incluir
    tramo_seleccionado = tramos[choice]
    calzada = tramo_seleccionado['calzada']
    eje = tramo_seleccionado['eje']
    p1_idx = tramo_seleccionado['p1_idx']
    p1 = secuencias_completas[(calzada, eje)][p1_idx]
    p2 = secuencias_completas[(calzada, eje)][p1_idx + 1]

    # Usar lógica centralizada para reconstruir la secuencia alrededor de (p1 -> p2)
    porticos_seleccionados = reconstruct_portico_sequence(
        df_porticos=df_porticos,
        p1=p1,
        p2=p2,
        max_anterior=max_anterior,
        max_posterior=max_posterior,
        prefer_family=True,
        deduplicate_slice=True,
    )

    print("\n✅ Pórticos seleccionados para el grafo:")
    print(f"   - Tramo base: {tramo_seleccionado['tramo_str']}")
    print(f"   - Pórticos incluidos: {porticos_seleccionados}")

    return porticos_seleccionados

def load_graph():

    """
    Muestra los archivos .pt disponibles, solicita al usuario que elija uno y lo carga.
    
    Returns:
        dict or None: El objeto del grafo cargado o None si se cancela o falla.
    """
    pattern = os.path.join(RESULTADOS_DIR, "highway_graph_*.pt")
    pt_files = sorted(glob.glob(pattern))
    
    if not pt_files:
        print(f"No se encontraron archivos de grafos (`highway_graph_*.pt`) en '{RESULTADOS_DIR}'.")
        return None
    
    if len(pt_files) == 1:
        pt_path = pt_files[0]
        print(f"\n✅ Cargando automáticamente el único grafo encontrado → {os.path.basename(pt_path)}")
    else:
        print("\nArchivos de grafo disponibles:")
        for i, f in enumerate(pt_files, 1):
            print(f"  [{i}] {os.path.basename(f)}")
        try:
            idx_str = input("\nSelecciona el número del archivo a cargar (o presione Enter para cancelar): ").strip()
            if not idx_str:
                print("Carga cancelada.")
                return None
            idx = int(idx_str) - 1
            if 0 <= idx < len(pt_files):
                pt_path = pt_files[idx]
            else:
                print("❌ Opción inválida.")
                return None
        except (ValueError, IndexError):
            print("❌ Selección inválida.")
            return None

    # --- Carga con barra de progreso ---
    import io
    file_size = os.path.getsize(pt_path)
    with open(pt_path, 'rb') as f:
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Cargando {os.path.basename(pt_path)}") as pbar:
            buffer = io.BytesIO(f.read())
            pbar.update(file_size)
    buffer.seek(0)
    obj = torch.load(buffer, weights_only=False)
    # --- Fin de carga con barra de progreso ---

    # Add filename to the loaded object
    obj['filename'] = os.path.basename(pt_path)
    print(f"\n✅ Grafo cargado desde '{os.path.basename(pt_path)}' con tipos de nodos: {list(obj['data'].node_types)}")
    return obj

def visualize_acc_subgraph(loaded_obj, layout='spring', spring_iters=50):
    _require_matplotlib()
    """
    Visualiza el subgrafo de influencia de un nodo PM marcado como accidente 
    o un nodo sintético, mostrando todos los tipos de aristas con colores 
    distintos y una clara dirección temporal hacia el evento.
    Distingue visualmente los nodos y aristas sintéticos si existen.
    """

    # --- 1. Extracción de Datos y Configuración ---
    if not (isinstance(loaded_obj, dict) and 'data' in loaded_obj):
        print("❌ Objeto de grafo no válido.")
        return
    data = loaded_obj['data']
    pm_index = loaded_obj.get('pm_index')

    # --- Identificación de Nodos de Interés (Accidente y Sintéticos) ---
    is_augmented = 'pm' in data.node_types and hasattr(data['pm'], 'is_synthetic')
    
    # Nodos PM marcados como accidentes (usando la nueva máscara)
    accident_pm_indices = []
    if 'pm' in data.node_types and hasattr(data['pm'], 'is_accident_pm'):
        if DEBUG:
            print(f"DEBUG: Found 'is_accident_pm' mask. Sum of True values: {torch.sum(data['pm'].is_accident_pm).item()}")
        accident_pm_indices = torch.where(data['pm'].is_accident_pm)[0].tolist()

    syn_pm_indices = []
    if is_augmented:
        syn_pm_indices = torch.where(data['pm'].is_synthetic)[0].tolist()

    # --- Paleta de Colores y Tamaños ---
    node_color_map = {
        'pm_accident': 'salmon', 
        'pm': 'skyblue',
        'pm_syn': '#ffcc00'
    }
    # Colores base por relación conocida; el resto se asigna dinámicamente
    base_edge_colors = {
        'temporal': 'green',
        'spatial': 'blue',
        'spatial_back': '#9b59b6',  # púrpura
        'st_fwd': '#e67e22'        # naranja
    }
    size_map = {'pm_accident': 300, 'pm': 150, 'pm_syn': 250}

    if pm_index is None:
        print("⚠️  Advertencia: No se encontró 'pm_index'. Las etiquetas de los nodos no tendrán detalles.")
    
    if not accident_pm_indices and not syn_pm_indices:
        print("No hay nodos PM marcados como accidente o nodos sintéticos para visualizar.")
        return

    # --- 2. Selección de Nodo de Interés (sin listar todos, solo rango) ---
    menu_title = ""
    candidate_indices = []
    selected_pool = None  # 'acc' o 'syn'

    if accident_pm_indices and syn_pm_indices:
        while True:
            print("--- ¿Qué tipo de nodo desea visualizar? ---")
            print("  [1] Nodos PM de Accidente")
            print("  [2] Nodos PM Sintéticos")
            print("  [q] Salir")
            choice_type = input("Seleccione una opción: ").strip().lower()
            if choice_type == 'q':
                return
            if choice_type == '1':
                menu_title = "--- Selección de nodo PM de ACCIDENTE ---"
                candidate_indices = accident_pm_indices
                selected_pool = 'acc'
                break
            elif choice_type == '2':
                menu_title = "--- Selección de nodo PM SINTÉTICO ---"
                candidate_indices = syn_pm_indices
                selected_pool = 'syn'
                break
            else:
                print("Opción inválida.")
    elif accident_pm_indices:
        menu_title = "--- Selección de nodo PM de ACCIDENTE ---"
        candidate_indices = accident_pm_indices
        selected_pool = 'acc'
    elif syn_pm_indices:
        menu_title = "--- Selección de nodo PM SINTÉTICO (No hay nodos de accidente) ---"
        candidate_indices = syn_pm_indices
        selected_pool = 'syn'
    else:
        print("No hay nodos PM marcados como accidente o nodos sintéticos para visualizar.")
        return

    start_node_type, start_node_idx = None, None
    max_nodes = 10

    # Mostrar solo el rango de selección sin listar cada elemento
    while True:
        try:
            print(f"\n{menu_title}")
            total = len(candidate_indices)
            if total == 0:
                print("No hay elementos disponibles en el rango.")
                return
            print(f"Total disponibles: {total} | Selección por posición en la lista")
            print(f"Rango válido: [0 .. {total-1}]  (ingrese 'q' para salir)")

            choice = input("Ingrese una posición del rango: ").strip().lower()
            if choice == 'q':
                return
            pos = int(choice)
            if 0 <= pos < total:
                start_node_type = 'pm'
                start_node_idx = candidate_indices[pos]
                # Mensaje corto de confirmación
                tipo_txt = "Accidente" if selected_pool == 'acc' else "Sintético"
                print(f"Seleccionado PM {tipo_txt} => posición {pos}, índice de grafo {start_node_idx}")
                break
            else:
                print("Índice fuera de rango.")
        except ValueError:
            print("Entrada inválida.")

    while True:
        try:
            prompt = f"¿Cuántos vecindarios previos al nodo seleccionado desea visualizar? (máximo {max_nodes}): "
            choice = input(prompt).strip().lower()
            if choice == 'q': return
            prev_nodes = int(choice)
            if 0 <= prev_nodes <= max_nodes: break
            print(f"Por favor, ingrese un número entre 0 y {max_nodes}.")
        except ValueError:
            print("Entrada inválida.")

    # --- 3. Construcción de Subgrafo ---
    # Reglas:
    #  - Temporal: sólo retrocede (predecesores) y aumenta profundidad (saltos)
    #  - Spatial (+ spatial_back si existe): expansión bidireccional al MISMO nivel (no aumenta saltos)
    from collections import deque, defaultdict

    temporal_rev_adj = defaultdict(list)
    spatial_bi_adj = defaultdict(list)

    for src_t, rel, dst_t in data.edge_types:
        edge_index = data[(src_t, rel, dst_t)].edge_index
        if edge_index.numel() == 0:
            continue
        for src, dst in edge_index.t().tolist():
            a = (src_t, src)
            b = (dst_t, dst)
            if rel == 'temporal':
                temporal_rev_adj[b].append(a)  # retrocede en el tiempo
            elif rel in ('spatial', 'spatial_back'):
                spatial_bi_adj[a].append(b)
                spatial_bi_adj[b].append(a)    # tratar espacial como bidireccional para visualización

    max_hops = max(0, int(prev_nodes))
    start = (start_node_type, start_node_idx)
    visited = {start}
    q = deque([(start, 0)])

    while q:
        node, depth = q.popleft()
        # Expansión espacial a mismo nivel
        for neigh in spatial_bi_adj.get(node, []):
            if neigh not in visited:
                visited.add(neigh)
                q.append((neigh, depth))
        # Retroceso temporal (aumenta profundidad)
        if depth < max_hops:
            for pred in temporal_rev_adj.get(node, []):
                if pred not in visited:
                    visited.add(pred)
                    q.append((pred, depth + 1))

    sub_nodes = visited

    # Asegurar: incluir explícitamente aristas 'spatial' que LLEGAN al nodo accidente seleccionado
    if start_node_type == 'pm':
        incoming_rel_types = []
        if ('pm', 'spatial', 'pm') in data.edge_types:
            incoming_rel_types.append(('pm', 'spatial', 'pm'))
        if ('pm', 'spatial_back', 'pm') in data.edge_types:
            incoming_rel_types.append(('pm', 'spatial_back', 'pm'))
        for et in incoming_rel_types:
            ei = data[et].edge_index
            if ei.numel() == 0:
                continue
            dst_eq_start = (ei[1] == start_node_idx).nonzero(as_tuple=True)[0].tolist()
            for epos in dst_eq_start:
                src_idx = int(ei[0, epos].item())
                sub_nodes.add(('pm', src_idx))

    # --- 4. Creación del Grafo de Visualización con NetworkX ---
    G = nx.DiGraph()
    custom_labels = {}

    for ntype, idx in sub_nodes:
        node_id = f"{ntype}_{idx}"
        is_syn = is_augmented and ntype == 'pm' and idx < len(data['pm'].is_synthetic) and data['pm'].is_synthetic[idx]
        is_accident_node = idx in accident_pm_indices

        tag = 'pm_syn' if is_syn else ('pm_accident' if is_accident_node else 'pm')
        G.add_node(node_id, tag=tag)
        
        label_str = ""
        if 'y' in data[ntype] and idx < len(data[ntype].y):
            label_val = data[ntype].y[idx].item()
            label_str = f"\nLabel: {label_val}"

        if is_syn:
            # Etiqueta para nodos sintéticos
            base_label = f"{node_id}\n(Syn)"
            custom_labels[node_id] = f"{base_label}{label_str}"
        elif ntype == 'pm' and pm_index:
            # Es un nodo PM real, intentar obtener detalles
            try:
                portico, ts_min = pm_index._rev[idx]
                dt = datetime.fromtimestamp(ts_min * 60)
                custom_labels[node_id] = f"P:{portico}\n{dt.strftime('%d-%m %H:%M')}{label_str}"
            except (IndexError, TypeError, KeyError):
                # Si falla la búsqueda en pm_index, es un nodo real sin mapeo
                custom_labels[node_id] = f"{node_id}\n(Real){label_str}"
        else:
            # Es un nodo real pero no es de tipo 'pm' o no hay pm_index
            custom_labels[node_id] = f"{node_id}\n(Real){label_str}"

    # Determinar las relaciones presentes en el grafo
    rels_present = [rel for (_, rel, _) in data.edge_types]

    # Construir mapa de colores incluyendo relaciones nuevas no mapeadas explícitamente
    edge_color_map = {}
    # 1) Asignar colores base cuando correspondan
    for rel in rels_present:
        if rel in base_edge_colors:
            edge_color_map[rel] = base_edge_colors[rel]
    # 2) Asignar colores a relaciones nuevas usando una paleta
    import itertools
    palette = list(matplotlib.cm.get_cmap('tab10').colors) + list(matplotlib.cm.get_cmap('Set2').colors)
    palette_cycle = itertools.cycle(palette)
    for rel in rels_present:
        if rel not in edge_color_map:
            edge_color_map[rel] = matplotlib.colors.to_hex(next(palette_cycle))

    edges_by_type = {rel: [] for rel in edge_color_map.keys()}
    syn_edges_by_type = {rel: [] for rel in edge_color_map.keys()}

    for i, (src_t, rel, dst_t) in enumerate(data.edge_types):
        edge_index = data[(src_t, rel, dst_t)].edge_index
        is_edge_syn = is_augmented and 'is_synthetic' in data[(src_t, rel, dst_t)]
        
        for k, (src, dst) in enumerate(edge_index.t().tolist()):
            if (src_t, src) in sub_nodes and (dst_t, dst) in sub_nodes:
                u, v = f"{src_t}_{src}", f"{dst_t}_{dst}"
                G.add_edge(u, v)
                if is_edge_syn and data[(src_t, rel, dst_t)].is_synthetic[k]:
                    syn_edges_by_type[rel].append((u,v))
                else:
                    edges_by_type[rel].append((u, v))

    # --- 5. Dibujo del Grafo ---
    if len(G.nodes()) < 2:
        print("\n[INFO] No hay suficientes nodos para dibujar un grafo significativo.")
        return
        
    plt.figure(figsize=(18, 14))
    
    pos = nx.spring_layout(G, iterations=spring_iters, seed=SEED, k=1.5 / math.sqrt(G.number_of_nodes()) if G.number_of_nodes() > 0 else 1)

    for tag, color in node_color_map.items():
        nx.draw_networkx_nodes(G, pos, nodelist=[n for n,d in G.nodes(data=True) if d['tag']==tag], node_color=color, node_size=size_map.get(tag, 100), label=tag)
    
    for rel, color in edge_color_map.items():
        nx.draw_networkx_edges(G, pos, edgelist=edges_by_type.get(rel, []), edge_color=color, arrowstyle='->', arrowsize=15, width=1.5, label=f"{rel} (Real)", connectionstyle='arc3,rad=0.1')

    for rel, color in edge_color_map.items():
        nx.draw_networkx_edges(G, pos, edgelist=syn_edges_by_type.get(rel, []), edge_color=color, arrowstyle='->', arrowsize=15, width=2.0, style='dashed', label=f"{rel} (Syn)", connectionstyle='arc3,rad=0.1')

    nx.draw_networkx_labels(G, pos, labels=custom_labels, font_size=8, font_color='black')

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Nodo: {t}', markerfacecolor=c, markersize=10) for t, c in node_color_map.items()] \
                    + [Line2D([0], [0], color=c, lw=2, label=f'Arista Real: {r}') for r, c in edge_color_map.items()] \
                    + [Line2D([0], [0], color=c, lw=2, linestyle='--', label=f'Arista Sintética: {r}') for r, c in edge_color_map.items()]
    plt.legend(handles=legend_elements, loc='upper right')
    
    title = f"Subgrafo de Influencia para el Nodo {start_node_type.upper()} {start_node_idx}"
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_graph_overview(loaded_obj):
    """
    Muestra un resumen detallado del grafo, incluyendo estadísticas de nodos,
    aristas, features y máscaras de división de datos.
    """
    if not isinstance(loaded_obj, dict) or 'data' not in loaded_obj:
        print("❌ Objeto de grafo no válido.")
        return

    data = loaded_obj['data']
    feature_cols = loaded_obj.get("feature_cols", [])

    print("\n" + "="*80)
    print(" " * 25 + "Resumen Detallado del Grafo")
    print("="*80)

    # --- 1. Resumen de Tipos de Nodos ---
    print("\n--- Tipos de Nodos ---")
    for node_type in data.node_types:
        print(f"\n▶ Nodo: '{node_type}'")
        node_store = data[node_type]
        print(f"  - Número de nodos: {node_store.num_nodes}")
        
        print("  - Atributos:")
        for key, value in node_store.items():
            if hasattr(value, 'shape'):
                if 'mask' not in key:
                    print(f"    - {key}: Tensor con forma {list(value.shape)}")
            else:
                print(f"    - {key}: (No es un tensor)")

        if node_type == 'pm' and feature_cols:
            print(f"  - Features ({len(feature_cols)} en total):")
            max_len = max(len(f) for f in feature_cols) if feature_cols else 0
            num_cols = 3
            col_width = max_len + 4
            for i in range(0, len(feature_cols), num_cols):
                line = "".join(f"{feat:<{col_width}}" for feat in feature_cols[i:i+num_cols])
                print(f"    {line}")
        elif 'x' in node_store and hasattr(node_store.x, 'shape') and node_store.x.shape[1] > 0:
             print(f"  - Features: {node_store.x.shape[1]} features sin nombre definido.")

    # --- 2. Resumen de Tipos de Aristas ---
    print("\n" + "-"*80)
    print("--- Tipos de Aristas ---")
    for edge_type in data.edge_types:
        src, rel, dst = edge_type
        print(f"\n▶ Arista: ('{src}', '{rel}', '{dst}')")
        edge_store = data[edge_type]
        print(f"  - Número de aristas: {edge_store.num_edges}")

        for key, value in edge_store.items():
            if hasattr(value, 'shape'):
                shape = list(value.shape)
                if key == 'edge_index':
                    print(f"    - {key}: Tensor de conectividad con forma {shape}")
                elif key == 'edge_attr':
                    print(f"    - {key}: Atributos de arista con forma {shape} ({shape[1]} features)")
                else:
                    print(f"    - {key}: Tensor con forma {shape}")
            else:
                print(f"    - {key}: (No es un tensor)")

    # --- 3. Resumen de Máscaras (División de Datos) y Distribución de Etiquetas ---
    print("\n" + "-"*80)
    print("--- División de Datos (Máscaras) y Distribución de Etiquetas ---")
    
    has_masks = False
    for node_type in data.node_types:
        node_store = data[node_type]
        
        mask_keys = [k for k in node_store.keys() if k.endswith('_mask') or k == 'is_accident_pm']
        
        if mask_keys:
            has_masks = True
            print(f"\n▶ Tipo de Nodo: '{node_type}'")
            
            for mask_name in sorted(mask_keys):
                if mask_name in node_store:
                    mask = node_store[mask_name]
                    true_count = mask.sum().item()
                    false_count = mask.numel() - true_count
                    print(f"  - Máscara '{mask_name}':")
                    print(f"    - True:  {true_count}")
                    print(f"    - False: {false_count}")

                    if 'y' in node_store and mask_name in ['train_mask', 'val_mask', 'test_mask']:
                        y = node_store.y
                        labels_in_mask = y[mask]
                        if labels_in_mask.numel() > 0:
                            unique_labels, counts = torch.unique(labels_in_mask, return_counts=True)
                            label_dist_str = ", ".join([f"Clase {l.item()}: {c.item()}" for l, c in zip(unique_labels, counts)])
                            print(f"    - Distribución de etiquetas en la máscara: {label_dist_str}")
                        else:
                            print("    - Distribución de etiquetas en la máscara: (sin nodos en la máscara)")
    
    if not has_masks:
        print("\n  - No se encontraron máscaras de división para los nodos.")

    # Aristas
    print("\n▶ Conteo de Aristas por Conjunto:")
    print("  (Una arista pertenece a un conjunto si AMBOS nodos (origen y destino) están en ese conjunto)")
    has_edge_masks = False
    for edge_type in data.edge_types:
        src_type, rel, dst_type = edge_type
        src_store = data[src_type]
        dst_store = data[dst_type]
        
        if all(k in src_store for k in ['train_mask', 'val_mask', 'test_mask']) and \
           all(k in dst_store for k in ['train_mask', 'val_mask', 'test_mask']):
            
            has_edge_masks = True
            edge_index = data[edge_type].edge_index
            src_nodes, dst_nodes = edge_index[0], edge_index[1]

            print(f"  - Tipo ('{src_type}', '{rel}', '{dst_type}'):")
            
            # Conteo para Train
            edge_train_mask = src_store.train_mask[src_nodes] & dst_store.train_mask[dst_nodes]
            train_count = edge_train_mask.sum().item()
            print(f"    - Train: {train_count}")
            
            # Conteo para Val
            edge_val_mask = src_store.val_mask[src_nodes] & dst_store.val_mask[dst_nodes]
            val_count = edge_val_mask.sum().item()
            print(f"    - Val:   {val_count}")

            # Conteo para Test
            edge_test_mask = src_store.test_mask[src_nodes] & dst_store.test_mask[dst_nodes]
            test_count = edge_test_mask.sum().item()
            print(f"    - Test:  {test_count}")

    if not has_edge_masks:
        print("  - No se encontraron máscaras en los nodos necesarios para calcular la división de aristas.")

    print("\n" + "="*80)
    input("\nPresiona Enter para continuar....")

def compute_minute_aggregates(df, freq='min'):
    """Agrega datos por minuto y categoría."""
    df = df.infer_objects(copy=False)
    df_indexed = df.set_index('FECHA')
    agg = df_indexed.groupby([
        'PORTICO',
        pd.Grouper(freq=freq),
        'CATEGORIA'
    ])['VELOCIDAD'].agg(['count', 'mean', 'std']).rename(columns={
        'count': 'flow',
        'mean': 'mean_speed',
        'std': 'speed_std'
    })
    agg['speed_std'] = agg['speed_std'].fillna(0)
    return agg.reset_index()

def generar_tabla_A(minute_agg):
    """
    Calcula y muestra la Tabla A de estadísticas descriptivas globales por categoría.
    El flujo se calcula como el total de vehículos por hora en cada pórtico,
    dividido entre 3 para obtener el flujo aproximado por pista (asumiendo 3 pistas).
    """
    print("\n" + "="*80)
    print(" " * 10 + "Tabla A - Estadísticas Descriptivas Globales por Categoría")
    print("="*80)

    if minute_agg is None or minute_agg.empty:
        print("No hay datos para generar la Tabla A.")
        return

    group_cols = ['PORTICO', 'CATEGORIA', 'FECHA_HORA']
    exported_tables = []

    def _imprimir_tabla(df_subset, titulo):
        """Calcula y despliega la tabla de estadísticas para un subconjunto."""
        if df_subset is None or df_subset.empty:
            print(f"\n{titulo}: No hay datos para esta franja horaria.")
            return

        df_local = df_subset[df_subset['CATEGORIA'] != 'Other'].copy()
        if df_local.empty:
            print(f"\n{titulo}: Sin datos tras excluir categoría 'Other'.")
            return

        df_local['FECHA_HORA'] = df_local['FECHA'].dt.floor('h')

        def _weighted_speed(series):
            weights = df_local.loc[series.index, 'flow']
            total_weight = weights.sum()
            return np.average(series, weights=weights) if total_weight > 0 else np.nan

        hourly_agg = df_local.groupby(group_cols).agg(
            flow_total=('flow', 'sum'),
            speed_weighted=('mean_speed', _weighted_speed)
        ).reset_index()

        if hourly_agg.empty:
            print(f"\n{titulo}: No hay datos horarios tras la agregación.")
            return

        # --- Cálculos de las variables ---
        hourly_agg['Flow [veh/h]'] = hourly_agg['flow_total'] / 3  # flujo por pista (3 pistas)
        hourly_agg['Speed [km/h]'] = hourly_agg['speed_weighted']
        hourly_agg['Density [veh/km]'] = hourly_agg.apply(
            lambda row: row['Flow [veh/h]'] / row['Speed [km/h]'] if row['Speed [km/h]'] > 0 else 0,
            axis=1
        )

        # --- Agregación para obtener estadísticas ---
        stats_agg = hourly_agg.groupby('CATEGORIA').agg({
            'Flow [veh/h]': ['mean', 'std', 'min', 'max'],
            'Speed [km/h]': ['mean', 'std', 'min', 'max'],
            'Density [veh/km]': ['mean', 'std', 'min', 'max']
        })

        if stats_agg.empty:
            print(f"\n{titulo}: No hay datos suficientes para generar estadísticas.")
            return

        # --- Reestructuración de la tabla ---
        stats_agg.columns.names = ['Variable', 'Estadística']
        
        df_final = stats_agg.stack(level='Variable', future_stack=True)
        
        df_final = df_final.rename(columns={
            'mean': 'Average',
            'std': 'Std.Dev',
            'min': 'Min',
            'max': 'Max'
        }).round(2)
        
        df_final = df_final[['Average', 'Std.Dev', 'Min', 'Max']]

        print(f"\n{titulo}")
        # --- Cálculo de la participación por categoría (Category Share) ---
        share_series = None
        category_total_flow = hourly_agg.groupby('CATEGORIA')['Flow [veh/h]'].sum()
        total_flow = category_total_flow.sum()
        if total_flow > 0:
            share = (category_total_flow / total_flow * 100).round(2)
            share_df = pd.DataFrame(share).rename(columns={'Flow [veh/h]': 'Share (%)'})
            share_series = share_df['Share (%)']
            print("Share por categoría:")
            print(share_df)
            print("-"*30)

        print(df_final.to_string())

        # Acumular para exportar
        export_df = df_final.reset_index().rename(columns={'level_0': 'CATEGORIA', 'level_1': 'Variable'})
        if share_series is not None:
            export_df['Share (%)'] = export_df['CATEGORIA'].map(share_series)
        export_df.insert(0, 'Tabla', titulo)
        exported_tables.append(export_df)

    df = minute_agg.copy()

    # Tabla general (24h)
    _imprimir_tabla(df, "Tabla A - Estadísticas Descriptivas Globales por Categoría")

    # Tablas por franja horaria
    df['hour'] = df['FECHA'].dt.hour
    df['day_type'] = df['FECHA'].dt.dayofweek.apply(lambda x: 'Weekday' if x < 5 else 'Weekend')

    # Tablas 24h por tipo de día
    _imprimir_tabla(df[df['day_type'] == 'Weekday'], "Tabla A - Lunes a Viernes (24h)")
    _imprimir_tabla(df[df['day_type'] == 'Weekend'], "Tabla A - Sábado y Domingo (24h)")

    franjas = [
        ("Tabla 1 (06:00-8:00)", (df['hour'] >= 6) & (df['hour'] < 8)),
        ("Tabla 2 (08:00-18:00)", (df['hour'] >= 8) & (df['hour'] < 18)),
        ("Tabla 3 (18:00-20:00)", (df['hour'] >= 18) & (df['hour'] < 20)),
        ("Tabla 4 (20:00-06:00)", (df['hour'] >= 20) | (df['hour'] < 6))
    ]

    for titulo, mask in franjas:
        franja_df = df[mask]
        # Todos los días en la franja
        _imprimir_tabla(franja_df, f"{titulo} - Todos los días")

        subsets = [
            ("Lunes a Viernes", franja_df[franja_df['day_type'] == 'Weekday']),
            ("Sábado y Domingo", franja_df[franja_df['day_type'] == 'Weekend'])
        ]
        for subtitulo, df_sub in subsets:
            _imprimir_tabla(df_sub, f"{titulo} - {subtitulo}")

    # Exportar todas las tablas generadas a un solo CSV
    if exported_tables:
        export_path = os.path.join(
            RESULTADOS_DIR,
            f"tabla_A_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        pd.concat(exported_tables, ignore_index=True).to_csv(export_path, index=False)
        print(f"\n✅ Tablas exportadas a CSV: {export_path}")

    print("="*80 + "\n")

def analisis_dinamica_temporal(minute_agg, show_tables=True, show_flow_plot=True, show_speed_plot=True, show_seasonality=True):
    """
    Realiza un análisis de la dinámica temporal del flujo vehicular, incluyendo
    tablas y gráficos para perfiles horarios y estacionalidad.
    """
    if show_flow_plot or show_speed_plot:
        _require_matplotlib()
        _require_seaborn()

    print("\n" + "="*80)
    print(" " * 20 + "1. Dinámica Temporal Multiescala")
    print("="*80)

    # --- Preparación de Datos ---
    minute_agg['hour'] = minute_agg['FECHA'].dt.hour
    minute_agg['day_type'] = minute_agg['FECHA'].dt.dayofweek.apply(
        lambda x: 'Weekend' if x >= 5 else 'Weekday'
    )
    
    if show_tables:
        # --- Perfil Horario Promedio (Tablas) ---
        print("\n--- Tablas de Perfil Horario Promedio (Weekday vs Weekend) ---")
        hourly_profile_table = minute_agg.groupby(['day_type', 'hour', 'CATEGORIA']).agg(
            avg_flow=('flow', 'mean'),
            avg_speed=('mean_speed', 'mean')
        ).round(2).unstack(level='CATEGORIA').fillna(0)
        
        if 'Weekday' in hourly_profile_table.index:
            print("\n[Weekday]")
            print(hourly_profile_table.loc['Weekday'])
        if 'Weekend' in hourly_profile_table.index:
            print("\n[Weekend]")
            print(hourly_profile_table.loc['Weekend'])

    if show_flow_plot or show_speed_plot:
        # --- Perfil Horario Promedio (Gráficos) ---
        print("\n--- Gráficos de Perfil Horario Promedio ---")
        
        hourly_profile_plot = minute_agg.groupby(['day_type', 'hour']).agg(
            total_flow=('flow', 'sum'),
            avg_speed=('mean_speed', 'mean')
        ).reset_index()

        if not hourly_profile_plot.empty:
            sns.set_style("white")
            palette = ["black", "orange"]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            if show_flow_plot:
                # --- Gráfico de Flujo ---
                plt.figure(figsize=(12, 6))
                ax_flow = sns.lineplot(data=hourly_profile_plot, x='hour', y='total_flow', hue='day_type', marker='o', palette=palette)
                #ax_flow.set_title('Perfil Horario de Flujo Vehicular Total (Weekday vs Weekend)', fontsize=16)
                ax_flow.set_ylabel('Total Average Flow (veh/min)')
                ax_flow.set_xlabel('Time of Day')
                ax_flow.set_xticks(range(0, 24))
                ax_flow.spines['top'].set_visible(False)
                ax_flow.spines['right'].set_visible(False)
                ax_flow.spines['bottom'].set_color('#333333')
                ax_flow.spines['left'].set_color('#333333')
                #ax_flow.legend(title='Tipo de Día')
                ax_flow.grid(False)
                
                plt.tight_layout()
                save_path_flow = os.path.join(RESULTADOS_DIR, f"perfil_horario_flujo_{timestamp}.png")
                plt.savefig(save_path_flow, format="png", dpi=300, bbox_inches="tight")
                print(f"\n✅ Gráfico de perfil de flujo guardado en: {save_path_flow}")
                plt.show()

            if show_speed_plot:
                # --- Gráfico de Velocidad ---
                plt.figure(figsize=(12, 6))
                ax_speed = sns.lineplot(data=hourly_profile_plot, x='hour', y='avg_speed', hue='day_type', marker='o', palette=palette)
                #ax_speed.set_title('Perfil Horario de Velocidad Media (Weekday vs Weekend)', fontsize=16)
                ax_speed.set_ylabel('Speed mean (km/h)')
                ax_speed.set_xlabel('Time of Day')
                ax_speed.set_xticks(range(0, 24))
                ax_speed.spines['top'].set_visible(False)
                ax_speed.spines['right'].set_visible(False)
                ax_speed.spines['bottom'].set_color('#333333')
                ax_speed.spines['left'].set_color('#333333')
                #ax_speed.legend(title='Tipo de Día')
                ax_speed.grid(False)

                plt.tight_layout()
                save_path_speed = os.path.join(RESULTADOS_DIR, f"perfil_horario_velocidad_{timestamp}.png")
                plt.savefig(save_path_speed, format="png", dpi=300, bbox_inches="tight")
                print(f"\n✅ Gráfico de perfil de velocidad guardado en: {save_path_speed}")
                plt.show()

        else:
            print("No hay suficientes datos para generar los gráficos de perfil horario.")

    if show_seasonality:
        # --- Factor de Estacionalidad Semanal ---
        print("\n--- Factor de Estacionalidad Semanal ---")
        daily_flow = minute_agg.groupby(minute_agg['FECHA'].dt.date)['flow'].sum()
        global_mean_flow = daily_flow.mean()
        
        if global_mean_flow > 0:
            seasonality_factor = (daily_flow / global_mean_flow)
            seasonality_df = seasonality_factor.reset_index()
            seasonality_df.columns = ['Fecha', 'Factor_Estacionalidad']
            seasonality_df['Dia_Semana'] = pd.to_datetime(seasonality_df['Fecha']).dt.day_name()
            
            # Agrupar por día de la semana y calcular el factor promedio
            weekly_seasonality = seasonality_df.groupby('Dia_Semana')['Factor_Estacionalidad'].mean().round(3)
            
            # Ordenar los días de la semana
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly_seasonality = weekly_seasonality.reindex(days_order).reset_index()
            
            print(weekly_seasonality.to_string(index=False))
        else:
            print("No hay suficiente flujo para calcular el factor de estacionalidad.")

def analisis_composicion_vehicular(minute_agg):
    """
    Analiza la composición vehicular diaria a partir del DataFrame agregado por minuto.
    """
    _require_matplotlib()
    _require_seaborn()
    print("\n" + "="*80)
    print(" " * 20 + "2. Composición Vehicular Variable en el Tiempo")
    print("="*80)

    if minute_agg is None or minute_agg.empty:
        print("No hay datos para analizar la composición vehicular.")
        return

    # --- Construcción de la serie diaria ---
    print("\n--- Calculando composición vehicular diaria desde datos agregados... ---")
    minute_agg = minute_agg.copy()
    minute_agg['FECHA_DIA'] = minute_agg['FECHA'].dt.floor('D')
    daily_composition = (
        minute_agg.groupby(['FECHA_DIA', 'CATEGORIA'])['flow']
        .sum()
        .unstack(fill_value=0)
    )

    if daily_composition.empty:
        print("No hay suficientes datos para calcular la composición diaria.")
        return

    # --- Gráfico de Serie Temporal Suavizada ---
    print("\n--- Gráfico de Serie Temporal Suavizada (7 días) ---")
    daily_composition_pct = daily_composition.apply(
        lambda x: x / x.sum() * 100 if x.sum() > 0 else x, axis=1
    )
    
    if len(daily_composition_pct) > 7:
        daily_composition_smooth = daily_composition_pct.rolling(window=7, min_periods=1).mean()
        
        sns.set_style("white")
        from itertools import cycle
        palette = cycle(["black", "orange", "#666666"])

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1]})
        fig.subplots_adjust(hspace=0.05)  # adjust space between axes

        # Plot the same data on both axes
        for cat in daily_composition_smooth.columns:
            color = next(palette)
            ax1.plot(daily_composition_smooth.index, daily_composition_smooth[cat], label=cat, color=color)
            ax2.plot(daily_composition_smooth.index, daily_composition_smooth[cat], label=cat, color=color)

        # Set y-limits for each subplot
        ax1.set_ylim(85, 100)  # Upper part of the plot
        ax2.set_ylim(0, 10)   # Lower part of the plot

        # Hide the spines between ax1 and ax2
        ax1.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax1.xaxis.tick_top()
        ax1.tick_params(labeltop=False)  # Don't put tick labels at the top
        ax2.xaxis.tick_bottom()

        # Add diagonal lines to indicate the break
        d = .015  # how big to make the diagonal lines in axes coordinates
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
        ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

        # Set titles and labels
        fig.suptitle('Composición Vehicular (Media Móvil de 7 Días)', fontsize=16)
        fig.text(0.04, 0.5, 'Porcentaje del Total (%)', va='center', rotation='vertical')
        ax2.set_xlabel('Fecha')
        
        # Common legend
        handles, labels = ax1.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        fig.legend(unique_labels.values(), unique_labels.keys(), title='Categoría', loc='upper right')

        ax1.grid(False)
        ax2.grid(False)

        plt.tight_layout(rect=[0.05, 0, 1, 0.95]) # Adjust layout to make room for suptitle and centered ylabel

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(RESULTADOS_DIR, f"composicion_vehicular_{timestamp}.png")
        plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
        print(f"\n✅ Gráfico de composición vehicular guardado en: {save_path}")
        plt.show()
    else:
        print("No hay suficientes datos para generar el gráfico de serie temporal (se requieren más de 7 días).")

    # --- Tabla de Percentiles de Composición ---
    print("\n--- Tabla de Percentiles de Composición Vehicular Diaria ---")
    if not daily_composition_pct.empty:
        percentiles = daily_composition_pct.quantile([0.1, 0.5, 0.9]).round(2)
        percentiles.index = ['p10', 'p50', 'p90']
        print(percentiles)
    else:
        print("No hay datos de composición para calcular percentiles.")

def analisis_velocidad_flujo(minute_agg, agg_freq_minutes=60):
    """
    Analiza la relación entre velocidad y flujo.
    """
    _require_matplotlib()
    _require_seaborn()
    _require_statsmodels()
    print("\n" + "="*80)
    print(" " * 20 + "3. Relación Velocidad-Flujo")
    print("="*80)

    if minute_agg.empty:
        print("No hay datos agregados para generar el gráfico de velocidad vs. flujo.")
        return

    # Agregación a nivel de autopista
    highway_agg = minute_agg.groupby(pd.Grouper(key='FECHA', freq=f'{agg_freq_minutes}min')).agg(
        total_flow=('flow', 'sum'),
        avg_speed=('mean_speed', 'mean')
    ).dropna()

    if len(highway_agg) < 20:
        print("No hay suficientes datos para generar un gráfico de velocidad-flujo significativo.")
        return

    # Submuestreo para visualización
    sample_df = highway_agg.sample(n=min(5000, len(highway_agg)), random_state=SEED)
    
    # Ajuste de la curva LOESS
    frac = 0.3
    lowess_smooth = lowess(sample_df['avg_speed'], sample_df['total_flow'], frac=frac)
    
    # Cálculo de estadísticas para la leyenda
    mean_speed_overall = highway_agg['avg_speed'].mean()
    q_95 = highway_agg['total_flow'].quantile(0.95)
    
    # Calcular la pendiente (derivada) de la curva LOESS
    # Se usa la mediana de las pendientes para robustez
    delta_x = np.diff(lowess_smooth[:, 0])
    delta_y = np.diff(lowess_smooth[:, 1])
    
    # Evitar división por cero y calcular pendientes válidas
    valid_mask = delta_x != 0
    slopes = np.full_like(delta_x, np.nan)
    np.divide(delta_y, delta_x, out=slopes, where=valid_mask)

    # Calcular la mediana de las pendientes finitas
    if np.any(np.isfinite(slopes)):
        median_slope = np.nanmedian(slopes)
    else:
        median_slope = np.nan

    # --- Creación del Gráfico ---
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(12, 9))

    # Gráfico de dispersión y curva LOESS
    ax.scatter(sample_df['total_flow'], sample_df['avg_speed'], alpha=0.5, label='Subsample', color='orange', edgecolors='none', s=30)
    ax.plot(lowess_smooth[:, 0], lowess_smooth[:, 1], color='black', lw=2.5, label='LOESS curve')
    
    # Títulos y etiquetas
    #ax.set_title(f'Relación Velocidad-Flujo (Agregado cada {agg_freq_minutes} min)', fontsize=18, pad=20)
    ax.set_xlabel('Total flow (veh/h)', fontsize=12)
    ax.set_ylabel('Speed mean (km/h)', fontsize=12)
    
    # Estética del gráfico
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#333333')
    ax.spines['left'].set_color('#333333')

    # --- Creación del cuadro de texto con estadísticas ---
    stats_text = (
        f"Span: {frac:.2f} (linear)\n"
        f"Mean speed (overall): {mean_speed_overall:.2f} km/h\n"
        f"Slope (median): {median_slope:.2f} km/h per veh/min\n"
        f"q_95: {q_95:.0f} veh/min"
    )

    # Añadir el cuadro de texto al gráfico
    #props = dict(boxstyle='round,pad=0.5', facecolor='none', alpha=0.5)
    #ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
    #        verticalalignment='top', horizontalalignment='right', bbox=props)

    # Leyenda principal
    ax.legend(loc='lower left')

    plt.tight_layout()

    # Guardar el gráfico
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(RESULTADOS_DIR, f"velocidad_flujo_{timestamp}.png")
    plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
    print(f"\n✅ Gráfico de velocidad-flujo guardado en: {save_path}")
    plt.show()

def generar_estadisticas_flujo_en_chunks():
    """
    Permite al usuario seleccionar archivos de flujo, los procesa en chunks y muestra
    estadísticas descriptivas detalladas y visualizaciones.
    """
    print("\n--- Generador de Estadísticas de Flujo (Modo Chunks) ---")
    
    data_folder = "Datos"
    try:
        csv_files = sorted([f for f in os.listdir(data_folder) if f.lower().endswith('.csv')])
    except FileNotFoundError:
        print(f"❌ Error: La carpeta '{data_folder}' no existe.")
        return
    
    if not csv_files:
        print(f"❌ No se encontraron archivos CSV en la carpeta '{data_folder}'.")
        return

    print("\nArchivos de flujo disponibles:")
    for i, f in enumerate(csv_files, 1):
        print(f"  [{i}] {f}")

    while True:
        try:
            selections = input("\nSelecciona los números de los archivos a analizar, separados por comas (ej: 1,3): ").strip()
            if not selections:
                print("Selección cancelada.")
                return
            indices = [int(s.strip()) - 1 for s in selections.split(',')]
            if not all(0 <= i < len(csv_files) for i in indices):
                raise ValueError("Uno o más números están fuera de rango.")
            selected_files = [os.path.join(data_folder, csv_files[i]) for i in indices]
            break
        except (ValueError, IndexError) as e:
            print(f"❌ Error en la selección: {e}. Inténtalo de nuevo.")

    all_minute_agg = []

    print("\nCargando y procesando archivos en chunks...")
    for file_path in selected_files:
        print(f"\nProcesando archivo: {file_path}")
        try:
            # Contar líneas para una barra de progreso más informativa
            chunksize = 1000000  
            num_lines = _count_lines(file_path)
            total_chunks = math.ceil(num_lines / chunksize) if num_lines > 0 else 1

            # Usar un iterador de chunks para no cargar todo en memoria
            chunk_iter = pd.read_csv(file_path, chunksize=chunksize, low_memory=False)
            
            for chunk in tqdm(chunk_iter, total=total_chunks, desc=f"Procesando {os.path.basename(file_path)}"):
                # Procesar cada chunk
                chunk.columns = [c.strip().upper() for c in chunk.columns]

                fecha_col = buscar_columna(chunk, "FECHA")
                velo_col = buscar_columna(chunk, "VELOCIDAD")
                cat_col = buscar_columna(chunk, "CATEGORIA")
                portico_col = buscar_columna(chunk, "PORTICO")

                if velo_col and velo_col in chunk.columns:
                    chunk.loc[:, velo_col] = pd.to_numeric(chunk[velo_col], errors='coerce')

                    # --- Detección y eliminación de outliers en la velocidad del chunk ---
                    if pd.api.types.is_numeric_dtype(chunk[velo_col]):
                        Q1 = chunk[velo_col].quantile(0.25)
                        Q3 = chunk[velo_col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = 1 #Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR

                        if pd.notnull(lower_bound) and pd.notnull(upper_bound):
                            valid_mask = (chunk[velo_col] >= lower_bound) & (chunk[velo_col] <= upper_bound)
                            if not valid_mask.all():
                                chunk = chunk[valid_mask].copy()

                chunk.loc[:, fecha_col] = pd.to_datetime(chunk[fecha_col], format='%d-%m-%Y %H:%M:%S', errors='coerce')
                chunk = chunk.dropna(subset=[fecha_col, velo_col])

                cat_map = {'1': 'Light', '2': 'Heavy', '3': 'Motorcycle'}
                chunk[cat_col] = chunk[cat_col].astype(str).map(cat_map).fillna('Other')

                df_proc_chunk = chunk[[fecha_col, portico_col, cat_col, velo_col]].rename(columns={
                    fecha_col: 'FECHA',
                    portico_col: 'PORTICO',
                    cat_col: 'CATEGORIA',
                    velo_col: 'VELOCIDAD'
                })

                minute_agg_chunk = compute_minute_aggregates(df_proc_chunk, freq='min')
                # Reducir dtypes para aliviar el consumo de memoria en la concatenación final
                minute_agg_chunk['flow'] = minute_agg_chunk['flow'].astype('int32', copy=False)
                minute_agg_chunk['mean_speed'] = minute_agg_chunk['mean_speed'].astype('float32', copy=False)
                minute_agg_chunk['speed_std'] = minute_agg_chunk['speed_std'].astype('float32', copy=False)

                all_minute_agg.append(minute_agg_chunk)

        except Exception as e:
            print(f"❌ Error procesando el archivo {file_path}: {e}")
            continue

    if not all_minute_agg:
        print("❌ No se pudieron procesar los datos de los archivos seleccionados.")
        return

    final_minute_agg = pd.concat(all_minute_agg, ignore_index=True)

    while True:
        print("\n--- Menú de Análisis de Flujo ---")
        print("  [1] Tabla A - Estadísticas Descriptivas Globales por Categoría")
        print("  [2] Dinámica Temporal Multiescala (Tablas y Gráficos)")
        print("  [3] Gráfico de perfil de flujo")
        print("  [4] Gráfico de perfil de velocidad")
        print("  [5] Composición Vehicular Variable en el Tiempo")
        print("  [6] Relación Velocidad-Flujo")
        print("  [7] Ejecutar TODO el flujo de análisis")
        print("\n  [0] Salir")
        
        choice = input("Seleccione una opción: ").strip()

        if choice == '1':
            generar_tabla_A(final_minute_agg.copy())
        elif choice == '2':
            analisis_dinamica_temporal(final_minute_agg.copy())
        elif choice == '3':
            analisis_dinamica_temporal(final_minute_agg.copy(), show_tables=False, show_flow_plot=True, show_speed_plot=False, show_seasonality=False)
        elif choice == '4':
            analisis_dinamica_temporal(final_minute_agg.copy(), show_tables=False, show_flow_plot=False, show_speed_plot=True, show_seasonality=False)
        elif choice == '5':
            analisis_composicion_vehicular(final_minute_agg.copy())
        elif choice == '6':
            analisis_velocidad_flujo(final_minute_agg.copy())
        elif choice == '7':
            print("\n--- Ejecutando todos los análisis ---")
            generar_tabla_A(final_minute_agg.copy())
            analisis_dinamica_temporal(final_minute_agg.copy())
            analisis_composicion_vehicular(final_minute_agg.copy())
            analisis_velocidad_flujo(final_minute_agg.copy())
            print("\n--- Todos los análisis completados ---")
        elif choice == '0':
            break
        else:
            print("❌ Opción no válida. Inténtelo de nuevo.")

    print("\n" + "="*80)
    print("✅ Análisis de flujo completado.")
    print("="*80 + "\n")

def analizar_puntos_negros_accidentes():
    """
    Analiza la concentración de accidentes por tramo y franja horaria para
    identificar los 'puntos negros'.
    """
    print("\n--- Análisis de Puntos Negros de Accidentes ---")

    # 1. Cargar datos
    print("\nPor favor, seleccione el archivo de PÓRTICOS.")
    df_porticos = load_porticos()
    if df_porticos is None or df_porticos.empty:
        print("❌ Carga de pórticos cancelada o fallida. Abortando análisis.")
        return

    print("\nPor favor, seleccione el archivo de ACCIDENTES.")
    df_acc = load_accidentes()
    if df_acc is None or df_acc.empty:
        print("❌ Carga de accidentes cancelada o fallida. Abortando análisis.")
        return

    # 2. Generar tramos (pórtico_inicio -> pórtico_fin)
    df_sorted = df_porticos.sort_values(by=["calzada", "eje", "orden"])
    siguiente_portico_map = {}
    for (calzada, eje), group in df_sorted.groupby(['calzada', 'eje']):
        porticos = group['portico'].tolist()
        for i in range(len(porticos) - 1):
            siguiente_portico_map[porticos[i]] = porticos[i+1]

    # 3. Procesar accidentes para asignar tramo y franja horaria
    df_acc['siguiente_portico'] = df_acc['ultimo_portico'].map(siguiente_portico_map)
    
    # Descartar accidentes cuyo pórtico anterior es el último de una secuencia
    df_acc.dropna(subset=['siguiente_portico'], inplace=True)
    if df_acc.empty:
        print("❌ No se pudieron asociar los accidentes a tramos definidos. Verifica que los pórticos y accidentes correspondan.")
        return

    df_acc['tramo'] = df_acc['ultimo_portico'].astype(str) + ' -> ' + df_acc['siguiente_portico'].astype(str)
    
    # Asignar franja horaria
    hours = df_acc['accidente_time'].dt.hour
    conditions = [
        (hours >= 6) & (hours < 10),
        (hours >= 18) & (hours < 22)
    ]
    choices = ['Mañana (06-10h)', 'Tarde (18-22h)']
    df_acc['franja_horaria'] = np.select(conditions, choices, default='Resto del día')

    # 4. Realizar el análisis estadístico (accidentes por tramo y franja)
    stats = df_acc.groupby(['tramo', 'franja_horaria']).size().reset_index(name='n_accidentes')

    # 4.a Calcular horas de cada franja_horaria a partir de la etiqueta (e.g., "06-10h")
    import re
    unique_franjas = stats['franja_horaria'].unique().tolist()
    horas_map = {}
    total_known = 0
    for fh in unique_franjas:
        m = re.search(r"(\d{2})-(\d{2})h", str(fh))
        if m:
            h0, h1 = int(m.group(1)), int(m.group(2))
            horas_map[fh] = max(0, h1 - h0)
            total_known += horas_map[fh]
        else:
            horas_map[fh] = None  # p.ej., "Resto del día"

    # Asignar horas restantes a 'Resto del día' (o etiquetas sin rango horario)
    resto_horas = max(0, 24 - total_known) if total_known > 0 else 24
    for fh in unique_franjas:
        if horas_map[fh] is None:
            horas_map[fh] = resto_horas

    stats['horas_franja'] = stats['franja_horaria'].map(horas_map).astype(float)
    # Evitar divisiones por cero
    stats['horas_franja'].replace({0.0: np.nan}, inplace=True)
    stats['accidentes_por_hora'] = stats['n_accidentes'] / stats['horas_franja']

    # Ordenar por tasa (accidentes/hora) y luego por número de accidentes
    stats.sort_values(by=['accidentes_por_hora', 'n_accidentes', 'tramo'], ascending=[False, False, True], inplace=True)

    if stats.empty:
        print("✅ No se encontraron accidentes en los tramos y franjas horarias analizadas.")
        return

    # 5. Mostrar resultados en la terminal
    punto_negro = stats.iloc[0]
    
    print("\n" + "="*80)
    print(" " * 25 + "RESULTADOS DEL ANÁLISIS DE PUNTOS NEGROS")
    print("="*80)
    print(f"\n🚨 ¡PUNTO NEGRO IDENTIFICADO! 🚨")
    print(f"  - Tramo: {punto_negro['tramo']}")
    print(f"  - Franja Horaria: {punto_negro['franja_horaria']}")
    print(f"  - Número de Accidentes: {int(punto_negro['n_accidentes'])}")
    print(f"  - Horas en Franja: {int(punto_negro['horas_franja'])}")
    print(f"  - Tasa (accidentes/hora): {float(punto_negro['accidentes_por_hora']):.4f}")
    print("-" * 80)
    
    print("\nTabla de Estadísticas Completas (ordenada por peligrosidad):\n")
    print(stats.to_string(index=False))
    print("\n" + "="*80)

    # 6. Guardar resultados en CSV
    timestamp = datetime.now().strftime('%d%m%Y_%H%M')
    filename = f"analisis_puntos_negros_{timestamp}.csv"
    output_path = os.path.join(RESULTADOS_DIR, filename)
    
    try:
        stats.to_csv(output_path, index=False, sep=';', decimal=',')
        print(f"\n✅ Resultados guardados exitosamente en: {output_path}")
    except Exception as e:
        print(f"\n❌ Error al guardar el archivo CSV: {e}")

def run_visualization_app():
    """Lanza la aplicación de visualización de Streamlit."""
    import subprocess
    import sys
    import os
    
    # Construir la ruta al script de streamlit usando el ejecutable de python del venv
    streamlit_script = os.path.join(os.path.dirname(sys.executable), 'streamlit')
    
    # Construir la ruta al script de visualización relativo a este archivo
    # __file__ es la ruta de utils.py, que está en 'src'
    visualization_script = os.path.join(os.path.dirname(__file__), 'visualization.py')
    
    command = [streamlit_script, 'run', visualization_script]
    
    print(f"\nLanzando visualizador...\nComando: {' '.join(command)}\n")
    try:
        # Usamos Popen para no bloquear el proceso principal. 
        # Streamlit gestiona su propio ciclo de vida.
        process = subprocess.Popen(command)
        process.wait() # Espera a que el proceso termine (cuando el usuario cierra la app)
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el ejecutable de streamlit en '{streamlit_script}'.")
        print("Asegúrate de que Streamlit está instalado en el entorno virtual correcto (`env/bin/pip install streamlit`).")
    except Exception as e:
        print(f"❌ Error al ejecutar la aplicación de Streamlit: {e}")

def menu_utilidades_csv():
    limpiar_pantalla()
    while True:
        print("===== Menú de Utilidades CSVs=====")
        print("  [1] Unir CSVs")
        print("  [2] Estandarizar CSVs de Flujos")
        print("  [3] Estadísticas descriptivas de un CSV")
        print("  [4] Visualizar estructura de pórticos")
        print("  [5] Generar Estadísticas de Flujos")
        print("  [6] Analizar puntos negros de accidentes")
        print("  [7] Inspeccionar Features CSV (porticos/días/horarios)")
        print("  [8] Visualizar Features Interactivamente (HTML)")
        print("  [9] Renderizar PDF de documentación (LaTeX)")
        print("")
        print("  [0] Volver")
        print("")
        choice = input("Elige una opción: ").strip()
        if choice == "1":
            unir_csvs_en_directorio()
        elif choice == "2":
            estandarizar_csvs_FLUJOS()
        elif choice == "3":
            generar_estadisticas_descriptivas()
        elif choice == "4":
            visualizar_estructura_porticos()
        elif choice == "5":
            generar_estadisticas_flujo_en_chunks()
        elif choice == "6":
            analizar_puntos_negros_accidentes()
        elif choice == "7":
            inspeccionar_features_csv()
        elif choice == "8":
            run_visualization_app()
        elif choice == "9":
            renderizar_documentacion_pdf()
        elif choice == "0":
            limpiar_pantalla()
            break
        else:
            print("❌ Opción no reconocida.")

# --------------------------------------------------------------------------- #
# Utilidad: Renderizar documentación LaTeX a PDF
# --------------------------------------------------------------------------- #
def renderizar_documentacion_pdf():
    """
    Compila 'pipeline_documentacion.tex' a PDF y lo deja en 'Resultados/docs/'.
    Requiere tener instalado 'latexmk' o 'pdflatex'/'xelatex' en el PATH.
    """
    try:
        import subprocess, shutil
    except Exception as e:
        print(f"❌ Error importando dependencias del sistema: {e}")
        return

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    tex_path = os.path.join(repo_root, 'pipeline_documentacion.tex')
    if not os.path.exists(tex_path):
        print(f"❌ No se encontró el archivo LaTeX en: {tex_path}")
        return

    out_dir = os.path.join(RESULTADOS_DIR, 'docs')
    os.makedirs(out_dir, exist_ok=True)

    # Preferir latexmk; fallback a (xe|pdf|lua)latex
    latexmk = shutil.which('latexmk')
    engines = [
        shutil.which('pdflatex'),
        shutil.which('xelatex'),
        shutil.which('lualatex'),
    ]
    engine = next((e for e in engines if e), None)

    if latexmk:
        cmd = [latexmk, '-pdf', '-interaction=nonstopmode', '-halt-on-error',
               f'-output-directory={out_dir}', tex_path]
        engine_name = 'latexmk'
    elif engine:
        cmd = [engine, '-interaction=nonstopmode', '-halt-on-error',
               f'-output-directory={out_dir}', tex_path]
        engine_name = os.path.basename(engine)
    else:
        print("❌ No se encontraron 'latexmk' ni 'pdflatex/xelatex/lualatex' en el PATH.")
        print("   Instala TeX Live o MacTeX. Ejemplos:")
        print("   • macOS: brew install --cask mactex-no-gui")
        print("   • Ubuntu/Debian: sudo apt-get install texlive-full")
        return

    print("\n▶️ Compilando documentación LaTeX → PDF…")
    print(f"   Motor: {engine_name}")
    try:
        subprocess.run(cmd, cwd=repo_root, check=True)
        # Segunda pasada si no usamos latexmk (para TOC, refs)
        if not latexmk and engine:
            subprocess.run(cmd, cwd=repo_root, check=True)
    except subprocess.CalledProcessError as e:
        print("❌ Error al compilar LaTeX. Revisa el log en consola.")
        return
    except FileNotFoundError:
        print("❌ No se pudo invocar el motor LaTeX. Verifica la instalación.")
        return

    pdf_out = os.path.join(out_dir, 'pipeline_documentacion.pdf')
    if os.path.exists(pdf_out):
        print(f"\n✅ PDF generado en: {pdf_out}")
    else:
        # Intentar localizar cualquier PDF recién generado
        candidates = sorted(glob.glob(os.path.join(out_dir, '*.pdf')), key=os.path.getmtime)
        if candidates:
            print(f"\n✅ PDF generado en: {candidates[-1]}")
        else:
            print(f"⚠️ Compilación finalizada, pero no se encontró el PDF en {out_dir}.")

# --------------------------------------------------------------------------- #
# Utilidad: Inspeccionar CSV de features (pm)
# --------------------------------------------------------------------------- #
def _select_features_csv() -> Optional[str]:
    """Lista archivos features_*.csv en RESULTADOS_DIR y devuelve la ruta.
    - Si solo hay uno, lo selecciona automáticamente.
    - Excluye archivos de nombres de features (features_selected_names_*).
    """
    files = sorted(glob.glob(os.path.join(RESULTADOS_DIR, "features_*.csv")))
    # Excluir listas de nombres y cualquier features_{texto}; mantener solo features_{números}.csv
    files = [
        f for f in files
        if not os.path.basename(f).startswith("features_selected_names_")
        and re.match(r"^features_\d", os.path.basename(f))
    ]
    if not files:
        print("❌ No se encontraron archivos 'features_*.csv' en la carpeta Resultados.")
        return None
    if len(files) == 1:
        print(f"✅ Cargando automáticamente: {os.path.basename(files[0])}")
        return files[0]
    print("Archivos de features disponibles:")
    for i, f in enumerate(files, 1):
        print(f"  [{i}] {os.path.basename(f)}")
    try:
        sel = int(input("Seleccione el archivo de features: ").strip()) - 1
        if 0 <= sel < len(files):
            return files[sel]
    except Exception:
        pass
    print("❌ Selección inválida.")
    return None

def inspeccionar_features_csv():
    """Explora un CSV de features (pm):
    - Filtra por pórticos, día, franja horaria.
    - Muestra cobertura de horas, conteos esperados (según DT_MINUTES), huecos/duplicados.
    - Imprime estadísticas de variables seleccionadas y permite exportar un subset.
    """
    path = _select_features_csv()
    if path is None:
        return
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"❌ Error al leer {path}: {e}")
        return

    portico_col = buscar_columna(df, "portico")
    if portico_col is None or 'ts_min' not in df.columns:
        print("❌ El CSV no contiene columnas requeridas: 'portico' y 'ts_min'.")
        return

    # Normalizar tipos
    df = df.copy()
    df['ts_min'] = df['ts_min'].astype(int)
    df[portico_col] = df[portico_col].astype(str)
    dt = pd.to_datetime(df['ts_min'], unit='m')
    df['datetime'] = dt
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour

    # Estadísticas globales
    print("\n--- Resumen Global ---")
    try:
        resum = df.groupby(portico_col).agg(rows=("ts_min","count"),
                                            ts_unique=("ts_min", pd.Series.nunique),
                                            first_ts=("ts_min","min"),
                                            last_ts=("ts_min","max")).reset_index()
        resum['first_time'] = pd.to_datetime(resum['first_ts'], unit='m')
        resum['last_time']  = pd.to_datetime(resum['last_ts'],  unit='m')
        print(resum[[portico_col,'rows','ts_unique','first_time','last_time']].to_string(index=False))
    except Exception as e:
        print(f"⚠️ No se pudo generar resumen global: {e}")

    unique_ports = sorted(df[portico_col].unique(), key=lambda x: (len(x), x))
    unique_dates = sorted(df['date'].unique())
    min_hour, max_hour = int(df['hour'].min()), int(df['hour'].max())
    print(f"\nPórticos disponibles ({len(unique_ports)}): {unique_ports}")
    print(f"Rango de fechas: {unique_dates[0]} → {unique_dates[-1]}  | Horas: {min_hour}–{max_hour}")

    # Selección interactiva
    ports_in = input("Ingrese pórticos separados por coma (Enter = todos): ").strip()
    if ports_in:
        sel_ports = {p.strip() for p in ports_in.split(',') if p.strip()}
        present_ports = set(df[portico_col].unique())
        missing_ports = sorted(list(sel_ports - present_ports))
        if missing_ports:
            print(f"⚠️ Los siguientes pórticos NO tienen filas en este CSV: {missing_ports}")
        df = df[df[portico_col].isin(present_ports & sel_ports)].copy()
        if df.empty:
            print("❌ No hay filas para los pórticos seleccionados.")
            return

    # Elegir día (opcional)
    choose_day = input("¿Filtrar por un día específico? (s/n): ").strip().lower()
    if choose_day in ('s','si','y','yes'):
        # Mostrar lista enumerada de días
        print("Días disponibles:")
        for i, d in enumerate(unique_dates, 1):
            print(f"  [{i}] {d}")
        try:
            idx = int(input("Seleccione el día: ").strip()) - 1
            if 0 <= idx < len(unique_dates):
                day = unique_dates[idx]
                df = df[df['date'] == day].copy()
            else:
                print("❌ Día inválido.")
                return
        except Exception:
            print("❌ Entrada inválida.")
            return

    # Elegir franja horaria (opcional)
    print("\nFranjas horarias:")
    print("  [1] Mañana (6–10)")
    print("  [2] Tarde (18–22)")
    print("  [3] Completo (sin filtro)")
    print("  [4] Personalizado (h_ini-h_fin)")
    fh = input("Seleccione franja [1/2/3/4]: ").strip()
    if fh == '1':
        h_ini, h_fin = 6, 10
    elif fh == '2':
        h_ini, h_fin = 18, 22
    elif fh == '4':
        try:
            rng = input("Ingrese rango horario (ej. 7-9): ").strip()
            h_ini, h_fin = map(int, rng.split('-'))
        except Exception:
            print("❌ Rango inválido.")
            return
    else:
        h_ini, h_fin = None, None

    present_all_ports = set(df[portico_col].unique())
    if h_ini is not None:
        before = len(df)
        df = df[(df['hour'] >= h_ini) & (df['hour'] < h_fin)].copy()
        print(f"Aplicado filtro horario {h_ini}:00–{h_fin}:00 → {len(df)} filas (antes {before}).")
        if df.empty:
            print("❌ Sin filas tras el filtro horario.")
            return
        after_ports = set(df[portico_col].unique())
        lost_ports = sorted(list(present_all_ports - after_ports))
        if lost_ports:
            print(f"⚠️ Pórticos que tienen datos fuera de esta franja pero NO dentro: {lost_ports}")

    # Chequeos de cobertura y huecos
    print("\n--- Cobertura y Calidad por Pórtico ---")
    for p in sorted(df[portico_col].unique(), key=lambda x: (len(x), x)):
        sub = df[df[portico_col]==p].sort_values('ts_min')
        cnt = len(sub)
        expected = None
        if h_ini is not None and 'date' in sub.columns:
            # Esperado: por día, número de bins en la franja seleccionada
            bins_per_day = int(((h_fin - h_ini) * 60) / max(1, DT_MINUTES))
            n_days = int(sub['date'].nunique())
            expected = bins_per_day * n_days
        # Huecos y duplicados
        diffs = sub['ts_min'].diff().dropna().astype(int)
        gaps = (diffs != DT_MINUTES).sum()
        dups = sub['ts_min'].duplicated().sum()
        first_t = pd.to_datetime(int(sub['ts_min'].min()), unit='m') if cnt>0 else None
        last_t  = pd.to_datetime(int(sub['ts_min'].max()), unit='m') if cnt>0 else None
        print(f"  - Pórtico {p}: filas={cnt} esperado={expected if expected is not None else '-'} huecos={gaps} duplicados={dups} rango=[{first_t} → {last_t}]")
        # Muestra de horas presentes
        hours_present = sub['hour'].value_counts().sort_index().to_dict()
        print(f"    horas_presentes: {hours_present}")

    # Estadísticas de variables
    print("\n--- Estadísticas de Variables (describe) ---")
    feature_cols = [c for c in df.columns if c not in [portico_col,'ts_min','datetime','date','hour']]
    if feature_cols:
        try:
            print(df[feature_cols].describe().transpose().to_string())
        except Exception as e:
            print(f"⚠️ No se pudo calcular describe(): {e}")
    else:
        print("(No hay columnas de features para describir)")

    # Muestras
    print("\n--- Muestras (primeras 10 filas ordenadas) ---")
    sample_cols = [portico_col,'ts_min','datetime','hour'] + feature_cols[:min(7, len(feature_cols))]
    sort_cols = [portico_col, 'ts_min'] if portico_col in df.columns else ['ts_min']
    print(df.sort_values(sort_cols).head(10)[sample_cols].to_string(index=False))

    # Exportar subset
    save = input("\n¿Exportar el subset filtrado a CSV? (s/n): ").strip().lower()
    if save in ('s','si','y','yes'):
        os.makedirs(RESULTADOS_DIR, exist_ok=True)
        out = os.path.join(RESULTADOS_DIR, f"features_subset_{datetime.now().strftime('%d%m%Y_%H%M')}.csv")
        try:
            df.to_csv(out, index=False)
            print(f"✅ Subset exportado a → {out}")
        except Exception as e:
            print(f"❌ Error al exportar: {e}")

def compute_n_syn_from_ratio(data, node_type, train_mask, minority_class, target_pos_ratio: float, max_new: Optional[int] = None) -> int:
    y_tr = data[node_type].y[train_mask]
    pos = int((y_tr == minority_class).sum().item())
    neg = int((y_tr != minority_class).sum().item())
    N   = pos + neg
    r   = float(target_pos_ratio)
    if N == 0 or r <= 0.0: return 0
    if r >= 0.999999: r = 0.999999
    n_syn = int(max(0, round((r * N - pos) / (1.0 - r))))
    if max_new is not None: n_syn = min(n_syn, int(max_new))
    return n_syn

def should_apply_smote(epoch: int, start_epoch: int, every_n_epochs: int) -> bool:
    if epoch < start_epoch: return False
    return ((epoch - start_epoch) % max(1, every_n_epochs)) == 0

class PMIndex:
    """Contenedor para mapeos de nodos PM."""
    def __init__(self, p_map, r_map):
        self._map = p_map
        self._rev = r_map
    def __len__(self):
        return len(self._rev)

def select_features():
    """
    Permite seleccionar un archivo de features y aplicar 
    diferentes métodos de selección para identificar las variables más informativas.
    """
    from src.features import compute_pm_features
    print("--- Selección de Caracteristicas ---")
    
    # 1. Seleccionar archivo de features
    csv_files = sorted(glob.glob(os.path.join("Resultados", "features_*.csv")))
    # Excluir archivos de nombres y cualquier features_{texto}; mantener solo features_{números}.csv
    csv_files = [
        f for f in csv_files
        if not os.path.basename(f).startswith("features_selected_names_")
        and re.match(r"^features_\d", os.path.basename(f))
    ]
    if not csv_files:
        print("No se encontraron archivos CSV de features en 'Resultados'.")
        return

    if len(csv_files) == 1:
        csv_path = csv_files[0]
        df = pd.read_csv(csv_path)
        print(f"✅ Cargado automáticamente: {os.path.basename(csv_path)}")
    else:
        print("Archivos de features disponibles:")
        for i, f in enumerate(csv_files, 1):
            print(f"  [{i}] {os.path.basename(f)}")
        try:
            sel = int(input("Seleccione el número de archivo CSV: ").strip()) - 1
            if not (0 <= sel < len(csv_files)):
                raise ValueError
            csv_path = csv_files[sel]
            df = pd.read_csv(csv_path)
            print(f"Cargado features desde -> {csv_path}")
        except (ValueError, IndexError):
            print("Selección inválida.")
            return

    # --- Generar la variable objetivo (y) ---
    print("Generando la variable objetivo (target) a partir de los datos de accidentes...")
    df_acc = load_accidentes()
    df_acc['ts_min'] = (df_acc["accidente_time"].dt.floor(f'{DT_MINUTES}min').astype("int64") // 60_000_000_000)
    portico_col = buscar_columna(df, "portico")

    # Crear un set de tuplas (pórtico, ts_min) para una búsqueda rápida
    accident_events = set(zip(df_acc['ultimo_portico'], df_acc['ts_min']))

    # Crear la columna target. 1 si la fila (pórtico, ts) corresponde a un accidente, 0 si no.
    df['target'] = df.apply(lambda row: 1 if (row[portico_col], row['ts_min']) in accident_events else 0, axis=1)

    if df['target'].sum() == 0:
        print("⚠️ Advertencia: No se encontró ninguna correspondencia entre los accidentes y las features del archivo CSV.")
        print("Asegúrese de que los rangos de fechas y los pórticos coincidan.")
        return

    print(f"  - Se encontraron {int(df['target'].sum())} eventos de accidente en el conjunto de datos.")

    # Separar features (X) y target (y)
    feature_cols = [c for c in df.columns if c not in [portico_col, "ts_min", "target"]]
    X = df[feature_cols]
    y = df['target']

    # --- Menú de Selección de Features ---
    while True:
        print("--- Métodos de Selección de Features ---")
        print("1. Filter: Umbral de Varianza")
        print("2. Filter: Correlación de Pearson")
        print("3. Wrapper: RFE (Recursive Feature Elimination)")
        print("4. Wrapper: SFS (Secuencial Feature Selection)")
        print("5. Combinado: Filters + Wrappers (Recomendado)")
        print("6. Filter: Importancia de Gini (Random Forest)")
        print("0. Salir")
        choice = input("Seleccione una opción: ").strip()

        selected_features = None
        if choice == '1':
            # Umbral de Varianza
            threshold = float(input("Ingrese el umbral de varianza (e.g., 0.01): "))
            selector = VarianceThreshold(threshold=threshold)
            selector.fit(X)
            selected_features = X.columns[selector.get_support()]
        elif choice == '2':
            # Correlación de Pearson
            threshold = float(input("Ingrese el umbral de correlación (e.g., 0.8): "))
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
            selected_features = X.columns.drop(to_drop)
        elif choice == '3':
            # RFE
            while True:
                try:
                    n_features_input = input(f"Ingrese el número de features a seleccionar (1 a {X.shape[1]}): ").strip()
                    if not n_features_input: continue
                    n_features = int(n_features_input)
                    if 1 <= n_features <= X.shape[1]:
                        break
                    print(f"❌ Por favor ingrese un número entre 1 y {X.shape[1]}.")
                except ValueError:
                    print("❌ Entrada inválida.")
            model = RandomForestClassifier(n_estimators=100, random_state=SEED)
            selector = RFE(estimator=model, n_features_to_select=n_features, step=1)
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()]
        elif choice == '4':
            # Selección Secuencial
            while True:
                try:
                    n_features_input = input(f"Ingrese el número de features a seleccionar (1 a {X.shape[1] - 1}): ").strip()
                    if not n_features_input: continue
                    n_features = int(n_features_input)
                    if 1 <= n_features < X.shape[1]:
                        break
                    print(f"❌ El número de features debe ser menor que el total ({X.shape[1]}).")
                except ValueError:
                    print("❌ Entrada inválida.")
            model = RandomForestClassifier(n_estimators=100, random_state=SEED)
            sfs = SequentialFeatureSelector(estimator=model, n_features_to_select=n_features, direction='forward', cv=2)
            sfs.fit(X, y)
            selected_features = X.columns[sfs.get_support()]
        elif choice == '5':
            # Combinado
            print("--- Ejecutando método combinado ---")
            # 1. Filtro de Varianza
            var_thresh = float(input("Paso 1: Ingrese el umbral de varianza (e.g., 0.01): "))
            var_selector = VarianceThreshold(threshold=var_thresh)
            var_selector.fit(X)
            
            variances = var_selector.variances_
            feature_variances = {name: var for name, var in zip(X.columns, variances)}
            removed_by_variance = [name for name, var in feature_variances.items() if var < var_thresh]

            X_var = X[X.columns[var_selector.get_support()]]
            print(f"  - Features después de filtro de varianza: {X_var.shape[1]}")
            if removed_by_variance:
                print("    - Features eliminadas por baja varianza:")
                for feat in removed_by_variance:
                    print(f"      - '{feat}' (Varianza: {feature_variances[feat]:.4f})")

            # 2. Filtro de Correlación
            corr_thresh = float(input("Paso 2: Ingrese el umbral de correlación (e.g., 0.8): "))
            corr_matrix = X_var.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > corr_thresh)]
            
            if to_drop:
                print("    - Features eliminadas por alta correlación:")
                for column in to_drop:
                    correlated_with = upper[column][upper[column] > corr_thresh].index.tolist()
                    for corr_col in correlated_with:
                        print(f"      - '{column}' (Correlación con '{corr_col}': {corr_matrix.loc[column, corr_col]:.4f})")

            X_filtered = X_var.drop(columns=to_drop)
            print(f"  - Features después de filtro de correlación: {X_filtered.shape[1]}")

            # 3. Wrappers
            while True:
                try:
                    n_features_input = input(f"Paso 3: Ingrese el número de features a seleccionar para RFE y SFS (1 a {X_filtered.shape[1] - 1}): ").strip()
                    if not n_features_input: continue
                    n_features = int(n_features_input)
                    if 1 <= n_features < X_filtered.shape[1]:
                        break
                    print(f"❌ El número de features debe ser menor que el total de features filtradas ({X_filtered.shape[1]}).")
                except ValueError:
                    print("❌ Entrada inválida.")
            model = RandomForestClassifier(n_estimators=20, random_state=SEED)
            
            # RFE
            print("  - Ejecutando RFE...")
            rfe_selector = RFE(estimator=model, n_features_to_select=n_features, step=1)
            rfe_selector.fit(X_filtered, y)
            rfe_features = X_filtered.columns[rfe_selector.get_support()]
            rfe_removed = X_filtered.columns.difference(rfe_features)
            print(f"    - Features seleccionadas por RFE: {len(rfe_features)}")
            if rfe_removed.any():
                print("    - Features eliminadas por RFE (menor importancia según el modelo):")
                print(f"      - {rfe_removed.tolist()}")

            # SFS
            print("  - Ejecutando SFS...")
            sfs_selector = SequentialFeatureSelector(estimator=model, n_features_to_select=n_features, direction='forward', cv=2)
            sfs_selector.fit(X_filtered, y)
            sfs_features = X_filtered.columns[sfs_selector.get_support()]
            sfs_removed = X_filtered.columns.difference(sfs_features)
            print(f"    - Features seleccionadas por SFS: {len(sfs_features)}")
            if sfs_removed.any():
                print("    - Features eliminadas por SFS (no aportaron mejora significativa):")
                print(f"      - {sfs_removed.tolist()}")

            # 4. Intersección
            selected_features = pd.Index(list(set(rfe_features) & set(sfs_features)))
            print(f"  - Features en la intersección de RFE y SFS: {len(selected_features)}")

        elif choice == '6':
            # Importancia de Gini
            print("--- Ejecutando selección por Importancia de Gini (Random Forest) ---")
            model = RandomForestClassifier(n_estimators=100, random_state=SEED)
            print("  - Entrenando Random Forest para calcular la importancia de las features...")
            model.fit(X, y)
            
            importances = model.feature_importances_
            feature_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
            
            print("  - Importancia de cada feature (Gini):")
            for name, imp in feature_importances.items():
                print(f"    - '{name}': {imp:.4f}")

            while True:
                try:
                    threshold = float(input("Ingrese el umbral de importancia para seleccionar features (e.g., 0.05): ").strip())
                    if threshold >= 0:
                        break
                    print("❌ El umbral debe ser un número positivo.")
                except ValueError:
                    print("❌ Entrada inválida.")

            selected_features = feature_importances[feature_importances >= threshold].index
            print(f"  - Features seleccionadas con importancia >= {threshold}: {len(selected_features)}")

        elif choice == '0':
            break
        else:
            print("Opción no válida.")
            continue

        if selected_features is not None:
            print(f"Features seleccionadas ({len(selected_features)}):{selected_features.tolist()}")
            
            # Guardar resultados
            save = input("¿Desea guardar los NOMBRES de estas features en un nuevo archivo CSV? (s/n): ").strip().lower()
            if save in ('s', 'si', 'y', 'yes'):
                output_df = pd.DataFrame({'feature_name': selected_features})
                timestamp = datetime.now().strftime('%d%m%Y_%H%M')
                filename = f"features_selected_names_{timestamp}.csv"
                output_path = os.path.join(RESULTADOS_DIR, filename)
                output_df.to_csv(output_path, index=False)
                print(f"Archivo con nombres de features guardado en → {output_path}")

def fusionar_grafos():
    """
    Permite seleccionar múltiples archivos de grafos .pt, los fusiona
    en un único grafo y lo guarda. Durante la fusión, recalcula las
    estadísticas de normalización (mu y sigma) y las máscaras de datos
    para todo el conjunto de datos.
    """
    print("▶️ Iniciando fusión de grafos...")
    
    # 1. Listar y seleccionar archivos de grafos
    pattern = os.path.join(RESULTADOS_DIR, "highway_graph_*.pt")
    pt_files = sorted(glob.glob(pattern))
    
    if len(pt_files) < 2:
        print(f"❌ Se necesitan al menos 2 grafos para fusionar. Se encontraron {len(pt_files)} en '{RESULTADOS_DIR}'.")
        return

    print("Archivos de grafo disponibles para fusionar:")
    for i, f in enumerate(pt_files, 1):
        print(f"  [{i}] {os.path.basename(f)}")
    
    try:
        selections = input("Selecciona los números de los archivos a fusionar, separados por comas (ej: 1,3,4): ").strip()
        if not selections:
            print("Fusión cancelada.")
            return
        indices = [int(s.strip()) - 1 for s in selections.split(',')]
        if not all(0 <= i < len(pt_files) for i in indices) or len(indices) < 2:
            raise ValueError("Selección inválida.")
        selected_files = [pt_files[i] for i in indices]
    except (ValueError, IndexError) as e:
        print(f"❌ Error en la selección: {e}")
        return

    # 2. Cargar los grafos seleccionados
    loaded_objs = []
    print("Cargando grafos...")
    for pt_path in selected_files:
        try:
            obj = torch.load(pt_path, map_location=torch.device('cpu'), weights_only=False)
            if 'data' not in obj or not isinstance(obj['data'], HeteroData):
                print(f"⚠️  Advertencia: El archivo '{os.path.basename(pt_path)}' no es un grafo válido y será omitido.")
                continue
            loaded_objs.append(obj)
            print(f"  - Cargado: {os.path.basename(pt_path)}")
        except Exception as e:
            print(f"⚠️  Error al cargar '{os.path.basename(pt_path)}': {e}. Será omitido.")

    if len(loaded_objs) < 2:
        print("❌ No se pudieron cargar suficientes grafos para la fusión.")
        return

    # 3. Fusionar los grafos y recalcular estadísticas
    print("Fusionando grafos y recalculando estadísticas de normalización...")
    merged_data = HeteroData()

    # 3.1. Unificar columnas de features de todos los grafos
    all_feature_cols = set()
    for obj in loaded_objs:
        if 'feature_cols' in obj:
            all_feature_cols.update(obj['feature_cols'])
    
    unified_feature_list = sorted(list(all_feature_cols))
    unified_feature_map = {feat: i for i, feat in enumerate(unified_feature_list)}
    num_unified_features = len(unified_feature_list)
    print(f"Unified to {num_unified_features} features.")

    # 3.2. Desnormalizar, alinear y concatenar features de nodos 'pm'
    processed_pm_x = []
    for obj in loaded_objs:
        if 'pm' in obj['data'].node_types and 'x' in obj['data']['pm']:
            # Desnormalizar
            mu_val = obj.get('mu', 0)
            sigma_val = obj.get('sigma', 1)
            mu_tensor = torch.tensor(mu_val, dtype=torch.float)
            sigma_tensor = torch.tensor(sigma_val, dtype=torch.float)
            x_unnorm = obj['data']['pm'].x * sigma_tensor + mu_tensor

            # Alinear a features unificadas
            current_feature_cols = obj.get('feature_cols', [])
            num_nodes = x_unnorm.shape[0]
            new_x = torch.zeros((num_nodes, num_unified_features), dtype=torch.float)

            for i, feat_name in enumerate(current_feature_cols):
                if feat_name in unified_feature_map:
                    target_idx = unified_feature_map[feat_name]
                    new_x[:, target_idx] = x_unnorm[:, i]
            
            processed_pm_x.append(new_x)

    # 3.3. Recalcular mu y sigma globales y re-normalizar
    if processed_pm_x:
        full_unnormalized_x = torch.cat(processed_pm_x, dim=0)
        mu = full_unnormalized_x.mean(dim=0).numpy()
        sigma = full_unnormalized_x.std(dim=0).numpy() + 1e-9
        merged_data['pm'].x = (full_unnormalized_x - torch.tensor(mu)) / torch.tensor(sigma)
        print("✅ Nuevas estadísticas (mu, sigma) calculadas y aplicadas.")
    
    # 3.4. Fusionar otros atributos de nodos ('y', etc.)
    all_node_types = set(nt for obj in loaded_objs for nt in obj['data'].node_types)
    for node_type in all_node_types:
        attrs_to_merge = ['y']  # No concatenar máscaras viejas
        if node_type != 'pm':
            attrs_to_merge.append('x')

        for attr in attrs_to_merge:
            attr_list = [obj['data'][node_type][attr] for obj in loaded_objs if node_type in obj['data'].node_types and attr in obj['data'][node_type]]
            if attr_list:
                merged_data[node_type][attr] = torch.cat(attr_list, dim=0)

    # 4. Fusionar Aristas
    all_edge_types = set(et for obj in loaded_objs for et in obj['data'].edge_types)
    node_offsets = {nt: [0] + [obj['data'][nt].num_nodes for obj in loaded_objs] for nt in all_node_types}
    node_offsets_cumulative = {nt: torch.tensor(node_offsets[nt]).cumsum(dim=0) for nt in all_node_types}

    for edge_type in all_edge_types:
        src_type, _, dst_type = edge_type
        edge_indices = []
        edge_attrs = []
        for i, obj in enumerate(loaded_objs):
            if edge_type in obj['data'].edge_types:
                edge_index = obj['data'][edge_type].edge_index
                offset = torch.tensor([[node_offsets_cumulative[src_type][i]], [node_offsets_cumulative[dst_type][i]]])
                edge_indices.append(edge_index + offset)
                if 'edge_attr' in obj['data'][edge_type]:
                    edge_attrs.append(obj['data'][edge_type].edge_attr)
        
        if edge_indices:
            merged_data[edge_type].edge_index = torch.cat(edge_indices, dim=1)
        if edge_attrs:
            # Alinear edge_attr también
            aligned_edge_attrs = []
            for i, obj in enumerate(loaded_objs):
                 if edge_type in obj['data'].edge_types and 'edge_attr' in obj['data'][edge_type]:
                    ea_unnorm = obj['data'][edge_type].edge_attr # Asumimos que no está normalizado
                    current_feature_cols = obj.get('feature_cols', [])
                    num_edges = ea_unnorm.shape[0]
                    new_ea = torch.zeros((num_edges, num_unified_features), dtype=torch.float)
                    for j, feat_name in enumerate(current_feature_cols):
                        if feat_name in unified_feature_map:
                            target_idx = unified_feature_map[feat_name]
                            new_ea[:, target_idx] = ea_unnorm[:, j]
                    aligned_edge_attrs.append(new_ea)
            if aligned_edge_attrs:
                merged_data[edge_type].edge_attr = torch.cat(aligned_edge_attrs, dim=0)

    # 5. Reconstruir PMIndex
    print("▶️ Reconstruyendo PMIndex...")
    all_rev_tuples = []
    node_idx_counter = 0
    for obj in loaded_objs:
        if 'pm_index' in obj and hasattr(obj['pm_index'], '_rev'):
            # La clave del rev map es el índice local, el valor es (portico, ts)
            rev_map = obj['pm_index']._rev
            # Asegurarse de que el rev_map es un diccionario para acceder por clave
            if isinstance(rev_map, list):
                rev_map = {i: v for i, v in enumerate(rev_map)}
            
            num_nodes_in_graph = obj['data']['pm'].num_nodes
            for local_idx in range(num_nodes_in_graph):
                all_rev_tuples.append(rev_map.get(local_idx))
                node_idx_counter += 1

    if all_rev_tuples:
        # Crear el mapeo inverso (tupla -> índice global)
        merged_map = {val: i for i, val in enumerate(all_rev_tuples) if val is not None}
        # El mapeo directo es simplemente la lista de tuplas, donde el índice es el node_id
        merged_pm_index = PMIndex(p_map=merged_map, r_map=all_rev_tuples)
        print("✅ PMIndex reconstruido.")
    else:
        merged_pm_index = PMIndex(p_map={}, r_map=[])
        print("⚠️ No se encontraron datos de PMIndex para fusionar. Se ha creado un índice vacío.")

    # 6. Recalcular máscaras de datos (División Estratificada)
    print("▶️ Recalculando máscaras de entrenamiento/validación/prueba (división estratificada por clase)...")
    num_pm_nodes = merged_data['pm'].num_nodes
    if num_pm_nodes > 0 and 'y' in merged_data['pm']:
        y = merged_data['pm'].y
        pos_indices = torch.where(y == 1)[0]
        neg_indices = torch.where(y == 0)[0]

        # Shuffle indices
        pos_indices = pos_indices[torch.randperm(len(pos_indices))]
        neg_indices = neg_indices[torch.randperm(len(neg_indices))]

        num_pos = len(pos_indices)
        num_neg = len(neg_indices)

        if num_pos < 3:
            print("⚠️ Advertencia: No hay suficientes nodos de clase 1 (<3) para una división train/val/test. Se asignará todo a entrenamiento.")
            train_indices = torch.cat([pos_indices, neg_indices])
            val_indices = torch.tensor([], dtype=torch.long)
            test_indices = torch.tensor([], dtype=torch.long)
        else:
            # Positive class split
            train_pos_end = int(num_pos * TRAIN_RATIO / 100)
            val_pos_end = train_pos_end + int(num_pos * VAL_RATIO / 100)
            train_pos = pos_indices[:train_pos_end]
            val_pos = pos_indices[train_pos_end:val_pos_end]
            test_pos = pos_indices[val_pos_end:]

            # Negative class split
            train_neg_end = int(num_neg * TRAIN_RATIO / 100)
            val_neg_end = train_neg_end + int(num_neg * VAL_RATIO / 100)
            train_neg = neg_indices[:train_neg_end]
            val_neg = neg_indices[train_neg_end:val_neg_end]
            test_neg = neg_indices[val_neg_end:]

            # Combine
            train_indices = torch.cat([train_pos, train_neg])
            val_indices = torch.cat([val_pos, val_neg])
            test_indices = torch.cat([test_pos, test_neg])

        # Create masks
        train_mask = torch.zeros(num_pm_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_pm_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_pm_nodes, dtype=torch.bool)
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True

        merged_data['pm'].train_mask = train_mask
        merged_data['pm'].val_mask = val_mask
        merged_data['pm'].test_mask = test_mask

        print(f"✅ Máscaras recalculadas: train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")

        # Propagar máscaras a nodos 'acc' si existen
        if 'acc' in merged_data.node_types and ('pm', 'triggers', 'acc') in merged_data.edge_types:
            num_acc_nodes = merged_data['acc'].num_nodes
            acc_train_mask = torch.zeros(num_acc_nodes, dtype=torch.bool)
            acc_val_mask = torch.zeros(num_acc_nodes, dtype=torch.bool)
            acc_test_mask = torch.zeros(num_acc_nodes, dtype=torch.bool)

            trigger_src = merged_data[('pm', 'triggers', 'acc')].edge_index[0]
            trigger_dst = merged_data[('pm', 'triggers', 'acc')].edge_index[1]

            for pm_idx, acc_idx in zip(trigger_src, trigger_dst):
                if train_mask[pm_idx]:
                    acc_train_mask[acc_idx] = True
                elif val_mask[pm_idx]:
                    acc_val_mask[acc_idx] = True
                else:
                    acc_test_mask[acc_idx] = True
            
            merged_data['acc'].train_mask = acc_train_mask
            merged_data['acc'].val_mask = acc_val_mask
            merged_data['acc'].test_mask = acc_test_mask
            print(f"  - Nodos 'acc' divididos: {acc_train_mask.sum()} (train), {acc_val_mask.sum()} (val), {acc_test_mask.sum()} (test)")

    else:
        print("⚠️ No hay nodos 'pm' o falta la etiqueta 'y' para recalcular las máscaras.")

    # 7. Guardar
    timestamp = datetime.now().strftime('%d%m%Y_%H%M')
    filename = f"highway_graph_FUSED_{timestamp}.pt"
    output_path = os.path.join(RESULTADOS_DIR, filename)

    torch.save({
        "data": merged_data,
        "pm_index": merged_pm_index,
        "mu": mu,
        "sigma": sigma,
        "feature_cols": unified_feature_list
    }, output_path)

    print(f"✅ Fusión completada. Grafo guardado en -> {filename}")
    print(f"{merged_data}")

def export_graph_to_csv(loaded_obj):
    """
    Exporta los datos de un grafo a un archivo CSV para fácil visualización.
    Incluye features de nodos, máscaras de datos y variables globales.
    """
    data = loaded_obj["data"]
    feature_cols = loaded_obj.get("feature_cols", [])
    mu = loaded_obj.get("mu")
    sigma = loaded_obj.get("sigma")
    pm_index = loaded_obj.get("pm_index")

    all_dfs = []

    # Procesar cada tipo de nodo
    for node_type in data.node_types:
        node_data = data[node_type]
        num_nodes = node_data.num_nodes
        
        if num_nodes == 0:
            continue

        records = []
        for i in range(num_nodes):
            record = {'node_type': node_type, 'node_id': i}
            
            # Añadir etiqueta si existe
            if 'y' in node_data:
                record['label'] = node_data.y[i].item()

            # Añadir máscaras si existen
            for mask in ['train_mask', 'val_mask', 'test_mask', 'is_synthetic', 'is_accident_pm']:
                if mask in node_data:
                    if DEBUG:
                        print(f"DEBUG: node_type={node_type}, mask={mask}, i={i}, node_data[mask].shape={node_data[mask].shape}")
                    record[mask] = node_data[mask][i].item()

            # Añadir información específica de PM
            if node_type == 'pm' and pm_index and hasattr(pm_index, '_rev') and i < len(pm_index._rev):
                try:
                    rev_info = pm_index._rev[i]
                    if rev_info:
                        portico, ts_min = rev_info
                        if portico is not None and ts_min is not None:
                            dt = datetime.fromtimestamp(ts_min * 60)
                            record["portico"] = portico
                            record["timestamp"] = dt.strftime('%Y-%m-%d %H:%M:%S')
                except (IndexError, TypeError, ValueError):
                    pass

            records.append(record)
        
        df_nodes = pd.DataFrame(records)

        # Añadir features
        if 'x' in node_data and hasattr(node_data.x, 'cpu'):
            features = node_data.x.cpu().detach().numpy()
            
            current_feature_cols = []
            if node_type == 'pm' and feature_cols:
                num_feats_expected = len(feature_cols)
                num_feats_actual = features.shape[1]
                if num_feats_expected == num_feats_actual:
                    current_feature_cols = feature_cols
                else:
                    print(f"⚠️  Advertencia: El número de `feature_cols` ({num_feats_expected}) no coincide con las features en el grafo ({num_feats_actual}) para nodos '{node_type}'. Se usarán nombres genéricos.")
                    current_feature_cols = [f'feature_{j}' for j in range(num_feats_actual)]
            else:
                current_feature_cols = [f'feature_{j}' for j in range(features.shape[1])]

            df_features = pd.DataFrame(features, columns=current_feature_cols)
            df_nodes = pd.concat([df_nodes, df_features], axis=1)

        all_dfs.append(df_nodes)

    # Combinar todos los dataframes de nodos
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        
        cols = ['node_type', 'node_id', 'portico', 'timestamp', 'label', 'train_mask', 'val_mask', 'test_mask', 'is_synthetic', 'is_accident_pm']
        existing_cols = [c for c in cols if c in final_df.columns]
        feature_name_cols = [c for c in final_df.columns if c not in existing_cols]
        final_df = final_df[existing_cols + feature_name_cols]

        timestamp = datetime.now().strftime('%d%m%Y_%H%M')
        filename = f"graph_export_nodes_{timestamp}.csv"
        output_path = os.path.join(RESULTADOS_DIR, filename)
        final_df.to_csv(output_path, index=False, sep=';', decimal=',')
        print(f"✅ Datos de nodos exportados → {output_path}")

    # Exportar aristas
    edge_dfs = []
    for edge_type in data.edge_types:
        edge_data = data[edge_type]
        if edge_data.num_edges == 0:
            continue
        
        src, rel, dst = edge_type
        edge_index = edge_data.edge_index.cpu().numpy()
        
        df_edge = pd.DataFrame({
            'source_type': src,
            'relation_type': rel,
            'destination_type': dst,
            'source_id': edge_index[0],
            'destination_id': edge_index[1]
        })
        
        if DEBUG:
            print(f"DEBUG: Edge type {edge_type}: 'edge_attr' exists: {'edge_attr' in edge_data}. Shape: {edge_data.edge_attr.shape if 'edge_attr' in edge_data else 'N/A'}")
        
        if 'edge_attr' in edge_data:
            edge_features = edge_data.edge_attr.cpu().numpy()
            df_edge_features = pd.DataFrame(edge_features, columns=[f'edge_attr_{j}' for j in range(edge_features.shape[1])])
            df_edge = pd.concat([df_edge, df_edge_features], axis=1)
            
        edge_dfs.append(df_edge)

    if edge_dfs:
        final_edges_df = pd.concat(edge_dfs, ignore_index=True)
        timestamp = datetime.now().strftime('%d%m%Y_%H%M')
        filename_edges = f"graph_export_edges_{timestamp}.csv"
        output_path_edges = os.path.join(RESULTADOS_DIR, filename_edges)
        final_edges_df.to_csv(output_path_edges, index=False, sep=';', decimal=',')
        print(f"✅ Datos de aristas exportados → {output_path_edges}")

    # Exportar estadísticas globales si existen
    if mu is not None and sigma is not None and feature_cols:
        try:
            stats_df = pd.DataFrame({
                'feature': feature_cols,
                'mu': mu.flatten(),
                'sigma': sigma.flatten()
            })
            timestamp = datetime.now().strftime('%d%m%Y_%H%M')
            filename_stats = f"graph_export_stats_{timestamp}.csv"
            output_path_stats = os.path.join(RESULTADOS_DIR, filename_stats)
            stats_df.to_csv(output_path_stats, index=False, sep=';', decimal=',')
            print(f"✅ Estadísticas de normalización exportadas → {output_path_stats}")
        except Exception as e:
            print(f"⚠️ No se pudieron exportar las estadísticas de normalización: {e}")

# =============================================================================
# GESTIÓN BASE DE DATOS DE FLUJOS (DUCKDB) - PORTED FROM SRC_SUMO
# =============================================================================
from dataclasses import dataclass
try:
    import duckdb
except ImportError:
    duckdb = None

DEFAULT_DUCKDB_FILE = Path("Datos") / "flujos.duckdb"
FLOW_TABLE_NAME = "flujos_duckdb"
FLOW_TABLE_SCHEMA = """
    FECHA TIMESTAMP,
    VELOCIDAD DOUBLE,
    CATEGORIA INTEGER,
    MATRICULA VARCHAR,
    PORTICO VARCHAR,
    CARRIL VARCHAR
"""

@dataclass(frozen=True)
class FlowColumns:
    timestamp: str = "FECHA"
    speed_kmh: str = "VELOCIDAD"
    class_id: str = "CATEGORIA"
    plate_id: str = "MATRICULA"
    portico: str = "PORTICO"
    lane: str = "CARRIL"

@dataclass(frozen=True)
class FlowDBSummary:
    row_count: int
    min_timestamp: Optional[pd.Timestamp]
    max_timestamp: Optional[pd.Timestamp]
    db_path: Path

@dataclass(frozen=True)
class FlowSampleSelection:
    date_start: Optional[pd.Timestamp] = None
    date_end: Optional[pd.Timestamp] = None
    row_limit: Optional[int] = None

def choose_file_filtered(start_folder: str, patterns) -> Optional[str]:
    """
    Igual a choose_file, pero solo muestra archivos CSV que apliquen a los patrones glob.
    """
    if isinstance(patterns, str):
        patterns = [patterns]
    patterns_l = [p.lower() for p in patterns]

    start_folder = os.path.abspath(start_folder)
    current_folder = start_folder

    while True:
        print(f"\n📁 Carpeta actual: {current_folder}")
        items = os.listdir(current_folder)
        dirs = sorted([f for f in items if os.path.isdir(os.path.join(current_folder, f))])
        files = []
        for f in items:
            fp = os.path.join(current_folder, f)
            if os.path.isfile(fp) and f.lower().endswith(".csv"):
                name_l = f.lower()
                if any(fnmatch.fnmatch(name_l, p) for p in patterns_l):
                    files.append(f)
        files = sorted(files)
        opciones = dirs + files

        if current_folder == start_folder:
            print("  [0] 🔙 Volver/Salir")
            offset = 1
        else:
            print("  [0] 🔙 Subir un nivel")
            offset = 1

        if not files and not dirs:
            print("  (No hay carpetas ni archivos que coincidan con el patrón.)")

        print(f"  ↳ Mostrando solo archivos: {', '.join(patterns)}")
        for i, name in enumerate(opciones):
            icon = "📂" if name in dirs else "📄"
            print(f"  [{i+offset}] {icon} {name}")

        choice = input("Seleccione una opción: ").strip()
        if choice == "0":
            if current_folder == start_folder:
                print("Cancelado.")
                return None
            current_folder = os.path.dirname(current_folder)
            continue

        try:
            idx = int(choice) - offset
            if 0 <= idx < len(opciones):
                selected = opciones[idx]
                selected_path = os.path.join(current_folder, selected)
                if os.path.isdir(selected_path):
                    current_folder = selected_path
                else:
                    print(f"Archivo seleccionado: {selected_path}\n")
                    return selected_path
            else:
                print("Selección fuera de rango.")
        except ValueError:
            print("Entrada inválida.")

def select_flujos_csv_path() -> Optional[str]:
    base_dir = "Datos"
    patterns = ["flujo*.csv"]
    try:
        top_files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
    except FileNotFoundError:
        top_files = []
    matches_top = [
        os.path.join(base_dir, f)
        for f in top_files
        if f.lower().endswith(".csv") and any(fnmatch.fnmatch(f.lower(), p) for p in patterns)
    ]
    if len(matches_top) == 1:
        file_path = matches_top[0]
        print(f"🔎 Detectado único archivo de flujos en '{base_dir}'. Cargando automáticamente → {file_path}")
        return file_path
    return choose_file_filtered(base_dir, patterns)

def _ensure_duckdb_available() -> None:
    if duckdb is None:
        raise ImportError("duckdb no está instalado. Ejecute `pip install duckdb`.")

def _get_duckdb_path(db_path: Optional[Path] = None) -> Path:
    path = Path(db_path) if db_path is not None else DEFAULT_DUCKDB_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def _connect_duckdb(read_only: bool = False, db_path: Optional[Path] = None):
    _ensure_duckdb_available()
    path = _get_duckdb_path(db_path)
    ro_flag = read_only and path.exists()
    try:
        return duckdb.connect(str(path), read_only=ro_flag)
    except duckdb.ConnectionException as exc:
        if ro_flag and "different configuration" in str(exc):
            return duckdb.connect(str(path), read_only=False)
        raise

def _create_flow_table(conn) -> None:
    conn.execute(f"CREATE TABLE IF NOT EXISTS {FLOW_TABLE_NAME} ({FLOW_TABLE_SCHEMA})")

def _flow_date_parse_expr(column: str = "fecha_txt") -> str:
    return (
        "COALESCE("
        f"try_strptime({column}, '%d/%m/%Y %H:%M:%S'),"
        f"try_strptime({column}, '%d-%m-%Y %H:%M:%S'),"
        f"try_strptime({column}, '%Y-%m-%d %H:%M:%S'),"
        f"try_strptime({column}, '%Y/%m/%d %H:%M:%S'),"
        f"try_strptime({column}, '%Y-%m-%dT%H:%M:%SZ'),"
        f"try_strptime({column}, '%Y-%m-%dT%H:%M:%S.%fZ'),"
        f"try_strptime({column}, '%Y-%m-%dT%H:%M:%S%z'),"
        f"try_strptime({column}, '%d/%m/%Y %H:%M'),"
        f"try_strptime({column}, '%Y-%m-%d %H:%M')"
        ")"
    )

def _flow_csv_select_sql() -> str:
    fecha_expr = _flow_date_parse_expr()
    return f"""
        WITH raw AS (
            SELECT *
            FROM read_csv_auto(?,
                SAMPLE_SIZE=-1,
                HEADER=TRUE,
                IGNORE_ERRORS=TRUE,
                ALL_VARCHAR=TRUE
            )
        ),
        normalized AS (
            SELECT
                raw.*,
                NULLIF(TRIM(raw.FECHA), '') AS fecha_txt,
                NULLIF(TRIM(raw.MATRICULA), '') AS matricula_txt,
                NULLIF(TRIM(raw.PORTICO), '') AS portico_txt,
                NULLIF(TRIM(raw.CARRIL), '') AS carril_txt,
                TRY_CAST(NULLIF(TRIM(raw.CATEGORIA), '') AS INTEGER) AS categoria_num,
                TRY_CAST(REPLACE(NULLIF(TRIM(raw.VELOCIDAD), ''), ',', '.') AS DOUBLE) AS velocidad_num
            FROM raw
        ),
        prepared AS (
            SELECT
                {fecha_expr} AS FECHA,
                velocidad_num AS VELOCIDAD,
                categoria_num AS CATEGORIA,
                matricula_txt AS MATRICULA,
                portico_txt AS PORTICO,
                carril_txt AS CARRIL
            FROM normalized
            WHERE categoria_num IS NOT NULL
              AND {fecha_expr} IS NOT NULL
        )
        SELECT * FROM prepared
    """

def import_flujos_to_duckdb(
    csv_path: Optional[str] = None,
    *,
    replace: bool = False,
    db_path: Optional[Path] = None,
    progress_callback: Optional[object] = None,
) -> int:
    csv_path = csv_path or select_flujos_csv_path()
    if csv_path is None:
        return 0
    csv_path = str(Path(csv_path).resolve())

    def _report(ratio: float, message: str) -> None:
        if progress_callback is None:
            return
        ratio_clamped = min(max(ratio, 0.0), 1.0)
        try:
            progress_callback(ratio_clamped, message)
        except Exception:
            pass
    def _stage(message: str, value: float) -> None:
        _report(value, message)

    conn = _connect_duckdb(read_only=False, db_path=db_path)
    select_sql = _flow_csv_select_sql()
    inserted = 0
    success = False
    try:
        _stage("Iniciando importación", 0.0)
        if replace:
            _stage("Reemplazando tabla", 0.1)
            conn.execute(f"DROP TABLE IF EXISTS {FLOW_TABLE_NAME}")
            conn.execute(f"CREATE TABLE {FLOW_TABLE_NAME} AS {select_sql}", [csv_path])
            inserted = conn.execute(f"SELECT COUNT(*) FROM {FLOW_TABLE_NAME}").fetchone()[0]
        else:
            _stage("Anexando datos", 0.1)
            _create_flow_table(conn)
            conn.execute(f"CREATE OR REPLACE TEMP TABLE __new_flujos AS {select_sql}", [csv_path])
            inserted = conn.execute("SELECT COUNT(*) FROM __new_flujos").fetchone()[0]
            if inserted:
                conn.execute(f"INSERT INTO {FLOW_TABLE_NAME} SELECT * FROM __new_flujos")
            conn.execute("DROP TABLE __new_flujos")
        success = True
        _stage("Completado", 1.0)
        return int(inserted or 0)
    finally:
        conn.close()

def clear_flow_table(db_path: Optional[Path] = None) -> int:
    conn = _connect_duckdb(read_only=False, db_path=db_path)
    try:
        _create_flow_table(conn)
        existing = conn.execute(f"SELECT COUNT(*) FROM {FLOW_TABLE_NAME}").fetchone()[0]
        conn.execute(f"DELETE FROM {FLOW_TABLE_NAME}")
        return int(existing or 0)
    finally:
        conn.close()

def get_flow_db_summary(db_path: Optional[Path] = None) -> FlowDBSummary:
    conn = _connect_duckdb(read_only=True, db_path=db_path)
    try:
        try:
             row = conn.execute(f"SELECT COUNT(*), MIN(FECHA), MAX(FECHA) FROM {FLOW_TABLE_NAME}").fetchone()
        except:
             return FlowDBSummary(0, None, None, _get_duckdb_path(db_path))
        
        min_ts = pd.to_datetime(row[1]) if row[1] else None
        max_ts = pd.to_datetime(row[2]) if row[2] else None
        return FlowDBSummary(int(row[0] or 0), min_ts, max_ts, _get_duckdb_path(db_path))
    finally:
        conn.close()

def get_flow_table_columns(db_path: Optional[Path] = None) -> List[dict]:
    conn = _connect_duckdb(read_only=False, db_path=db_path)
    try:
        _create_flow_table(conn)
        info = conn.execute(f"PRAGMA table_info('{FLOW_TABLE_NAME}')").fetchall()
    finally:
        conn.close()
    cols = []
    for row in info:
        cols.append({"cid": row[0], "name": row[1], "type": row[2]})
    return cols

def get_flow_table_sample(limit: int = 5, db_path: Optional[Path] = None) -> pd.DataFrame:
    conn = _connect_duckdb(read_only=True, db_path=db_path)
    try:
        return conn.execute(f"SELECT * FROM {FLOW_TABLE_NAME} LIMIT ?", [limit]).df()
    finally:
        conn.close()
