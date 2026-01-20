import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
import time
from typing import Callable

# --- Streamlit context-safe cache decorator ---
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx as _get_ctx
    _HAS_ST_CTX = _get_ctx() is not None
except Exception:
    _HAS_ST_CTX = False

def _identity_decorator(func: Callable = None, **kwargs):
    # No-op decorator to mimic @st.cache_data outside Streamlit context
    if func is None:
        def _wrap(f):
            return f
        return _wrap
    return func

cache_data = st.cache_data if _HAS_ST_CTX else _identity_decorator

# --- Funciones de Lógica de Dominio (Copiadas/Adaptadas de utils.py) ---

def find_candidate_porticos(acc_km, porticos_df, eje, calzada):
    try:
        accident_km = float(str(acc_km).replace(",", "."))
    except (ValueError, AttributeError):
        return {"posterior": None, "cercano": None}

    df = porticos_df.copy()
    df["Eje_norm"] = df["eje"].astype(str).str.strip().str.upper()
    df["Calzada_norm"] = df["calzada"].astype(str).str.strip().str.upper()
    eje_filtro = str(eje).strip().upper()
    calzada_filtro = str(calzada).strip().upper()
    df_filtrado = df[(df["Eje_norm"] == eje_filtro) & (df["Calzada_norm"] == calzada_filtro)].copy()
    
    if df_filtrado.empty:
        return {"posterior": None, "cercano": None}
    
    df_filtrado["km_num"] = pd.to_numeric(df_filtrado["km"].astype(str).str.replace(",", "."), errors='coerce')
    df_filtrado.dropna(subset=["km_num"], inplace=True)
    df_filtrado = df_filtrado.sort_values(by="orden", ascending=True).reset_index(drop=True)
    
    if len(df_filtrado) < 2:
        return {"posterior": None, "cercano": df_filtrado.iloc[0].to_dict() if not df_filtrado.empty else None}

    direction = "asc" if df_filtrado.loc[0, "km_num"] <= df_filtrado.loc[1, "km_num"] else "desc"
    
    for i in range(len(df_filtrado) - 1):
        km_current = df_filtrado.loc[i, "km_num"]
        km_next = df_filtrado.loc[i+1, "km_num"]
        if (direction == "asc" and km_current <= accident_km <= km_next) or \
           (direction == "desc" and km_current >= accident_km >= km_next):
            return {"posterior": df_filtrado.loc[i].to_dict(), "cercano": df_filtrado.loc[i+1].to_dict()}
    
    return {"posterior": None, "cercano": None} # No se encontró en ningún tramo intermedio

@cache_data
def load_gantry_data(file_path):
    if not file_path or not os.path.exists(file_path):
        return None
    try:
        df = pd.read_csv(file_path, sep=';', encoding='utf-8')
    except Exception:
        df = pd.read_csv(file_path, sep=';', encoding='latin-1')
    
    # Adaptado de buscar_columna y load_porticos
    col_map = {
        next(c for c in df.columns if c.lower().strip() == 'cod_portico'): "portico",
        next(c for c in df.columns if c.lower().strip() == 'km'): "km",
        next(c for c in df.columns if c.lower().strip() == 'calzada'): "calzada",
        next(c for c in df.columns if c.lower().strip() == 'orden'): "orden",
        next(c for c in df.columns if c.lower().strip() == 'eje'): "eje",
    }
    df_porticos = df[list(col_map.keys())].rename(columns=col_map)
    return df_porticos

@cache_data
def load_accident_data(file_path, gantry_df):
    if not file_path or gantry_df is None or not os.path.exists(file_path):
        return pd.DataFrame()
    try:
        # Especificar el separador punto y coma (;)
        df = pd.read_csv(file_path, sep=';', encoding='utf-8')
    except Exception:
        df = pd.read_csv(file_path, sep=';', encoding='latin-1')

    # Lógica adaptada de load_accidentes
    df.columns = [c.lower().strip() for c in df.columns]
    df = df[df['tipo'].str.lower() == "accidente"].copy()
    # Parseo robusto de timestamp (evita warning de inferencia de formato)
    s_start = df['fechas inicio'].astype(str).str.strip() + " " + df['hora inicio'].astype(str).str.strip()
    try:
        df['timestamp'] = pd.to_datetime(s_start, errors='coerce', dayfirst=True, format='mixed')
    except TypeError:
        df['timestamp'] = pd.to_datetime(s_start, errors='coerce', dayfirst=True)
    df.dropna(subset=['timestamp', 'km.', 'eje', 'calzada'], inplace=True)

    def get_last_gantry(row):
        candidates = find_candidate_porticos(row['km.'], gantry_df, row['eje'], row['calzada'])
        if candidates and candidates['posterior']:
            return candidates['posterior']['portico']
        return None

    df['portico'] = df.apply(get_last_gantry, axis=1)
    return df.dropna(subset=['portico'])

# --- Interfaz Principal de Streamlit ---

def run_visualization():
    st.set_page_config(layout="wide")
    st.sidebar.header("Controles")

    # 1. Selección de archivos
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    # Selector para Features
    features_path = os.path.join(project_root, 'Resultados')
    all_feature_files = glob.glob(os.path.join(features_path, 'features_*.csv'))
    # Solo archivos features_{números}.csv (excluir selected_names y features_{texto})
    import re as _re
    feature_files = [
        f for f in all_feature_files
        if 'selected_names' not in os.path.basename(f)
        and _re.match(r'^features_\d', os.path.basename(f))
        and 'subset' not in os.path.basename(f)
    ]
    if not feature_files:
        st.error("No se encontraron archivos de features (`features_*.csv`) en `Resultados/`.")
        st.stop()
    selected_feature_file = st.sidebar.selectbox(
        "Archivo de Features:",
        sorted(feature_files, key=os.path.basename, reverse=True),
        format_func=os.path.basename
    )

    # Selector para archivo de nombres de features seleccionadas (opcional)
    selected_names_files = sorted(
        glob.glob(os.path.join(features_path, 'features_selected_names_*.csv')),
        key=os.path.basename,
        reverse=True
    )
    names_file_options = ["No aplicar filtro de variables"] + selected_names_files
    selected_names_file = st.sidebar.selectbox(
        "Filtro de variables:",
        names_file_options,
        format_func=lambda x: os.path.basename(x) if isinstance(x, str) and os.path.exists(x) else str(x)
    )

    # Selector para Pórticos
    datos_path = os.path.join(project_root, 'Datos')
    portico_files = glob.glob(os.path.join(datos_path, '*orticos.csv')) # Flexible a Porticos o porticos
    selected_portico_file = st.sidebar.selectbox(
        "Archivo de Pórticos:",
        sorted(portico_files, key=os.path.basename, reverse=True),
        format_func=os.path.basename
    )

    # Selector para Accidentes (Opcional)
    event_files = glob.glob(os.path.join(datos_path, '*ventos*.csv'))
    event_files.insert(0, "No mostrar accidentes")
    selected_event_file = st.sidebar.selectbox(
        "Archivo de Accidentes (Opcional):",
        sorted(event_files, key=os.path.basename, reverse=True),
        format_func=lambda x: os.path.basename(x) if x != "No mostrar accidentes" else "No mostrar accidentes"
    )

    # 2. Carga de datos
    df = load_data(selected_feature_file)
    gantry_df = load_gantry_data(selected_portico_file)
    accident_df = load_accident_data(selected_event_file if selected_event_file != "No mostrar accidentes" else None, gantry_df)

    if df is None:
        st.stop()

    # 3. Filtros en la barra lateral
    portico_col_name = next((col for col in ['portico', 'id_portico'] if col in df.columns), None)
    if not portico_col_name:
        st.error("No se encontró columna de pórtico en el archivo de features.")
        st.stop()

    st.sidebar.subheader("Filtros y Agrupaciones")
    all_vars = sorted([col for col in df.select_dtypes(include=['number']).columns if col not in [portico_col_name, 'ts_min']])

    # Aplicar filtro de variables si se seleccionó un archivo de nombres
    if isinstance(selected_names_file, str) and os.path.exists(selected_names_file):
        try:
            df_names = pd.read_csv(selected_names_file)
            # Buscar columna 'feature_name' (case-insensitive). Si no, usar la primera columna.
            col_candidates = [c for c in df_names.columns if c.strip().lower() == 'feature_name']
            col = col_candidates[0] if col_candidates else df_names.columns[0]
            selected_name_list = (
                df_names[col]
                .dropna()
                .astype(str)
                .str.strip()
                .tolist()
            )
            filtered_all_vars = [v for v in all_vars if v in selected_name_list]
            if filtered_all_vars:
                all_vars = sorted(filtered_all_vars)
        except Exception as e:
            st.warning(f"No se pudo aplicar el filtro de variables desde '{os.path.basename(selected_names_file)}': {e}")

    selected_variables = st.sidebar.multiselect(
        "Variables a visualizar:",
        all_vars,
        default=(all_vars if all_vars else None)
    )
    selected_porticos = st.sidebar.multiselect("Pórticos:", sorted(df[portico_col_name].unique()), default=sorted(df[portico_col_name].unique()))
    min_time, max_time = df['timestamp'].min(), df['timestamp'].max()
    # Convertir pandas Timestamps a python datetimes para compatibilidad con el slider de Streamlit
    py_min_time = min_time.to_pydatetime()
    py_max_time = max_time.to_pydatetime()
    start_time, end_time = st.sidebar.slider("Ventana temporal:", min_value=py_min_time, max_value=py_max_time, value=(py_min_time, py_max_time))
    chart_type = st.sidebar.selectbox(
        "Tipo de Gráfico:",
        ["Líneas", "Dispersión (Scatter)", "Dispersión 3D", "Histograma", "Box Plot"]
    )

    # Selectores para 3D (elegir 3 features: X, Y, Z)
    x_var = y_var = z_var = None
    if chart_type == "Dispersión 3D":
        st.sidebar.markdown("---")
        st.sidebar.subheader("Ejes 3D")
        x_var = st.sidebar.selectbox("Eje X:", all_vars, key="x_var_3d")
        y_options = [v for v in all_vars if v != x_var] or all_vars
        y_var = st.sidebar.selectbox("Eje Y:", y_options, key="y_var_3d")
        z_options = [v for v in all_vars if v not in {x_var, y_var}] or all_vars
        z_var = st.sidebar.selectbox("Eje Z:", z_options, key="z_var_3d")

    normalize = st.sidebar.checkbox("Normalizar variables (Min-Max)", value=False)

    st.sidebar.markdown("---")
    if st.sidebar.button("🛑 Finalizar"):
        st.info("Cerrando la aplicación de visualización...")
        time.sleep(2)
        os._exit(0)

    # 4. Filtrado de datos
    required_vars = selected_variables if chart_type != "Dispersión 3D" else [x_var, y_var, z_var]
    if not required_vars or not selected_porticos:
        st.info("Selecciona variables y pórticos para generar la gráfica.")
        st.stop()

    filtered_df = df[(df[portico_col_name].isin(selected_porticos)) & (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]

    # 5. Normalización y Gráfica
    if filtered_df.empty:
        st.warning("No hay datos para los filtros seleccionados.")
    else:
        df_to_plot = filtered_df.copy()
        vars_to_normalize = selected_variables if chart_type != "Dispersión 3D" else [x_var, y_var, z_var]
        if normalize and vars_to_normalize:
            for col in vars_to_normalize:
                min_val = df_to_plot[col].min()
                max_val = df_to_plot[col].max()
                if max_val > min_val:
                    df_to_plot[col] = (df_to_plot[col] - min_val) / (max_val - min_val)
        
        fig = create_figure(
            df_to_plot,
            chart_type,
            selected_variables if chart_type != "Dispersión 3D" else [x_var, y_var, z_var],
            portico_col_name,
        )

        if fig:
            # Añadir marcadores de accidentes como anotaciones
            if not accident_df.empty and chart_type in ["Líneas", "Dispersión (Scatter)"]:
                accidents_in_view = accident_df[
                    (accident_df['portico'].isin(selected_porticos)) &
                    (accident_df['timestamp'] >= start_time) &
                    (accident_df['timestamp'] <= end_time)
                ]
                
                annotations = list(fig.layout.annotations) if fig.layout.annotations else []
                if not accidents_in_view.empty and selected_variables:
                    y_col = selected_variables[0]
                    for _, acc in accidents_in_view.iterrows():
                        y_val = None
                        closest_feature_rows = df_to_plot[df_to_plot[portico_col_name] == acc['portico']]
                        if not closest_feature_rows.empty:
                            y_val = pd.to_numeric(closest_feature_rows[y_col], errors='coerce').mean()
                        
                        if not pd.notna(y_val):
                            y_val = df_to_plot[y_col].mean()

                        annotations.append(
                            dict(
                                x=acc['timestamp'],
                                y=y_val,
                                xref="x",
                                yref="y",
                                text="X",
                                showarrow=False,
                                font=dict(
                                    family="sans serif",
                                    size=16,
                                    color="red"
                                ),
                                align="center",
                                bordercolor="white",
                                borderwidth=1,
                                bgcolor="rgba(255,255,255,0.5)"
                            )
                        )
                if annotations:
                    fig.update_layout(annotations=annotations)
            elif not accident_df.empty and chart_type == "Dispersión 3D":
                # Hacer que las mediciones normales sean 'x' pequeñas
                for tr in fig.data:
                    try:
                        tr.update(marker=dict(size=1, symbol='x'))
                    except Exception:
                        tr.update(marker=dict(size=3))

                accidents_in_view = accident_df[
                    (accident_df['portico'].isin(selected_porticos)) &
                    (accident_df['timestamp'] >= start_time) &
                    (accident_df['timestamp'] <= end_time)
                ]
                if not accidents_in_view.empty and x_var and y_var and z_var:
                    xs, ys, zs, texts = [], [], [], []
                    # Preindex by portico for speed
                    df_by_port = {p: g.sort_values('timestamp') for p, g in df_to_plot.groupby(portico_col_name)}
                    for _, acc in accidents_in_view.iterrows():
                        p = acc['portico']
                        sub = df_by_port.get(p)
                        if sub is None or sub.empty:
                            continue
                        # Encontrar el timestamp más cercano al accidente
                        try:
                            nearest_idx = (sub['timestamp'] - acc['timestamp']).abs().idxmin()
                            row = sub.loc[nearest_idx]
                        except Exception:
                            # Fallback: tomar la fila más reciente previa al accidente
                            sub_prior = sub[sub['timestamp'] <= acc['timestamp']]
                            if sub_prior.empty:
                                row = sub.iloc[0]
                            else:
                                row = sub_prior.iloc[-1]
                        xv = pd.to_numeric(row.get(x_var, None), errors='coerce')
                        yv = pd.to_numeric(row.get(y_var, None), errors='coerce')
                        zv = pd.to_numeric(row.get(z_var, None), errors='coerce')
                        if pd.notna(xv) and pd.notna(yv) and pd.notna(zv):
                            xs.append(float(xv)); ys.append(float(yv)); zs.append(float(zv))
                            texts.append(f"{p}<br>{acc['timestamp']}")
                    if xs:
                        fig.add_trace(go.Scatter3d(
                            x=xs, y=ys, z=zs,
                            mode='markers',
                            marker=dict(size=6, color='red'),
                            name='Accidentes',
                            text=texts,
                            hovertemplate="%{text}<extra></extra>")
                        )
            elif not accident_df.empty and chart_type == "Dispersión 3D":
                accidents_in_view = accident_df[
                    (accident_df['portico'].isin(selected_porticos)) &
                    (accident_df['timestamp'] >= start_time) &
                    (accident_df['timestamp'] <= end_time)
                ]
                if not accidents_in_view.empty and x_var and y_var and z_var:
                    xs, ys, zs, texts = [], [], [], []
                    for _, acc in accidents_in_view.iterrows():
                        rows = df_to_plot[df_to_plot[portico_col_name] == acc['portico']]
                        if rows.empty:
                            continue
                        xv = pd.to_numeric(rows[x_var], errors='coerce').mean()
                        yv = pd.to_numeric(rows[y_var], errors='coerce').mean()
                        zv = pd.to_numeric(rows[z_var], errors='coerce').mean()
                        if pd.notna(xv) and pd.notna(yv) and pd.notna(zv):
                            xs.append(xv); ys.append(yv); zs.append(zv)
                            texts.append(f"{acc['portico']}<br>{acc['timestamp']}")
                    if xs:
                        fig.add_trace(go.Scatter3d(
                            x=xs, y=ys, z=zs,
                            mode='markers',
                            marker=dict(size=5, color='red'),
                            name='Accidentes',
                            text=texts,
                            hovertemplate="%{text}<extra></extra>")
                        )
            
            st.plotly_chart(fig, width="stretch")
        else:
            st.error(f"El tipo de gráfico '{chart_type}' aún no está implementado.")

@cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    if 'ts_min' in df.columns:
        df['timestamp'] = pd.to_datetime(df['ts_min'], unit='m')
        return df
    st.error(f"El archivo {os.path.basename(file_path)} no contiene la columna 'ts_min'.")
    return None

def create_figure(df, chart_type, selected_variables, portico_col):
    if not selected_variables:
        return go.Figure(layout={
            "title": "Por favor, selecciona al menos una variable.",
            "xaxis": {"visible": False}, "yaxis": {"visible": False}
        })

    if chart_type == "Líneas":
        long_df = df.melt(id_vars=['timestamp', portico_col], value_vars=selected_variables, var_name='variable', value_name='valor')
        return px.line(long_df, x='timestamp', y='valor', color='variable', line_dash=portico_col, title="Evolución Temporal de Variables")
    
    elif chart_type == "Dispersión (Scatter)":
        y_var = selected_variables[0]
        if len(selected_variables) > 1:
            st.warning(f"El gráfico de dispersión solo muestra la primera variable seleccionada: {y_var}.")
        return px.scatter(df, x='timestamp', y=y_var, color=portico_col, title=f'Dispersión de {y_var} vs Tiempo')

    elif chart_type == "Dispersión 3D":
        # selected_variables en este modo debe ser [x, y, z]
        if len(selected_variables) < 3:
            return go.Figure(layout={
                "title": "Seleccione tres variables para X, Y y Z.",
                "xaxis": {"visible": False}, "yaxis": {"visible": False}
            })
        x_var, y_var, z_var = selected_variables[:3]
        return px.scatter_3d(df, x=x_var, y=y_var, z=z_var, color=portico_col,
                             title=f"Dispersión 3D: {x_var} vs {y_var} vs {z_var}")

    elif chart_type == "Histograma":
        long_df = df.melt(value_vars=selected_variables, var_name='variable', value_name='valor')
        return px.histogram(long_df, x='valor', color='variable', facet_col='variable', 
                            title="Distribución de Variables", histnorm='probability density',
                            facet_col_wrap=min(3, len(selected_variables)))

    elif chart_type == "Box Plot":
        long_df = df.melt(id_vars=[portico_col], value_vars=selected_variables, var_name='variable', value_name='valor')
        return px.box(long_df, x='variable', y='valor', color=portico_col, 
                      title="Distribución de Variables por Pórtico")

    return None



if __name__ == "__main__":
    run_visualization()
