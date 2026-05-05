import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
import glob
import os
import subprocess
from typing import Optional

# Add src to path if needed
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
RESULTS_DIR = ROOT_DIR / "Resultados"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import necessary modules
try:
    from src.clustering import CLUSTER_DB_PATH
except ImportError:
    CLUSTER_DB_PATH = None

# Import MARL logic
from src.marl_core import DEFAULT_STATE_COLS, MARLManager, MAIRLManager
from src.sumo_simulation_app import (
    _build_sumo_subprocess_env,
    find_sumo_binary, 
    SIMULATION_DIR,
    _render_flow_loader,
    _render_porticos_loader
)
from src.SUMO import (
    build_sumo_fcd_transition_dataset,
    is_sumo_fcd_transition_dataset,
    run_sumo_pipeline,
    FlowColumns,
)

def _init_state():
    if "marl_manager" not in st.session_state:
        st.session_state["marl_manager"] = None
    if "marl_training_active" not in st.session_state:
        st.session_state["marl_training_active"] = False
    if "marl_rewards_log" not in st.session_state:
        st.session_state["marl_rewards_log"] = []
    if "marl_cluster_stats" not in st.session_state:
        st.session_state["marl_cluster_stats"] = {}
    if "marl_agent_config" not in st.session_state:
        st.session_state["marl_agent_config"] = []
    if "marl_cluster_df" not in st.session_state:
        st.session_state["marl_cluster_df"] = None
    if "mairl_transition_df" not in st.session_state:
        st.session_state["mairl_transition_df"] = None
    if "mairl_avi_transition_df" not in st.session_state:
        st.session_state["mairl_avi_transition_df"] = None
    if "mairl_transition_source" not in st.session_state:
        st.session_state["mairl_transition_source"] = None
    if "mairl_fcd_path" not in st.session_state:
        st.session_state["mairl_fcd_path"] = str(SIMULATION_DIR / "sumo_fcd.xml")
    if "mairl_sumo_result" not in st.session_state:
        st.session_state["mairl_sumo_result"] = None
    if "mairl_feature_cols" not in st.session_state:
        st.session_state["mairl_feature_cols"] = []
    if "mairl_metrics_log" not in st.session_state:
        st.session_state["mairl_metrics_log"] = []
    if "mairl_manager" not in st.session_state:
        st.session_state["mairl_manager"] = None
    if "mairl_last_metrics" not in st.session_state:
        st.session_state["mairl_last_metrics"] = {}

def load_cluster_data(csv_path: Path):
    try:
        df = pd.read_csv(csv_path)
        # Clean column names to avoid whitespace issues
        df.columns = df.columns.str.strip()
        st.session_state["marl_cluster_df"] = df.copy()
        
        if "cluster_label" not in df.columns:
            st.error("El archivo CSV debe contener una columna 'cluster_label'.")
            return None
        
        # Check if it's a descriptive file (already aggregated)
        is_descriptive = "avg_speed_kmh_mean" in df.columns
        
        if is_descriptive:
            # Descriptive format: One row per cluster
            stats = {}
            for _, row in df.iterrows():
                label = row["cluster_label"]
                stats[label] = {
                    "avg_speed_kmh": row.get("avg_speed_kmh_mean"),
                    "avg_headway_s": row.get("avg_headway_s_mean"),
                    "lane_change_rate": row.get("lane_change_rate_mean"),
                    "plate": row.get("avg_speed_kmh_count", 1) 
                }
        else:
            # Raw format: One row per vehicle -> Aggregate
            stats = df.groupby("cluster_label").agg({
                "avg_speed_kmh": "mean",
                "avg_headway_s": "mean",
                "lane_change_rate": "mean",
                "plate": "count" # Count size
            }).to_dict(orient="index")
        
        return stats
    except Exception as e:
        st.error(f"Error cargando CSV: {e}")
        return None

def _agent_config_from_transitions(df: pd.DataFrame):
    if df is None or df.empty or "agent_id" not in df.columns:
        return []
    config = []
    for agent_id, group in df.groupby("agent_id", sort=True):
        config.append(
            {
                "id": str(agent_id),
                "name": f"Policy_{agent_id}",
                "stats": {"transitions": int(len(group))},
            }
        )
    return config


def _training_agent_config(df: pd.DataFrame):
    config = st.session_state.get("marl_agent_config", [])
    if df is None or df.empty or "agent_id" not in df.columns:
        return config
    available = set(df["agent_id"].astype(str))
    configured = {str(agent.get("id")) for agent in config}
    if config and configured.issubset(available):
        return config
    return _agent_config_from_transitions(df)


def _set_fcd_transitions(transition_df: pd.DataFrame, fcd_path: Path) -> None:
    st.session_state["mairl_transition_df"] = transition_df
    st.session_state["mairl_transition_source"] = "sumo_fcd"
    st.session_state["mairl_fcd_path"] = str(fcd_path)
    st.session_state["mairl_manager"] = None
    st.session_state["mairl_metrics_log"] = []
    st.session_state["mairl_last_metrics"] = {}
    config = _agent_config_from_transitions(transition_df)
    if config:
        st.session_state["marl_agent_config"] = config
        st.session_state["marl_num_agents"] = len(config)


def _build_transitions_from_fcd(fcd_path: Path) -> Optional[pd.DataFrame]:
    try:
        transitions = build_sumo_fcd_transition_dataset(fcd_path)
    except Exception as e:
        st.error(f"Error leyendo FCD SUMO: {e}")
        return None
    if transitions.empty:
        st.error("El FCD no contiene pares consecutivos válidos para construir transiciones.")
        return None
    _set_fcd_transitions(transitions, fcd_path)
    st.success(f"Transiciones FCD reconstruidas: {len(transitions):,}.")
    return transitions


def render_configuration():
    st.header("Configuración de Reward Learning")
    
    st.subheader("1. Trayectorias expertas desde SUMO FCD")
    st.caption(
        "MA-AIRL entrena con transiciones reales s, a, s_next formadas desde muestras "
        "consecutivas del FCD de SUMO. No hay fallback a s'=s."
    )

    default_cfg = SIMULATION_DIR / "sample.sumocfg"
    cfg_value = str(default_cfg) if default_cfg.exists() else ""
    cfg_path = st.text_input("Archivo .sumocfg", value=cfg_value, key="mairl_sumo_cfg_path")
    default_tripinfo = SIMULATION_DIR / "tripinfo.xml"
    tripinfo_path = st.text_input(
        "Salida tripinfo.xml",
        value=str(default_tripinfo),
        key="mairl_tripinfo_path",
    )
    fcd_path_value = st.text_input(
        "Salida/importación FCD XML",
        value=st.session_state.get("mairl_fcd_path", str(SIMULATION_DIR / "sumo_fcd.xml")),
        key="mairl_fcd_output_path",
    )
    fcd_period = st.number_input(
        "Periodo FCD (segundos)",
        value=1.0,
        min_value=0.1,
        step=0.5,
        key="mairl_fcd_period",
    )

    col_run, col_import = st.columns(2)
    with col_run:
        if st.button("Ejecutar SUMO y construir transiciones FCD", type="primary"):
            if not cfg_path:
                st.error("Ingrese la ruta del archivo .sumocfg.")
            else:
                cfg_file = Path(cfg_path).expanduser()
                if not cfg_file.exists():
                    st.error("El archivo .sumocfg no existe.")
                else:
                    tripinfo_file = Path(tripinfo_path).expanduser()
                    fcd_file = Path(fcd_path_value).expanduser()
                    tripinfo_file.parent.mkdir(parents=True, exist_ok=True)
                    fcd_file.parent.mkdir(parents=True, exist_ok=True)

                    executable = "sumo-gui" if st.session_state.get("marl_gui") else "sumo"
                    sumo_bin = find_sumo_binary(executable)
                    if sumo_bin is None:
                        st.error(f"No se encontró el ejecutable '{executable}'. Configure SUMO_HOME o el PATH.")
                    else:
                        cmd = [
                            str(sumo_bin),
                            "-c",
                            str(cfg_file),
                            "--tripinfo-output",
                            str(tripinfo_file),
                            "--fcd-output",
                            str(fcd_file),
                            "--device.fcd.period",
                            f"{float(fcd_period):.3f}",
                            "--fcd-output.acceleration",
                            "true",
                            "--no-step-log",
                            "true",
                            "--duration-log.disable",
                            "true",
                        ]
                        with st.spinner("Ejecutando SUMO con salida FCD..."):
                            try:
                                subprocess.run(cmd, check=True, env=_build_sumo_subprocess_env())
                            except subprocess.CalledProcessError as exc:
                                st.error(f"SUMO falló (código {exc.returncode}).")
                            else:
                                _build_transitions_from_fcd(fcd_file)

    with col_import:
        if st.button("Importar FCD existente"):
            fcd_file = Path(fcd_path_value).expanduser()
            if not fcd_file.exists():
                st.error("El archivo FCD no existe.")
            else:
                _build_transitions_from_fcd(fcd_file)

    transition_df = st.session_state.get("mairl_transition_df")
    if transition_df is not None and not transition_df.empty:
        st.metric("Transiciones FCD expertas", f"{len(transition_df):,}")
        if "temporal_split" in transition_df.columns:
            split_counts = transition_df["temporal_split"].value_counts().rename_axis("split").reset_index(name="transitions")
            st.dataframe(split_counts, width="stretch")
        st.dataframe(transition_df.head(200), width="stretch")

        st.subheader("2. MA-AIRL: features del estado")
        numeric_cols = transition_df.select_dtypes(include=[np.number, bool]).columns.tolist()
        state_options = [
            c
            for c in numeric_cols
            if c.startswith("state_") or c.startswith("flow_context_") or c == "vms_active"
        ]
        default_cols = [c for c in DEFAULT_STATE_COLS if c in state_options]
        selected_cols = st.multiselect(
            "Columnas para estado s",
            options=state_options,
            default=default_cols or state_options[:5],
            key="mairl_feature_cols",
        )
        if not selected_cols:
            st.warning("Seleccione al menos una columna de estado para MA-AIRL.")

    st.subheader("3. Clústeres opcionales")
    st.caption("Los clústeres quedan como baseline o agrupación experimental; no reemplazan las trayectorias FCD.")
    
    # Scan for CSV files
    csv_files = glob.glob(str(RESULTS_DIR / "cluster_*.csv"))
    csv_options = [Path(p).name for p in csv_files]
    
    selected_file = st.selectbox(
        "Archivo de clústeres opcional:",
        options=csv_options,
        index=0 if csv_options else None
    )
    
    if selected_file:
        file_path = RESULTS_DIR / selected_file
        if st.button("Cargar y Analizar Clústeres"):
            stats = load_cluster_data(file_path)
            if stats:
                st.session_state["marl_cluster_stats"] = stats
                st.success(f"Se cargaron {len(stats)} clústeres exitosamente.")
                
                # Auto-configure agents
                agent_config = []
                for label, data in stats.items():
                    agent_config.append({
                        "id": label,
                        "name": f"Cluster_{label}",
                        "stats": data
                    })
                st.session_state["marl_agent_config"] = agent_config
                # Force update of the number input
                st.session_state["marl_num_agents"] = len(agent_config)
                st.rerun() # Rerun to refresh the UI immediately
    
    # Display Stats
    if st.session_state["marl_cluster_stats"]:
        st.write("Estadísticas de clústeres opcionales:")
        st.dataframe(pd.DataFrame(st.session_state["marl_cluster_stats"]).T)

    st.subheader("4. Parámetros de Simulación (SUMO)")
    col1, col2 = st.columns(2)
    with col1:
        st.number_input("Duración de Episodio (segundos)", value=3600, step=600, key="marl_duration")
        st.number_input("Paso de Simulación (segundos)", value=0.5, step=0.1, key="marl_step")
    with col2:
        st.checkbox("Usar GUI de SUMO", value=False, key="marl_gui")
        st.selectbox("Estrategia de Emisión", ["Matriz OD"], key="marl_emission")

    st.subheader("5. Diagnóstico AVI y pórticos")
    st.caption("Este bloque reconstruye transiciones AVI parciales para inspección; MA-AIRL no las usa como fallback.")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("##### Carga de flujos")
        flujos_df = _render_flow_loader()

    with col_b:
        st.markdown("##### Carga de pórticos")
        porticos_df = _render_porticos_loader()

    if st.button("Reconstruir transiciones AVI para diagnóstico"):
        if flujos_df is None or flujos_df.empty:
            st.error("Debe cargar flujos primero.")
        elif porticos_df is None or porticos_df.empty:
            st.error("Debe cargar pórticos primero.")
        else:
            with st.spinner("Reconstruyendo trayectorias parciales y transiciones IRL..."):
                try:
                    result = run_sumo_pipeline(
                        flujos_df,
                        porticos_df,
                        flow_cols=FlowColumns(),
                        output_dir=SIMULATION_DIR,
                        segment_filter=None,
                    )
                except Exception as e:
                    st.error(f"Error reconstruyendo transiciones: {e}")
                    return
            st.session_state["mairl_sumo_result"] = result
            st.session_state["mairl_sumo_result"] = result
            st.session_state["mairl_avi_transition_df"] = result.irl_transitions
            st.success(f"Transiciones AVI diagnósticas reconstruidas: {len(result.irl_transitions):,}.")

    avi_df = st.session_state.get("mairl_avi_transition_df")
    if avi_df is not None and not avi_df.empty:
        st.metric("Transiciones AVI diagnósticas", f"{len(avi_df):,}")
        st.dataframe(avi_df.head(100), width="stretch")

    st.subheader("6. Configuración de políticas compartidas")
    
    # Use config from state if available
    current_config = st.session_state.get("marl_agent_config", [])
    num_clusters = st.number_input("Número de políticas", value=len(current_config) if current_config else 3, min_value=1, max_value=20, key="marl_num_agents")
    
    # Adjust config list size if number changed manually
    if len(current_config) != num_clusters:
        # If growing, add defaults. If shrinking, slice.
        if num_clusters < len(current_config):
            current_config = current_config[:num_clusters]
        else:
             for i in range(len(current_config), num_clusters):
                 current_config.append({"id": i, "name": f"Policy_{i}", "stats": {}})
        st.session_state["marl_agent_config"] = current_config

    cols = st.columns(3)
    updated_config = []
    for i, agent in enumerate(current_config):
        with cols[i % 3]:
            new_name = st.text_input(f"Nombre política {i+1}", value=agent["name"], key=f"marl_agent_name_{i}")
            agent["name"] = new_name
            if agent["stats"]:
                if "transitions" in agent["stats"]:
                    st.caption(f"Transiciones: {agent['stats'].get('transitions', 0):,}")
                else:
                    st.caption(f"Speed: {agent['stats'].get('avg_speed_kmh',0):.1f}, LC: {agent['stats'].get('lane_change_rate',0):.2f}")
            updated_config.append(agent)

    if st.button("Generar Rutas de Tráfico (Background Traffic)"):
        if flujos_df is None or flujos_df.empty:
            st.error("Debe cargar flujos primero.")
        elif porticos_df is None or porticos_df.empty:
            st.error("Debe cargar pórticos primero.")
        else:
            with st.spinner("Generando rutas SUMO desde flujos..."):
                try:
                    result = run_sumo_pipeline(
                        flujos_df,
                        porticos_df,
                        flow_cols=FlowColumns(),
                        output_dir=SIMULATION_DIR,
                        segment_filter=None
                    )
                    
                    if result.sumo_trips_path and result.sumo_trips_path.exists():
                        st.success(f"Rutas generadas en: {result.sumo_trips_path}")
                        st.session_state["marl_background_traffic_path"] = str(result.sumo_trips_path)
                    else:
                        st.error("El pipeline no generó trips válidos.")
                except Exception as e:
                    st.error(f"Error generando rutas: {e}")

    traffic_path = st.session_state.get("marl_background_traffic_path")
    if traffic_path:
        st.info(f"Rutas de fondo activas: {Path(traffic_path).name}")

def render_training():
    st.header("Entrenamiento de Reward Learning")

    algorithm = st.selectbox(
        "Algoritmo",
        options=["MA-AIRL (SUMO FCD)", "CEM (legacy)"],
        index=0,
        key="marl_algorithm",
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Control")

        if algorithm == "MA-AIRL (SUMO FCD)":
            iterations = st.number_input("Iteraciones", value=200, min_value=1, step=50, key="mairl_iterations")
            batch_size = st.number_input("Batch size", value=256, min_value=8, step=16, key="mairl_batch")
            gamma = st.slider("Gamma (descuento)", min_value=0.80, max_value=0.99, value=0.99, step=0.01, key="mairl_gamma")
            policy_lr = st.number_input("LR Policy", value=3e-4, format="%.5f", key="mairl_policy_lr")
            reward_lr = st.number_input("LR Reward", value=3e-4, format="%.5f", key="mairl_reward_lr")
            hidden = st.selectbox("Hidden units", options=[32, 64, 128], index=1, key="mairl_hidden")
            device = st.selectbox("Device", options=["auto", "cpu"], index=0, key="mairl_device")

            if st.button("Iniciar Entrenamiento", type="primary", key="mairl_start"):
                expert_df = st.session_state.get("mairl_transition_df")
                if expert_df is None or expert_df.empty:
                    st.error("Debe generar o importar transiciones FCD de SUMO antes de entrenar MA-AIRL.")
                elif (
                    st.session_state.get("mairl_transition_source") != "sumo_fcd"
                    or not is_sumo_fcd_transition_dataset(expert_df)
                ):
                    st.error("MA-AIRL requiere transiciones FCD de SUMO; no se permite fallback a AVI ni a s'=s.")
                else:
                    train_df = expert_df
                    if "temporal_split" in expert_df.columns:
                        train_df = expert_df[expert_df["temporal_split"] == "train"].copy()
                    if train_df.empty:
                        st.error("La partición temporal de entrenamiento no contiene transiciones válidas.")
                        return
                    st.session_state["marl_training_active"] = True
                    st.session_state["mairl_metrics_log"] = []

                    config = _training_agent_config(train_df)
                    if not config:
                        st.warning("No hay políticas configuradas. Usando una política compartida.")
                        config = [{"id": "shared_policy", "name": "SharedPolicy", "stats": {}}]

                    try:
                        manager = MAIRLManager(
                            expert_df=train_df,
                            agent_config=config,
                            feature_cols=st.session_state.get("mairl_feature_cols") or None,
                            hidden_sizes=(hidden, hidden),
                            gamma=float(gamma),
                            policy_lr=float(policy_lr),
                            reward_lr=float(reward_lr),
                            device=device,
                        )
                    except Exception as e:
                        st.error(f"Error inicializando MA-AIRL: {e}")
                        st.session_state["marl_training_active"] = False
                        return

                    st.session_state["mairl_manager"] = manager

                    progress = st.progress(0)
                    status = st.empty()
                    chart = st.empty()

                    for i in range(int(iterations)):
                        if not st.session_state["marl_training_active"]:
                            break

                        status.write(f"Iteración {i+1}/{int(iterations)}...")
                        metrics = manager.train_step(batch_size=int(batch_size))
                        metrics["iteration"] = i + 1
                        st.session_state["mairl_metrics_log"].append(metrics)
                        st.session_state["mairl_last_metrics"] = metrics

                        df_metrics = pd.DataFrame(st.session_state["mairl_metrics_log"]).set_index("iteration")
                        chart.line_chart(df_metrics[["disc_loss", "policy_loss", "reward_mean"]])

                        progress.progress((i + 1) / int(iterations))
                        time.sleep(0.05)

                    status.write("Entrenamiento completado.")
                    st.session_state["marl_training_active"] = False

            if st.button("Detener", key="mairl_stop"):
                st.session_state["marl_training_active"] = False
                st.warning("Entrenamiento detenido.")

            if st.session_state.get("mairl_manager") is not None and st.button("Exportar perfiles MA-AIRL"):
                output_path = SIMULATION_DIR / "mairl_policy_profiles.json"
                try:
                    st.session_state["mairl_manager"].export_policy_profiles(output_path)
                    st.success(f"Perfiles exportados en: {output_path}")
                except Exception as e:
                    st.error(f"Error exportando perfiles: {e}")
        else:
            sumo_bin = find_sumo_binary("sumo")
            if not sumo_bin:
                st.error("No se encontró el binario de SUMO. Configure SUMO_HOME.")
                return

            episodes = st.number_input("Episodios por Iteración", value=5, min_value=1, key="marl_episodes")
            iterations = st.number_input("Iteraciones", value=10, min_value=1, key="marl_iters_legacy")

            if st.button("Iniciar Entrenamiento", type="primary", key="marl_start_legacy"):
                st.session_state["marl_training_active"] = True
                st.session_state["marl_rewards_log"] = []

                manager = MARLManager(SIMULATION_DIR, sumo_cmd=[str(sumo_bin)])

                config = st.session_state.get("marl_agent_config", [])
                if not config:
                    st.warning("No hay agentes configurados. Usando defaults.")
                    for i in range(3):
                        manager.add_agent(f"Cluster_{i}")
                else:
                    for agent in config:
                        manager.add_agent(agent["name"], target_stats=agent["stats"])

                st.session_state["marl_manager"] = manager

                progress = st.progress(0)
                status = st.empty()
                chart = st.empty()

                for i in range(int(iterations)):
                    if not st.session_state["marl_training_active"]:
                        break

                    status.write(f"Iteración {i+1}/{int(iterations)}...")
                    iter_rewards = []
                    for e in range(int(episodes)):
                        rewards = manager.run_episode(i * int(episodes) + e)
                        if rewards:
                            iter_rewards.append(np.mean(list(rewards.values())))

                    avg_reward = float(np.mean(iter_rewards)) if iter_rewards else 0.0
                    st.session_state["marl_rewards_log"].append(avg_reward)
                    chart.line_chart(st.session_state["marl_rewards_log"])

                    progress.progress((i + 1) / int(iterations))
                    time.sleep(0.1)

                status.write("Entrenamiento completado.")
                st.session_state["marl_training_active"] = False

            if st.button("Detener", key="marl_stop_legacy"):
                st.session_state["marl_training_active"] = False
                st.warning("Entrenamiento detenido.")

    with col2:
        st.subheader("Métricas en Tiempo Real")
        if algorithm == "MA-AIRL (SUMO FCD)":
            if st.session_state["mairl_metrics_log"]:
                df_metrics = pd.DataFrame(st.session_state["mairl_metrics_log"]).set_index("iteration")
                st.line_chart(df_metrics[["disc_loss", "policy_loss", "reward_mean"]])
                last = st.session_state.get("mairl_last_metrics", {})
                if last:
                    st.metric("Disc loss", f"{last.get('disc_loss', 0):.4f}")
                    st.metric("Policy loss", f"{last.get('policy_loss', 0):.4f}")
                    st.metric("Reward mean", f"{last.get('reward_mean', 0):.4f}")
            else:
                st.info("Inicie el entrenamiento MA-AIRL para ver métricas.")
        else:
            if st.session_state["marl_rewards_log"]:
                st.line_chart(st.session_state["marl_rewards_log"])
            else:
                st.info("Inicie el entrenamiento para ver métricas.")

def render_visualization():
    st.header("Análisis y Visualización")
    st.write("Recompensas y políticas aprendidas desde transiciones SUMO FCD.")

    if st.session_state.get("mairl_metrics_log"):
        st.subheader("MA-AIRL: Métricas de Entrenamiento")
        df_metrics = pd.DataFrame(st.session_state["mairl_metrics_log"]).set_index("iteration")
        st.dataframe(df_metrics)
        manager = st.session_state.get("mairl_manager")
        if manager is not None:
            st.subheader("Acciones aprendidas por política compartida")
            try:
                params = manager.get_policy_actions()
                df_params = pd.DataFrame(params).T
                st.dataframe(df_params)
            except Exception as e:
                st.warning(f"No se pudieron obtener parámetros aprendidos: {e}")
    elif st.session_state.get("marl_manager"):
        history = st.session_state["marl_manager"].history
        if history:
            df_hist = pd.DataFrame(history)
            st.write("Historial de Entrenamiento (CEM):", df_hist)
        else:
            st.info("No hay historial disponible.")
    else:
        st.info("No hay historial disponible.")

    st.subheader("Limitaciones y próximos pasos")
    st.markdown(
        r"""
        - \(s'\) y las acciones se estiman desde pares consecutivos del FCD de SUMO.
        - La calidad de las recompensas depende de la resolución del FCD y de rutas SUMO válidas.
        - SUMOPy queda como referencia conceptual; no se añade como dependencia por su stack histórico Python 2.7/wxPython.
        """
    )

def main(set_page_config: bool = True, show_exit_button: bool = True) -> None:
    _init_state()
    if set_page_config:
        st.set_page_config(page_title="Reward Learning", layout="wide")
    
    st.title("Reward Learning for Driver Behavior")
    
    tabs = st.tabs(["Configuración", "Entrenamiento", "Análisis"])
    
    with tabs[0]:
        render_configuration()
    
    with tabs[1]:
        render_training()
    
    with tabs[2]:
        render_visualization()

if __name__ == "__main__":
    main()
