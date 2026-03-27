# SUMO - Drivers Behavior Modeling and Simulation

Aplicación multipágina en Streamlit y utilidades CLI para analizar flujos vehiculares, agrupar comportamiento de conductores, predecir accidentes, construir grafos para GNN, ejecutar módulos especializados de NLP y drift, y correr un pipeline de simulación con SUMO.

El repositorio usa como ejes principales:

- `Datos/` para insumos persistentes.
- `Resultados/` para artefactos, modelos y salidas intermedias.
- `simulación/` para archivos propios del flujo SUMO.
- `src/` para la lógica de negocio y las apps de Streamlit.

## Módulos Activos

El menú principal se define en `streamlit_main.py` y hoy expone estos módulos:

### Data & Gestión

- `Flow database` (`src/flow_database_app.py`): importa, consulta y administra la base DuckDB de flujos.
- `Files` (`src/files_app.py`): navega y administra artefactos en `Resultados/` y carpetas similares.
- `GitHub Sync` (`src/github_sync_app.py`): sincroniza el repo local con el remoto y permite purgar del remoto archivos ya trackeados que ahora están cubiertos por `.gitignore`, sin borrar la copia local.

### Análisis & Modelos

- `Clustering` (`src/clustering_tabs_app.py`): genera variables por patente y ejecuta K-Means, GMM y HDBSCAN.
- `Crash prediction` (`src/cluster_accident_app.py`): entrena y evalúa modelos de predicción de accidentes usando variables de flujo y, opcionalmente, variables agregadas por cluster.
- `NLP in Severity` (`src/nlp_severity_app.py`): construye datasets granulares con texto para modelar severidad.
- `Drift detection` (`src/drift_detection_app.py`): replica estrategias de recalibración y detección de drift para predicción de crash en tiempo real.
- `Graph Neural Network` (`src/graph_builder_app.py`, `src/gnn_main.py`): construye grafos espaciotemporales, ejecuta entrenamiento GNN y soporta balanceo con GraphSMOTE e ImGAGN.
- `Multi Agent RL` (`src/multi_agent_rl_app.py`): módulo experimental de aprendizaje por refuerzo multiagente sobre el flujo SUMO.

### Simulación & Visualización

- `Events` (`src/events_map_app.py`): visualiza eventos en mapa usando `Datos/eventos.duckdb`.
- `Experiments Live` (`src/experiments_live_app.py`): monitorea en vivo bases SQLite `Resultados/experiment_live_*.sqlite`.
- `Simulación SUMO` (`src/sumo_simulation_app.py`): ejecuta el pipeline SUMO, genera `sumo_trips.rou.xml`, corre `duarouter` y luego `sumo`.

### Configuración

- `Notification system` (`src/notification_system.py`): configura notificaciones por correo usando `email_config.json`.
- `LaTeX` (`src/latex_viewer_app.py`): abre y, si `latexmk` está instalado, compila documentación `.tex`.
- `Test` (`src/test_page.py`): página de pruebas.

## Estructura Del Proyecto

```text
.
├── streamlit_main.py
├── main.py
├── start_app.command
├── start_app_windows.bat
├── venv_start.py
├── src/
├── Datos/
├── Resultados/
├── simulación/
├── docs/
├── DRIFT/
├── NLP/
└── tests/
```

## Requisitos

### Requisitos base

- Python `3.10` a `3.12`.
- Recomendado: Python `3.12`.
- No usar Python `3.13+` si se va a trabajar con PyTorch Geometric y NeighborLoader.
- `pip`, `venv` y dependencias de `requirements.txt`.

### Dependencias opcionales según módulo

- SUMO (`sumo`, `duarouter`) disponible por `PATH` o vía `SUMO_HOME`.
- `latexmk` para compilar `.tex` desde el visor LaTeX.
- Git configurado si se va a usar `GitHub Sync`.

## Instalación Recomendada

La forma más segura de preparar el entorno es usar el bootstrap del repo, porque valida la versión de Python e instala primero el stack de PyTorch Geometric.

```bash
python venv_start.py --python python3.12
source .venv/bin/activate
```

Si `python3.12` ya es tu `python` por defecto:

```bash
python venv_start.py
source .venv/bin/activate
```

En Windows:

```bat
py -3.12 venv_start.py
.venv\Scripts\activate
```

## Arranque De La Aplicación

### Opción recomendada

macOS:

```bash
./start_app.command
```

Windows:

```bat
start_app_windows.bat
```

### Opción manual

```bash
source .venv/bin/activate
streamlit run streamlit_main.py
```

## Entrypoints Disponibles

- `streamlit_main.py`: launcher principal de toda la suite.
- `main.py`: CLI alterna para tres bloques concretos: Flow database, SUMO y Clustering.

CLI:

```bash
source .venv/bin/activate
python main.py
```

## Insumos Esperados

### 1. Flujos vehiculares

El flujo de trabajo estándar persiste los datos en:

- `Datos/flujos.duckdb`
- tabla `flujos_duckdb`

Columnas estándar esperadas por las utilidades base:

- `FECHA`
- `VELOCIDAD`
- `CATEGORIA`
- `MATRICULA`
- `PORTICO`
- `CARRIL`

Si todavía no existe `Datos/flujos.duckdb`, el módulo `Flow database` permite importar un CSV de flujos y dejarlo listo para el resto del pipeline.

### 2. Pórticos

Archivo esperado:

- `Datos/Porticos.csv`

El loader asume separador `;` y necesita, al menos:

- `cod_portico`
- `Km`
- `Calzada`
- `Orden`
- `Eje`

Columnas opcionales útiles para SUMO y visualización:

- `edge_id_sumo`
- `lane_id_sumo`
- `pos_m`
- `lat`
- `lon`
- `lat-lon`

### 3. Eventos y accidentes

Según el módulo, el repo utiliza archivos como:

- `Datos/Eventos-2018-2021.csv`
- `Datos/Eventos-2022-2024.csv`
- `Datos/eventos.duckdb`

Estos insumos alimentan principalmente `Events`, `Crash prediction`, `NLP in Severity` y `Graph Neural Network`.

### 4. Archivos SUMO

Para ejecutar la parte de simulación se usan, por defecto, estos archivos:

- `simulación/highway.net.xml`
- `simulación/sample.sumocfg`

Y se generan:

- `simulación/sumo_trips.rou.xml`
- `simulación/routes.rou.xml`
- `simulación/tripinfo.xml`
- `simulación/sumo_depart_summary.xml`

## Orden Recomendado De Ejecución Del Pipeline

### Pipeline general

1. Crear y activar el entorno virtual.
2. Iniciar la app con `streamlit run streamlit_main.py` o con `./start_app.command`.
3. Entrar a `Flow database` e importar o validar `Datos/flujos.duckdb`.
4. Verificar `Datos/Porticos.csv` antes de usar módulos que dependen de georreferencia, segmentación o SUMO.
5. Ejecutar `Clustering` si se requieren features por patente o agregados por cluster.
6. Ejecutar `Crash prediction` para construir features por pórtico-intervalo, entrenar modelos y evaluar resultados.
7. Ejecutar `Graph Neural Network` si se desea construir grafos y entrenar variantes GNN sobre el problema de accidentes.
8. Usar `Experiments Live` para monitorear optimizaciones y experimentos que escriben SQLite en `Resultados/`.
9. Usar `Events`, `NLP in Severity`, `Drift detection` o `Multi Agent RL` según el caso de estudio.

### Pipeline SUMO

1. Confirmar que `SUMO_HOME` esté configurado o que `sumo` y `duarouter` estén en el `PATH`.
2. Asegurar que `Datos/flujos.duckdb` y `Datos/Porticos.csv` estén disponibles.
3. Abrir `Simulación SUMO`.
4. Ejecutar el pipeline para reconstruir trayectorias y generar `simulación/sumo_trips.rou.xml`.
5. Ejecutar la pestaña de `duarouter` para generar `routes.rou.xml`.
6. Ejecutar la pestaña de `sumo` para producir `tripinfo.xml`.

También se puede hacer por CLI:

```bash
source .venv/bin/activate
python main.py
```

Luego seleccionar:

- `1` para Flow database
- `2` para SUMO
- `3` para Clustering

## Artefactos Generados

Dependiendo del módulo, el pipeline genera artefactos como:

- `Datos/flujos.duckdb`
- `Resultados/cluster_features*.duckdb`
- `Resultados/cluster_kmeans_k*.csv`
- `Resultados/cluster_gmm_k*.csv`
- `Resultados/cluster_hdbscan.csv`
- `Resultados/cluster_summary*.csv`
- `Resultados/experiment_live_*.sqlite`
- modelos, embeddings, grafos y reportes GNN en `Resultados/`
- artefactos SUMO en `simulación/`

## Validación Rápida

Antes de correr experimentos largos, conviene verificar:

- que `Flow database` detecte correctamente la cobertura temporal de `Datos/flujos.duckdb`
- que `Datos/Porticos.csv` cargue sin errores
- que `Resultados/` tenga permisos de escritura
- que `sumo` y `duarouter` respondan si se usará la simulación
- que `latexmk` esté instalado si se quiere compilar documentación desde la app

## Troubleshooting

- Si falla la instalación de PyG, usa Python `3.12` y vuelve a crear `.venv`.
- Si `duarouter` o `sumo` no aparecen, exporta `SUMO_HOME` o agrega `SUMO_HOME/bin` al `PATH`.
- Si el visor LaTeX abre PDFs pero no compila `.tex`, falta `latexmk`.
- Si no existe `Datos/flujos.duckdb`, importa primero el CSV desde `Flow database`.
- Si el pipeline SUMO no genera rutas, revisa que `Porticos.csv` incluya `edge_id_sumo`, `lane_id_sumo` y `pos_m`.

## Testing

```bash
source .venv/bin/activate
pytest
```
