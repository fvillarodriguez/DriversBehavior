# Revisión de código base: hallazgos y tareas propuestas

## 1) Tarea para corregir un error tipográfico
**Hallazgo:** En `src/gnn_main.py` aparece el comentario `FallBck`, que parece un error tipográfico de `Fallback`.

**Tarea propuesta:**
- Corregir el comentario a `Fallback` y revisar comentarios cercanos para evitar más errores tipográficos en bloques de configuración crítica.
- Criterio de aceptación: no quedan ocurrencias de `FallBck` en el repositorio.

## 2) Tarea para corregir una falla
**Hallazgo:** La prueba `tests/test_gnn_feature_engineering_outputs.py` falla durante la recolección con `ModuleNotFoundError: No module named 'src'` cuando se ejecuta `pytest` desde la raíz del proyecto.

**Tarea propuesta:**
- Estandarizar la resolución de imports en pruebas (por ejemplo, con `tests/conftest.py` o configuración de `pytest`/`PYTHONPATH`) para que `from src...` funcione consistentemente.
- Criterio de aceptación: `pytest -q tests/test_gnn_feature_engineering_outputs.py` pasa la fase de colección sin errores de importación.

## 3) Tarea para corregir discrepancia en comentarios/documentación
**Hallazgo:** En `tests/test_notification.py`, el comentario `# Add src to path` no coincide con el código: se añade la raíz del repositorio (`..`) y no el directorio `src`.

**Tarea propuesta:**
- Alinear comentario y comportamiento: o bien actualizar el comentario para reflejar que se agrega la raíz del repo, o ajustar el código para agregar explícitamente `src`.
- Criterio de aceptación: comentario y código describen exactamente la misma acción.

## 4) Tarea para mejorar una prueba
**Hallazgo:** `tests/test_notification.py` cubre el flujo exitoso de envío de correo, pero no valida rutas de error (configuración incompleta, excepción de SMTP) ni asegura restauración robusta cuando el archivo de configuración original está vacío.

**Tarea propuesta:**
- Agregar pruebas negativas para `send_notification_email` (faltan credenciales/destinatarios y excepción en SMTP).
- Robustecer `tearDown` para distinguir entre `None` y cadena vacía al restaurar `email_config.json`.
- Criterio de aceptación: nuevas pruebas fallan antes de la corrección y pasan después, con cobertura explícita de errores y limpieza de estado.
