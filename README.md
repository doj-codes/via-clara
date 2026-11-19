# VíaClara v2 — Aplicación Web Funcional Completa (Rúbrica D1, D2, D3 y D4)

## Novedades de esta versión para cumplir el 100% de la Entrega 3:
1. **CSS 100% Nativo Integrado (Cero dependencia de CDN externos):** Nunca más se perderán los estilos por bloqueos de DNS (`cdn.tailwindcss.com` eliminado por completo).
2. **Entrenamiento desde la Interfaz Web (Criterio D2):** Pestaña dedicada que permite ajustar proporción de prueba, tasa de aprendizaje, épocas y semilla, entrenando en vivo la **Regresión Lineal Múltiple (OLS)** y la **Clasificación Logística Multiclase (One-vs-Rest)** sobre el archivo `dataset_vial.csv` (450 registros).
3. **Todas las Métricas Exigidas (Criterio C3):**
   - Regresión: **R², RMSE y MAE**.
   - Clasificación Multiclase (One-vs-Rest): **Matriz de Confusión 3×3, Accuracy, Precision, Recall, F1-Score y Specificity (Especificidad)** por cada clase.
4. **Predicción por Imagen y por Variables de Entrada X (Criterio D3):** Permite tanto subir fotografías reales (`crack-seg` / `RDD2022`) como ingresar manualmente los valores numéricos $(X_1, X_2, X_3, X_4)$.

## Ejecución Local con `uv` o `pip`
```bash
uv pip install -r requirements.txt
uv run uvicorn main:app --reload --port 8000
```
