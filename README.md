# VíaClara — Sistema de Evaluación de Daño Vial (FastAPI + OpenCV + NumPy)

Proyecto listo para desplegar en **Render** en un solo servicio web (Backend en Python sirve también el Frontend minimalista).

## Cómo probar en local (WSL2 / Ubuntu / Windows)
```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Abre `http://localhost:8000` en tu navegador.

## Cómo desplegar en Render.com en 2 minutos
1. Sube el contenido de esta carpeta a un repositorio de GitHub.
2. En Render.com haz clic en **New +** -> **Web Service** y conecta tu repositorio.
3. Si Render te pide los comandos:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
