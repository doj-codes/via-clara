from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np
import cv2
import base64
import os

app = FastAPI(title="VíaClara - Inspección Vial Inteligente")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pesos de la Regresión Lineal Múltiple calibrados:
# y_hat = BETA_0 + B1*(Densidad de Fisuras) + B2*(Área de Cavidad) + B3*(Rugosidad Superficial)
BETA_0 = 4.2
BETA_WEIGHTS = np.array([52.0, 31.5, 12.3])

def process_image_array(img_bgr: np.ndarray, noise_sensitivity: int = 45):
    img_bgr = cv2.resize(img_bgr, (420, 420))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Preprocesamiento: Filtro Bilateral y Gaussiano para borrar poros del cemento pero conservar bordes de grietas
    smoothed = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    blurred = cv2.GaussianBlur(smoothed, (5, 5), 0)

    # 2. Detección adaptativa local: una grieta real es más oscura que su vecindario inmediato
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=25, C=10
    )

    # 3. Gradiente espacial (Sobel) para asegurar que hay cambio estructural brusco
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    edge_mask = (grad_mag > 32).astype(np.uint8) * 255

    # Combinar oscuridad local + borde estructural
    combined = cv2.bitwise_and(thresh, edge_mask)

    # Cerrar micro-cortes en la grieta y limpiar puntos sueltos
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    # 4. Filtrado de Componentes Conectados: descarta manchas o poros pequeños del concreto (anti-ruido)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    final_crack_mask = np.zeros_like(closed)
    
    min_area = max(15, int(noise_sensitivity))
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(w, h) / (min(w, h) + 1e-5)
        # Conservamos si tiene área suficiente o si es una fisura alargada clara
        if area >= min_area or (area >= 22 and aspect >= 2.2):
            final_crack_mask[labels == i] = 255

    # Detectar cavidad profunda (bache / núcleo muy oscuro)
    mean_lum = float(np.mean(blurred))
    dark_threshold = max(45.0, mean_lum * 0.58)
    cavity_raw = ((blurred < dark_threshold) & (final_crack_mask > 0)).astype(np.uint8) * 255

    # 5. Extracción de Características Numéricas Normalizadas (X1, X2, X3 en rango [0, 1])
    total_pixels = float(gray.size)
    crack_ratio = float(np.sum(final_crack_mask > 0)) / total_pixels
    cavity_ratio = float(np.sum(cavity_raw > 0)) / total_pixels
    roughness_raw = float(np.std(blurred))

    x1_fisuras = float(np.clip(crack_ratio / 0.085, 0.0, 1.0))
    x2_cavidad = float(np.clip(cavity_ratio / 0.035, 0.0, 1.0))
    x3_rugosidad = float(np.clip(roughness_raw / 62.0, 0.0, 1.0))

    X_vec = np.array([x1_fisuras, x2_cavidad, x3_rugosidad])

    # 6. Regresión Lineal Múltiple: y_hat = b0 + b1*x1 + b2*x2 + b3*x3
    y_hat = float(np.clip(BETA_0 + np.dot(BETA_WEIGHTS, X_vec), 0.0, 100.0))

    # 7. Clasificación Logística (Sigmoide para Probabilidad de Intervención Prioritaria)
    z = (y_hat - 50.0) / 9.5
    prob_urgente = float(1.0 / (1.0 + np.exp(-z)))

    # 8. Clustering K-Means (Distancia a 3 centroides viales pre-calculados)
    centroids = np.array([
        [0.06, 0.02, 0.18],  # Clúster 0: Vía Óptima / Leve
        [0.42, 0.18, 0.45],  # Clúster 1: Fisuración Moderada
        [0.82, 0.65, 0.72],  # Clúster 2: Daño Estructural Severo
    ])
    dists = np.linalg.norm(centroids - X_vec, axis=1)
    cluster_id = int(np.argmin(dists))
    cluster_names = ["Desgaste Superficial Leve", "Agrietamiento Moderado", "Falla Estructural Crítica"]

    # 9. Crear imagen procesada limpia sin ruido de fondo
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # Pintar fisuras detectadas en Azul (#2563EB -> BGR: 235, 99, 37)
    vis[final_crack_mask > 0] = [235, 99, 37]
    # Pintar cavidades profundas en Rojo (#EF4444 -> BGR: 68, 68, 239)
    vis[cavity_raw > 0] = [68, 68, 239]

    _, buffer_orig = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    _, buffer_proc = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    orig_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer_orig).decode("utf-8")
    proc_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer_proc).decode("utf-8")

    return {
        "severity_score": round(y_hat, 1),
        "urgent_probability": round(prob_urgente * 100, 1),
        "cluster_id": cluster_id,
        "cluster_label": cluster_names[cluster_id],
        "features": {
            "crack_density": round(x1_fisuras * 100, 1),
            "cavity_area": round(x2_cavidad * 100, 1),
            "roughness": round(x3_rugosidad * 100, 1),
        },
        "equation": {
            "b0": BETA_0,
            "b1": float(BETA_WEIGHTS[0]),
            "b2": float(BETA_WEIGHTS[1]),
            "b3": float(BETA_WEIGHTS[2]),
            "x1": round(x1_fisuras, 3),
            "x2": round(x2_cavidad, 3),
            "x3": round(x3_rugosidad, 3),
        },
        "original_image": orig_b64,
        "processed_image": proc_b64,
    }


def generate_synthetic_sample(preset_type: str) -> np.ndarray:
    rng = np.random.default_rng(42 if preset_type == "sano" else (101 if preset_type == "lineal" else (202 if preset_type == "red" else 309)))
    base = rng.normal(148, 8, (420, 420)).astype(np.uint8)
    img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    if preset_type == "lineal":
        pts = np.array([[210, 10], [204, 110], [218, 210], [202, 310], [212, 410]], np.int32)
        cv2.polylines(img, [pts], False, (35, 35, 35), thickness=4, lineType=cv2.LINE_AA)
    elif preset_type == "red":
        pts1 = np.array([[80, 40], [160, 180], [220, 290], [340, 390]], np.int32)
        pts2 = np.array([[340, 50], [230, 170], [150, 280], [70, 380]], np.int32)
        pts3 = np.array([[40, 210], [200, 220], [380, 200]], np.int32)
        cv2.polylines(img, [pts1, pts2, pts3], False, (32, 32, 32), thickness=4, lineType=cv2.LINE_AA)
    elif preset_type == "bache":
        cv2.circle(img, (210, 210), 68, (22, 22, 22), -1)
        cv2.circle(img, (235, 195), 52, (15, 15, 15), -1)
        pts = np.array([[90, 210], [210, 210], [350, 250]], np.int32)
        cv2.polylines(img, [pts], False, (30, 30, 30), thickness=6, lineType=cv2.LINE_AA)

    return img


@app.post("/api/analyze")
async def analyze_pavement(
    file: UploadFile = File(...),
    sensitivity: int = Form(65)
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return process_image_array(img_bgr, noise_sensitivity=sensitivity)


@app.get("/api/preset/{preset_id}")
async def analyze_preset(preset_id: str, sensitivity: int = 65):
    img_bgr = generate_synthetic_sample(preset_id)
    return process_image_array(img_bgr, noise_sensitivity=sensitivity)


@app.get("/api/health")
async def health():
    return {"status": "ok", "engine": "OpenCV + NumPy Linear Regression"}


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join("static", "index.html"))
