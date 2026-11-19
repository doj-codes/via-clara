from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import pandas as pd
import cv2
import base64
import os

app = FastAPI(title="VíaClara v2 — Plataforma Integral de Entrenamiento, Evaluación y Predicción Vial")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset_vial.csv")

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

class ModelState:
    def __init__(self):
        self.theta_ols = np.array([4.038, 43.978, 29.870, 10.109, 11.540])
        self.mu_X = np.array([0.4021, 0.2719, 0.4503, 0.4316])
        self.std_X = np.array([0.2921, 0.2663, 0.2292, 0.2447])
        self.W_ovr = np.zeros((3, 5))
        self.last_metrics = None
        self.train_models(test_ratio=0.20, lr=0.25, epochs=1200, seed=42)

    def train_models(self, test_ratio: float = 0.20, lr: float = 0.25, epochs: int = 1200, seed: int = 42):
        df = pd.read_csv(DATASET_PATH)
        n_samples = len(df)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n_samples)
        train_size = int((1.0 - test_ratio) * n_samples)
        train_idx, test_idx = idx[:train_size], idx[train_size:]

        X_cols = ["x1_crack_density", "x2_cavity_area", "x3_roughness", "x4_contrast_drop"]
        X_train = df.iloc[train_idx][X_cols].values
        y_reg_train = df.iloc[train_idx]["y_severity_index"].values
        y_clf_train = df.iloc[train_idx]["y_damage_level"].values

        X_test = df.iloc[test_idx][X_cols].values
        y_reg_test = df.iloc[test_idx]["y_severity_index"].values
        y_clf_test = df.iloc[test_idx]["y_damage_level"].values

        # 1. Entrenamiento Regresión Lineal Múltiple (OLS)
        X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
        X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
        self.theta_ols = np.linalg.pinv(X_train_b.T @ X_train_b) @ X_train_b.T @ y_reg_train

        y_reg_pred_test = X_test_b @ self.theta_ols
        ss_res = float(np.sum((y_reg_test - y_reg_pred_test) ** 2))
        ss_tot = float(np.sum((y_reg_test - np.mean(y_reg_test)) ** 2))
        r2_test = 1.0 - (ss_res / (ss_tot + 1e-8))
        rmse_test = float(np.sqrt(np.mean((y_reg_test - y_reg_pred_test) ** 2)))
        mae_test = float(np.mean(np.abs(y_reg_test - y_reg_pred_test)))

        # 2. Estandarización Z-score sin fuga de información (solo con X_train)
        self.mu_X = X_train.mean(axis=0)
        self.std_X = X_train.std(axis=0) + 1e-8
        X_train_std = np.c_[np.ones((X_train.shape[0], 1)), (X_train - self.mu_X) / self.std_X]
        X_test_std = np.c_[np.ones((X_test.shape[0], 1)), (X_test - self.mu_X) / self.std_X]

        # 3. Entrenamiento Clasificación Multiclase One-vs-Rest (OvR)
        self.W_ovr = np.zeros((3, X_train_std.shape[1]))
        for c in range(3):
            y_bin = (y_clf_train == c).astype(float)
            w = np.zeros(X_train_std.shape[1])
            for _ in range(epochs):
                probs = sigmoid(X_train_std @ w)
                grad = (X_train_std.T @ (probs - y_bin)) / len(y_bin)
                w -= lr * grad
            self.W_ovr[c] = w

        probs_test = sigmoid(X_test_std @ self.W_ovr.T)
        y_clf_pred_test = np.argmax(probs_test, axis=1)

        cm = np.zeros((3, 3), dtype=int)
        for t, p in zip(y_clf_test, y_clf_pred_test):
            cm[int(t), int(p)] += 1

        acc_test = float(np.trace(cm) / np.sum(cm))
        class_names = ["Clase 0: Leve / Óptimo", "Clase 1: Moderado", "Clase 2: Crítico / Severo"]
        per_class = []
        for c in range(3):
            tp = int(cm[c, c])
            fp = int(np.sum(cm[:, c]) - tp)
            fn = int(np.sum(cm[c, :]) - tp)
            tn = int(np.sum(cm) - (tp + fp + fn))
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            spec = tn / (tn + fp + 1e-8)
            per_class.append({
                "class_id": c,
                "class_name": class_names[c],
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": round(float(prec * 100), 2),
                "recall": round(float(rec * 100), 2),
                "f1": round(float(f1 * 100), 2),
                "specificity": round(float(spec * 100), 2),
            })

        self.last_metrics = {
            "n_total": n_samples,
            "n_train": int(train_size),
            "n_test": int(n_samples - train_size),
            "hyperparams": {"test_ratio": test_ratio, "lr": lr, "epochs": epochs, "seed": seed},
            "regression": {
                "r2": round(r2_test, 4),
                "rmse": round(rmse_test, 4),
                "mae": round(mae_test, 4),
                "betas": [round(float(b), 3) for b in self.theta_ols],
            },
            "classification": {
                "accuracy": round(acc_test * 100, 2),
                "confusion_matrix": cm.tolist(),
                "per_class": per_class,
            }
        }
        return self.last_metrics

    def predict_from_features(self, x1: float, x2: float, x3: float, x4: float):
        x_vec = np.array([x1, x2, x3, x4], dtype=float)
        y_hat = float(np.clip(self.theta_ols[0] + np.dot(self.theta_ols[1:], x_vec), 0.0, 100.0))
        x_std = np.r_[1.0, (x_vec - self.mu_X) / self.std_X]
        probs_raw = sigmoid(x_std @ self.W_ovr.T)
        probs_norm = probs_raw / (np.sum(probs_raw) + 1e-8)
        pred_class = int(np.argmax(probs_norm))
        class_labels = ["Clase 0: Condición Óptima / Leve", "Clase 1: Deterioro Moderado", "Clase 2: Falla Estructural Crítica"]
        return {
            "severity_score": round(y_hat, 2),
            "predicted_class_id": pred_class,
            "predicted_class_label": class_labels[pred_class],
            "class_probabilities": [round(float(p * 100), 1) for p in probs_norm],
            "urgent_probability": round(float(100.0 / (1.0 + np.exp(-(y_hat - 50.0) / 9.5))), 1),
            "features": {
                "x1_crack_density": round(x1, 4),
                "x2_cavity_area": round(x2, 4),
                "x3_roughness": round(x3, 4),
                "x4_contrast_drop": round(x4, 4),
            },
            "betas": [round(float(b), 3) for b in self.theta_ols]
        }

STATE = ModelState()

class TrainRequest(BaseModel):
    test_ratio: float = 0.20
    lr: float = 0.25
    epochs: int = 1200
    seed: int = 42

class ManualPredictRequest(BaseModel):
    x1_crack_density: float
    x2_cavity_area: float
    x3_roughness: float
    x4_contrast_drop: float

def process_image_array(img_bgr: np.ndarray, noise_sensitivity: int = 65):
    img_bgr = cv2.resize(img_bgr, (420, 420))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    smoothed = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    blurred = cv2.GaussianBlur(smoothed, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=25, C=10
    )

    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    edge_mask = (grad_mag > 32).astype(np.uint8) * 255

    combined = cv2.bitwise_and(thresh, edge_mask)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    final_crack_mask = np.zeros_like(closed)
    min_area = max(15, int(noise_sensitivity))
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(w, h) / (min(w, h) + 1e-5)
        if area >= min_area or (area >= 22 and aspect >= 2.2):
            final_crack_mask[labels == i] = 255

    mean_lum = float(np.mean(blurred))
    dark_threshold = max(45.0, mean_lum * 0.58)
    cavity_raw = ((blurred < dark_threshold) & (final_crack_mask > 0)).astype(np.uint8) * 255

    total_pixels = float(gray.size)
    crack_ratio = float(np.sum(final_crack_mask > 0)) / total_pixels
    cavity_ratio = float(np.sum(cavity_raw > 0)) / total_pixels
    roughness_raw = float(np.std(blurred))

    if np.any(final_crack_mask > 0):
        crack_lum = float(np.mean(blurred[final_crack_mask > 0]))
        contrast_drop = max(0.0, (mean_lum - crack_lum) / (mean_lum + 1e-5))
    else:
        contrast_drop = 0.05

    x1 = float(np.clip(crack_ratio / 0.085, 0.0, 1.0))
    x2 = float(np.clip(cavity_ratio / 0.035, 0.0, 1.0))
    x3 = float(np.clip(roughness_raw / 62.0, 0.0, 1.0))
    x4 = float(np.clip(contrast_drop / 0.45, 0.0, 1.0))

    pred = STATE.predict_from_features(x1, x2, x3, x4)

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    vis[final_crack_mask > 0] = [235, 99, 37]
    vis[cavity_raw > 0] = [68, 68, 239]

    _, buffer_orig = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    _, buffer_proc = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    pred["original_image"] = "data:image/jpeg;base64," + base64.b64encode(buffer_orig).decode("utf-8")
    pred["processed_image"] = "data:image/jpeg;base64," + base64.b64encode(buffer_proc).decode("utf-8")
    return pred

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

@app.post("/api/train")
async def train_endpoint(req: TrainRequest):
    return STATE.train_models(test_ratio=req.test_ratio, lr=req.lr, epochs=req.epochs, seed=req.seed)

@app.get("/api/metrics")
async def get_metrics():
    return STATE.last_metrics

@app.post("/api/predict-manual")
async def predict_manual_endpoint(req: ManualPredictRequest):
    return STATE.predict_from_features(req.x1_crack_density, req.x2_cavity_area, req.x3_roughness, req.x4_contrast_drop)

@app.post("/api/analyze")
async def analyze_pavement(file: UploadFile = File(...), sensitivity: int = Form(65)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return process_image_array(img_bgr, noise_sensitivity=sensitivity)

@app.get("/api/preset/{preset_id}")
async def analyze_preset(preset_id: str, sensitivity: int = 65):
    img_bgr = generate_synthetic_sample(preset_id)
    return process_image_array(img_bgr, noise_sensitivity=sensitivity)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join("static", "index.html"))
