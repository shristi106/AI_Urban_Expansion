import sys
from pathlib import Path
import io
import json

import numpy as np
import torch

from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(BASE_DIR / "src"))

from model_hybrid import HybridSiameseCNNViT
from model_tue import HybridTUE


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")


# ============================================================
# LOAD OSCD FINAL MODEL
# ============================================================

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "hybrid_siamese_vit_augmented.pth"
)

model = HybridSiameseCNNViT(
    in_channels=13
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint)

model.to(device)
model.eval()

print("Final OSCD model loaded successfully!")


# ============================================================
# LOAD TUE-CD MODEL
# ============================================================

TUE_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "tue_hybrid_siamese_vit.pth"
)

tue_model = HybridTUE(
    in_channels=3
)

tue_checkpoint = torch.load(
    TUE_MODEL_PATH,
    map_location=device
)

tue_model.load_state_dict(
    tue_checkpoint
)

tue_model.to(device)
tue_model.eval()

print("TUE-CD model loaded successfully!")


# ============================================================
# HELPER: CREATE SENTINEL-2 RGB PREVIEW
# ============================================================

def create_rgb_image(image):
    # Sentinel-2:
    # B2 = index 1
    # B3 = index 2
    # B4 = index 3
    rgb = image[[3, 2, 1]].astype(np.float32)

    rgb_min = rgb.min()
    rgb_max = rgb.max()

    rgb = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-8)
    rgb = np.transpose(rgb, (1, 2, 0))
    rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

    return Image.fromarray(rgb)


def rgb_to_13bands(pil_img):
    """
    Convert a standard RGB PIL Image to a 13-band Sentinel-2 reflectance array
    (13, 128, 128) scaled 0-10000.
    """
    resized = pil_img.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
    arr = np.array(resized, dtype=np.float32)

    # Scale 0-255 to Sentinel-2 top-of-atmosphere reflectance (0-10000)
    scale = 10000.0 / 255.0
    r = arr[:, :, 0] * scale
    g = arr[:, :, 1] * scale
    b = arr[:, :, 2] * scale

    bands = np.zeros((13, 128, 128), dtype=np.float32)
    # Sentinel-2 band indices:
    # B1: Coastal aerosol
    bands[0] = b * 0.9
    # B2: Blue
    bands[1] = b
    # B3: Green
    bands[2] = g
    # B4: Red
    bands[3] = r
    # B5, B6, B7: Red-Edge
    nir = (g * 0.4 + r * 0.6) * 1.3
    bands[4] = nir * 0.7
    bands[5] = nir * 0.85
    bands[6] = nir * 0.95
    # B8, B8A: NIR
    bands[7] = nir
    bands[8] = nir
    # B9: Water vapor
    bands[9] = nir * 0.8
    # B10: Cirrus
    bands[10] = r * 0.5
    # B11, B12: SWIR
    bands[11] = r * 0.9
    bands[12] = r * 0.75

    return bands


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# STATUS / HEALTH API
# ============================================================

@app.route("/api/status")
@app.route("/health")
def api_status():
    return jsonify({
        "status": "online",
        "model_loaded": True,
        "device": str(device),
        "primary_model": "Enhanced Siamese CNN + Vision Transformer",
        "dataset": "OSCD Sentinel-2 Multispectral",
        "parameters": 767777,
        "cross_dataset_model": "TUE-CD Siamese CNN + ViT (WorldView-2)",
        "tue_loaded": True
    })


# ============================================================
# CURATED SAMPLE PATCHES API
# ============================================================

@app.route("/api/samples")
def api_samples():
    samples_file = BASE_DIR / "web" / "static" / "samples" / "samples.json"
    if samples_file.exists():
        with open(samples_file, "r") as f:
            data = json.load(f)
        return jsonify({"success": True, "samples": data})
    return jsonify({"success": False, "samples": []})


# ============================================================
# BENCHMARKS API
# ============================================================

@app.route("/api/benchmarks")
def api_benchmarks():
    return jsonify({
        "dataset": "OSCD (Sentinel-2 13-Band Multispectral)",
        "metrics": [
            {
                "model": "Enhanced Siamese CNN + ViT",
                "f1": 47.44,
                "iou": 31.10,
                "rank": 1,
                "highlight": True,
                "type": "Our Best Approach"
            },
            {
                "model": "Siamese CNN + Vision Transformer",
                "f1": 45.93,
                "iou": 29.81,
                "rank": 2,
                "highlight": False,
                "type": "Hybrid Architecture"
            },
            {
                "model": "Swin Transformer + U-Net",
                "f1": 39.18,
                "iou": 24.36,
                "rank": 3,
                "highlight": False,
                "type": "Hierarchical ViT"
            },
            {
                "model": "U-Net + Transformer",
                "f1": 34.23,
                "iou": 20.65,
                "rank": 4,
                "highlight": False,
                "type": "Encoder-Decoder"
            },
            {
                "model": "CNN + Change Detection Network",
                "f1": 32.00,
                "iou": 19.04,
                "rank": 5,
                "highlight": False,
                "type": "CNN Baseline"
            },
            {
                "model": "ResNet + LSTM",
                "f1": 2.48,
                "iou": 1.25,
                "rank": 6,
                "highlight": False,
                "type": "Recurrent Temporal"
            }
        ],
        "cross_validation_10fold": {
            "status": "completed",
            "mean_f1": 24.88,
            "std_f1": 10.96,
            "mean_iou": 15.90,
            "std_iou": 8.10,
            "folds": [
                {"fold": 1, "f1": 30.40, "iou": 20.00},
                {"fold": 2, "f1": 30.52, "iou": 19.56},
                {"fold": 3, "f1": 21.70, "iou": 12.90},
                {"fold": 4, "f1": 27.15, "iou": 16.77},
                {"fold": 5, "f1": 20.10, "iou": 12.51},
                {"fold": 6, "f1": 45.19, "iou": 31.49},
                {"fold": 7, "f1": 10.17, "iou": 5.60},
                {"fold": 8, "f1": 35.18, "iou": 23.84},
                {"fold": 9, "f1": 17.20, "iou": 10.11},
                {"fold": 10, "f1": 11.23, "iou": 6.23}
            ]
        },
        "test_set_evaluation": {
            "model": "Enhanced Siamese CNN + ViT",
            "enhancement": "Noise-Robust Spatial + Spectral Augmentation",
            "accuracy": 95.44,
            "precision": 58.74,
            "recall": 39.79,
            "f1": 47.44,
            "iou": 31.10,
            "evaluation_type": "Untouched Test Evaluation"
        }
    })


# ============================================================
# LITERATURE & PAPERS API (5 Included 2026 Papers)
# ============================================================

@app.route("/api/papers")
def api_papers():
    return jsonify({
        "papers": [
            {
                "id": 1,
                "file": "01.pdf",
                "title": "TranSiamUNet based transformer-augmented Siamese-U-Net for precise change detection in satellite imagery",
                "authors": "Farid Ali, Soha Safwat Labib, Ayat Mahmoud, Ibrahim Eldesouky Fattoh",
                "year": 2026,
                "journal": "Scientific Reports (Nature Portfolio)",
                "doi": "10.1038/s41598-026-43164-w",
                "key_idea": "Integrates a Vision Transformer module into a Siamese U-Net backbone to capture long-range contextual dependencies between multi-temporal satellite scenes.",
                "relevance": "Direct theoretical foundation for our primary architecture combining shared Siamese convolutional encoders with multi-head self-attention and U-Net skip connections."
            },
            {
                "id": 2,
                "file": "02.pdf",
                "title": "Automated Urban Expansion and Land-Use Change Detection Using Deep Learning Ensembles on Sentinel-2 Imagery",
                "authors": "Emadoddin Hemmati, Niloofar Alizadeh, Fatemeh Mahmoudzadeh, Shahin Jafari, Hamed Amini Amirkolaee",
                "year": 2026,
                "journal": "ISPRS Annals of Photogrammetry, Remote Sensing and Spatial Information Sciences",
                "doi": "10.5194/isprs-annals-X-4-W8-2025-349-2026",
                "key_idea": "Decomposes complex urban transition dynamics into specialized binary semantic tasks evaluated on Sentinel-2 multi-spectral bands.",
                "relevance": "Validates our problem framing of Sentinel-2 multi-temporal imagery for urban expansion tracking and pixel-level binary change segmentation."
            },
            {
                "id": 3,
                "file": "03.pdf",
                "title": "S2C: A Noise-Resistant Difference Learning Framework for Unsupervised Change Detection in VHR Remote Sensing Images",
                "authors": "Lei Ding, Xibing Zuo, Haitao Guo, Jun Lu, Zhihui Gong, Xuanguang Liu, Jicang Lu",
                "year": 2026,
                "journal": "Information Engineering University Research",
                "doi": "In Press (2026)",
                "key_idea": "Presents a semantic-to-change framework with explicit noise-resistant contrastive difference learning for high-resolution satellite imagery.",
                "relevance": "Directly motivated our noise-robust spatial and spectral augmentation strategy (band scaling, Gaussian perturbation, and independent temporal jittering) that boosted test F1 to 47.44%."
            },
            {
                "id": 4,
                "file": "04.pdf",
                "title": "MCD-Mamba: mamba-based framework for infrastructure change detection using multimodal satellite data",
                "authors": "Meruyert Kenzhebay, Mehak Khan, Abdul Hanan, Reza Arghandeh",
                "year": 2026,
                "journal": "State-Space Model Geospatial Analytics",
                "doi": "In Press (2026)",
                "key_idea": "Explores selective state-space sequence modeling (Mamba) as a linear-complexity alternative to quadratic self-attention for remote sensing change detection.",
                "relevance": "Serves as an architectural benchmark comparison for global sequence modeling vs. our 256-token Vision Transformer attention mechanism."
            },
            {
                "id": 5,
                "file": "05.pdf",
                "title": "A GCN-Mamba-based model for remote sensing change detection using local-global feature aggregation and dual-perspective adaptive fusion",
                "authors": "Xueli Chang, Hongyang Dai, Guangqi Xie, Ting Bai, Liqiao Tian",
                "year": 2026,
                "journal": "Remote Sensing Machine Intelligence",
                "doi": "In Press (2026)",
                "key_idea": "Combines Graph Convolutional Networks (GCN) with state-space models to model topological geometric structures and long-range temporal relations simultaneously.",
                "relevance": "Provides comparative insights into spatial topological aggregation vs. convolutional inductive bias in our Siamese CNN encoder."
            }
        ]
    })


# ============================================================
# 10-FOLD CROSS-VALIDATION API
# ============================================================

@app.route("/api/cross-validation")
def api_cross_validation():
    return jsonify({
        "status": "completed",
        "title": "10-Fold Scene-Level Cross-Validation",
        "subtitle": "Evaluation across spatially separated OSCD scenes",
        "mean_f1": 24.88,
        "std_f1": 10.96,
        "mean_iou": 15.90,
        "std_iou": 8.10,
        "folds": [
            {"fold": 1, "f1": 30.40, "iou": 20.00},
            {"fold": 2, "f1": 30.52, "iou": 19.56},
            {"fold": 3, "f1": 21.70, "iou": 12.90},
            {"fold": 4, "f1": 27.15, "iou": 16.77},
            {"fold": 5, "f1": 20.10, "iou": 12.51},
            {"fold": 6, "f1": 45.19, "iou": 31.49},
            {"fold": 7, "f1": 10.17, "iou": 5.60},
            {"fold": 8, "f1": 35.18, "iou": 23.84},
            {"fold": 9, "f1": 17.20, "iou": 10.11},
            {"fold": 10, "f1": 11.23, "iou": 6.23}
        ]
    })


# ============================================================
# OSCD PREDICTION API
# Supports:
# 1) Direct NPZ file via request.files['file'] (Original contract)
# 2) Two separate image files via request.files['t1'] & request.files['t2']
# 3) Sample selection via form/json param 'sample_id'
# ============================================================

@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        t1 = None
        t2 = None
        t1_rgb = None
        t2_rgb = None

        # ----------------------------------------------------
        # OPTION 1: Curated sample patch selected by ID
        # ----------------------------------------------------
        sample_id = None
        if request.is_json:
            sample_id = request.json.get("sample_id")
        else:
            sample_id = request.form.get("sample_id")

        if sample_id:
            # Map sample_id to npz file
            sample_map = {
                "sample_1": "changed_0005.npz",
                "sample_2": "changed_0018.npz",
                "sample_3": "changed_0000.npz",
                "sample_4": "unchanged_0000.npz"
            }
            npz_name = sample_map.get(sample_id, f"{sample_id}.npz")
            sample_path = BASE_DIR / "web" / "static" / "samples" / npz_name
            if not sample_path.exists():
                sample_path = BASE_DIR / "dataset" / "val" / npz_name

            if sample_path.exists():
                npz = np.load(sample_path)
                t1 = npz["image1"]
                t2 = npz["image2"]
                t1_rgb = create_rgb_image(t1)
                t2_rgb = create_rgb_image(t2)
            else:
                return jsonify({
                    "success": False,
                    "error": f"Sample patch '{sample_id}' not found."
                }), 404

        # ----------------------------------------------------
        # OPTION 2: Single file uploaded (NPZ file)
        # ----------------------------------------------------
        elif "file" in request.files and request.files["file"].filename != "":
            uploaded_file = request.files["file"]
            filename = uploaded_file.filename.lower()

            if filename.endswith(".npz"):
                file_bytes = uploaded_file.read()
                npz = np.load(io.BytesIO(file_bytes))
                t1 = npz["image1"]
                t2 = npz["image2"]

                if t1.shape != (13, 128, 128) or t2.shape != (13, 128, 128):
                    return jsonify({
                        "success": False,
                        "error": f"Invalid NPZ shapes. Expected (13, 128, 128), got T1: {t1.shape}, T2: {t2.shape}"
                    }), 400

                t1_rgb = create_rgb_image(t1)
                t2_rgb = create_rgb_image(t2)
            else:
                # User uploaded a single image file instead of NPZ without providing both T1 & T2
                return jsonify({
                    "success": False,
                    "error": "Single file upload must be a 13-band .npz patch. For standard images, upload both Time T1 and Time T2."
                }), 400

        # ----------------------------------------------------
        # OPTION 3: Two separate images (T1 and T2)
        # ----------------------------------------------------
        elif ("t1" in request.files or "image1" in request.files) and ("t2" in request.files or "image2" in request.files):
            file_t1 = request.files.get("t1") or request.files.get("image1")
            file_t2 = request.files.get("t2") or request.files.get("image2")

            if not file_t1 or not file_t2 or file_t1.filename == "" or file_t2.filename == "":
                return jsonify({
                    "success": False,
                    "error": "Both Time T1 and Time T2 image files are required."
                }), 400

            img1 = Image.open(io.BytesIO(file_t1.read()))
            img2 = Image.open(io.BytesIO(file_t2.read()))

            t1 = rgb_to_13bands(img1)
            t2 = rgb_to_13bands(img2)

            t1_rgb = img1.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
            t2_rgb = img2.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)

        else:
            return jsonify({
                "success": False,
                "error": "Please provide an NPZ satellite patch, select a sample scene, or upload both T1 and T2 satellite images."
            }), 400

        # ----------------------------------------------------
        # NORMALIZATION (0-10000 -> 0-1)
        # ----------------------------------------------------
        t1_normalized = t1.astype(np.float32) / 10000.0
        t2_normalized = t2.astype(np.float32) / 10000.0

        # ----------------------------------------------------
        # CONVERT TO TORCH TENSOR
        # ----------------------------------------------------
        t1_tensor = torch.from_numpy(t1_normalized).unsqueeze(0).to(device)
        t2_tensor = torch.from_numpy(t2_normalized).unsqueeze(0).to(device)

        # ----------------------------------------------------
        # MODEL PREDICTION (REAL PYTORCH INFERENCE)
        # ----------------------------------------------------
        with torch.no_grad():
            output = model(t1_tensor, t2_tensor)
            probability = torch.sigmoid(output)
            prediction = (probability > 0.5).float()

        # ----------------------------------------------------
        # CHANGE STATISTICS
        # ----------------------------------------------------
        change_pixels = int(prediction.sum().item())
        total_pixels = prediction.numel()
        change_percentage = (change_pixels / total_pixels) * 100.0

        # Average model confidence for detected changed pixels
        mask_np = prediction[0, 0].cpu().numpy().astype(np.uint8)
        prob_np = probability[0, 0].cpu().numpy()

        if change_pixels > 0:
            avg_confidence = float(prob_np[mask_np == 1].mean())
        else:
            avg_confidence = float((1.0 - prob_np).mean())

        # ----------------------------------------------------
        # CREATE CHANGE MAP (Red: 255, 80, 80)
        # ----------------------------------------------------
        change_map = np.zeros((128, 128, 3), dtype=np.uint8)
        change_map[mask_np == 1] = [255, 80, 80]
        change_image = Image.fromarray(change_map)

        # ----------------------------------------------------
        # CREATE OVERLAY IMAGE (T2 base + Alpha blend changed pixels)
        # ----------------------------------------------------
        t2_base = np.array(t2_rgb).copy()
        # Highlight changed pixels with vivid geospatial red/coral
        overlay_arr = t2_base.copy()
        alpha = 0.65
        for c in range(3):
            overlay_arr[mask_np == 1, 0] = np.uint8(overlay_arr[mask_np == 1, 0] * (1 - alpha) + 255 * alpha)
            overlay_arr[mask_np == 1, 1] = np.uint8(overlay_arr[mask_np == 1, 1] * (1 - alpha) + 60 * alpha)
            overlay_arr[mask_np == 1, 2] = np.uint8(overlay_arr[mask_np == 1, 2] * (1 - alpha) + 60 * alpha)
        overlay_image = Image.fromarray(overlay_arr)

        # ----------------------------------------------------
        # SAVE IMAGES TO STATIC/PREDICTIONS
        # ----------------------------------------------------
        prediction_dir = BASE_DIR / "web" / "static" / "predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)

        t1_path = prediction_dir / "t1.png"
        t2_path = prediction_dir / "t2.png"
        change_path = prediction_dir / "change_map.png"
        overlay_path = prediction_dir / "overlay.png"

        t1_rgb.save(t1_path)
        t2_rgb.save(t2_path)
        change_image.save(change_path)
        overlay_image.save(overlay_path)

        # ----------------------------------------------------
        # RETURN RESULT
        # ----------------------------------------------------
        return jsonify({
            "success": True,
            "t1_image": "/static/predictions/t1.png",
            "t2_image": "/static/predictions/t2.png",
            "change_map": "/static/predictions/change_map.png",
            "overlay_image": "/static/predictions/overlay.png",
            "changed_pixels": change_pixels,
            "total_pixels": total_pixels,
            "change_percentage": round(change_percentage, 2),
            "confidence": round(avg_confidence * 100, 1),
            "input_resolution": "128 × 128 (10m GSD)",
            "model": "Enhanced Siamese CNN + ViT",
            "dataset": "OSCD Sentinel-2 Multispectral"
        })

    except Exception as e:
        print("OSCD Prediction error:", str(e))
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# TUE-CD RESULTS API
# ============================================================

@app.route("/tue-result")
def tue_result():
    result_image = (
        BASE_DIR
        / "results"
        / "tue_cd_best_test_result_red_overlay.png"
    )

    return jsonify({
        "success": True,
        "dataset": "TUE-CD",
        "note": "TUE-CD is an independent building-damage change-detection experiment and is not the primary urban-expansion dataset.",
        "model": "Siamese CNN + Vision Transformer",
        "sensor": "WorldView-2",
        "input_type": "RGB",
        "test_images": 167,
        "accuracy": 95.03,
        "precision": 50.84,
        "recall": 40.93,
        "f1": 43.50,
        "iou": 29.60,
        "best_sample": "train(617).png",
        "best_sample_f1": 89.96,
        "best_sample_iou": 81.76,
        "result_image": "/tue-result-image",
        "result_exists": result_image.exists()
    })


# ============================================================
# TUE-CD RESULT IMAGE
# ============================================================

@app.route("/tue-result-image")
def tue_result_image():
    result_image = (
        BASE_DIR
        / "results"
        / "tue_cd_best_test_result_red_overlay.png"
    )

    if not result_image.exists():
        return jsonify({
            "success": False,
            "error": "TUE-CD result image not found."
        }), 404

    return send_file(
        result_image,
        mimetype="image/png"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    import os
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )