from pathlib import Path
import sys
import time
import uuid
import os

import cv2
import torch
import torch.nn as nn
from flask import Flask, render_template, request
from torchvision import models, transforms
from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from preprocess import ben_graham_preprocess


MODEL_PATH = ROOT_DIR / "models" / "best_efficientnet_b3.pth"

UPLOAD_DIR = ROOT_DIR / "web" / "static" / "uploads"
PROCESSED_DIR = ROOT_DIR / "web" / "static" / "processed"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]

CLASS_DESCRIPTIONS = {
    "No DR": "No visible diabetic retinopathy.",
    "Mild": "Early-stage diabetic retinopathy signs may be present.",
    "Moderate": "Moderate retinal abnormalities may be present.",
    "Severe": "Severe diabetic retinopathy signs may be present.",
    "Proliferative DR": "Advanced diabetic retinopathy signs may be present.",
}

NUM_CLASSES = 5
IMAGE_SIZE = 224

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model():
    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    return model


def load_model():
    model = build_model()
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


model = load_model()

inference_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def allowed_file(filename):
    allowed_extensions = {".jpg", ".jpeg", ".png"}
    return Path(filename).suffix.lower() in allowed_extensions


def predict_image(image_path, processed_save_path):
    start_time = time.time()

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError("Cannot read uploaded image.")

    processed_bgr = ben_graham_preprocess(image_bgr)
    cv2.imwrite(str(processed_save_path), processed_bgr)

    image_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
    tensor = inference_transform(image_rgb)
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    predicted_index = int(probabilities.argmax())
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])

    inference_time_ms = round((time.time() - start_time) * 1000, 2)

    results = []
    for i, class_name in enumerate(CLASS_NAMES):
        results.append({
            "class_name": class_name,
            "probability": float(probabilities[i]),
            "percent": round(float(probabilities[i]) * 100, 2),
        })

    return {
        "predicted_class": predicted_class,
        "description": CLASS_DESCRIPTIONS[predicted_class],
        "confidence": round(confidence * 100, 2),
        "inference_time_ms": inference_time_ms,
        "results": results,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    uploaded_image = None
    processed_image = None

    if request.method == "POST":
        if "image" not in request.files:
            error = "No image file was uploaded."
            return render_template("index.html", prediction=prediction, error=error)

        file = request.files["image"]

        if file.filename == "":
            error = "Please select an image file."
            return render_template("index.html", prediction=prediction, error=error)

        if not allowed_file(file.filename):
            error = "Only JPG, JPEG, and PNG files are supported."
            return render_template("index.html", prediction=prediction, error=error)

        file_id = uuid.uuid4().hex
        ext = Path(file.filename).suffix.lower()

        upload_filename = f"{file_id}{ext}"
        processed_filename = f"{file_id}_processed.png"

        upload_path = UPLOAD_DIR / upload_filename
        processed_path = PROCESSED_DIR / processed_filename

        file.save(upload_path)

        try:
            prediction = predict_image(upload_path, processed_path)
            uploaded_image = f"static/uploads/{upload_filename}"
            processed_image = f"static/processed/{processed_filename}"
        except Exception as exc:
            error = f"Prediction failed: {exc}"

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        uploaded_image=uploaded_image,
        processed_image=processed_image,
        device=str(device),
    )


if __name__ == "__main__":
    print(f"Using device: {device}")
    print(f"Loading model from: {MODEL_PATH}")
  port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port, debug=False)