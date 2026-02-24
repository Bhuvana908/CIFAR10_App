from __future__ import annotations

import io
import os
from typing import Dict, List, Tuple

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


app = Flask(__name__)
CORS(app)


class CIFAR10_CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cifar10_model.pth")

device = torch.device("cpu")
DEVICE = device
model = CIFAR10_CNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
MODEL = model
print("Model loaded from:", MODEL_PATH)

CLASS_NAMES = [
    "Airplane",
    "Car",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
]

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ]
)


def prepare_image(file_bytes: bytes) -> torch.Tensor:
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Invalid image file") from exc

    tensor = TRANSFORM(image).unsqueeze(0)
    return tensor.to(DEVICE)


def predict_topk(image_tensor: torch.Tensor, k: int = 3) -> Tuple[str, float, List[Dict[str, float | str]]]:
    with torch.no_grad():
        outputs = MODEL(image_tensor)
        probs = F.softmax(outputs, dim=1)
        topk_conf, topk_idx = torch.topk(probs, k=min(k, probs.shape[1]), dim=1)

    top1_idx = int(topk_idx[0][0].item())
    top1_label = CLASS_NAMES[top1_idx]
    top1_confidence = float(topk_conf[0][0].item())

    topk_list: List[Dict[str, float | str]] = []
    for conf, idx in zip(topk_conf[0].tolist(), topk_idx[0].tolist()):
        topk_list.append({"class_name": CLASS_NAMES[int(idx)], "confidence": float(conf)})

    return top1_label, top1_confidence, topk_list


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict_endpoint():
    if "file" not in request.files:
        return (
            jsonify({"error": "No file part in the request. Use form-data with key 'file'."}),
            400,
        )

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected for upload."}), 400

    try:
        file_bytes = file.read()
        if not file_bytes:
            return jsonify({"error": "Uploaded file is empty."}), 400

        img_tensor = prepare_image(file_bytes)

        if not os.path.exists(MODEL_PATH):
            return (
                jsonify(
                    {
                        "error": "Model weights not found on server.",
                        "detail": f"Expected weights file at '{MODEL_PATH}'.",
                    }
                ),
                500,
            )

        label, confidence, top3 = predict_topk(img_tensor, k=3)

        return jsonify(
            {
                "prediction": label,
                "confidence": confidence,
                "top3": top3,
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        return jsonify({"error": "Internal server error during prediction."}), 500


if __name__ == "__main__":
    app.run(debug=True)
