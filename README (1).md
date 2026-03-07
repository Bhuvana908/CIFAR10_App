# CIFAR-10 Image Classifier

A web application that classifies images into one of 10 categories using a custom Convolutional Neural Network (CNN) trained on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. Built with PyTorch and Flask, with a drag-and-drop frontend interface.

---

## Features

- **Image upload** via drag-and-drop or file browser
- **Real-time classification** using a pretrained CNN
- **Top-3 predictions** displayed with confidence bars
- **Emoji-enhanced results** for a friendly UI
- Supports PNG, JPG/JPEG, WEBP, and GIF formats

---

## Supported Classes

| # | Class    |
|---|----------|
| 0 | Airplane |
| 1 | Car      |
| 2 | Bird     |
| 3 | Cat      |
| 4 | Deer     |
| 5 | Dog      |
| 6 | Frog     |
| 7 | Horse    |
| 8 | Ship     |
| 9 | Truck    |

---

## Model Architecture

The `CIFAR10_CNN` model is a custom CNN with three convolutional blocks followed by a fully connected classifier:

- **Conv Block 1:** `Conv2d(3→32)` × 2, BatchNorm, ReLU, MaxPool, Dropout(0.25)
- **Conv Block 2:** `Conv2d(32→64)` × 2, BatchNorm, ReLU, MaxPool, Dropout(0.25)
- **Conv Block 3:** `Conv2d(64→128)`, BatchNorm, ReLU, MaxPool, Dropout(0.25)
- **Classifier:** `Linear(128×4×4 → 512)`, ReLU, Dropout(0.5), `Linear(512 → 10)`

Input images are resized to **32×32** and normalized using CIFAR-10 dataset statistics:
- Mean: `(0.4914, 0.4822, 0.4465)`
- Std: `(0.2470, 0.2435, 0.2616)`

---

## Project Structure

```
cifar10_app/
├── app.py                 # Flask application & model inference
├── cifar10_model.pth      # Pretrained model weights
├── requirements.txt       # Python dependencies
├── Procfile               # Deployment config (e.g. Heroku)
├── static/
│   ├── script.js          # Frontend interaction logic
│   └── style.css          # Stylesheet
└── templates/
    └── index.html         # Main HTML page
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone <your-repo-url>
cd cifar10_app
pip install -r requirements.txt
```

> **Note:** The `requirements.txt` uses the CPU-only PyTorch build (`torch==2.10.0+cpu`) via `--extra-index-url https://download.pytorch.org/whl/cpu`. This keeps the install size smaller and is sufficient for inference.

### Running Locally

```bash
python app.py
```

The app will start on `http://0.0.0.0:5000` by default. You can override the port with the `PORT` environment variable:

```bash
PORT=8080 python app.py
```

---

## API

### `GET /`

Serves the main web UI.

### `POST /predict`

Accepts a multipart form upload and returns a JSON prediction.

**Request:**
```
Content-Type: multipart/form-data
Body: file=<image file>
```

**Response (200 OK):**
```json
{
  "prediction": "Dog",
  "confidence": 0.8731,
  "top3": [
    { "class_name": "Dog",  "confidence": 0.8731 },
    { "class_name": "Cat",  "confidence": 0.0812 },
    { "class_name": "Deer", "confidence": 0.0241 }
  ]
}
```

**Error responses:**
- `400` — No file provided, empty file, or invalid image format
- `500` — Model weights missing or internal server error

---

## Deployment

A `Procfile` is included for deployment on platforms like **Heroku**:

```
web: gunicorn app:app
```

Make sure `gunicorn` is installed (it is included in `requirements.txt`).

---

## Dependencies

| Package        | Purpose                          |
|----------------|----------------------------------|
| Flask          | Web framework                    |
| flask-cors     | Cross-origin request support     |
| torch (CPU)    | Deep learning inference          |
| torchvision    | Image transforms                 |
| Pillow         | Image loading and preprocessing  |
| numpy          | Numerical utilities              |
| gunicorn       | Production WSGI server           |

---

## Notes

- All images are automatically resized to **32×32 pixels** before inference — the model was trained at this resolution, so very high-resolution images may lose fine detail.
- The model runs on **CPU only** by default.
- Model weights (`cifar10_model.pth`) must be present in the same directory as `app.py`.
