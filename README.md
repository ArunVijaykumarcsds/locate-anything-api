# LocateAnything-3B API

A production-ready FastAPI server for **[nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B)** — NVIDIA's vision-language model for precise object localization, text detection, GUI grounding, and pointing from natural language.

> **License notice:** The model is released under the NVIDIA non-commercial research license. This API wrapper is MIT-licensed. Do not use in commercial products.

---

## What it does

Send an image + a text prompt → get back bounding boxes or point coordinates.

| Endpoint | Task |
|---|---|
| `POST /detect` | Locate objects by category (e.g. `person,car,bicycle`) |
| `POST /ground/single` | Locate one instance matching a phrase |
| `POST /ground/multi` | Locate all instances matching a phrase |
| `POST /text/detect` | Find all text in the image |
| `POST /text/ground` | Find a specific text string |
| `POST /gui/ground` | Locate a UI element (box or click-point) |
| `POST /point` | Point to a described object |
| `POST /predict/raw` | Send any custom prompt |
| `GET  /health` | Liveness + readiness check |

All responses return pixel coordinates **and** normalised `[0, 1]` coordinates.

---

## Requirements

| Requirement | Minimum |
|---|---|
| Python | 3.10+ |
| CUDA GPU | 8 GB VRAM (BF16) |
| Disk | 10 GB free (model cache) |
| RAM | 16 GB |

Tested on: NVIDIA A100, H100, L40, RTX 4090.

---

## Quick start

```bash
# Clone or unzip the project
cd locate-anything-api

# Install PyTorch for your CUDA version first (example: CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt

# Start the server (model downloads automatically on first run, ~8 GB)
python run.py
```

Server starts at **http://localhost:8000**
Interactive docs at **http://localhost:8000/docs**

---

## Example request

```bash
curl -X POST http://localhost:8000/detect \
  -F "image=@photo.jpg" \
  -F "categories=person,car,bicycle" \
  -F "generation_mode=hybrid"
```

```json
{
  "answer": "<raw model output>",
  "boxes": [
    {
      "x1": 120.5, "y1": 45.0, "x2": 340.2, "y2": 210.8,
      "x1_norm": 0.15, "y1_norm": 0.09, "x2_norm": 0.43, "y2_norm": 0.42
    }
  ],
  "points": [],
  "image_size": { "width": 800, "height": 600 }
}
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `nvidia/LocateAnything-3B` | HuggingFace repo ID or local path |
| `DEVICE` | `cuda` | `cuda` or `cpu` (cpu is very slow) |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Port |
| `LOG_LEVEL` | `info` | Uvicorn log level |

Copy `.env.example` to `.env` to override any of these.

---

## Project structure

```
locate-anything-api/
├── app/
│   ├── __init__.py
│   ├── config.py       # Settings loaded from env / .env
│   ├── main.py         # FastAPI app, routes, lifespan
│   ├── schemas.py      # Pydantic request/response models
│   └── worker.py       # Model loading and inference
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── README.md
├── installation.md
├── deployment.md
├── requirements.txt
├── run.py              # Uvicorn entry point
└── test_client.py      # Local smoke-test script
```

---

## Further reading

- [installation.md](installation.md) — step-by-step local setup
- [deployment.md](deployment.md) — Docker, GitHub, and free cloud deployment
- [nvidia/LocateAnything-3B model card](https://huggingface.co/nvidia/LocateAnything-3B)
