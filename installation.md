# Installation Guide

Complete step-by-step instructions for setting up LocateAnything-3B API locally.

---

## Prerequisites

Before starting, confirm you have:

- **Python 3.10 or newer**
  ```bash
  python3 --version
  ```

- **pip** (comes with Python)
  ```bash
  pip --version
  ```

- **An NVIDIA GPU with at least 8 GB VRAM**
  ```bash
  nvidia-smi
  ```

- **CUDA 11.8 or newer** (check the `CUDA Version` field in `nvidia-smi` output)

- **10 GB of free disk space** (model weights download to `~/.cache/huggingface/`)

- **Git** (optional, only needed if you clone instead of unzip)

---

## Step 1 — Get the project

**Option A — Unzip the downloaded archive:**
```bash
unzip locate-anything-api.zip
cd locate-anything-api
```

**Option B — Clone from GitHub (if you have pushed it):**
```bash
git clone https://github.com/YOUR_USERNAME/locate-anything-api.git
cd locate-anything-api
```

---

## Step 2 — Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

---

## Step 3 — Install PyTorch

Install PyTorch **before** `requirements.txt`. Use the version matching your CUDA:

**CUDA 12.1 (most common on modern GPUs):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CPU only (very slow — for testing syntax only, not real inference):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

To find your CUDA version:
```bash
nvidia-smi | grep "CUDA Version"
```

---

## Step 4 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

Expected output ends with something like:
```
Successfully installed fastapi-... uvicorn-... transformers-4.57.1 ...
```

---

## Step 5 — Configure (optional)

The server works with all defaults. To override any setting:

```bash
cp .env.example .env
```

Then open `.env` and edit as needed. The only value you might want to change:

```
# Use a local model directory instead of downloading from HuggingFace
MODEL_PATH=/path/to/local/nvidia-LocateAnything-3B

# Use CPU (testing only — inference takes minutes per image)
DEVICE=cpu
```

---

## Step 6 — Start the server

```bash
python run.py
```

On first run, the model (~8 GB) downloads automatically to `~/.cache/huggingface/`.
This takes 5–15 minutes depending on your connection.

You will see:
```
[worker] Loading tokenizer from 'nvidia/LocateAnything-3B' ...
[worker] Loading processor ...
[worker] Loading model weights (BF16) ...
[worker] ✅ Model ready.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

## Step 7 — Verify it works

**Browser:** open http://localhost:8000/docs — you should see the interactive API UI.

**Terminal health check:**
```bash
curl http://localhost:8000/health
```
Expected:
```json
{"status": "ok", "model_loaded": true}
```

**Full smoke test** (requires an image file):
```bash
python test_client.py path/to/any/photo.jpg
```

---

## Stopping the server

Press `Ctrl+C` in the terminal where `python run.py` is running.

---

## Troubleshooting

### `CUDA out of memory`
The model needs ~8 GB VRAM. Close other GPU processes and retry:
```bash
# Check what is using the GPU
nvidia-smi

# Kill a specific PID if needed
kill -9 <PID>
```

### `ModuleNotFoundError: No module named 'torch'`
PyTorch was not installed. Repeat Step 3.

### `trust_remote_code` warning
This is expected. The model ships custom code that must be loaded with `trust_remote_code=True`. The flag is already set in `worker.py`.

### Model download is slow
Set a custom cache directory on a faster disk:
```bash
export HF_HOME=/fast-disk/huggingface-cache
python run.py
```

### Port 8000 already in use
Change the port in `.env`:
```
PORT=8001
```
Then restart with `python run.py`.

### `ImportError` on `decord`
`decord` has no Windows wheel for Python 3.12+. Use Linux or WSL2, or install from source.
