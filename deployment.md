# Deployment Guide

Instructions for running the API with Docker, pushing to GitHub, and deploying
for free using cloud platforms that offer GPU access.

---

## 1. Run locally (recap)

```bash
# Activate your virtual environment first
source venv/bin/activate

# Start the server
python run.py
```

Server: http://localhost:8000
Docs:   http://localhost:8000/docs

---

## 2. Run with Docker

Docker lets you run the API in a reproducible, isolated container.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed (required to pass your GPU into the container)

Verify the toolkit works:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Build the image

```bash
cd locate-anything-api
docker build -t locate-anything-api:v1 .
```

Build takes 3–5 minutes (downloads Python packages). Model weights are **not** baked in — they download at container start from HuggingFace cache.

### Run the container

```bash
docker run --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  locate-anything-api:v1
```

The `-v` flag mounts your local HuggingFace cache into the container so the
model does not re-download every time you restart.

### Verify

```bash
curl http://localhost:8000/health
```

### Run with a custom port or device

```bash
docker run --gpus all \
  -p 9000:9000 \
  -e PORT=9000 \
  -e DEVICE=cuda \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  locate-anything-api:v1
```

### Stop the container

```bash
# Find the container ID
docker ps

# Stop it
docker stop <CONTAINER_ID>
```

---

## 3. Push to GitHub

### First time setup

```bash
cd locate-anything-api

# Initialise git (skip if already a repo)
git init

# Stage all files (.gitignore already excludes .env, __pycache__, model cache)
git add .

# Commit
git commit -m "feat: LocateAnything-3B FastAPI server v1"
```

Create a new **empty** repository on https://github.com/new (do not add a README or .gitignore — the project already has them).

```bash
# Add the remote (replace YOUR_USERNAME and REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/locate-anything-api.git

# Push
git branch -M main
git push -u origin main
```

### Subsequent pushes

```bash
git add .
git commit -m "fix: <describe what changed>"
git push
```

### What is excluded from Git

The `.gitignore` already excludes:
- `.env` — never commit secrets
- `__pycache__/`, `*.pyc`
- `.venv/`, `venv/`
- `.cache/` — model weights (too large for GitHub)

---

## 4. Deploy as an API for free

The model requires a GPU. Below are platforms that offer free GPU time.

---

### Option A — Hugging Face Spaces (easiest, recommended)

HuggingFace Spaces can run a FastAPI app on a free T4 GPU (16 GB VRAM).

**Step 1.** Create a Space at https://huggingface.co/new-space

- Owner: your username
- Space name: `locate-anything-api`
- SDK: **Docker**
- Hardware: **T4 small** (free tier)
- Visibility: Public

**Step 2.** Push your code to the Space (it has its own git remote):

```bash
# Add the Space as a remote
git remote add space https://huggingface.co/spaces/YOUR_HF_USERNAME/locate-anything-api

# Push
git push space main
```

**Step 3.** The Space builds the Docker image automatically and starts the server.
Your API is live at:
```
https://YOUR_HF_USERNAME-locate-anything-api.hf.space
```

**Step 4.** Check it:
```bash
curl https://YOUR_HF_USERNAME-locate-anything-api.hf.space/health
```

**Notes:**
- Free T4 Spaces sleep after 48 hours of inactivity — wake them with a GET request.
- The first cold start takes 5–10 minutes (model download).
- Do not commit `.env` — set secrets via Space Settings → Variables and secrets.

---

### Option B — Google Colab (testing only, not persistent)

Colab gives you a free T4 GPU for interactive sessions. Not suitable for a
persistent public API, but useful for development testing.

In a Colab notebook cell:

```python
# Install dependencies
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q
!pip install -r requirements.txt -q

# Start server in background
import subprocess, time
proc = subprocess.Popen(["python", "run.py"])
time.sleep(10)  # wait for startup

# Expose with ngrok (install ngrok first: pip install pyngrok)
from pyngrok import ngrok
tunnel = ngrok.connect(8000)
print("Public URL:", tunnel.public_url)
```

Your API is accessible at the printed ngrok URL for the duration of the session.

---

### Option C — Kaggle Notebooks (free T4/P100, 30 h/week)

Similar to Colab. Kaggle gives 30 GPU hours per week for free.

1. Create a new notebook at https://www.kaggle.com/code
2. Enable GPU accelerator (Settings → Accelerator → GPU T4 x2)
3. Upload the project as a Kaggle dataset or clone from GitHub:
   ```python
   !git clone https://github.com/YOUR_USERNAME/locate-anything-api.git
   %cd locate-anything-api
   !pip install -r requirements.txt -q
   ```
4. Run the server and expose with ngrok (same as Colab above).

---

### Option D — Modal (free $30/month credit)

[Modal](https://modal.com) is a serverless GPU platform with a generous free tier.
It requires a small wrapper script but gives you a persistent HTTPS endpoint.

```bash
pip install modal
modal setup        # authenticate
```

Create `modal_deploy.py` next to `run.py`:

```python
import modal

app = modal.App("locate-anything-api")
image = modal.Image.from_dockerfile("Dockerfile")

@app.function(
    image=image,
    gpu="T4",
    timeout=300,
    keep_warm=1,
)
@modal.web_endpoint(method="GET")
def health():
    import requests
    return requests.get("http://localhost:8000/health").json()
```

Then deploy:
```bash
modal deploy modal_deploy.py
```

For a full FastAPI integration see: https://modal.com/docs/guide/webhooks

---

## Choosing a platform

| Platform | GPU | Free limit | Persistent | Best for |
|---|---|---|---|---|
| HF Spaces | T4 16 GB | Unlimited (sleeps) | Yes | Sharing a demo |
| Google Colab | T4 | ~4 h/session | No | Development |
| Kaggle | T4/P100 | 30 h/week | No | Development |
| Modal | T4/A10G | $30/month credit | Yes | Real API endpoint |

---

## Testing a deployed endpoint

Replace `BASE_URL` with your deployment URL:

```bash
# Health check
curl https://YOUR_DEPLOYMENT_URL/health

# Object detection
curl -X POST https://YOUR_DEPLOYMENT_URL/detect \
  -F "image=@photo.jpg" \
  -F "categories=person,car" \
  -F "generation_mode=hybrid"

# Or use the test client (edit BASE_URL at the top of the file first)
python test_client.py photo.jpg
```
