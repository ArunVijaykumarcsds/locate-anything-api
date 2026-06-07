# ── LocateAnything-3B FastAPI Server ─────────────────────────────────────────
# Base: PyTorch 2.3.0 + CUDA 12.1 + cuDNN 8 (runtime, not devel)
# Supports: NVIDIA Ampere (A100), Hopper (H100), Lovelace (L40/RTX4090)
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
# torch/torchvision are already in the base image; requirements.txt lists them
# as minimum versions so pip will skip reinstalling if versions match.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ ./app/
COPY run.py .

# (Optional) Pre-download model weights at build time.
# Uncomment to bake the model into the image (~8 GB larger image).
# Requires HF_TOKEN build-arg if model is gated:
#   docker build --build-arg HF_TOKEN=hf_... -t locate-anything-api .
# ARG HF_TOKEN
# RUN if [ -n "$HF_TOKEN" ]; then \
#       python -c " \
#         from transformers import AutoModel, AutoTokenizer, AutoProcessor; \
#         kw = dict(trust_remote_code=True); \
#         AutoTokenizer.from_pretrained('nvidia/LocateAnything-3B', **kw); \
#         AutoProcessor.from_pretrained('nvidia/LocateAnything-3B', **kw); \
#         AutoModel.from_pretrained('nvidia/LocateAnything-3B', **kw); \
#       "; \
#     fi

EXPOSE 8000

# Health-check so orchestrators know when the model is loaded
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "run.py"]
