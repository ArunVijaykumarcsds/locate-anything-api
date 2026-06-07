"""
LocateAnything-3B FastAPI Server
nvidia/LocateAnything-3B — Vision-Language Grounding Model
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Optional
import io

from PIL import Image

from app.worker import LocateAnythingWorker
from app.schemas import PredictResponse
from app.config import settings


# ── Single global worker (loaded once at startup) ─────────────────────────────
worker: Optional[LocateAnythingWorker] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup; release on shutdown."""
    global worker
    print(f"Loading model from: {settings.MODEL_PATH}")
    worker = LocateAnythingWorker(
        model_path=settings.MODEL_PATH,
        device=settings.DEVICE,
    )
    print("✅ Model loaded and ready.")
    yield
    worker = None


app = FastAPI(
    title="LocateAnything-3B API",
    description=(
        "FastAPI wrapper for nvidia/LocateAnything-3B — "
        "a vision-language model for fast and high-quality visual grounding. "
        "Supports object detection, phrase grounding, GUI grounding, "
        "scene text detection, and pointing."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_worker() -> LocateAnythingWorker:
    """Return the global worker; raise 503 if not yet loaded."""
    if worker is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Try again in a moment.")
    return worker


def load_image(file: UploadFile) -> Image.Image:
    """Read an uploaded file and return a PIL RGB image."""
    contents = file.file.read()
    try:
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or unreadable image file.")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
def health():
    """Liveness + readiness check."""
    return {"status": "ok", "model_loaded": worker is not None}


# ── Object Detection ──────────────────────────────────────────────────────────

@app.post("/detect", response_model=PredictResponse, tags=["Grounding"])
async def detect(
    image: UploadFile = File(..., description="RGB image file (JPEG / PNG)"),
    categories: str = Form(..., description="Comma-separated categories e.g. 'person,car,bicycle'"),
    generation_mode: str = Form("hybrid", description="fast | slow | hybrid"),
    max_new_tokens: int = Form(2048, ge=64, le=8192),
):
    """
    **Object / Document Layout Detection**

    Locate all instances of the given categories.
    Returns bounding boxes in pixel and normalised [0,1] coordinates.
    """
    w = get_worker()
    img = load_image(image)
    cats = [c.strip() for c in categories.split(",") if c.strip()]
    if not cats:
        raise HTTPException(status_code=422, detail="Provide at least one category.")

    result = w.detect(img, cats, generation_mode=generation_mode, max_new_tokens=max_new_tokens)
    iw, ih = img.size
    return PredictResponse(
        answer=result["answer"],
        boxes=w.parse_boxes(result["answer"], iw, ih),
        image_size={"width": iw, "height": ih},
    )


# ── Phrase Grounding ──────────────────────────────────────────────────────────

@app.post("/ground/single", response_model=PredictResponse, tags=["Grounding"])
async def ground_single(
    image: UploadFile = File(...),
    phrase: str = Form(..., description="Natural-language referring expression"),
    generation_mode: str = Form("hybrid"),
    max_new_tokens: int = Form(2048, ge=64, le=8192),
):
    """Locate a **single** instance matching the phrase."""
    w = get_worker()
    img = load_image(image)
    result = w.ground_single(img, phrase, generation_mode=generation_mode, max_new_tokens=max_new_tokens)
    iw, ih = img.size
    return PredictResponse(
        answer=result["answer"],
        boxes=w.parse_boxes(result["answer"], iw, ih),
        image_size={"width": iw, "height": ih},
    )


@app.post("/ground/multi", response_model=PredictResponse, tags=["Grounding"])
async def ground_multi(
    image: UploadFile = File(...),
    phrase: str = Form(..., description="Natural-language referring expression"),
    generation_mode: str = Form("hybrid"),
    max_new_tokens: int = Form(2048, ge=64, le=8192),
):
    """Locate **all** instances matching the phrase."""
    w = get_worker()
    img = load_image(image)
    result = w.ground_multi(img, phrase, generation_mode=generation_mode, max_new_tokens=max_new_tokens)
    iw, ih = img.size
    return PredictResponse(
        answer=result["answer"],
        boxes=w.parse_boxes(result["answer"], iw, ih),
        image_size={"width": iw, "height": ih},
    )


# ── Text / OCR ────────────────────────────────────────────────────────────────

@app.post("/text/detect", response_model=PredictResponse, tags=["Text & OCR"])
async def detect_text(
    image: UploadFile = File(...),
    generation_mode: str = Form("hybrid"),
    max_new_tokens: int = Form(2048, ge=64, le=8192),
):
    """Detect **all scene text** in the image (returns bounding boxes)."""
    w = get_worker()
    img = load_image(image)
    result = w.detect_text(img, generation_mode=generation_mode, max_new_tokens=max_new_tokens)
    iw, ih = img.size
    return PredictResponse(
        answer=result["answer"],
        boxes=w.parse_boxes(result["answer"], iw, ih),
        image_size={"width": iw, "height": ih},
    )


@app.post("/text/ground", response_model=PredictResponse, tags=["Text & OCR"])
async def ground_text(
    image: UploadFile = File(...),
    phrase: str = Form(..., description="The text string to locate"),
    generation_mode: str = Form("hybrid"),
    max_new_tokens: int = Form(2048, ge=64, le=8192),
):
    """Locate a **specific text string** in the image."""
    w = get_worker()
    img = load_image(image)
    result = w.ground_text(img, phrase, generation_mode=generation_mode, max_new_tokens=max_new_tokens)
    iw, ih = img.size
    return PredictResponse(
        answer=result["answer"],
        boxes=w.parse_boxes(result["answer"], iw, ih),
        image_size={"width": iw, "height": ih},
    )


# ── GUI Grounding ─────────────────────────────────────────────────────────────

@app.post("/gui/ground", response_model=PredictResponse, tags=["GUI"])
async def gui_ground(
    image: UploadFile = File(...),
    phrase: str = Form(..., description="GUI element description e.g. 'the search button'"),
    output_type: str = Form("box", description="'box' or 'point'"),
    generation_mode: str = Form("hybrid"),
    max_new_tokens: int = Form(2048, ge=64, le=8192),
):
    """
    **GUI Element Grounding**

    Locate a UI element by natural-language description.
    `output_type=point` → click-point; `output_type=box` → bounding box.
    """
    if output_type not in ("box", "point"):
        raise HTTPException(status_code=422, detail="output_type must be 'box' or 'point'.")
    w = get_worker()
    img = load_image(image)
    result = w.ground_gui(img, phrase, output_type=output_type,
                          generation_mode=generation_mode, max_new_tokens=max_new_tokens)
    iw, ih = img.size
    return PredictResponse(
        answer=result["answer"],
        boxes=w.parse_boxes(result["answer"], iw, ih),
        points=w.parse_points(result["answer"], iw, ih),
        image_size={"width": iw, "height": ih},
    )


# ── Pointing ──────────────────────────────────────────────────────────────────

@app.post("/point", response_model=PredictResponse, tags=["Grounding"])
async def point(
    image: UploadFile = File(...),
    phrase: str = Form(..., description="Object or region to point to"),
    generation_mode: str = Form("hybrid"),
    max_new_tokens: int = Form(2048, ge=64, le=8192),
):
    """Return a **point coordinate** for the described object/region."""
    w = get_worker()
    img = load_image(image)
    result = w.point(img, phrase, generation_mode=generation_mode, max_new_tokens=max_new_tokens)
    iw, ih = img.size
    return PredictResponse(
        answer=result["answer"],
        points=w.parse_points(result["answer"], iw, ih),
        image_size={"width": iw, "height": ih},
    )


# ── Raw / Custom Prompt ───────────────────────────────────────────────────────

@app.post("/predict/raw", response_model=PredictResponse, tags=["Advanced"])
async def predict_raw(
    image: UploadFile = File(...),
    prompt: str = Form(..., description="Custom natural-language prompt"),
    generation_mode: str = Form("hybrid"),
    max_new_tokens: int = Form(2048, ge=64, le=8192),
    temperature: float = Form(0.7, ge=0.0, le=2.0),
):
    """
    **Raw / Custom Prompt**

    Send any prompt directly to the model.
    Useful for custom task templates not covered by the convenience endpoints.
    """
    w = get_worker()
    img = load_image(image)
    result = w.predict(img, prompt, generation_mode=generation_mode,
                       max_new_tokens=max_new_tokens, temperature=temperature)
    iw, ih = img.size
    return PredictResponse(
        answer=result["answer"],
        boxes=w.parse_boxes(result["answer"], iw, ih),
        points=w.parse_points(result["answer"], iw, ih),
        image_size={"width": iw, "height": ih},
    )
