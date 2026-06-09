"""
main.py — FastAPI app for locate-anything-api.

Changes vs original:
- All form handlers now accept `strict: bool = Form(False)`
- strict is forwarded to the corresponding worker.predict_* function
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image
import io

from .schemas import PredictResponse
from . import worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    worker.load_model()
    yield


app = FastAPI(
    title="LocateAnything-3B API",
    description="Natural language object localisation powered by NVIDIA LocateAnything-3B",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _load_image(upload: UploadFile) -> Image.Image:
    return Image.open(io.BytesIO(upload.file.read())).convert("RGB")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect", response_model=PredictResponse)
async def detect(
    image: UploadFile = File(...),
    categories: str = Form(...),
    generation_mode: str = Form("hybrid"),
    strict: bool = Form(False),          # ← NEW
):
    img = _load_image(image)
    return worker.predict_detect(img, categories, generation_mode, strict=strict)


@app.post("/ground/single", response_model=PredictResponse)
async def ground_single(
    image: UploadFile = File(...),
    query: str = Form(...),
    generation_mode: str = Form("hybrid"),
    strict: bool = Form(False),          # ← NEW
):
    img = _load_image(image)
    return worker.predict_ground_single(img, query, generation_mode, strict=strict)


@app.post("/ground/multi", response_model=PredictResponse)
async def ground_multi(
    image: UploadFile = File(...),
    query: str = Form(...),
    generation_mode: str = Form("hybrid"),
    strict: bool = Form(False),          # ← NEW
):
    img = _load_image(image)
    return worker.predict_ground_multi(img, query, generation_mode, strict=strict)


@app.post("/text/detect", response_model=PredictResponse)
async def text_detect(
    image: UploadFile = File(...),
    strict: bool = Form(False),          # ← NEW
):
    img = _load_image(image)
    return worker.predict_text_detect(img, strict=strict)


@app.post("/text/ground", response_model=PredictResponse)
async def text_ground(
    image: UploadFile = File(...),
    text: str = Form(...),
    strict: bool = Form(False),          # ← NEW
):
    img = _load_image(image)
    return worker.predict_text_ground(img, text, strict=strict)


@app.post("/gui/ground", response_model=PredictResponse)
async def gui_ground(
    image: UploadFile = File(...),
    query: str = Form(...),
    output_type: str = Form("box"),
    strict: bool = Form(False),          # ← NEW
):
    img = _load_image(image)
    return worker.predict_gui_ground(img, query, output_type, strict=strict)


@app.post("/point", response_model=PredictResponse)
async def point(
    image: UploadFile = File(...),
    query: str = Form(...),
    strict: bool = Form(False),          # ← NEW
):
    img = _load_image(image)
    return worker.predict_point(img, query, strict=strict)


@app.post("/predict/raw", response_model=PredictResponse)
async def predict_raw(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    # NOTE: no strict here — caller owns the prompt on this endpoint
):
    img = _load_image(image)
    return worker.predict_raw(img, prompt)
