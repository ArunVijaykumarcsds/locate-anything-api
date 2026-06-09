"""
worker.py — Model loading and inference for locate-anything-api.

Key changes vs original:
- _build_strict_prefix() injects "If not found output NONE." into prompts
- _is_none_response() detects when the model says it found nothing
- run_inference() now accepts strict: bool and returns not_found metadata
- All task-specific helpers (detect, ground, text, gui, point) pass strict through
"""

import logging
import re
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from .config import settings
from .schemas import BoundingBox, ImageSize, Point, PredictResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level model state (loaded once at startup)
# ---------------------------------------------------------------------------

_model: Any = None
_processor: Any = None


def load_model() -> None:
    """Load LocateAnything-3B from HuggingFace (or local path)."""
    global _model, _processor

    logger.info("Loading model from %s …", settings.model_path)

    _processor = AutoProcessor.from_pretrained(
        settings.model_path, trust_remote_code=True
    )
    _model = AutoModelForCausalLM.from_pretrained(
        settings.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(settings.device)
    _model.eval()

    logger.info("Model ready on %s", settings.device)


def get_model():
    if _model is None or _processor is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _model, _processor


# ---------------------------------------------------------------------------
# Strict-mode helpers
# ---------------------------------------------------------------------------

_STRICT_PREFIX = (
    "IMPORTANT: Only locate objects that EXACTLY match the description. "
    "If no such object exists in the image, output the single word NONE and nothing else.\n\n"
)

_NONE_PATTERN = re.compile(r"^\s*none\.?\s*$", re.IGNORECASE)


def _build_strict_prefix(strict: bool) -> str:
    return _STRICT_PREFIX if strict else ""


def _is_none_response(text: str) -> bool:
    """Return True if the model signalled 'not found' via NONE output."""
    return bool(_NONE_PATTERN.match(text.strip()))


# ---------------------------------------------------------------------------
# Coordinate parsing helpers  (unchanged from original — adapt as needed)
# ---------------------------------------------------------------------------

def _parse_boxes(text: str, img_w: int, img_h: int) -> list[BoundingBox]:
    """
    Extract bounding boxes from model output text.
    LocateAnything-3B outputs boxes as <box>x1,y1,x2,y2</box> in pixel coords
    OR as normalised floats — adjust the regex to match actual model output.
    """
    boxes = []
    # Pattern for pixel-coord boxes: <box>120,45,340,210</box>
    for m in re.finditer(r"<box>\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*</box>", text):
        x1, y1, x2, y2 = map(float, m.groups())
        boxes.append(BoundingBox(
            x1=x1, y1=y1, x2=x2, y2=y2,
            x1_norm=x1 / img_w, y1_norm=y1 / img_h,
            x2_norm=x2 / img_w, y2_norm=y2 / img_h,
        ))
    return boxes


def _parse_points(text: str, img_w: int, img_h: int) -> list[Point]:
    """Extract point coordinates from model output."""
    points = []
    for m in re.finditer(r"<point>\s*([0-9.]+)\s*,\s*([0-9.]+)\s*</point>", text):
        x, y = map(float, m.groups())
        points.append(Point(x=x, y=y, x_norm=x / img_w, y_norm=y / img_h))
    return points


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def run_inference(image: Image.Image, prompt: str, strict: bool = False) -> PredictResponse:
    """
    Run the model and return a structured PredictResponse.

    If strict=True and the model outputs NONE, boxes/points will be empty
    and not_found=True will be set in the response.
    """
    model, processor = get_model()

    full_prompt = _build_strict_prefix(strict) + prompt

    inputs = processor(text=full_prompt, images=image, return_tensors="pt").to(settings.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512)

    answer = processor.decode(output_ids[0], skip_special_tokens=True).strip()

    img_w, img_h = image.size
    image_size = ImageSize(width=img_w, height=img_h)

    # --- Strict-mode: detect NONE response ---
    if strict and _is_none_response(answer):
        logger.info("Model returned NONE (strict mode) for prompt: %r", prompt[:80])
        return PredictResponse(
            answer=answer,
            boxes=[],
            points=[],
            image_size=image_size,
            not_found=True,
            not_found_reason=(
                "The model could not find any object matching the description "
                "in this image."
            ),
        )

    boxes = _parse_boxes(answer, img_w, img_h)
    points = _parse_points(answer, img_w, img_h)

    return PredictResponse(
        answer=answer,
        boxes=boxes,
        points=points,
        image_size=image_size,
        not_found=False,
    )


# ---------------------------------------------------------------------------
# Task-specific prompt builders  (keep in sync with main.py endpoint calls)
# ---------------------------------------------------------------------------

def predict_detect(image: Image.Image, categories: str, generation_mode: str, strict: bool) -> PredictResponse:
    prompt = f"Detect all instances of: {categories}. Generation mode: {generation_mode}."
    return run_inference(image, prompt, strict=strict)


def predict_ground_single(image: Image.Image, query: str, generation_mode: str, strict: bool) -> PredictResponse:
    prompt = f"Ground the following (single instance): {query}. Generation mode: {generation_mode}."
    return run_inference(image, prompt, strict=strict)


def predict_ground_multi(image: Image.Image, query: str, generation_mode: str, strict: bool) -> PredictResponse:
    prompt = f"Ground all instances of: {query}. Generation mode: {generation_mode}."
    return run_inference(image, prompt, strict=strict)


def predict_text_detect(image: Image.Image, strict: bool) -> PredictResponse:
    prompt = "Detect all text in the image."
    return run_inference(image, prompt, strict=strict)


def predict_text_ground(image: Image.Image, text: str, strict: bool) -> PredictResponse:
    prompt = f"Find the text string: \"{text}\"."
    return run_inference(image, prompt, strict=strict)


def predict_gui_ground(image: Image.Image, query: str, output_type: str, strict: bool) -> PredictResponse:
    prompt = f"Locate the UI element: {query}. Output type: {output_type}."
    return run_inference(image, prompt, strict=strict)


def predict_point(image: Image.Image, query: str, strict: bool) -> PredictResponse:
    prompt = f"Point to: {query}."
    return run_inference(image, prompt, strict=strict)


def predict_raw(image: Image.Image, prompt: str) -> PredictResponse:
    # Raw endpoint — no strict injection, caller owns the prompt
    return run_inference(image, prompt, strict=False)
