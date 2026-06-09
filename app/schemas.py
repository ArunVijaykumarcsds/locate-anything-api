"""
Pydantic request/response schemas for locate-anything-api.

Changes from original:
- Added `strict: bool = False` to all request bodies
- Added `not_found: bool` to PredictResponse
- Added `not_found_reason: str | None` to PredictResponse
"""

from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    x1_norm: float
    y1_norm: float
    x2_norm: float
    y2_norm: float


class Point(BaseModel):
    x: float
    y: float
    x_norm: float
    y_norm: float


class ImageSize(BaseModel):
    width: int
    height: int


# ---------------------------------------------------------------------------
# Response — shared by all endpoints
# ---------------------------------------------------------------------------

class PredictResponse(BaseModel):
    answer: str = Field(description="Raw model output string")
    boxes: list[BoundingBox] = Field(default_factory=list)
    points: list[Point] = Field(default_factory=list)
    image_size: ImageSize

    # --- NEW FIELDS ---
    not_found: bool = Field(
        default=False,
        description="True when strict=True and the model could not find the target"
    )
    not_found_reason: Optional[str] = Field(
        default=None,
        description="Human-readable explanation when not_found is True"
    )


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class DetectRequest(BaseModel):
    categories: str = Field(
        description="Comma-separated object categories, e.g. 'person,car,bicycle'"
    )
    generation_mode: str = Field(default="hybrid")
    # --- NEW ---
    strict: bool = Field(
        default=False,
        description=(
            "When True, the model is instructed to return NONE if no matching "
            "object exists. Prevents hallucinated results."
        )
    )


class GroundRequest(BaseModel):
    query: str = Field(description="Natural-language phrase to ground")
    generation_mode: str = Field(default="hybrid")
    # --- NEW ---
    strict: bool = Field(
        default=False,
        description="When True, returns not_found=True instead of a hallucinated result."
    )


class TextGroundRequest(BaseModel):
    text: str = Field(description="The text string to locate in the image")
    # --- NEW ---
    strict: bool = Field(default=False)


class GuiGroundRequest(BaseModel):
    query: str = Field(description="Description of the UI element to locate")
    output_type: str = Field(default="box", description="'box' or 'point'")
    # --- NEW ---
    strict: bool = Field(default=False)


class PointRequest(BaseModel):
    query: str = Field(description="Description of the object to point to")
    # --- NEW ---
    strict: bool = Field(default=False)


class RawPredictRequest(BaseModel):
    prompt: str = Field(description="Custom prompt sent directly to the model")
    # NOTE: strict is intentionally omitted for /predict/raw — caller owns the prompt
