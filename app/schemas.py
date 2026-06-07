"""
Pydantic schemas for LocateAnything-3B API request/response models.
"""

from pydantic import BaseModel, Field
from typing import Literal


class BoundingBox(BaseModel):
    """Bounding box in both pixel and normalised [0, 1] coordinates."""
    x1: float = Field(..., description="Left edge in pixels")
    y1: float = Field(..., description="Top edge in pixels")
    x2: float = Field(..., description="Right edge in pixels")
    y2: float = Field(..., description="Bottom edge in pixels")
    x1_norm: float = Field(..., description="Left edge normalised [0,1]")
    y1_norm: float = Field(..., description="Top edge normalised [0,1]")
    x2_norm: float = Field(..., description="Right edge normalised [0,1]")
    y2_norm: float = Field(..., description="Bottom edge normalised [0,1]")


class Point(BaseModel):
    """Point coordinate in both pixel and normalised [0, 1] coordinates."""
    x: float = Field(..., description="X coordinate in pixels")
    y: float = Field(..., description="Y coordinate in pixels")
    x_norm: float = Field(..., description="X coordinate normalised [0,1]")
    y_norm: float = Field(..., description="Y coordinate normalised [0,1]")


class ImageSize(BaseModel):
    width: int
    height: int


class PredictResponse(BaseModel):
    """Standard response for all grounding endpoints."""
    answer: str = Field(..., description="Raw model token output (for debugging)")
    boxes: list[BoundingBox] = Field(default_factory=list, description="Detected bounding boxes")
    points: list[Point] = Field(default_factory=list, description="Detected point coordinates")
    image_size: dict = Field(..., description="Original image dimensions {width, height}")
