"""
LocateAnything-3B API — Local Test Client
Usage:
    python test_client.py <path_to_image.jpg>

Requires the API server to be running:
    python run.py
"""

import sys
import json
import requests
from pathlib import Path

BASE_URL = "http://localhost:8000"
IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "example.jpg"


def post(endpoint: str, extra_data: dict, image_path: str = IMAGE_PATH) -> dict:
    with open(image_path, "rb") as f:
        files = {"image": (Path(image_path).name, f, "image/jpeg")}
        r = requests.post(f"{BASE_URL}{endpoint}", files=files, data=extra_data, timeout=120)
    r.raise_for_status()
    return r.json()


def pretty(label: str, res: dict) -> None:
    print(f"\n{'─'*60}")
    print(f"▶ {label}")
    print(f"  answer (first 120 chars): {str(res.get('answer',''))[:120]}")
    print(f"  boxes : {json.dumps(res.get('boxes', [])[:2], indent=4)}")
    print(f"  points: {json.dumps(res.get('points', [])[:2], indent=4)}")
    print(f"  image : {res.get('image_size')}")


if __name__ == "__main__":
    if not Path(IMAGE_PATH).exists():
        print(f"❌ Image not found: {IMAGE_PATH}")
        print("   Usage: python test_client.py path/to/image.jpg")
        sys.exit(1)

    print(f"Testing against {BASE_URL} with image: {IMAGE_PATH}\n")

    # Health check
    h = requests.get(f"{BASE_URL}/health", timeout=10).json()
    print(f"Health: {h}")
    if not h.get("model_loaded"):
        print("❌ Model not loaded yet — wait for startup to finish and retry.")
        sys.exit(1)

    # Object detection
    res = post("/detect", {"categories": "person,car,bicycle", "generation_mode": "hybrid"})
    pretty("Object Detection  (person, car, bicycle)", res)

    # Phrase grounding — multi
    res = post("/ground/multi", {"phrase": "people wearing hats"})
    pretty("Phrase Grounding  (people wearing hats)", res)

    # Scene text detection
    res = post("/text/detect", {"generation_mode": "hybrid"})
    pretty("Scene Text Detection", res)

    # Pointing
    res = post("/point", {"phrase": "the largest object in the scene"})
    pretty("Pointing  (largest object)", res)

    # Raw prompt
    res = post("/predict/raw", {
        "prompt": "Locate all the instances that matches the following description: dog.</c>cat.",
        "generation_mode": "hybrid",
    })
    pretty("Raw prompt  (dog, cat)", res)

    print(f"\n{'─'*60}")
    print("✅ All tests passed")
