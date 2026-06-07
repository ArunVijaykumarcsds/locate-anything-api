"""
LocateAnythingWorker
====================
Stateful worker that loads nvidia/LocateAnything-3B once and serves all
perception tasks: object detection, phrase grounding, text detection,
GUI grounding, and pointing.

Ported from the official NVIDIA model card:
https://huggingface.co/nvidia/LocateAnything-3B

Model weights (BF16, ~8 GB total):
  model.safetensors.index.json
  model-00001-of-00002.safetensors
  model-00002-of-00002.safetensors

Loaded via:
  AutoTokenizer.from_pretrained(..., trust_remote_code=True)
  AutoProcessor.from_pretrained(..., trust_remote_code=True)
  AutoModel.from_pretrained(..., trust_remote_code=True)
"""

import re
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor


class LocateAnythingWorker:
    """Loads the model once and serves all perception queries."""

    def __init__(
        self,
        model_path: str = "nvidia/LocateAnything-3B",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.dtype = dtype

        # trust_remote_code=True is REQUIRED — the model ships custom modelling
        # code that registers process_vision_info, py_apply_chat_template, and
        # the Parallel Box Decoding (PBD) generate() logic on the processor/model.
        print(f"[worker] Loading tokenizer from '{model_path}' ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        print(f"[worker] Loading processor ...")
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

        print(f"[worker] Loading model weights (BF16) ...")
        # AutoModel loads the full sharded checkpoint:
        #   model-00001-of-00002.safetensors
        #   model-00002-of-00002.safetensors
        # guided by model.safetensors.index.json
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device).eval()

        print("[worker] ✅ Model ready.")

    # ── Core inference ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        question: str,
        generation_mode: str = "hybrid",
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        verbose: bool = False,
    ) -> dict:
        """
        Run a single inference pass.

        Args:
            image:           PIL RGB image.
            question:        Natural-language prompt for the model.
            generation_mode: 'fast'  — MTP only (fastest, good for simple scenes)
                             'slow'  — pure autoregressive (slowest, most robust)
                             'hybrid'— MTP with AR fallback (default, best overall)
            max_new_tokens:  Max tokens to generate (NVIDIA recommends 8192).
            temperature:     Sampling temperature.
            verbose:         Print per-step decoding info.

        Returns:
            dict with key 'answer' (str) containing the raw model output.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

        # py_apply_chat_template and process_vision_info are injected by the
        # model's custom remote code when trust_remote_code=True is set.
        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            return_tensors="pt",
        ).to(self.device)

        # Cast pixel_values to the model's working dtype (BF16)
        pixel_values = inputs["pixel_values"].to(self.dtype)
        input_ids = inputs["input_ids"]
        image_grid_hws = inputs.get("image_grid_hws", None)

        response = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            image_grid_hws=image_grid_hws,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode=generation_mode,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=verbose,
        )

        # generate() returns either a plain string or a (answer, history, stats) tuple
        result: dict = {
            "answer": response[0] if isinstance(response, tuple) else response
        }
        if isinstance(response, tuple) and len(response) >= 3:
            result["history"] = response[1]
            result["stats"] = response[2]
        return result

    # ── Convenience task methods ───────────────────────────────────────────────

    def detect(self, image: Image.Image, categories: list[str], **kwargs) -> dict:
        """Object detection / document layout analysis (multi-category)."""
        cats = "</c>".join(categories)
        prompt = f"Locate all the instances that matches the following description: {cats}."
        return self.predict(image, prompt, **kwargs)

    def ground_single(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Phrase grounding — single instance."""
        prompt = f"Locate a single instance that matches the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def ground_multi(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Phrase grounding — all matching instances."""
        prompt = f"Locate all the instances that match the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def ground_text(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Text grounding — locate a specific text string."""
        prompt = f"Please locate the text referred as {phrase}."
        return self.predict(image, prompt, **kwargs)

    def detect_text(self, image: Image.Image, **kwargs) -> dict:
        """Scene text detection — find all text in the image."""
        prompt = "Detect all the text in box format."
        return self.predict(image, prompt, **kwargs)

    def ground_gui(
        self,
        image: Image.Image,
        phrase: str,
        output_type: str = "box",
        **kwargs,
    ) -> dict:
        """GUI element grounding. output_type='box' or 'point'."""
        if output_type == "point":
            prompt = f"Point to: {phrase}."
        else:
            prompt = f"Locate the region that matches the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def point(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Pointing — return a point coordinate for the described object."""
        prompt = f"Point to: {phrase}."
        return self.predict(image, prompt, **kwargs)

    # ── Output parsers ─────────────────────────────────────────────────────────

    @staticmethod
    def parse_boxes(
        answer: str, image_width: int, image_height: int
    ) -> list[dict]:
        """
        Parse model output into pixel-coordinate bounding boxes.

        The model outputs coordinates as integers normalised to [0, 1000]:
            <box><x1><y1><x2><y2></box>

        Returns a list of dicts with pixel coords + normalised coords.
        """
        boxes = []
        # 4-coord pattern — bounding box
        for m in re.finditer(
            r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer
        ):
            x1, y1, x2, y2 = (int(g) for g in m.groups())
            boxes.append(
                {
                    "x1": round(x1 / 1000 * image_width, 2),
                    "y1": round(y1 / 1000 * image_height, 2),
                    "x2": round(x2 / 1000 * image_width, 2),
                    "y2": round(y2 / 1000 * image_height, 2),
                    "x1_norm": round(x1 / 1000, 4),
                    "y1_norm": round(y1 / 1000, 4),
                    "x2_norm": round(x2 / 1000, 4),
                    "y2_norm": round(y2 / 1000, 4),
                }
            )
        return boxes

    @staticmethod
    def parse_points(
        answer: str, image_width: int, image_height: int
    ) -> list[dict]:
        """
        Parse model output into pixel-coordinate points.

        Point format (2 coords only):
            <box><x><y></box>

        Note: the regex is intentionally anchored to exactly 2 groups so it
        does NOT match 4-coord bounding boxes.
        """
        points = []
        # 2-coord pattern — point only (negative lookbehind on extra tokens)
        for m in re.finditer(
            r"<box><(\d+)><(\d+)></box>", answer
        ):
            # Skip if this match is actually part of a 4-coord box
            full_match = m.group(0)
            # A point box has exactly 2 number tokens between <box> and </box>
            token_count = len(re.findall(r"<\d+>", full_match))
            if token_count != 2:
                continue
            x, y = int(m.group(1)), int(m.group(2))
            points.append(
                {
                    "x": round(x / 1000 * image_width, 2),
                    "y": round(y / 1000 * image_height, 2),
                    "x_norm": round(x / 1000, 4),
                    "y_norm": round(y / 1000, 4),
                }
            )
        return points
