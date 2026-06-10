from __future__ import annotations

import io
import sys
import warnings
from contextlib import nullcontext, redirect_stderr, redirect_stdout
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .masks import bbox_and_area, mask_to_png_data, simple_prompt_mask
from .schemas import MaskCandidate, SamPromptRequest


class SamService:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.model = None
        self.processor = None
        self.device = "cpu"
        self._image_cache: dict[str, Any] = {}
        self.last_error: str | None = None
        self.load_notes: str = ""
        self.allow_fallback = os.environ.get("FOODSEG_ALLOW_FAKE_SAM", "").lower() in {
            "1",
            "true",
            "yes",
        }

    def _ensure_loaded(self, threshold: float) -> bool:
        if self.processor is not None:
            self.processor.confidence_threshold = threshold
            return True
        try:
            sys.path.insert(0, str(self.repo_root / "sam3"))
            import torch
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            if torch.version.cuda is None:
                raise RuntimeError(
                    "PyTorch was installed without CUDA support. Install torch and torchvision from the cu130 wheel index before running SAM."
                )
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            bpe_path = self.repo_root / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
            ckpt_path = self.repo_root / "sam3" / "assets" / "sam3.1_multiplex.pt"
            warnings.filterwarnings(
                "ignore",
                message="Importing from timm.models.layers is deprecated.*",
                category=FutureWarning,
            )
            load_log = io.StringIO()
            with redirect_stdout(load_log), redirect_stderr(load_log):
                self.model = build_sam3_image_model(
                    bpe_path=str(bpe_path),
                    checkpoint_path=str(ckpt_path),
                    device=self.device,
                )
            self.load_notes = load_log.getvalue().strip()
            self.model.eval()
            self.processor = Sam3Processor(
                self.model, device=self.device, confidence_threshold=threshold
            )
            self.last_error = None
            return True
        except Exception as exc:
            self.last_error = str(exc)
            return False

    def predict(self, image_path: Path, request: SamPromptRequest, width: int, height: int) -> list[MaskCandidate]:
        if not self._ensure_loaded(request.confidence_threshold):
            return self._fallback_predict(request, width, height)

        try:
            image = Image.open(image_path).convert("RGB")
            with self._torch_autocast():
                if request.prompt_type == "point":
                    masks, scores = self._predict_geometric_points(
                        image, request, width, height
                    )
                    self.last_error = None
                    return self._format_masks(masks, scores, "point")

                state = self.processor.set_image(image)
                if request.prompt_type == "text":
                    if not request.text:
                        raise ValueError("Text prompt is required")
                    state = self.processor.set_text_prompt(request.text, state)
                elif request.prompt_type == "box":
                    if not request.boxes:
                        raise ValueError("At least one box is required")
                    for box in request.boxes:
                        norm = [
                            (box.x + box.width / 2.0) / width,
                            (box.y + box.height / 2.0) / height,
                            box.width / width,
                            box.height / height,
                        ]
                        state = self.processor.add_geometric_prompt(norm, bool(box.label), state)
            masks = [m[0].detach().cpu().numpy().astype(bool) for m in state.get("masks", [])]
            scores = [float(s.detach().cpu().item()) for s in state.get("scores", [])]
            self.last_error = None
            return self._format_masks(masks, scores, request.prompt_type)
        except Exception as exc:
            self.last_error = str(exc)
            return self._fallback_predict(request, width, height)

    def _torch_autocast(self):
        if self.device != "cuda":
            return nullcontext()
        import torch

        return torch.autocast("cuda", dtype=torch.bfloat16)

    def _predict_geometric_points(
        self,
        image: Image.Image,
        request: SamPromptRequest,
        width: int,
        height: int,
    ) -> tuple[list[np.ndarray], list[float]]:
        if not request.points:
            raise ValueError("At least one point is required")
        import torch

        state = self.processor.set_image(image)
        dummy_text_outputs = self.model.backbone.forward_text(["visual"], device=self.device)
        state["backbone_out"].update(dummy_text_outputs)
        state["geometric_prompt"] = self.model._get_dummy_prompt()
        for point in request.points:
            coords = torch.tensor(
                [point.x / width, point.y / height],
                device=self.device,
                dtype=torch.float32,
            ).view(1, 1, 2)
            labels = torch.tensor(
                [point.label],
                device=self.device,
                dtype=torch.long,
            ).view(1, 1)
            state["geometric_prompt"].append_points(coords, labels)
        state = self.processor._forward_grounding(state)
        masks = [m[0].detach().cpu().numpy().astype(bool) for m in state.get("masks", [])]
        scores = [float(s.detach().cpu().item()) for s in state.get("scores", [])]
        positive_points = [(int(p.x), int(p.y)) for p in request.points if p.label == 1]
        negative_points = [(int(p.x), int(p.y)) for p in request.points if p.label == 0]
        filtered: list[tuple[np.ndarray, float]] = []
        for idx, mask in enumerate(masks):
            contains_positive = all(
                0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]
                for x, y in positive_points
            )
            excludes_negative = all(
                not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x])
                for x, y in negative_points
            )
            if contains_positive and excludes_negative:
                filtered.append((mask, scores[idx] if idx < len(scores) else 0.0))
        if filtered:
            filtered.sort(key=lambda item: item[1], reverse=True)
            return [item[0] for item in filtered[:5]], [item[1] for item in filtered[:5]]
        return masks, scores

    def _format_masks(self, masks: list[np.ndarray], scores: list[float], prompt_type: str) -> list[MaskCandidate]:
        candidates: list[MaskCandidate] = []
        scored_masks = [
            (idx, mask, scores[idx] if idx < len(scores) else None)
            for idx, mask in enumerate(masks)
        ]
        scored_masks.sort(key=lambda item: item[2] if item[2] is not None else -1.0, reverse=True)
        for idx, mask, score in scored_masks:
            bbox, area = bbox_and_area(mask)
            if area <= 0:
                continue
            candidates.append(
                MaskCandidate(
                    mask_png=mask_to_png_data(mask),
                    bbox=bbox,
                    area=area,
                    score=score,
                    prompt_type=prompt_type,
                )
            )
        return candidates

    def _fallback_predict(self, request: SamPromptRequest, width: int, height: int) -> list[MaskCandidate]:
        if not self.allow_fallback:
            return []
        mask = simple_prompt_mask(width, height, boxes=request.boxes, points=request.points)
        bbox, area = bbox_and_area(mask)
        return [
            MaskCandidate(
                mask_png=mask_to_png_data(mask),
                bbox=bbox,
                area=area,
                score=0.0,
                prompt_type=f"{request.prompt_type}:fallback",
            )
        ]
