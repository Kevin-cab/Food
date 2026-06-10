from __future__ import annotations

import base64
import io
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def mask_from_png_data(data: str) -> np.ndarray:
    if "," in data:
        data = data.split(",", 1)[1]
    raw = base64.b64decode(data)
    img = Image.open(io.BytesIO(raw)).convert("L")
    return np.array(img) > 0


def mask_to_png_data(mask: np.ndarray) -> str:
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def load_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 0


def copy_mask(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def bbox_and_area(mask: np.ndarray) -> tuple[list[float], int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0], 0
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)], int(mask.sum())


def simple_prompt_mask(width: int, height: int, boxes: list | None = None, points: list | None = None) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if boxes:
        for box in boxes:
            x = int(max(0, min(width, box.x)))
            y = int(max(0, min(height, box.y)))
            w = int(max(1, min(width - x, box.width)))
            h = int(max(1, min(height - y, box.height)))
            if box.label:
                mask[y : y + h, x : x + w] = True
            else:
                mask[y : y + h, x : x + w] = False
    elif points:
        yy, xx = np.ogrid[:height, :width]
        radius = max(8, min(width, height) // 10)
        for pt in points:
            disk = (xx - pt.x) ** 2 + (yy - pt.y) ** 2 <= radius**2
            if pt.label:
                mask |= disk
            else:
                mask &= ~disk
    else:
        margin_x = width // 4
        margin_y = height // 4
        mask[margin_y : height - margin_y, margin_x : width - margin_x] = True
    return mask

