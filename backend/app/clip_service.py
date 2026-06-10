from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .db import ProjectDb
from .project_service import ProjectService
from .schemas import ImageRecord, SearchResult


class ClipService:
    def __init__(self, repo_root: Path, allow_fallback: bool = False):
        self.repo_root = repo_root
        self.model = None
        self.preprocess = None
        self.device = "cpu"
        self.model_name = "ViT-B/32"
        self.last_error: str | None = None
        self.allow_fallback = allow_fallback

    def _ensure_loaded(self) -> bool:
        if self.model is not None:
            return True
        try:
            import torch
            import clip

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            self.model.eval()
            return True
        except Exception as exc:
            message = str(exc)
            if "No module named 'clip'" in message:
                message = "OpenAI CLIP is not installed in this venv. Install it with: .\\.venv\\Scripts\\python.exe -m pip install git+https://github.com/openai/CLIP.git"
            self.last_error = message
            return False

    def available(self) -> bool:
        return self._ensure_loaded()

    def status(self, project: ProjectDb, project_service: ProjectService) -> dict:
        with project.connect() as conn:
            indexed = int(conn.execute("SELECT COUNT(*) AS n FROM embeddings").fetchone()["n"])
        total = len(project_service.list_images(limit=1_000_000))
        return {
            "available": self.available(),
            "indexed": indexed,
            "total": total,
            "model": self.model_name,
            "device": self.device,
            "error": self.last_error,
        }

    def index(self, project: ProjectDb, project_service: ProjectService, force: bool = False, limit: int | None = None) -> int:
        if not self._ensure_loaded() and not self.allow_fallback:
            raise RuntimeError(f"CLIP is unavailable: {self.last_error}")
        images = project_service.list_images(limit=limit or 1_000_000, offset=0)
        done = 0
        for image in images:
            with project.connect() as conn:
                exists = conn.execute(
                    "SELECT image_id FROM embeddings WHERE image_id=?", (image.id,)
                ).fetchone()
            if exists and not force:
                continue
            vector = self.encode_image(project.root / image.file_name)
            vec_path = project.embeddings_dir / f"{image.id}.npy"
            np.save(vec_path, vector)
            with project.connect() as conn:
                conn.execute(
                    """
                    INSERT INTO embeddings(image_id, model, vector_path)
                    VALUES (?, ?, ?)
                    ON CONFLICT(image_id) DO UPDATE SET
                      model=excluded.model,
                      vector_path=excluded.vector_path,
                      updated_at=CURRENT_TIMESTAMP
                    """,
                    (image.id, self.model_name, vec_path.relative_to(project.meta_dir).as_posix()),
                )
            done += 1
        return done

    def encode_image(self, path: Path) -> np.ndarray:
        if self._ensure_loaded():
            import torch

            image = self.preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model.encode_image(image)
                features = features / features.norm(dim=-1, keepdim=True)
            return features[0].detach().cpu().numpy().astype("float32")
        if not self.allow_fallback:
            raise RuntimeError(f"CLIP is unavailable: {self.last_error}")
        return self._fallback_vector(path.name)

    def encode_text(self, text: str) -> np.ndarray:
        if self._ensure_loaded():
            import torch
            import clip

            tokens = clip.tokenize([text]).to(self.device)
            with torch.no_grad():
                features = self.model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
            return features[0].detach().cpu().numpy().astype("float32")
        if not self.allow_fallback:
            raise RuntimeError(f"CLIP is unavailable: {self.last_error}")
        return self._fallback_vector(text)

    def search(
        self,
        project: ProjectDb,
        images: list[ImageRecord],
        *,
        text: str | None = None,
        image_id: int | None = None,
        limit: int = 24,
    ) -> list[SearchResult]:
        query = self._query_vector(project, text=text, image_id=image_id)
        vectors: list[tuple[ImageRecord, np.ndarray]] = []
        with project.connect() as conn:
            rows = conn.execute("SELECT image_id, vector_path FROM embeddings").fetchall()
        by_id = {img.id: img for img in images}
        for row in rows:
            img = by_id.get(int(row["image_id"]))
            if img is None:
                continue
            vec_path = project.meta_dir / row["vector_path"]
            if vec_path.exists():
                vectors.append((img, np.load(vec_path)))
        if not vectors:
            return []
        results = [
            SearchResult(image=img, score=float(np.dot(query, vec) / (np.linalg.norm(vec) + 1e-8)))
            for img, vec in vectors
            if image_id is None or img.id != image_id
        ]
        return sorted(results, key=lambda r: r.score, reverse=True)[:limit]

    def _query_vector(self, project: ProjectDb, *, text: str | None, image_id: int | None) -> np.ndarray:
        if text:
            return self.encode_text(text)
        if image_id is not None:
            with project.connect() as conn:
                row = conn.execute(
                    "SELECT vector_path FROM embeddings WHERE image_id=?", (image_id,)
                ).fetchone()
            if row is not None:
                return np.load(project.meta_dir / row["vector_path"])
        raise ValueError("Provide text or image_id for CLIP search")

    def _fallback_vector(self, seed: str, dim: int = 512) -> np.ndarray:
        digest = hashlib.sha256(seed.encode("utf-8")).digest()
        repeated = (digest * ((dim // len(digest)) + 1))[:dim]
        vec = np.frombuffer(repeated, dtype=np.uint8).astype("float32") - 127.5
        return vec / (np.linalg.norm(vec) + 1e-8)
