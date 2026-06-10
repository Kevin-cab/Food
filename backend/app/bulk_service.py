from __future__ import annotations

import base64
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .annotations import AnnotationService
from .db import ProjectDb, bbox_json, parse_bbox
from .masks import mask_from_png_data, mask_to_png_data, save_mask
from .project_service import ProjectService
from .schemas import (
    BulkJobCreateRequest,
    BulkJobRecord,
    ImageRecord,
    ReviewCandidateRecord,
    SamPromptRequest,
)


def _job_from_row(row) -> BulkJobRecord:
    return BulkJobRecord(
        id=int(row["id"]),
        kind=row["kind"],
        status=row["status"],
        payload=json.loads(row["payload_json"]),
        result=json.loads(row["result_json"] or "{}"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class BulkJobService:
    def __init__(self, annotations: AnnotationService, sam, clip) -> None:
        self.annotations = annotations
        self.sam = sam
        self.clip = clip
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.lock = threading.Lock()

    def create_job(self, project: ProjectDb, projects: ProjectService, request: BulkJobCreateRequest) -> BulkJobRecord:
        payload = request.model_dump()
        images = self._select_images(project, projects, request)
        payload["image_ids"] = [image.id for image in images]
        result = {
            "processed": 0,
            "total": len(images),
            "created": 0,
            "error": None,
        }
        with project.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO jobs(kind, status, payload_json, result_json)
                VALUES ('bulk_sam', 'queued', ?, ?)
                """,
                (json.dumps(payload), json.dumps(result)),
            )
            job_id = int(cur.lastrowid)
        job = self.get_job(project, job_id)
        self.executor.submit(self._run_job, project.root, job_id)
        return job

    def list_jobs(self, project: ProjectDb, limit: int = 20) -> list[BulkJobRecord]:
        with project.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM jobs
                WHERE kind='bulk_sam'
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [_job_from_row(row) for row in rows]

    def get_job(self, project: ProjectDb, job_id: int) -> BulkJobRecord:
        with project.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"Job not found: {job_id}")
        return _job_from_row(row)

    def delete_job(self, project: ProjectDb, job_id: int) -> dict[str, int]:
        job = self.get_job(project, job_id)
        if job.status in {"queued", "running"}:
            raise RuntimeError("Cannot delete a queued or running bulk job.")
        self._delete_job_files(project, job_id)
        with project.connect() as conn:
            conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        return {"deleted": job_id}

    def delete_finished_jobs(self, project: ProjectDb) -> dict[str, list[int]]:
        with project.connect() as conn:
            rows = conn.execute(
                """
                SELECT id FROM jobs
                WHERE kind='bulk_sam' AND status IN ('completed', 'failed', 'cancelled')
                ORDER BY id DESC
                """
            ).fetchall()
        deleted: list[int] = []
        for row in rows:
            job_id = int(row["id"])
            self._delete_job_files(project, job_id)
            with project.connect() as conn:
                conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
            deleted.append(job_id)
        return {"deleted": deleted}

    def cancel_job(self, project: ProjectDb, job_id: int) -> BulkJobRecord:
        job = self.get_job(project, job_id)
        if job.status == "queued":
            self._set_job(project, job_id, "cancelled", cancel_requested=True, error=None)
            return self.get_job(project, job_id)
        if job.status == "running":
            self._set_job(project, job_id, "running", cancel_requested=True)
            return self.get_job(project, job_id)
        return job

    def list_candidates(self, project: ProjectDb, projects: ProjectService, job_id: int, status: str = "pending") -> list[ReviewCandidateRecord]:
        with project.connect() as conn:
            rows = conn.execute(
                """
                SELECT rc.*, i.file_name, i.width, i.height
                FROM review_candidates rc
                JOIN images i ON i.id=rc.image_id
                WHERE rc.job_id=? AND (?='all' OR rc.status=?)
                ORDER BY rc.image_id, rc.rank, rc.id
                """,
                (job_id, status, status),
            ).fetchall()
        return [self._candidate_from_row(row) for row in rows]

    def get_candidate(self, project: ProjectDb, candidate_id: int) -> ReviewCandidateRecord:
        with project.connect() as conn:
            row = conn.execute(
                """
                SELECT rc.*, i.file_name, i.width, i.height
                FROM review_candidates rc
                JOIN images i ON i.id=rc.image_id
                WHERE rc.id=?
                """,
                (candidate_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Review candidate not found: {candidate_id}")
        return self._candidate_from_row(row)

    def update_candidate(self, project: ProjectDb, candidate_id: int, category_name: str) -> ReviewCandidateRecord:
        clean = category_name.strip() or "food"
        self.get_candidate(project, candidate_id)
        with project.connect() as conn:
            conn.execute(
                "UPDATE review_candidates SET category_name=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (clean, candidate_id),
            )
        return self.get_candidate(project, candidate_id)

    def rename_pending_candidates(self, project: ProjectDb, job_id: int, category_name: str) -> dict[str, object]:
        self.get_job(project, job_id)
        clean = category_name.strip() or "food"
        with project.connect() as conn:
            rows = conn.execute(
                """
                SELECT id FROM review_candidates
                WHERE job_id=? AND status='pending'
                ORDER BY image_id, rank, id
                """,
                (job_id,),
            ).fetchall()
            ids = [int(row["id"]) for row in rows]
            conn.execute(
                """
                UPDATE review_candidates
                SET category_name=?, updated_at=CURRENT_TIMESTAMP
                WHERE job_id=? AND status='pending'
                """,
                (clean, job_id),
            )
        return {"updated": len(ids), "ids": ids, "category_name": clean}

    def candidate_mask_png(self, project: ProjectDb, candidate_id: int) -> str:
        candidate = self.get_candidate(project, candidate_id)
        path = project.meta_dir / candidate.mask_path
        raw = path.read_bytes()
        return f"data:image/png;base64,{base64.b64encode(raw).decode('ascii')}"

    def accept_candidate(self, project: ProjectDb, candidate_id: int) -> ReviewCandidateRecord:
        candidate = self.get_candidate(project, candidate_id)
        if candidate.status == "accepted":
            return candidate
        mask_png = self.candidate_mask_png(project, candidate_id)
        ann = self.annotations.create_from_png(
            project,
            candidate.image.id,
            candidate.category_name,
            mask_png,
            status="accepted",
            score=candidate.score,
        )
        with project.connect() as conn:
            conn.execute(
                """
                UPDATE review_candidates
                SET status='accepted', annotation_id=?, updated_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (ann.id, candidate_id),
            )
        return self.get_candidate(project, candidate_id)

    def accept_pending_candidates(self, project: ProjectDb, job_id: int) -> dict[str, object]:
        self.get_job(project, job_id)
        with project.connect() as conn:
            rows = conn.execute(
                """
                SELECT id FROM review_candidates
                WHERE job_id=? AND status='pending'
                ORDER BY image_id, rank, id
                """,
                (job_id,),
            ).fetchall()
        accepted_ids: list[int] = []
        annotation_ids: list[int] = []
        for row in rows:
            candidate_id = int(row["id"])
            accepted = self.accept_candidate(project, candidate_id)
            accepted_ids.append(candidate_id)
            if accepted.annotation_id is not None:
                annotation_ids.append(int(accepted.annotation_id))
        return {
            "accepted": len(accepted_ids),
            "ids": accepted_ids,
            "annotation_ids": annotation_ids,
        }

    def link_accepted_annotation(self, project: ProjectDb, candidate_id: int, annotation_id: int) -> ReviewCandidateRecord:
        self.get_candidate(project, candidate_id)
        with project.connect() as conn:
            conn.execute(
                """
                UPDATE review_candidates
                SET status='accepted', annotation_id=?, updated_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (annotation_id, candidate_id),
            )
        return self.get_candidate(project, candidate_id)

    def reject_candidate(self, project: ProjectDb, candidate_id: int) -> ReviewCandidateRecord:
        self.get_candidate(project, candidate_id)
        with project.connect() as conn:
            conn.execute(
                "UPDATE review_candidates SET status='rejected', updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (candidate_id,),
            )
        return self.get_candidate(project, candidate_id)

    def reopen_candidate(self, project: ProjectDb, candidate_id: int) -> ReviewCandidateRecord:
        self.get_candidate(project, candidate_id)
        with project.connect() as conn:
            conn.execute(
                """
                UPDATE review_candidates
                SET status='pending', annotation_id=NULL, updated_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (candidate_id,),
            )
        return self.get_candidate(project, candidate_id)

    def _select_images(self, project: ProjectDb, projects: ProjectService, request: BulkJobCreateRequest) -> list[ImageRecord]:
        images = projects.list_images(limit=1_000_000)
        if request.mode == "clip_filtered":
            if request.clip_image_ids:
                wanted = set(request.clip_image_ids)
                images = [image for image in images if image.id in wanted]
            else:
                matches = self.clip.search(project, images, text=request.text, limit=request.max_images or 50)
                images = [match.image for match in matches]
        if request.max_images is not None:
            images = images[: request.max_images]
        return images

    def _run_job(self, project_root: Path, job_id: int) -> None:
        project = ProjectDb(project_root)
        project.init()
        projects = ProjectService()
        projects.project = project
        try:
            job = self.get_job(project, job_id)
            if job.status == "cancelled":
                return
            payload = job.payload
            image_ids = payload.get("image_ids", [])
            self._set_job(project, job_id, "running", processed=0, total=len(image_ids), created=0, error=None)
            created = 0
            for index, image_id in enumerate(image_ids, start=1):
                if self._cancel_requested(project, job_id):
                    current = self.get_job(project, job_id).result
                    self._set_job(
                        project,
                        job_id,
                        "cancelled",
                        processed=int(current.get("processed", index - 1)),
                        total=len(image_ids),
                        created=int(current.get("created", created)),
                        cancel_requested=True,
                        error=None,
                    )
                    return
                image = projects.get_image(int(image_id))
                prompt_text = payload["text"]
                prompt = SamPromptRequest(
                    image_id=image.id,
                    prompt_type="text",
                    text=prompt_text,
                    category_name=payload.get("category_name") or prompt_text,
                    confidence_threshold=float(payload.get("confidence_threshold", 0.3)),
                )
                candidates = self.sam.predict(projects.image_path(image.id), prompt, image.width, image.height)
                filtered = [
                    candidate
                    for candidate in candidates
                    if candidate.score is None or candidate.score >= float(payload.get("confidence_threshold", 0.3))
                ][: int(payload.get("top_k", 3))]
                for rank, candidate in enumerate(filtered, start=1):
                    self._save_candidate(project, job_id, image.id, payload, candidate, rank)
                    created += 1
                self._set_job(project, job_id, "running", processed=index, total=len(image_ids), created=created, error=None)
            final_status = "cancelled" if self._cancel_requested(project, job_id) else "completed"
            self._set_job(project, job_id, final_status, processed=len(image_ids), total=len(image_ids), created=created, error=None)
        except Exception as exc:
            current = self.get_job(project, job_id).result
            self._set_job(
                project,
                job_id,
                "failed",
                processed=int(current.get("processed", 0)),
                total=int(current.get("total", 0)),
                created=int(current.get("created", 0)),
                error=str(exc),
            )

    def _save_candidate(self, project: ProjectDb, job_id: int, image_id: int, payload: dict, candidate, rank: int) -> None:
        mask = mask_from_png_data(candidate.mask_png)
        rel_mask = f"review_candidates/{job_id}/{image_id}_{rank}.png"
        save_mask(mask, project.meta_dir / rel_mask)
        with project.connect() as conn:
            conn.execute(
                """
                INSERT INTO review_candidates(
                  job_id, image_id, category_name, prompt_text, mask_path,
                  bbox_json, area, score, rank, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                """,
                (
                    job_id,
                    image_id,
                    payload.get("category_name") or payload["text"],
                    payload["text"],
                    rel_mask,
                    bbox_json(candidate.bbox),
                    candidate.area,
                    candidate.score,
                    rank,
                ),
            )

    def _set_job(self, project: ProjectDb, job_id: int, status: str, **result) -> None:
        with project.connect() as conn:
            row = conn.execute("SELECT result_json FROM jobs WHERE id=?", (job_id,)).fetchone()
            merged = json.loads(row["result_json"] or "{}") if row is not None else {}
            merged.update(result)
            conn.execute(
                """
                UPDATE jobs
                SET status=?, result_json=?, updated_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (status, json.dumps(merged), job_id),
            )

    def _cancel_requested(self, project: ProjectDb, job_id: int) -> bool:
        job = self.get_job(project, job_id)
        return job.status == "cancelled" or bool(job.result.get("cancel_requested"))

    def _delete_job_files(self, project: ProjectDb, job_id: int) -> None:
        with project.connect() as conn:
            rows = conn.execute("SELECT mask_path FROM review_candidates WHERE job_id=?", (job_id,)).fetchall()
        for row in rows:
            mask_path = project.meta_dir / row["mask_path"]
            try:
                if mask_path.exists() and mask_path.is_file():
                    mask_path.unlink()
            except OSError:
                pass
        job_dir = project.review_dir / str(job_id)
        try:
            if job_dir.exists():
                shutil.rmtree(job_dir)
        except OSError:
            pass

    def _candidate_from_row(self, row) -> ReviewCandidateRecord:
        return ReviewCandidateRecord(
            id=int(row["id"]),
            job_id=int(row["job_id"]),
            image=ImageRecord(
                id=int(row["image_id"]),
                file_name=row["file_name"],
                width=int(row["width"]),
                height=int(row["height"]),
            ),
            category_name=row["category_name"],
            prompt_text=row["prompt_text"],
            mask_path=row["mask_path"],
            bbox=parse_bbox(row["bbox_json"]),
            area=int(row["area"]),
            score=row["score"],
            rank=int(row["rank"]),
            status=row["status"],
            annotation_id=row["annotation_id"],
            created_at=row["created_at"],
        )
