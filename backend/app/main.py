from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .annotations import AnnotationService
from .bulk_service import BulkJobService
from .clip_service import ClipService
from .exporter import export_coco, export_workspace_coco, scan_export_workspace as scan_export_workspace_helper
from .project_service import ProjectService
from .qa import validate_project
from .schemas import (
    AnnotationCreateRequest,
    AnnotationUpdateRequest,
    BulkJobCreateRequest,
    BulkConceptRequest,
    ClipIndexRequest,
    ExportCocoRequest,
    ClipSearchRequest,
    MaskReplaceRequest,
    ProjectOpenRequest,
    PropagateRequest,
    ReviewCandidateOpenResponse,
    ReviewCandidateLinkRequest,
    ReviewCandidatesBulkUpdateRequest,
    ReviewCandidateUpdateRequest,
    SamPromptRequest,
    ExportWorkspaceResponse,
)
from .sam_service import SamService

REPO_ROOT = Path(__file__).resolve().parents[2]

app = FastAPI(title="FoodSegmentation Local Annotation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

projects = ProjectService()
annotations = AnnotationService()
sam = SamService(REPO_ROOT)
clip = ClipService(REPO_ROOT)
bulk_jobs = BulkJobService(annotations, sam, clip)


def project_or_400():
    try:
        return projects.require()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "sam_loaded": sam.processor is not None,
        "sam_device": sam.device,
        "sam_error": sam.last_error,
        "sam_load_notes": sam.load_notes,
        "sam_fake_fallback": sam.allow_fallback,
        "clip_error": clip.last_error,
    }


@app.post("/api/projects/open")
def open_project(request: ProjectOpenRequest):
    try:
        return projects.open_project(request.path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/export/workspace/scan")
def scan_export_workspace(request: ProjectOpenRequest):
    try:
        return scan_export_workspace_helper(Path(request.path))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/projects/current")
def current_project():
    return projects.summary()


@app.get("/api/projects/index-status")
def project_index_status():
    project_or_400()
    return projects.index_status()


@app.get("/api/images")
def list_images(
    limit: int = Query(100, le=1000),
    offset: int = 0,
    q: str | None = None,
    mask_filter: Literal["all", "with_masks", "without_masks"] = "all",
    class_name: str | None = None,
    count_op: Literal["lt", "lte", "eq", "gte", "gt"] | None = None,
    count_value: int | None = Query(default=None, ge=0),
):
    project_or_400()
    return projects.list_images_page(limit=limit, offset=offset, q=q, mask_filter=mask_filter, class_name=class_name, count_op=count_op, count_value=count_value)


@app.get("/api/images/{image_id}")
def get_image(image_id: int):
    try:
        return projects.get_image(image_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/images/{image_id}/file")
def image_file(image_id: int):
    try:
        path = projects.image_path(image_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path, headers={"Cache-Control": "no-store"})


@app.get("/api/images/{image_id}/annotations")
def list_annotations(image_id: int):
    return annotations.list_for_image(project_or_400(), image_id)


@app.post("/api/annotations")
def create_annotation(request: AnnotationCreateRequest):
    return annotations.create_from_png(
        project_or_400(),
        request.image_id,
        request.category_name,
        request.mask_png,
        status=request.status,
        score=request.score,
    )


@app.patch("/api/annotations/{annotation_id}")
def update_annotation(annotation_id: int, request: AnnotationUpdateRequest):
    try:
        return annotations.update(
            project_or_400(),
            annotation_id,
            category_name=request.category_name,
            visible=request.visible,
            status=request.status,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/annotations/{annotation_id}")
def delete_annotation(annotation_id: int):
    try:
        annotations.delete(project_or_400(), annotation_id)
        return {"deleted": annotation_id}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/annotations/{annotation_id}/mask")
def annotation_mask(annotation_id: int):
    project = project_or_400()
    try:
        ann = annotations.get(project, annotation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    path = project.meta_dir / ann.mask_path
    if not path.exists():
        raise HTTPException(status_code=404, detail="Mask file is missing")
    return FileResponse(path, headers={"Cache-Control": "no-store"})


@app.post("/api/annotations/{annotation_id}/mask")
def replace_mask(annotation_id: int, request: MaskReplaceRequest):
    try:
        return annotations.replace_mask(project_or_400(), annotation_id, request.mask_png)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/annotations/{annotation_id}/undo")
def undo_annotation(annotation_id: int):
    return annotations.undo(project_or_400(), annotation_id)


@app.post("/api/annotations/{annotation_id}/redo")
def redo_annotation(annotation_id: int):
    return annotations.redo(project_or_400(), annotation_id)


@app.post("/api/sam/prompt")
def sam_prompt(request: SamPromptRequest):
    project = project_or_400()
    image = projects.get_image(request.image_id)
    candidates = sam.predict(projects.image_path(request.image_id), request, image.width, image.height)
    accepted = []
    if request.accept:
        category = request.category_name or request.text or "food"
        for candidate in candidates:
            ann = annotations.create_from_png(
                project,
                request.image_id,
                category,
                candidate.mask_png,
                status="accepted",
                score=candidate.score,
            )
            candidate.annotation = ann
            accepted.append(ann.id)
    annotations.log_prompt(project, request.image_id, request.prompt_type, request.model_dump(), accepted[0] if accepted else None)
    return {"candidates": candidates, "sam_error": sam.last_error}


@app.post("/api/clip/index")
def clip_index(request: ClipIndexRequest):
    project = project_or_400()
    try:
        indexed = clip.index(project, projects, force=request.force, limit=request.limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    status = clip.status(project, projects)
    return {"indexed": indexed, **status}


@app.get("/api/clip/status")
def clip_status():
    project = project_or_400()
    return clip.status(project, projects)


@app.post("/api/clip/search")
def clip_search(request: ClipSearchRequest):
    project = project_or_400()
    images = projects.list_images(limit=1_000_000)
    try:
        return clip.search(project, images, text=request.text, image_id=request.image_id, limit=request.limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/bulk/jobs")
def create_bulk_job(request: BulkJobCreateRequest):
    project = project_or_400()
    try:
        return bulk_jobs.create_job(project, projects, request)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/bulk/jobs")
def list_bulk_jobs(limit: int = Query(20, le=100)):
    return bulk_jobs.list_jobs(project_or_400(), limit=limit)


@app.delete("/api/bulk/jobs/finished")
def delete_finished_bulk_jobs():
    return bulk_jobs.delete_finished_jobs(project_or_400())


@app.get("/api/bulk/jobs/{job_id}")
def get_bulk_job(job_id: int):
    try:
        return bulk_jobs.get_job(project_or_400(), job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/bulk/jobs/{job_id}")
def delete_bulk_job(job_id: int):
    try:
        return bulk_jobs.delete_job(project_or_400(), job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/bulk/jobs/{job_id}/cancel")
def cancel_bulk_job(job_id: int):
    try:
        return bulk_jobs.cancel_job(project_or_400(), job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/bulk/jobs/{job_id}/candidates")
def list_bulk_candidates(job_id: int, status: str = "pending"):
    try:
        return bulk_jobs.list_candidates(project_or_400(), projects, job_id, status=status)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.patch("/api/bulk/jobs/{job_id}/candidates")
def update_bulk_job_candidates(job_id: int, request: ReviewCandidatesBulkUpdateRequest):
    try:
        return bulk_jobs.rename_pending_candidates(project_or_400(), job_id, request.category_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/bulk/jobs/{job_id}/candidates/accept")
def accept_bulk_job_candidates(job_id: int):
    try:
        return bulk_jobs.accept_pending_candidates(project_or_400(), job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/bulk/candidates/{candidate_id}/mask")
def bulk_candidate_mask(candidate_id: int):
    project = project_or_400()
    try:
        candidate = bulk_jobs.get_candidate(project, candidate_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    path = project.meta_dir / candidate.mask_path
    if not path.exists():
        raise HTTPException(status_code=404, detail="Mask file is missing")
    return FileResponse(path, headers={"Cache-Control": "no-store"})


@app.get("/api/bulk/candidates/{candidate_id}/open")
def open_bulk_candidate(candidate_id: int):
    project = project_or_400()
    try:
        candidate = bulk_jobs.get_candidate(project, candidate_id)
        mask_png = bulk_jobs.candidate_mask_png(project, candidate_id)
        return ReviewCandidateOpenResponse(candidate=candidate, mask_png=mask_png)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.patch("/api/bulk/candidates/{candidate_id}")
def update_bulk_candidate(candidate_id: int, request: ReviewCandidateUpdateRequest):
    try:
        return bulk_jobs.update_candidate(project_or_400(), candidate_id, request.category_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/bulk/candidates/{candidate_id}/accept")
def accept_bulk_candidate(candidate_id: int):
    try:
        return bulk_jobs.accept_candidate(project_or_400(), candidate_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/bulk/candidates/{candidate_id}/link-accepted")
def link_accepted_bulk_candidate(candidate_id: int, request: ReviewCandidateLinkRequest):
    try:
        return bulk_jobs.link_accepted_annotation(project_or_400(), candidate_id, request.annotation_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/bulk/candidates/{candidate_id}/reject")
def reject_bulk_candidate(candidate_id: int):
    try:
        return bulk_jobs.reject_candidate(project_or_400(), candidate_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/bulk/candidates/{candidate_id}/reopen")
def reopen_bulk_candidate(candidate_id: int):
    try:
        return bulk_jobs.reopen_candidate(project_or_400(), candidate_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/bulk/concept")
def bulk_concept(request: BulkConceptRequest):
    project = project_or_400()
    clip.index(project, projects, force=False)
    matches = clip.search(
        project,
        projects.list_images(limit=1_000_000),
        text=request.text,
        limit=request.limit,
    )
    created = []
    for result in matches:
        prompt = SamPromptRequest(
            image_id=result.image.id,
            prompt_type="text",
            text=request.text,
            category_name=request.category_name or request.text,
            confidence_threshold=request.confidence_threshold,
        )
        candidates = sam.predict(projects.image_path(result.image.id), prompt, result.image.width, result.image.height)
        for candidate in candidates[:1]:
            ann = annotations.create_from_png(
                project,
                result.image.id,
                request.category_name or request.text,
                candidate.mask_png,
                status="accepted" if request.accept else "pending",
                score=candidate.score,
            )
            created.append(ann)
    return {"matches": matches, "annotations": created, "sam_error": sam.last_error, "clip_error": clip.last_error}


@app.post("/api/propagate")
def propagate(request: PropagateRequest):
    project = project_or_400()
    source = annotations.get(project, request.annotation_id)
    clip.index(project, projects, force=False)
    matches = clip.search(
        project,
        projects.list_images(limit=1_000_000),
        image_id=source.image_id,
        limit=request.limit,
    )
    created = []
    if request.run_sam:
        prompt_text = source.category_name or "food"
        for result in matches:
            prompt = SamPromptRequest(
                image_id=result.image.id,
                prompt_type="text",
                text=prompt_text,
                category_name=prompt_text,
            )
            candidates = sam.predict(projects.image_path(result.image.id), prompt, result.image.width, result.image.height)
            if candidates:
                created.append(
                    annotations.create_from_png(
                        project,
                        result.image.id,
                        prompt_text,
                        candidates[0].mask_png,
                        status="pending",
                        score=candidates[0].score,
                    )
                )
    return {"matches": matches, "annotations": created}


@app.get("/api/qa/validate")
def qa_validate():
    return validate_project(project_or_400())


@app.post("/api/export/coco")
def export(request: ExportCocoRequest | None = Body(default=None)):
    if request and request.root and request.folder_splits:
        try:
            return export_workspace_coco(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return export_coco(project_or_400(), request)
