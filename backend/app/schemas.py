from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ProjectOpenRequest(BaseModel):
    path: str


class ProjectSummary(BaseModel):
    root: str
    db_path: str
    image_count: int


class ProjectIndexStatus(BaseModel):
    status: Literal["idle", "indexing", "completed", "failed"]
    indexed_count: int
    total_seen: int | None = None
    current_file: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None


class ImageRecord(BaseModel):
    id: int
    file_name: str
    width: int
    height: int
    accepted_object_count: int = 0
    matching_class_count: int | None = None


class ExportWorkspaceImageRecord(BaseModel):
    id: int
    file_name: str
    width: int
    height: int
    annotation_count: int


class ExportWorkspaceFolderRecord(BaseModel):
    folder: str
    image_count: int
    annotated_image_count: int
    annotation_count: int
    images: list[ExportWorkspaceImageRecord]


class ExportWorkspaceResponse(BaseModel):
    root: str
    folders: list[ExportWorkspaceFolderRecord]


class ExportFolderSplit(BaseModel):
    folder: str
    splits: dict[Literal["train", "val", "test"], list[int]]


class ExportCocoRequest(BaseModel):
    root: str | None = None
    combined: bool = False
    splits: dict[Literal["train", "val", "test"], list[int]] | None = None
    folder_splits: list[ExportFolderSplit] = Field(default_factory=list)


class ImagePage(BaseModel):
    items: list[ImageRecord]
    total: int
    limit: int
    offset: int
    has_more: bool


class CategoryRecord(BaseModel):
    id: int
    name: str


class AnnotationRecord(BaseModel):
    id: int
    image_id: int
    category_id: int | None = None
    category_name: str | None = None
    bbox: list[float]
    area: int
    iscrowd: int = 0
    mask_path: str
    version: int
    status: str = "accepted"
    score: float | None = None
    visible: bool = True


class PointPrompt(BaseModel):
    x: float
    y: float
    label: int = Field(ge=0, le=1)


class BoxPrompt(BaseModel):
    x: float
    y: float
    width: float
    height: float
    label: int = 1


class SamPromptRequest(BaseModel):
    image_id: int
    prompt_type: Literal["text", "box", "point"]
    text: str | None = None
    points: list[PointPrompt] = []
    boxes: list[BoxPrompt] = []
    category_name: str | None = None
    confidence_threshold: float = 0.1
    accept: bool = False


class MaskCandidate(BaseModel):
    mask_png: str
    bbox: list[float]
    area: int
    score: float | None = None
    prompt_type: str
    annotation: AnnotationRecord | None = None


class AnnotationCreateRequest(BaseModel):
    image_id: int
    category_name: str
    mask_png: str
    status: Literal["accepted", "pending"] = "accepted"
    score: float | None = None


class AnnotationUpdateRequest(BaseModel):
    category_name: str | None = None
    visible: bool | None = None
    status: Literal["accepted", "pending", "rejected"] | None = None


class MaskReplaceRequest(BaseModel):
    mask_png: str


class ClipIndexRequest(BaseModel):
    force: bool = False
    limit: int | None = None


class ClipStatus(BaseModel):
    available: bool
    indexed: int
    total: int
    model: str
    device: str
    error: str | None = None


class ClipSearchRequest(BaseModel):
    text: str | None = None
    image_id: int | None = None
    limit: int = 24


class SearchResult(BaseModel):
    image: ImageRecord
    score: float


class BulkConceptRequest(BaseModel):
    text: str
    category_name: str | None = None
    limit: int = 50
    confidence_threshold: float = 0.5
    accept: bool = False


class BulkJobCreateRequest(BaseModel):
    mode: Literal["all", "clip_filtered"] = "all"
    text: str
    category_name: str | None = None
    confidence_threshold: float = 0.3
    top_k: int = Field(default=3, ge=1, le=100)
    max_images: int | None = Field(default=None, ge=1)
    clip_image_ids: list[int] = []


class BulkJobRecord(BaseModel):
    id: int
    kind: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    payload: dict
    result: dict
    created_at: str
    updated_at: str


class ReviewCandidateRecord(BaseModel):
    id: int
    job_id: int
    image: ImageRecord
    category_name: str
    prompt_text: str
    mask_path: str
    bbox: list[float]
    area: int
    score: float | None = None
    rank: int
    status: Literal["pending", "accepted", "rejected"]
    annotation_id: int | None = None
    created_at: str


class ReviewCandidateOpenResponse(BaseModel):
    candidate: ReviewCandidateRecord
    mask_png: str


class ReviewCandidateLinkRequest(BaseModel):
    annotation_id: int


class ReviewCandidateUpdateRequest(BaseModel):
    category_name: str


class ReviewCandidatesBulkUpdateRequest(BaseModel):
    category_name: str


class PropagateRequest(BaseModel):
    annotation_id: int
    limit: int = 24
    run_sam: bool = True


class QaIssue(BaseModel):
    severity: Literal["warning", "error"]
    code: str
    message: str
    image_id: int | None = None
    annotation_id: int | None = None


class ExportResponse(BaseModel):
    export_dir: str
    coco_json: str
    mask_count: int
    issues: list[QaIssue]
    split_coco_jsons: dict[str, str] = Field(default_factory=dict)
    folder_exports: list[dict] = Field(default_factory=list)
