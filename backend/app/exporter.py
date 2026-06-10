from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .db import ProjectDb, parse_bbox
from .masks import load_mask
from .qa import validate_project
from .schemas import ExportCocoRequest, ExportResponse, ExportWorkspaceFolderRecord, ExportWorkspaceImageRecord, ExportWorkspaceResponse


def mask_to_polygons(mask: np.ndarray) -> list[list[float]]:
    contours, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[float]] = []
    for contour in contours:
        if len(contour) < 3:
            continue
        flattened = contour.reshape(-1, 2).astype(float).flatten().tolist()
        if len(flattened) >= 6:
            polygons.append(flattened)
    return polygons


SPLITS = ("train", "val", "test")


def export_coco(project: ProjectDb, request: ExportCocoRequest | None = None) -> ExportResponse:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = project.exports_dir / timestamp
    masks_dir = export_dir / "masks"
    semantic_dir = export_dir / "semantic_rgb"
    masks_dir.mkdir(parents=True, exist_ok=True)
    semantic_dir.mkdir(parents=True, exist_ok=True)
    split_by_image = _split_by_image(request)
    split_coco_jsons: dict[str, str] = {}
    with project.connect() as conn:
        images = [dict(row) for row in conn.execute("SELECT * FROM images ORDER BY id").fetchall()]
        categories = [dict(row) for row in conn.execute("SELECT * FROM categories ORDER BY id").fetchall()]
        annotations = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM annotations WHERE status='accepted' ORDER BY id"
            ).fetchall()
        ]

    selected_image_ids = set(split_by_image) if split_by_image else None
    images = [
        {**image, **({"split": split_by_image[int(image["id"])]} if int(image["id"]) in split_by_image else {})}
        for image in images
        if selected_image_ids is None or int(image["id"]) in selected_image_ids
    ]
    image_ids = {int(image["id"]) for image in images}

    coco_annotations = []
    polygon_sidecar = []
    anns_by_image: dict[int, list[dict]] = {}
    copied = 0
    for ann in annotations:
        image_id = int(ann["image_id"])
        if image_id not in image_ids:
            continue
        mask_abs = project.meta_dir / ann["mask_path"]
        if not mask_abs.exists():
            continue
        mask = load_mask(mask_abs)
        polygons = mask_to_polygons(mask)
        split = split_by_image.get(image_id)
        dst_dir = masks_dir / split if split else masks_dir
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{ann['id']}.png"
        shutil.copy2(mask_abs, dst)
        copied += 1
        coco_annotation = {
            "id": int(ann["id"]),
            "image_id": image_id,
            "category_id": int(ann["category_id"]) if ann["category_id"] else 0,
            "segmentation": polygons,
            "bbox": parse_bbox(ann["bbox_json"]),
            "area": int(ann["area"]),
            "iscrowd": int(ann["iscrowd"]),
            "mask_path": str(dst.relative_to(export_dir)).replace("\\", "/"),
            "version": int(ann["version"]),
        }
        if split:
            coco_annotation["split"] = split
        coco_annotations.append(coco_annotation)
        polygon_sidecar.append({"annotation_id": int(ann["id"]), "polygons": polygons})
        anns_by_image.setdefault(image_id, []).append(ann)

    for image in images:
        image_id = int(image["id"])
        semantic = np.zeros((int(image["height"]), int(image["width"]), 3), dtype=np.uint8)
        for ann in anns_by_image.get(image_id, []):
            mask_abs = project.meta_dir / ann["mask_path"]
            if not mask_abs.exists():
                continue
            mask = load_mask(mask_abs)
            color = _color_for_id(int(ann["category_id"] or ann["id"]))
            semantic[mask] = color
        split = split_by_image.get(image_id)
        dst_dir = semantic_dir / split if split else semantic_dir
        dst_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(semantic, mode="RGB").save(dst_dir / f"{image_id}.png")

    payload = {
        "info": {
            "description": "FoodSegmentation local export",
            "version": timestamp,
            "date_created": datetime.now().isoformat(timespec="seconds"),
        },
        "images": images,
        "categories": categories,
        "annotations": coco_annotations,
    }
    if split_by_image:
        payload["splits"] = {split: sorted(image_id for image_id, assigned in split_by_image.items() if assigned == split) for split in SPLITS}
    coco_path = export_dir / "annotations_coco.json"
    polygon_path = export_dir / "polygons.json"
    qa_path = export_dir / "qa_report.json"
    issues = validate_project(project)
    coco_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if split_by_image:
        for split in SPLITS:
            split_image_ids = {image_id for image_id, assigned in split_by_image.items() if assigned == split}
            split_payload = {
                **payload,
                "images": [image for image in images if int(image["id"]) in split_image_ids],
                "annotations": [ann for ann in coco_annotations if int(ann["image_id"]) in split_image_ids],
            }
            split_path = export_dir / f"annotations_{split}_coco.json"
            split_path.write_text(json.dumps(split_payload, indent=2), encoding="utf-8")
            split_coco_jsons[split] = str(split_path)
    polygon_path.write_text(json.dumps(polygon_sidecar, indent=2), encoding="utf-8")
    qa_path.write_text(json.dumps([issue.model_dump() for issue in issues], indent=2), encoding="utf-8")
    return ExportResponse(
        export_dir=str(export_dir),
        coco_json=str(coco_path),
        mask_count=copied,
        issues=issues,
        split_coco_jsons=split_coco_jsons,
    )


def export_workspace_coco(request: ExportCocoRequest) -> ExportResponse:
    if not request.root:
        raise ValueError("Export root is required for workspace export")
    if not request.folder_splits:
        raise ValueError("No folder splits were provided for workspace export")

    root = Path(request.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Export root does not exist: {root}")

    folder_exports = []
    issues = []
    split_coco_jsons: dict[str, str] = {}
    mask_count = 0
    first_coco_json = ""

    for folder_split in request.folder_splits:
        project_root = (root / folder_split.folder).resolve()
        try:
            project_root.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Export folder escapes root: {folder_split.folder}") from exc
        if not (project_root / ".foodseg" / "annotations.db").exists():
            raise ValueError(f"Export folder is missing annotations: {folder_split.folder}")

        exported = export_coco(ProjectDb(project_root), ExportCocoRequest(splits=folder_split.splits))
        if not first_coco_json:
            first_coco_json = exported.coco_json
        mask_count += exported.mask_count
        issues.extend(exported.issues)
        for split, path in exported.split_coco_jsons.items():
            split_coco_jsons[f"{folder_split.folder}/{split}"] = path
        folder_exports.append(
            {
                "folder": folder_split.folder,
                "export_dir": exported.export_dir,
                "coco_json": exported.coco_json,
                "mask_count": exported.mask_count,
                "split_coco_jsons": exported.split_coco_jsons,
            }
        )

    return ExportResponse(
        export_dir=str(root),
        coco_json=first_coco_json,
        mask_count=mask_count,
        issues=issues,
        split_coco_jsons=split_coco_jsons,
        folder_exports=folder_exports,
    )


def _split_by_image(request: ExportCocoRequest | None) -> dict[int, str]:
    if request is None or not request.splits:
        return {}
    split_by_image: dict[int, str] = {}
    for split in SPLITS:
        for image_id in request.splits.get(split, []):
            split_by_image[int(image_id)] = split
    return split_by_image


def scan_export_workspace(root: Path) -> ExportWorkspaceResponse:
    resolved_root = root.expanduser().resolve()
    if not resolved_root.exists() or not resolved_root.is_dir():
        raise ValueError(f"Export root does not exist: {resolved_root}")

    folders: list[ExportWorkspaceFolderRecord] = []
    seen: set[Path] = set()
    for meta_dir in resolved_root.rglob(".foodseg"):
        if not meta_dir.is_dir():
            continue
        folder = meta_dir.parent.resolve()
        if folder in seen:
            continue
        seen.add(folder)
        db_path = meta_dir / "annotations.db"
        if not db_path.exists():
            continue
        project = ProjectDb(folder)
        try:
            with project.connect() as conn:
                rows = conn.execute(
                    """
                    SELECT i.id, i.file_name, i.width, i.height, COUNT(a.id) AS annotation_count
                    FROM images i
                    LEFT JOIN annotations a ON a.image_id = i.id AND a.status='accepted'
                    GROUP BY i.id, i.file_name, i.width, i.height
                    HAVING annotation_count > 0
                    ORDER BY i.file_name
                    """
                ).fetchall()
        except Exception:
            continue

        images = [
            ExportWorkspaceImageRecord(
                id=int(row["id"]),
                file_name=row["file_name"],
                width=int(row["width"]),
                height=int(row["height"]),
                annotation_count=int(row["annotation_count"]),
            )
            for row in rows
        ]
        if not images:
            continue
        folders.append(
            ExportWorkspaceFolderRecord(
                folder=str(folder.relative_to(resolved_root)).replace("\\", "/"),
                image_count=len({image.file_name for image in images}),
                annotated_image_count=len(images),
                annotation_count=sum(image.annotation_count for image in images),
                images=images,
            )
        )

    folders.sort(key=lambda item: item.folder)
    return ExportWorkspaceResponse(root=str(resolved_root), folders=folders)


def _color_for_id(value: int) -> tuple[int, int, int]:
    palette = [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 190),
    ]
    return palette[value % len(palette)]
