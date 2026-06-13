from __future__ import annotations

import json
import shutil
import csv
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


def export_combined_workspace_coco(request: ExportCocoRequest) -> ExportResponse:
    if not request.root:
        raise ValueError("Export root is required for combined export")
    if not request.folder_splits:
        raise ValueError("No folder splits were provided for combined export")

    root = Path(request.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Export root does not exist: {root}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = root / f"combined_coco_export_{timestamp}"
    images_dir = export_dir / "images"
    masks_dir = export_dir / "masks"
    semantic_dir = export_dir / "semantic_rgb"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    semantic_dir.mkdir(parents=True, exist_ok=True)

    images = []
    categories_by_name: dict[str, int] = {}
    categories = []
    annotations = []
    polygon_sidecar = []
    manifest_rows = []
    split_image_ids: dict[str, list[int]] = {split: [] for split in SPLITS}
    issues = []
    mask_count = 0
    next_image_id = 1
    next_annotation_id = 1

    for folder_split in request.folder_splits:
        project_root = _resolve_export_folder(root, folder_split.folder)
        project = ProjectDb(project_root)
        split_by_image = _split_by_image(ExportCocoRequest(splits=folder_split.splits))
        selected_image_ids = set(split_by_image)
        folder_prefix = folder_split.folder.strip("./").replace("\\", "/")
        folder_prefix = folder_prefix or project_root.name
        folder_slug = _safe_export_name(folder_prefix)

        with project.connect() as conn:
            project_images = [dict(row) for row in conn.execute("SELECT * FROM images ORDER BY id").fetchall()]
            project_categories = [dict(row) for row in conn.execute("SELECT * FROM categories ORDER BY id").fetchall()]
            project_annotations = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM annotations WHERE status='accepted' ORDER BY id"
                ).fetchall()
            ]

        project_category_names = {int(category["id"]): str(category["name"]) for category in project_categories}
        category_map: dict[int, int] = {}

        def combined_category_id(source_category_id: int) -> int:
            if source_category_id <= 0:
                return 0
            name = project_category_names.get(source_category_id)
            if not name:
                return 0
            if name not in categories_by_name:
                categories_by_name[name] = len(categories_by_name) + 1
                categories.append({"id": categories_by_name[name], "name": name})
            category_map[source_category_id] = categories_by_name[name]
            return categories_by_name[name]

        new_image_by_old: dict[int, dict] = {}
        image_copy_by_old: dict[int, str] = {}
        for image in project_images:
            old_image_id = int(image["id"])
            if old_image_id not in selected_image_ids:
                continue
            split = split_by_image[old_image_id]
            src_image = project.root / image["file_name"]
            image_suffix = src_image.suffix or ".jpg"
            image_name = f"{folder_slug}_{old_image_id}{image_suffix}"
            image_dst_dir = images_dir / split
            image_dst_dir.mkdir(parents=True, exist_ok=True)
            image_dst = image_dst_dir / image_name
            if src_image.exists():
                shutil.copy2(src_image, image_dst)
            image_rel = str(image_dst.relative_to(export_dir)).replace("\\", "/")
            new_image = {
                "id": next_image_id,
                "file_name": image_rel,
                "width": int(image["width"]),
                "height": int(image["height"]),
                "source_folder": folder_split.folder,
                "source_image_id": old_image_id,
                "source_file_name": image["file_name"],
                "split": split,
            }
            images.append(new_image)
            split_image_ids[split].append(next_image_id)
            new_image_by_old[old_image_id] = new_image
            image_copy_by_old[old_image_id] = image_rel
            next_image_id += 1

        anns_by_image: dict[int, list[dict]] = {}
        mask_paths_by_image: dict[int, list[str]] = {}
        for ann in project_annotations:
            old_image_id = int(ann["image_id"])
            new_image = new_image_by_old.get(old_image_id)
            if not new_image:
                continue
            mask_abs = project.meta_dir / ann["mask_path"]
            if not mask_abs.exists():
                continue
            mask = load_mask(mask_abs)
            polygons = mask_to_polygons(mask)
            split = str(new_image["split"])
            dst_dir = masks_dir / split
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{folder_slug}_{ann['id']}.png"
            shutil.copy2(mask_abs, dst)
            category_id = int(ann["category_id"]) if ann["category_id"] else 0
            category_id = combined_category_id(category_id)
            mask_rel = str(dst.relative_to(export_dir)).replace("\\", "/")
            new_annotation = {
                "id": next_annotation_id,
                "image_id": int(new_image["id"]),
                "category_id": category_id,
                "segmentation": polygons,
                "bbox": parse_bbox(ann["bbox_json"]),
                "area": int(ann["area"]),
                "iscrowd": int(ann["iscrowd"]),
                "mask_path": mask_rel,
                "version": int(ann["version"]),
                "source_folder": folder_split.folder,
                "source_annotation_id": int(ann["id"]),
                "split": split,
            }
            annotations.append(new_annotation)
            polygon_sidecar.append({"annotation_id": next_annotation_id, "polygons": polygons})
            anns_by_image.setdefault(old_image_id, []).append(ann)
            mask_paths_by_image.setdefault(old_image_id, []).append(mask_rel)
            next_annotation_id += 1
            mask_count += 1

        for old_image_id, new_image in new_image_by_old.items():
            semantic = np.zeros((int(new_image["height"]), int(new_image["width"]), 3), dtype=np.uint8)
            for ann in anns_by_image.get(old_image_id, []):
                mask_abs = project.meta_dir / ann["mask_path"]
                if not mask_abs.exists():
                    continue
                mask = load_mask(mask_abs)
                category_id = int(ann["category_id"]) if ann["category_id"] else 0
                color = _color_for_id(category_map.get(category_id, combined_category_id(category_id)))
                semantic[mask] = color
            split = str(new_image["split"])
            dst_dir = semantic_dir / split
            dst_dir.mkdir(parents=True, exist_ok=True)
            semantic_path = dst_dir / f"{folder_slug}_{new_image['id']}.png"
            Image.fromarray(semantic, mode="RGB").save(semantic_path)
            manifest_rows.append(
                {
                    "image_path": image_copy_by_old[old_image_id],
                    "masks": ";".join(mask_paths_by_image.get(old_image_id, [])),
                    "semantic_rgb": str(semantic_path.relative_to(export_dir)).replace("\\", "/"),
                    "split": split,
                }
            )

        issues.extend(validate_project(project))

    payload = {
        "info": {
            "description": "FoodSegmentation combined workspace export",
            "version": timestamp,
            "date_created": datetime.now().isoformat(timespec="seconds"),
        },
        "images": images,
        "categories": categories,
        "annotations": annotations,
        "splits": {split: sorted(ids) for split, ids in split_image_ids.items()},
    }
    coco_path = export_dir / "annotations_coco.json"
    polygon_path = export_dir / "polygons.json"
    qa_path = export_dir / "qa_report.json"
    manifest_path = export_dir / "manifest.csv"
    labels_path = export_dir / "labels.json"
    coco_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    split_coco_jsons = {}
    for split in SPLITS:
        ids = set(split_image_ids[split])
        split_payload = {
            **payload,
            "images": [image for image in images if int(image["id"]) in ids],
            "annotations": [ann for ann in annotations if int(ann["image_id"]) in ids],
        }
        split_path = export_dir / f"annotations_{split}_coco.json"
        split_path.write_text(json.dumps(split_payload, indent=2), encoding="utf-8")
        split_coco_jsons[split] = str(split_path)
    polygon_path.write_text(json.dumps(polygon_sidecar, indent=2), encoding="utf-8")
    qa_path.write_text(json.dumps([issue.model_dump() for issue in issues], indent=2), encoding="utf-8")
    labels_path.write_text(json.dumps(_labels_payload(categories), indent=2), encoding="utf-8")
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "masks", "semantic_rgb", "split"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    return ExportResponse(
        export_dir=str(export_dir),
        coco_json=str(coco_path),
        mask_count=mask_count,
        issues=issues,
        split_coco_jsons=split_coco_jsons,
    )


def _resolve_export_folder(root: Path, folder: str) -> Path:
    project_root = (root / folder).resolve()
    try:
        project_root.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Export folder escapes root: {folder}") from exc
    if not (project_root / ".foodseg" / "annotations.db").exists():
        raise ValueError(f"Export folder is missing annotations: {folder}")
    return project_root


def _safe_export_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return safe.strip("_") or "folder"


def _labels_payload(categories: list[dict]) -> dict[str, dict]:
    labels = {"0": {"name": "background", "color": [0, 0, 0]}}
    for category in sorted(categories, key=lambda item: int(item["id"])):
        category_id = int(category["id"])
        labels[str(category_id)] = {
            "name": str(category["name"]),
            "color": list(_color_for_id(category_id)),
        }
    return labels


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
    if value <= 0:
        return (0, 0, 0)
    mixed = (value * 2654435761) & 0xFFFFFF
    r = ((mixed >> 16) + 64) % 256
    g = (((mixed >> 8) & 0xFF) + 64) % 256
    b = ((mixed & 0xFF) + 64) % 256
    return (r, g, b)
