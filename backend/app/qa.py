from __future__ import annotations

import itertools

import numpy as np

from .db import ProjectDb
from .masks import load_mask
from .schemas import QaIssue


def validate_project(project: ProjectDb) -> list[QaIssue]:
    issues: list[QaIssue] = []
    with project.connect() as conn:
        images = conn.execute("SELECT id, width, height FROM images").fetchall()
        anns = conn.execute(
            """
            SELECT a.*, c.name AS category_name
            FROM annotations a
            LEFT JOIN categories c ON c.id=a.category_id
            WHERE a.status != 'rejected'
            """
        ).fetchall()
        cat_counts = conn.execute(
            """
            SELECT c.name, COUNT(a.id) AS n
            FROM categories c
            LEFT JOIN annotations a ON a.category_id=c.id AND a.status='accepted'
            GROUP BY c.id
            """
        ).fetchall()
    image_dims = {int(row["id"]): (int(row["width"]), int(row["height"])) for row in images}
    anns_by_image: dict[int, list] = {}
    for ann in anns:
        image_id = int(ann["image_id"])
        width, height = image_dims.get(image_id, (0, 0))
        area = int(ann["area"])
        total = max(1, width * height)
        mask_path = project.meta_dir / ann["mask_path"]
        if not mask_path.exists():
            issues.append(QaIssue(severity="error", code="missing_mask", message="Mask file is missing", image_id=image_id, annotation_id=int(ann["id"])))
        if area <= 0:
            issues.append(QaIssue(severity="error", code="empty_mask", message="Mask has zero area", image_id=image_id, annotation_id=int(ann["id"])))
        if area / total < 0.001:
            issues.append(QaIssue(severity="warning", code="tiny_mask", message="Mask covers less than 0.1% of the image", image_id=image_id, annotation_id=int(ann["id"])))
        if area / total > 0.9:
            issues.append(QaIssue(severity="warning", code="large_mask", message="Mask covers more than 90% of the image", image_id=image_id, annotation_id=int(ann["id"])))
        anns_by_image.setdefault(image_id, []).append(ann)
    for image_id, image_anns in anns_by_image.items():
        for a, b in itertools.combinations(image_anns, 2):
            path_a = project.meta_dir / a["mask_path"]
            path_b = project.meta_dir / b["mask_path"]
            if not path_a.exists() or not path_b.exists():
                continue
            ma = load_mask(path_a)
            mb = load_mask(path_b)
            inter = int(np.logical_and(ma, mb).sum())
            smaller = max(1, min(int(a["area"]), int(b["area"])))
            if inter / smaller > 0.8:
                issues.append(QaIssue(severity="warning", code="overlap", message="Two masks overlap by more than 80% of the smaller mask", image_id=image_id, annotation_id=int(a["id"])))
    for row in cat_counts:
        if int(row["n"]) == 0:
            issues.append(QaIssue(severity="warning", code="missing_category", message=f"Category has no accepted annotations: {row['name']}"))
    return issues
