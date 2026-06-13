from __future__ import annotations

import json
from pathlib import Path

from .db import ProjectDb, bbox_json, get_or_create_category, parse_bbox
from .masks import bbox_and_area, copy_mask, load_mask, mask_from_png_data, save_mask
from .schemas import AnnotationRecord


def annotation_from_row(row) -> AnnotationRecord:
    return AnnotationRecord(
        id=int(row["id"]),
        image_id=int(row["image_id"]),
        category_id=row["category_id"],
        category_name=row["category_name"],
        bbox=parse_bbox(row["bbox_json"]),
        area=int(row["area"]),
        iscrowd=int(row["iscrowd"]),
        mask_path=row["mask_path"],
        version=int(row["version"]),
        status=row["status"],
        score=row["score"],
        visible=bool(row["visible"]),
    )


class AnnotationService:
    def list_for_image(self, project: ProjectDb, image_id: int) -> list[AnnotationRecord]:
        with project.connect() as conn:
            rows = conn.execute(
                """
                SELECT a.*, c.name AS category_name
                FROM annotations a
                LEFT JOIN categories c ON c.id=a.category_id
                WHERE a.image_id=?
                ORDER BY a.id
                """,
                (image_id,),
            ).fetchall()
        return [annotation_from_row(row) for row in rows]

    def get(self, project: ProjectDb, annotation_id: int) -> AnnotationRecord:
        with project.connect() as conn:
            row = conn.execute(
                """
                SELECT a.*, c.name AS category_name
                FROM annotations a
                LEFT JOIN categories c ON c.id=a.category_id
                WHERE a.id=?
                """,
                (annotation_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Annotation not found: {annotation_id}")
        return annotation_from_row(row)

    def create_from_mask(
        self,
        project: ProjectDb,
        image_id: int,
        category_name: str,
        mask,
        status: str = "accepted",
        score: float | None = None,
    ) -> AnnotationRecord:
        bbox, area = bbox_and_area(mask)
        with project.connect() as conn:
            category_id = get_or_create_category(conn, category_name)
            cur = conn.execute(
                """
                INSERT INTO annotations(image_id, category_id, bbox_json, area, mask_path, status, score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (image_id, category_id, bbox_json(bbox), area, "", status, score),
            )
            annotation_id = int(cur.lastrowid)
            rel_mask = f"masks/{image_id}/{annotation_id}.png"
            conn.execute(
                "UPDATE annotations SET mask_path=? WHERE id=?",
                (rel_mask, annotation_id),
            )
        save_mask(mask, project.meta_dir / rel_mask)
        return self.get(project, annotation_id)

    def create_from_png(
        self,
        project: ProjectDb,
        image_id: int,
        category_name: str,
        mask_png: str,
        status: str = "accepted",
        score: float | None = None,
    ) -> AnnotationRecord:
        return self.create_from_mask(
            project,
            image_id,
            category_name,
            mask_from_png_data(mask_png),
            status=status,
            score=score,
        )

    def update(self, project: ProjectDb, annotation_id: int, **changes) -> AnnotationRecord:
        with project.connect() as conn:
            if changes.get("category_name") is not None:
                category_id = get_or_create_category(conn, changes["category_name"])
                conn.execute(
                    "UPDATE annotations SET category_id=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (category_id, annotation_id),
                )
            if changes.get("visible") is not None:
                conn.execute(
                    "UPDATE annotations SET visible=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (1 if changes["visible"] else 0, annotation_id),
                )
            if changes.get("status") is not None:
                conn.execute(
                    "UPDATE annotations SET status=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                    (changes["status"], annotation_id),
                )
        return self.get(project, annotation_id)

    def bulk_rename_class(
        self,
        project: ProjectDb,
        from_category_name: str,
        to_category_name: str,
        status: str = "accepted",
    ) -> dict[str, object]:
        source = from_category_name.strip()
        target = to_category_name.strip()
        if not source:
            raise ValueError("Source class name is required")
        if not target:
            raise ValueError("Target class name is required")
        with project.connect() as conn:
            source_row = conn.execute("SELECT id FROM categories WHERE name=?", (source,)).fetchone()
            if source_row is None:
                return {"from_category_name": source, "to_category_name": target, "updated": 0, "status": status}
            target_id = get_or_create_category(conn, target)
            if status == "all":
                cur = conn.execute(
                    """
                    UPDATE annotations
                    SET category_id=?, updated_at=CURRENT_TIMESTAMP
                    WHERE category_id=?
                    """,
                    (target_id, int(source_row["id"])),
                )
            else:
                cur = conn.execute(
                    """
                    UPDATE annotations
                    SET category_id=?, updated_at=CURRENT_TIMESTAMP
                    WHERE category_id=? AND status=?
                    """,
                    (target_id, int(source_row["id"]), status),
                )
        return {"from_category_name": source, "to_category_name": target, "updated": int(cur.rowcount), "status": status}

    def delete(self, project: ProjectDb, annotation_id: int) -> None:
        ann = self.get(project, annotation_id)
        mask_paths = [ann.mask_path]
        with project.connect() as conn:
            rows = conn.execute(
                """
                SELECT mask_path FROM annotation_revisions WHERE annotation_id=?
                UNION ALL
                SELECT mask_path FROM annotation_redo WHERE annotation_id=?
                """,
                (annotation_id, annotation_id),
            ).fetchall()
            mask_paths.extend(row["mask_path"] for row in rows)
            conn.execute("DELETE FROM annotations WHERE id=?", (annotation_id,))
        for rel_path in set(mask_paths):
            mask_path = project.meta_dir / rel_path
            if mask_path.exists():
                mask_path.unlink()

    def replace_mask(self, project: ProjectDb, annotation_id: int, mask_png: str) -> AnnotationRecord:
        ann = self.get(project, annotation_id)
        old_abs = project.meta_dir / ann.mask_path
        revision_rel = f"masks/{ann.image_id}/{annotation_id}_v{ann.version}.png"
        revision_abs = project.meta_dir / revision_rel
        has_previous_mask = old_abs.exists()
        if old_abs.exists():
            copy_mask(old_abs, revision_abs)
        mask = mask_from_png_data(mask_png)
        bbox, area = bbox_and_area(mask)
        with project.connect() as conn:
            conn.execute("DELETE FROM annotation_redo WHERE annotation_id=?", (annotation_id,))
            if has_previous_mask:
                conn.execute(
                    """
                    INSERT INTO annotation_revisions(annotation_id, version, mask_path, bbox_json, area)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (annotation_id, ann.version, revision_rel, bbox_json(ann.bbox), ann.area),
                )
            conn.execute(
                """
                UPDATE annotations
                SET version=version+1, bbox_json=?, area=?, updated_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (bbox_json(bbox), area, annotation_id),
            )
        save_mask(mask, old_abs)
        return self.get(project, annotation_id)

    def undo(self, project: ProjectDb, annotation_id: int) -> AnnotationRecord:
        ann = self.get(project, annotation_id)
        current_abs = project.meta_dir / ann.mask_path
        with project.connect() as conn:
            rev = conn.execute(
                """
                SELECT * FROM annotation_revisions
                WHERE annotation_id=?
                ORDER BY version DESC, id DESC
                LIMIT 1
                """,
                (annotation_id,),
            ).fetchone()
            if rev is None:
                return ann
            redo_rel = f"masks/{ann.image_id}/{annotation_id}_redo_v{ann.version}.png"
            rev_abs = project.meta_dir / rev["mask_path"]
            if not rev_abs.exists():
                conn.execute("DELETE FROM annotation_revisions WHERE id=?", (int(rev["id"]),))
                return ann
            has_current_mask = current_abs.exists()
            if has_current_mask:
                copy_mask(current_abs, project.meta_dir / redo_rel)
                conn.execute(
                    """
                    INSERT INTO annotation_redo(annotation_id, version, mask_path, bbox_json, area)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (annotation_id, ann.version, redo_rel, bbox_json(ann.bbox), ann.area),
                )
            copy_mask(rev_abs, current_abs)
            conn.execute(
                """
                UPDATE annotations
                SET version=?, mask_path=?, bbox_json=?, area=?, updated_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (int(rev["version"]), ann.mask_path, rev["bbox_json"], int(rev["area"]), annotation_id),
            )
            conn.execute("DELETE FROM annotation_revisions WHERE id=?", (int(rev["id"]),))
        return self.get(project, annotation_id)

    def redo(self, project: ProjectDb, annotation_id: int) -> AnnotationRecord:
        ann = self.get(project, annotation_id)
        current_abs = project.meta_dir / ann.mask_path
        with project.connect() as conn:
            redo = conn.execute(
                """
                SELECT * FROM annotation_redo
                WHERE annotation_id=?
                ORDER BY version ASC, id DESC
                LIMIT 1
                """,
                (annotation_id,),
            ).fetchone()
            if redo is None:
                return ann
            undo_rel = f"masks/{ann.image_id}/{annotation_id}_v{ann.version}.png"
            redo_abs = project.meta_dir / redo["mask_path"]
            if not redo_abs.exists():
                conn.execute("DELETE FROM annotation_redo WHERE id=?", (int(redo["id"]),))
                return ann
            has_current_mask = current_abs.exists()
            if has_current_mask:
                copy_mask(current_abs, project.meta_dir / undo_rel)
                conn.execute(
                    """
                    INSERT INTO annotation_revisions(annotation_id, version, mask_path, bbox_json, area)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (annotation_id, ann.version, undo_rel, bbox_json(ann.bbox), ann.area),
                )
            copy_mask(redo_abs, current_abs)
            conn.execute(
                """
                UPDATE annotations
                SET version=?, mask_path=?, bbox_json=?, area=?, updated_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (int(redo["version"]), ann.mask_path, redo["bbox_json"], int(redo["area"]), annotation_id),
            )
            conn.execute("DELETE FROM annotation_redo WHERE id=?", (int(redo["id"]),))
        return self.get(project, annotation_id)

    def log_prompt(self, project: ProjectDb, image_id: int, prompt_type: str, payload: dict, annotation_id: int | None = None) -> None:
        with project.connect() as conn:
            conn.execute(
                """
                INSERT INTO prompt_history(image_id, annotation_id, prompt_type, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (image_id, annotation_id, prompt_type, json.dumps(payload)),
            )
