from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from .db import ProjectDb
from .schemas import ImagePage, ImageRecord, ProjectIndexStatus, ProjectSummary

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
COUNT_OPERATORS = {"lt": "<", "lte": "<=", "eq": "=", "gte": ">=", "gt": ">"}


class ProjectService:
    def __init__(self) -> None:
        self.project: ProjectDb | None = None
        self._index_lock = threading.Lock()
        self._index_generation = 0
        self._index_status = ProjectIndexStatus(status="idle", indexed_count=0)
        self._index_thread: threading.Thread | None = None

    def require(self) -> ProjectDb:
        if self.project is None:
            raise RuntimeError("No project is open")
        return self.project

    def open_project(self, path: str) -> ProjectSummary:
        cleaned = path.strip().strip('"').strip("'")
        root = Path(cleaned).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise ValueError(f"Project folder does not exist: {root}")
        project = ProjectDb(root)
        project.init()
        self.project = project
        count = self.image_count()
        self.start_indexing(project)
        return ProjectSummary(
            root=str(project.root),
            db_path=str(project.db_path),
            image_count=count,
        )

    def start_indexing(self, project: ProjectDb | None = None) -> None:
        project = project or self.require()
        with self._index_lock:
            self._index_generation += 1
            generation = self._index_generation
            self._index_status = ProjectIndexStatus(
                status="indexing",
                indexed_count=self.image_count(project),
                total_seen=0,
                current_file=None,
                started_at=datetime.now(UTC).isoformat(),
                finished_at=None,
                error=None,
            )
        thread = threading.Thread(target=self.index_images, args=(project.root, generation), daemon=True)
        self._index_thread = thread
        thread.start()

    def wait_for_index(self, timeout: float | None = None) -> None:
        thread = self._index_thread
        if thread is not None:
            thread.join(timeout=timeout)

    def index_status(self) -> ProjectIndexStatus:
        with self._index_lock:
            return self._index_status

    def _set_index_status(self, generation: int, **changes) -> None:
        with self._index_lock:
            if generation != self._index_generation:
                return
            current = self._index_status.model_dump()
            current.update(changes)
            self._index_status = ProjectIndexStatus(**current)

    def index_images(self, root: Path | None = None, generation: int | None = None) -> int:
        project = ProjectDb(root) if root is not None else self.require()
        project.init()
        count = 0
        try:
            with project.connect() as conn:
                conn.execute("CREATE TEMP TABLE IF NOT EXISTS indexed_files(file_name TEXT PRIMARY KEY)")
                conn.execute("DELETE FROM indexed_files")
                batch = 0
                for path in sorted(project.root.rglob("*")):
                    if ".foodseg" in path.parts or path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    rel = path.relative_to(project.root).as_posix()
                    if generation is not None:
                        self._set_index_status(generation, total_seen=count + 1, current_file=rel)
                    try:
                        with Image.open(path) as img:
                            width, height = img.size
                    except Exception:
                        continue
                    conn.execute("INSERT OR IGNORE INTO indexed_files(file_name) VALUES (?)", (rel,))
                    conn.execute(
                        """
                        INSERT INTO images(file_name, width, height)
                        VALUES (?, ?, ?)
                        ON CONFLICT(file_name) DO UPDATE SET
                          width=excluded.width,
                          height=excluded.height
                        """,
                        (rel, width, height),
                    )
                    count += 1
                    batch += 1
                    if batch >= 100:
                        conn.commit()
                        batch = 0
                        if generation is not None:
                            self._set_index_status(generation, indexed_count=self.image_count(project), total_seen=count)
                conn.execute(
                    """
                    DELETE FROM images
                    WHERE file_name NOT IN (SELECT file_name FROM indexed_files)
                    """
                )
                conn.commit()
            if generation is not None:
                self._set_index_status(
                    generation,
                    status="completed",
                    indexed_count=self.image_count(project),
                    total_seen=count,
                    current_file=None,
                    finished_at=datetime.now(UTC).isoformat(),
                    error=None,
                )
        except Exception as exc:
            if generation is not None:
                self._set_index_status(
                    generation,
                    status="failed",
                    current_file=None,
                    finished_at=datetime.now(UTC).isoformat(),
                    error=str(exc),
                )
            raise
        return count

    def _image_summary_query(
        self,
        q: str | None = None,
        mask_filter: str = "all",
        class_name: str | None = None,
        count_op: str | None = None,
        count_value: int | None = None,
    ) -> tuple[str, list[object]]:
        where: list[str] = []
        select_params: list[object] = []
        where_params: list[object] = []
        class_name = class_name.strip() if class_name else None
        if q:
            where.append("i.file_name LIKE ?")
            where_params.append(f"%{q}%")
        class_select = "NULL AS matching_class_count"
        if class_name:
            class_select = "SUM(CASE WHEN c.name = ? THEN 1 ELSE 0 END) AS matching_class_count"
            select_params.append(class_name)
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        outer_where: list[str] = []
        outer_params: list[object] = []
        if mask_filter == "with_masks":
            outer_where.append("accepted_object_count > 0")
        elif mask_filter == "without_masks":
            outer_where.append("accepted_object_count = 0")
        if class_name:
            outer_where.append("COALESCE(matching_class_count, 0) > 0")
        if count_op and count_value is not None:
            operator = COUNT_OPERATORS.get(count_op)
            if operator:
                outer_where.append(f"accepted_object_count {operator} ?")
                outer_params.append(int(count_value))
        outer_where_sql = f"WHERE {' AND '.join(outer_where)}" if outer_where else ""
        query = f"""
            SELECT * FROM (
              SELECT
                i.id,
                i.file_name,
                i.width,
                i.height,
                COUNT(a.id) AS accepted_object_count,
                {class_select}
              FROM images i
              LEFT JOIN annotations a ON a.image_id = i.id AND a.status = 'accepted'
              LEFT JOIN categories c ON c.id = a.category_id
              {where_sql}
              GROUP BY i.id
            ) summary
            {outer_where_sql}
        """
        return query, select_params + where_params + outer_params

    def image_count(
        self,
        project: ProjectDb | None = None,
        q: str | None = None,
        mask_filter: str = "all",
        class_name: str | None = None,
        count_op: str | None = None,
        count_value: int | None = None,
    ) -> int:
        project = project or self.require()
        summary_query, params = self._image_summary_query(q, mask_filter, class_name, count_op, count_value)
        with project.connect() as conn:
            return int(conn.execute(f"SELECT COUNT(*) AS n FROM ({summary_query}) filtered", params).fetchone()["n"])

    def list_images_page(
        self,
        limit: int = 100,
        offset: int = 0,
        q: str | None = None,
        mask_filter: str = "all",
        class_name: str | None = None,
        count_op: str | None = None,
        count_value: int | None = None,
    ) -> ImagePage:
        items = self.list_images(limit=limit, offset=offset, q=q, mask_filter=mask_filter, class_name=class_name, count_op=count_op, count_value=count_value)
        total = self.image_count(q=q, mask_filter=mask_filter, class_name=class_name, count_op=count_op, count_value=count_value)
        return ImagePage(items=items, total=total, limit=limit, offset=offset, has_more=offset + len(items) < total)

    def list_images(
        self,
        limit: int = 100,
        offset: int = 0,
        q: str | None = None,
        mask_filter: str = "all",
        class_name: str | None = None,
        count_op: str | None = None,
        count_value: int | None = None,
    ) -> list[ImageRecord]:
        project = self.require()
        query, params = self._image_summary_query(q, mask_filter, class_name, count_op, count_value)
        with project.connect() as conn:
            rows = conn.execute(
                f"{query} ORDER BY file_name LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
        return [ImageRecord(**dict(row)) for row in rows]

    def list_all_images(self) -> list[ImageRecord]:
        project = self.require()
        with project.connect() as conn:
            rows = conn.execute(
                "SELECT id, file_name, width, height FROM images ORDER BY file_name"
            ).fetchall()
        return [ImageRecord(**dict(row)) for row in rows]

    def legacy_index_images(self) -> int:
        project = self.require()
        count = 0
        with project.connect() as conn:
            for path in sorted(project.root.rglob("*")):
                if ".foodseg" in path.parts or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                rel = path.relative_to(project.root).as_posix()
                try:
                    with Image.open(path) as img:
                        width, height = img.size
                except Exception:
                    continue
                conn.execute(
                    """
                    INSERT INTO images(file_name, width, height)
                    VALUES (?, ?, ?)
                    ON CONFLICT(file_name) DO UPDATE SET
                      width=excluded.width,
                      height=excluded.height
                    """,
                    (rel, width, height),
                )
                count += 1
        return count

    def get_image(self, image_id: int) -> ImageRecord:
        project = self.require()
        with project.connect() as conn:
            row = conn.execute(
                """
                SELECT
                  i.id,
                  i.file_name,
                  i.width,
                  i.height,
                  COUNT(a.id) AS accepted_object_count,
                  NULL AS matching_class_count
                FROM images i
                LEFT JOIN annotations a ON a.image_id = i.id AND a.status = 'accepted'
                WHERE i.id=?
                GROUP BY i.id
                """,
                (image_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Image not found: {image_id}")
        return ImageRecord(**dict(row))

    def image_path(self, image_id: int) -> Path:
        image = self.get_image(image_id)
        path = (self.require().root / image.file_name).resolve()
        if not str(path).startswith(str(self.require().root)):
            raise ValueError("Image path escapes project root")
        return path

    def summary(self) -> ProjectSummary:
        project = self.require()
        return ProjectSummary(
            root=str(project.root),
            db_path=str(project.db_path),
            image_count=self.image_count(project),
        )
