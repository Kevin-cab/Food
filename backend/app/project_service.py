from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from .db import ProjectDb
from .schemas import ImagePage, ImageRecord, ProjectIndexStatus, ProjectSummary

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


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

    def image_count(self, project: ProjectDb | None = None, q: str | None = None) -> int:
        project = project or self.require()
        query = "SELECT COUNT(*) AS n FROM images"
        params: tuple = ()
        if q:
            query += " WHERE file_name LIKE ?"
            params = (f"%{q}%",)
        with project.connect() as conn:
            return int(conn.execute(query, params).fetchone()["n"])

    def list_images_page(self, limit: int = 100, offset: int = 0, q: str | None = None) -> ImagePage:
        items = self.list_images(limit=limit, offset=offset, q=q)
        total = self.image_count(q=q)
        return ImagePage(items=items, total=total, limit=limit, offset=offset, has_more=offset + len(items) < total)

    def list_images(self, limit: int = 100, offset: int = 0, q: str | None = None) -> list[ImageRecord]:
        project = self.require()
        where = ""
        params: tuple = ()
        if q:
            where = "WHERE file_name LIKE ?"
            params = (f"%{q}%",)
        with project.connect() as conn:
            rows = conn.execute(
                f"SELECT id, file_name, width, height FROM images {where} ORDER BY file_name LIMIT ? OFFSET ?",
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
                "SELECT id, file_name, width, height FROM images WHERE id=?", (image_id,)
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
