from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS images (
  id INTEGER PRIMARY KEY,
  file_name TEXT NOT NULL UNIQUE,
  width INTEGER NOT NULL,
  height INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS categories (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS annotations (
  id INTEGER PRIMARY KEY,
  image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  category_id INTEGER REFERENCES categories(id),
  bbox_json TEXT NOT NULL,
  area INTEGER NOT NULL,
  iscrowd INTEGER NOT NULL DEFAULT 0,
  mask_path TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 1,
  status TEXT NOT NULL DEFAULT 'accepted',
  score REAL,
  visible INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS annotation_revisions (
  id INTEGER PRIMARY KEY,
  annotation_id INTEGER NOT NULL REFERENCES annotations(id) ON DELETE CASCADE,
  version INTEGER NOT NULL,
  mask_path TEXT NOT NULL,
  bbox_json TEXT NOT NULL,
  area INTEGER NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS annotation_redo (
  id INTEGER PRIMARY KEY,
  annotation_id INTEGER NOT NULL REFERENCES annotations(id) ON DELETE CASCADE,
  version INTEGER NOT NULL,
  mask_path TEXT NOT NULL,
  bbox_json TEXT NOT NULL,
  area INTEGER NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS prompt_history (
  id INTEGER PRIMARY KEY,
  image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  annotation_id INTEGER REFERENCES annotations(id) ON DELETE SET NULL,
  prompt_type TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS embeddings (
  image_id INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
  model TEXT NOT NULL,
  vector_path TEXT NOT NULL,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY,
  kind TEXT NOT NULL,
  status TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  result_json TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS review_candidates (
  id INTEGER PRIMARY KEY,
  job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  category_name TEXT NOT NULL,
  prompt_text TEXT NOT NULL,
  mask_path TEXT NOT NULL,
  bbox_json TEXT NOT NULL,
  area INTEGER NOT NULL,
  score REAL,
  rank INTEGER NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  annotation_id INTEGER REFERENCES annotations(id) ON DELETE SET NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


class ProjectDb:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.meta_dir = self.root / ".foodseg"
        self.db_path = self.meta_dir / "annotations.db"
        self.masks_dir = self.meta_dir / "masks"
        self.exports_dir = self.meta_dir / "exports"
        self.embeddings_dir = self.meta_dir / "embeddings"
        self.review_dir = self.meta_dir / "review_candidates"

    def init(self) -> None:
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.review_dir.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return dict(row)


def bbox_json(bbox: list[float]) -> str:
    return json.dumps([float(v) for v in bbox])


def parse_bbox(raw: str) -> list[float]:
    return [float(v) for v in json.loads(raw)]


def get_or_create_category(conn: sqlite3.Connection, name: str) -> int:
    clean = name.strip() or "food"
    conn.execute("INSERT OR IGNORE INTO categories(name) VALUES (?)", (clean,))
    row = conn.execute("SELECT id FROM categories WHERE name=?", (clean,)).fetchone()
    return int(row["id"])
