from __future__ import annotations

import tempfile
import time
import unittest
import json
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from backend.app.annotations import AnnotationService
from backend.app.bulk_service import BulkJobService
from backend.app.clip_service import ClipService
from backend.app.exporter import export_coco, export_combined_workspace_coco, export_workspace_coco, merge_combined_coco_exports
from backend.app.masks import load_mask, mask_to_png_data, simple_prompt_mask
from backend.app.project_service import ProjectService
from backend.app.qa import validate_project
from backend.app.schemas import BulkJobCreateRequest, ExportCocoRequest, MaskCandidate, MergeCombinedExportRequest


class FakeSam:
    last_error = None

    def predict(self, image_path, prompt, width, height):
        mask = simple_prompt_mask(width, height)
        return [
            MaskCandidate(
                mask_png=mask_to_png_data(mask),
                bbox=[0, 0, width / 2, height / 2],
                area=int(mask.sum()),
                score=0.9,
                prompt_type=prompt.prompt_type,
                annotation=None,
            )
        ]


class SlowFakeSam(FakeSam):
    def predict(self, image_path, prompt, width, height):
        time.sleep(0.15)
        return super().predict(image_path, prompt, width, height)


class BackendCoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        Image.new("RGB", (64, 48), (180, 80, 30)).save(self.root / "a.jpg")
        Image.new("RGB", (64, 48), (40, 160, 120)).save(self.root / "b.jpg")
        self.projects = ProjectService()
        self.summary = self.projects.open_project(str(self.root))
        self.projects.wait_for_index(timeout=5)
        self.project = self.projects.require()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_project_annotation_export_flow(self) -> None:
        images = self.projects.list_images()
        self.assertEqual(len(images), 2)
        mask = simple_prompt_mask(images[0].width, images[0].height)
        ann = AnnotationService().create_from_png(
            self.project,
            images[0].id,
            "rice",
            mask_to_png_data(mask),
        )
        self.assertGreater(ann.area, 0)
        issues = validate_project(self.project)
        self.assertTrue(all(issue.severity in {"warning", "error"} for issue in issues))
        exported = export_coco(self.project)
        self.assertTrue(Path(exported.coco_json).exists())
        self.assertEqual(exported.mask_count, 1)

    def test_image_filters_and_object_counts(self) -> None:
        images = self.projects.list_images()
        service = AnnotationService()
        mask = simple_prompt_mask(images[0].width, images[0].height)
        service.create_from_png(self.project, images[0].id, "rice", mask_to_png_data(mask))
        service.create_from_png(self.project, images[0].id, "soup", mask_to_png_data(mask))
        service.create_from_png(self.project, images[1].id, "rice", mask_to_png_data(mask), status="rejected")

        all_images = self.projects.list_images()
        counts = {image.file_name: image.accepted_object_count for image in all_images}
        self.assertEqual(counts["a.jpg"], 2)
        self.assertEqual(counts["b.jpg"], 0)

        with_masks = self.projects.list_images(mask_filter="with_masks")
        self.assertEqual([image.file_name for image in with_masks], ["a.jpg"])

        without_masks = self.projects.list_images(mask_filter="without_masks")
        self.assertEqual([image.file_name for image in without_masks], ["b.jpg"])

        rice_images = self.projects.list_images(class_name="rice")
        self.assertEqual([image.file_name for image in rice_images], ["a.jpg"])
        self.assertEqual(rice_images[0].matching_class_count, 1)

        self.assertEqual([image.file_name for image in self.projects.list_images(count_op="lt", count_value=1)], ["b.jpg"])
        self.assertEqual([image.file_name for image in self.projects.list_images(count_op="lte", count_value=0)], ["b.jpg"])
        self.assertEqual([image.file_name for image in self.projects.list_images(count_op="eq", count_value=2)], ["a.jpg"])
        self.assertEqual([image.file_name for image in self.projects.list_images(count_op="gte", count_value=2)], ["a.jpg"])
        self.assertEqual([image.file_name for image in self.projects.list_images(count_op="gt", count_value=1)], ["a.jpg"])

        combined = self.projects.list_images(q="a", class_name="rice", count_op="gte", count_value=1)
        self.assertEqual([image.file_name for image in combined], ["a.jpg"])

    def test_bulk_rename_class_updates_accepted_annotations_only(self) -> None:
        images = self.projects.list_images()
        service = AnnotationService()
        mask = simple_prompt_mask(images[0].width, images[0].height)
        accepted = service.create_from_png(self.project, images[0].id, "dog", mask_to_png_data(mask))
        pending = service.create_from_png(self.project, images[1].id, "dog", mask_to_png_data(mask), status="pending")

        result = service.bulk_rename_class(self.project, "dog", "cat")

        self.assertEqual(result["updated"], 1)
        self.assertEqual(service.get(self.project, accepted.id).category_name, "cat")
        self.assertEqual(service.get(self.project, pending.id).category_name, "dog")

    def test_annotation_undo_redo_and_delete_mask_files(self) -> None:
        images = self.projects.list_images()
        service = AnnotationService()
        mask = simple_prompt_mask(images[0].width, images[0].height)
        ann = service.create_from_png(self.project, images[0].id, "rice", mask_to_png_data(mask))
        original_path = self.project.meta_dir / ann.mask_path
        edited = np.zeros_like(mask)
        edited[2:12, 2:12] = True
        updated = service.replace_mask(self.project, ann.id, mask_to_png_data(edited))
        revision_path = self.project.meta_dir / f"masks/{ann.image_id}/{ann.id}_v{ann.version}.png"
        self.assertTrue(original_path.exists())
        self.assertTrue(revision_path.exists())
        undone = service.undo(self.project, ann.id)
        self.assertEqual(undone.version, ann.version)
        self.assertEqual(int(load_mask(original_path).sum()), int(mask.sum()))
        redone = service.redo(self.project, ann.id)
        self.assertEqual(redone.version, updated.version)
        self.assertEqual(int(load_mask(original_path).sum()), int(edited.sum()))
        service.delete(self.project, ann.id)
        self.assertFalse(original_path.exists())
        self.assertFalse(revision_path.exists())

    def test_missing_mask_is_reported_and_skipped_by_export(self) -> None:
        images = self.projects.list_images()
        service = AnnotationService()
        mask = simple_prompt_mask(images[0].width, images[0].height)
        ann = service.create_from_png(self.project, images[0].id, "rice", mask_to_png_data(mask))
        (self.project.meta_dir / ann.mask_path).unlink()
        issues = validate_project(self.project)
        self.assertTrue(any(issue.code == "missing_mask" and issue.annotation_id == ann.id for issue in issues))
        exported = export_coco(self.project)
        self.assertEqual(exported.mask_count, 0)

    def test_export_coco_uses_assigned_splits(self) -> None:
        images = self.projects.list_images()
        service = AnnotationService()
        for image in images:
            mask = simple_prompt_mask(image.width, image.height)
            service.create_from_png(self.project, image.id, "rice", mask_to_png_data(mask))

        exported = export_coco(
            self.project,
            ExportCocoRequest(splits={"train": [images[0].id], "val": [images[1].id], "test": []}),
        )

        train_payload = json.loads(Path(exported.split_coco_jsons["train"]).read_text(encoding="utf-8"))
        val_payload = json.loads(Path(exported.split_coco_jsons["val"]).read_text(encoding="utf-8"))
        test_payload = json.loads(Path(exported.split_coco_jsons["test"]).read_text(encoding="utf-8"))
        self.assertEqual([image["id"] for image in train_payload["images"]], [images[0].id])
        self.assertEqual([image["id"] for image in val_payload["images"]], [images[1].id])
        self.assertEqual(test_payload["images"], [])
        self.assertTrue(all(annotation["mask_path"].startswith("masks/train/") for annotation in train_payload["annotations"]))
        self.assertTrue(all(annotation["mask_path"].startswith("masks/val/") for annotation in val_payload["annotations"]))

    def test_workspace_export_uses_folder_scoped_splits(self) -> None:
        images = self.projects.list_images()
        service = AnnotationService()
        mask = simple_prompt_mask(images[0].width, images[0].height)
        service.create_from_png(self.project, images[0].id, "rice", mask_to_png_data(mask))

        exported = export_workspace_coco(
            ExportCocoRequest(
                root=str(self.root),
                folder_splits=[
                    {
                        "folder": ".",
                        "splits": {"train": [images[0].id], "val": [], "test": []},
                    }
                ],
            )
        )

        self.assertEqual(exported.mask_count, 1)
        self.assertEqual(len(exported.folder_exports), 1)
        self.assertIn("./train", exported.split_coco_jsons)

    def test_combined_workspace_export_writes_to_root(self) -> None:
        images = self.projects.list_images()
        service = AnnotationService()
        mask = simple_prompt_mask(images[0].width, images[0].height)
        service.create_from_png(self.project, images[0].id, "rice", mask_to_png_data(mask))

        exported = export_combined_workspace_coco(
            ExportCocoRequest(
                root=str(self.root),
                combined=True,
                folder_splits=[
                    {
                        "folder": ".",
                        "splits": {"train": [images[0].id], "val": [], "test": []},
                    }
                ],
            )
        )

        export_dir = Path(exported.export_dir)
        self.assertEqual(export_dir.parent, self.root.resolve())
        self.assertTrue(export_dir.name.startswith("combined_coco_export_"))
        self.assertTrue(any((export_dir / "images" / "train").iterdir()))
        self.assertTrue((export_dir / "manifest.csv").exists())
        labels = json.loads((export_dir / "labels.json").read_text(encoding="utf-8"))
        self.assertEqual(labels["0"], {"name": "background", "color": [0, 0, 0]})
        self.assertEqual(labels["1"]["name"], "rice")
        self.assertEqual(len(labels["1"]["color"]), 3)
        train_payload = json.loads(Path(exported.split_coco_jsons["train"]).read_text(encoding="utf-8"))
        self.assertEqual(len(train_payload["images"]), 1)
        self.assertEqual(train_payload["images"][0]["split"], "train")
        with (export_dir / "manifest.csv").open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(rows[0]["split"], "train")
        self.assertTrue(rows[0]["image_path"].startswith("images/train/"))
        self.assertTrue(rows[0]["masks"].startswith("masks/train/"))
        self.assertTrue(rows[0]["semantic_rgb"].startswith("semantic_rgb/train/"))

    def test_combined_workspace_export_canonicalizes_context_class_names(self) -> None:
        folder = self.root / "pig_s_blood_soup"
        folder.mkdir()
        Image.new("RGB", (32, 24), (180, 80, 30)).save(folder / "pig_s_blood_soup_1.jpg")
        projects = ProjectService()
        projects.open_project(str(folder))
        projects.wait_for_index(timeout=5)
        project = projects.require()
        image = projects.list_images()[0]
        mask = simple_prompt_mask(image.width, image.height)
        AnnotationService().create_from_png(project, image.id, "food", mask_to_png_data(mask))

        exported = export_combined_workspace_coco(
            ExportCocoRequest(
                root=str(self.root),
                combined=True,
                folder_splits=[
                    {
                        "folder": "pig_s_blood_soup",
                        "splits": {"train": [image.id], "val": [], "test": []},
                    }
                ],
            )
        )

        export_dir = Path(exported.export_dir)
        payload = json.loads(Path(exported.coco_json).read_text(encoding="utf-8"))
        labels = json.loads((export_dir / "labels.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["categories"], [{"id": 1, "name": "pig_blood_soup"}])
        self.assertEqual(labels["1"]["name"], "pig_blood_soup")
        self.assertTrue(any(issue.code == "category_name_corrected" for issue in exported.issues))
        with (export_dir / "manifest.csv").open(encoding="utf-8", newline="") as file:
            row = next(csv.DictReader(file))
        semantic = np.array(Image.open(export_dir / row["semantic_rgb"]).convert("RGB"))
        colors = {tuple(color) for color in semantic.reshape(-1, 3) if tuple(color) != (0, 0, 0)}
        self.assertEqual(colors, {tuple(labels["1"]["color"])})

    def _write_combined_export_fixture(
        self,
        name: str,
        image_rel: str,
        split: str = "train",
        category_name: str = "rice",
    ) -> Path:
        export_dir = self.root / name
        image_path = export_dir / image_rel
        mask_rel = f"masks/{split}/{name}_1.png"
        semantic_rel = f"semantic_rgb/{split}/{name}_1.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        (export_dir / mask_rel).parent.mkdir(parents=True, exist_ok=True)
        (export_dir / semantic_rel).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (24, 20), (120, 80, 40)).save(image_path)
        mask = np.zeros((20, 24), dtype=np.uint8)
        mask[3:12, 4:15] = 255
        Image.fromarray(mask, mode="L").save(export_dir / mask_rel)
        semantic = np.zeros((20, 24, 3), dtype=np.uint8)
        semantic[mask > 0] = [10, 20, 30]
        Image.fromarray(semantic, mode="RGB").save(export_dir / semantic_rel)
        payload = {
            "info": {"description": "fixture"},
            "images": [
                {
                    "id": 1,
                    "file_name": image_rel,
                    "width": 24,
                    "height": 20,
                    "split": split,
                }
            ],
            "categories": [{"id": 1, "name": category_name}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "segmentation": [[4, 3, 14, 3, 14, 11, 4, 11]],
                    "bbox": [4, 3, 11, 9],
                    "area": int((mask > 0).sum()),
                    "iscrowd": 0,
                    "mask_path": mask_rel,
                    "version": 1,
                    "split": split,
                }
            ],
            "splits": {"train": [1] if split == "train" else [], "val": [1] if split == "val" else [], "test": [1] if split == "test" else []},
        }
        (export_dir / "annotations_coco.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (export_dir / "polygons.json").write_text(json.dumps([{"annotation_id": 1, "polygons": payload["annotations"][0]["segmentation"]}], indent=2), encoding="utf-8")
        (export_dir / "labels.json").write_text(json.dumps({"0": {"name": "background", "color": [0, 0, 0]}, "1": {"name": category_name, "color": [10, 20, 30]}}, indent=2), encoding="utf-8")
        (export_dir / "qa_report.json").write_text("[]", encoding="utf-8")
        with (export_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["image_path", "masks", "semantic_rgb", "split"])
            writer.writeheader()
            writer.writerow({"image_path": image_rel, "masks": mask_rel, "semantic_rgb": semantic_rel, "split": split})
        return export_dir

    def test_merge_combined_exports_preserves_splits_and_unifies_categories(self) -> None:
        first = self._write_combined_export_fixture("combined_coco_export_a", "images/train/a.jpg", "train", "rice")
        second = self._write_combined_export_fixture("combined_coco_export_b", "images/val/b.jpg", "val", "rice")
        output_root = self.root / "merged"
        output_root.mkdir()

        exported = merge_combined_coco_exports(
            MergeCombinedExportRequest(output_root=str(output_root), export_dirs=[str(first), str(second)])
        )

        export_dir = Path(exported.export_dir)
        self.assertTrue(export_dir.name.startswith("merged_combined_coco_export_"))
        payload = json.loads(Path(exported.coco_json).read_text(encoding="utf-8"))
        self.assertEqual([image["id"] for image in payload["images"]], [1, 2])
        self.assertEqual([annotation["id"] for annotation in payload["annotations"]], [1, 2])
        self.assertEqual(payload["categories"], [{"id": 1, "name": "rice"}])
        self.assertEqual(payload["splits"]["train"], [1])
        self.assertEqual(payload["splits"]["val"], [2])
        self.assertTrue((export_dir / "images/train/a.jpg").exists())
        self.assertTrue((export_dir / "images/val/b.jpg").exists())
        for annotation in payload["annotations"]:
            self.assertEqual(annotation["category_id"], 1)
            self.assertTrue((export_dir / annotation["mask_path"]).exists())
            self.assertIn("source_export", annotation)
        with (export_dir / "manifest.csv").open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual({row["split"] for row in rows}, {"train", "val"})
        for row in rows:
            self.assertTrue((export_dir / row["image_path"]).exists())
            self.assertTrue((export_dir / row["semantic_rgb"]).exists())
            for mask_path in row["masks"].split(";"):
                self.assertTrue((export_dir / mask_path).exists())

    def test_merge_combined_exports_rejects_duplicate_image_paths(self) -> None:
        first = self._write_combined_export_fixture("combined_coco_export_a", "images/train/a.jpg", "train", "rice")
        second = self._write_combined_export_fixture("combined_coco_export_b", "images/train/a.jpg", "train", "rice")
        output_root = self.root / "merged"
        output_root.mkdir()

        with self.assertRaises(ValueError) as ctx:
            merge_combined_coco_exports(
                MergeCombinedExportRequest(output_root=str(output_root), export_dirs=[str(first), str(second)])
            )
        self.assertIn("Duplicate image path", str(ctx.exception))

    def test_clip_fallback_search(self) -> None:
        clip = ClipService(Path.cwd(), allow_fallback=True)
        count = clip.index(self.project, self.projects, force=True)
        self.assertEqual(count, 2)
        results = clip.search(self.project, self.projects.list_images(), text="rice", limit=2)
        self.assertEqual(len(results), 2)
        self.assertGreaterEqual(results[0].score, results[1].score)

    def test_bulk_review_candidate_flow(self) -> None:
        service = BulkJobService(AnnotationService(), FakeSam(), ClipService(Path.cwd(), allow_fallback=True))
        request = BulkJobCreateRequest(
            mode="all",
            text="rice",
            confidence_threshold=0.3,
            top_k=3,
            max_images=1,
        )
        job = service.create_job(self.project, self.projects, request)
        service.executor.shutdown(wait=True)
        job = service.get_job(self.project, job.id)
        self.assertEqual(job.status, "completed")
        candidates = service.list_candidates(self.project, self.projects, job.id)
        self.assertEqual(len(candidates), 1)
        renamed = service.rename_pending_candidates(self.project, job.id, "cool rice")
        self.assertEqual(renamed["updated"], 1)
        candidates = service.list_candidates(self.project, self.projects, job.id)
        self.assertEqual(candidates[0].category_name, "cool rice")
        mask_path = self.project.meta_dir / candidates[0].mask_path
        self.assertTrue(mask_path.exists())
        accepted = service.accept_pending_candidates(self.project, job.id)
        self.assertEqual(accepted["accepted"], 1)
        candidates = service.list_candidates(self.project, self.projects, job.id)
        self.assertEqual(len(candidates), 0)
        accepted_candidates = service.list_candidates(self.project, self.projects, job.id, status="accepted")
        self.assertEqual(len(accepted_candidates), 1)
        self.assertIsNotNone(accepted_candidates[0].annotation_id)
        deleted = service.delete_job(self.project, job.id)
        self.assertEqual(deleted["deleted"], job.id)
        self.assertFalse(mask_path.exists())
        with self.assertRaises(KeyError):
            service.get_job(self.project, job.id)

    def test_bulk_job_can_be_cancelled_while_running(self) -> None:
        service = BulkJobService(AnnotationService(), SlowFakeSam(), ClipService(Path.cwd(), allow_fallback=True))
        request = BulkJobCreateRequest(
            mode="all",
            text="rice",
            confidence_threshold=0.3,
            top_k=1,
            max_images=2,
        )
        job = service.create_job(self.project, self.projects, request)
        deadline = time.time() + 2
        while time.time() < deadline and service.get_job(self.project, job.id).status == "queued":
            time.sleep(0.02)
        service.cancel_job(self.project, job.id)
        service.executor.shutdown(wait=True)
        job = service.get_job(self.project, job.id)
        self.assertEqual(job.status, "cancelled")
        self.assertLessEqual(job.result.get("processed", 0), 2)


if __name__ == "__main__":
    unittest.main()
