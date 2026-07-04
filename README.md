# FoodSegmentation Annotator

Local MVP annotation tool for segmentation-first TaiwanFood101-style image annotation. It uses a FastAPI backend, a React/Vite frontend, SQLite metadata, filesystem PNG masks, SAM 3.1 integration, and CLIP-style similarity search.

## Structure

- `backend/app/`: FastAPI service, project indexing, SQLite repositories, SAM/CLIP services, QA, COCO export.
- `frontend/`: React/TypeScript annotation UI.
- `sam3/`: existing SAM 3 checkout and local `assets/sam3.1_multiplex.pt` checkpoint.
- `tests/`: lightweight backend unit tests using Python `unittest`.

Project data is written inside the selected image folder under `.foodseg/`:

- `.foodseg/annotations.db`
- `.foodseg/masks/{image_id}/{annotation_id}.png`
- `.foodseg/embeddings/*.npy`
- `.foodseg/exports/{timestamp}/annotations_coco.json`

## SAM 3 Setup

This app expects a local SAM 3 checkout in the project root. Clone the SAM 3
repository into a folder named `sam3`:

```powershell
git clone <SAM3_REPO_URL> sam3
```

Then download the SAM 3.1 checkpoint from Hugging Face and place it at:

```text
sam3/assets/sam3.1_multiplex.pt
```

The tokenizer vocabulary should also be available at:

```text
sam3/assets/bpe_simple_vocab_16e6.txt.gz
```

Do not commit the checkpoint file to normal Git. It is a large model artifact;
use Git LFS or keep it as a local download.

## Run Backend

This project needs the CUDA 13 PyTorch wheels for an RTX 50-series GPU. If
you are setting up the environment from scratch, install the Python
dependencies with the cu130 index so `torch` and `torchvision` do not fall back
to CPU-only builds.

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade -r requirements.txt
```

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Open API docs at `http://127.0.0.1:8000/docs`.

## Run Frontend

Install Node.js first if `node`/`npm` are not available, then:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

## Usage

1. Start the backend.
2. Start the frontend.
3. Enter a local image folder path, for example `sam3/assets/images`, and click `Open`.
4. Select an image, use point or text prompts to create mask candidates, and refine them as needed.
5. Accept masks into the object list, use CLIP search or bulk concept mode, then run QA and export COCO.

## App Guide

### Main Workflow

1. Open a project folder that contains images. The app indexes supported image files and stores metadata, masks, embeddings, and exports under that folder's `.foodseg/` directory.
2. Pick an image from the left sidebar. Use filename, mask status, class name, and object-count filters to focus the list.
3. Set the `Class Name` field before saving masks. New accepted masks use this class name unless you rename them later.
4. Create mask candidates with point or text prompts, then refine selected masks with brush or erase.
5. Review unsaved candidates in the Editor tab. Accept one candidate, accept all candidates, rename them, toggle visibility, or clear them.
6. Manage saved objects in the Editor tab. Rename objects, toggle visibility, accept or reject pending objects, delete objects, merge accepted masks, and run QA.
7. Use Bulk Review for CLIP search, similar-image search, and queued SAM jobs across many images.
8. Use Export to split annotated folders into train/val/test COCO exports, create combined exports, or merge combined exports from multiple annotators.

### Features

- Local annotation projects with SQLite metadata and PNG mask files.
- SAM 3.1 point and text prompting, with deterministic fallback behavior when SAM is unavailable.
- Positive and negative point refinement, including draggable prompt points.
- Manual mask cleanup with brush and erase tools for selected candidates or saved objects.
- Candidate and saved-object visibility toggles, inline renaming, undo/redo, and object deletion.
- CLIP indexing, text search, and similar-image search when OpenAI CLIP is installed.
- Bulk SAM review jobs with all-image or CLIP-filtered modes, job cancellation, pending review queues, and accept/reject actions.
- Propagation from the selected saved object to visually similar images for review.
- QA checks before export.
- COCO instance segmentation export, per-folder train/val/test split export, combined dataset export, and combined-export merging.

### Shortcuts

Keyboard shortcuts are ignored while typing in inputs, textareas, selects, or editable text.

| Shortcut | Action |
| --- | --- |
| `Ctrl+Z` / `Cmd+Z` | Undo the latest candidate or editor action. |
| `N` | Select the positive point tool. |
| `B` | Select the brush tool. |
| `V` | Select the erase tool. |
| `G` | Run the current SAM text prompt. |
| `H` | Accept the selected mask candidate as a saved object. |
| `R` | Select positive point mode and refine the selected candidate. |
| `T` | Select positive point mode and create a new candidate per click. |
| `D` | Go to the previous image. |
| `F` | Go to the next image. |
| `Ctrl++` / `Cmd++` | Increase brush size. |
| `Ctrl+-` / `Cmd+-` | Decrease brush size. |

### Mouse Controls

- Scroll over the image stage to zoom toward the cursor.
- Drag with the View tool to pan.
- Middle-click and drag to pan from any tool.
- Click with the point tools to add a prompt point and run SAM.
- Hold `Shift` while clicking with a point tool to add a negative point.
- Drag an existing point to rerun the point-set prompt.
- Hold `Shift` and click an existing point to toggle it between positive and negative.
- Hold `Alt` and click an existing point to remove it.
- Select a candidate or saved object, then use brush or erase to edit its mask.
- Drag resize handles on a selected mask outline to adjust its bounding area.

The backend lazily loads SAM 3.1 from:

- `sam3/assets/sam3.1_multiplex.pt`
- `sam3/assets/bpe_simple_vocab_16e6.txt.gz`

If SAM or OpenAI CLIP is not importable, the service falls back to deterministic local behavior so the UI, persistence, QA, and export paths remain testable. Install OpenAI CLIP separately for real CLIP embeddings:

```powershell
.\.venv\Scripts\python.exe -m pip install git+https://github.com/openai/CLIP.git
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

## Notes

- Single-user local app only.
- SQLite and PNG masks are the source of truth.
- Export follows COCO instance segmentation fields and includes `mask_path` and `version` for local traceability.
- Pending masks from bulk/propagation are intentionally reviewable before final export.
