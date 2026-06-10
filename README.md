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
4. Select an image, choose point/box/text tools, and run SAM prompts.
5. Accept masks into the object list, use CLIP search or bulk concept mode, then run QA and export COCO.

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

