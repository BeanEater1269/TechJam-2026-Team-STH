import io
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))  # so `inference`/`judge_runs` resolve regardless of
                                    # whether this module was loaded as `webdemo.main`
                                    # (uvicorn's default) or run directly from webdemo/

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402

from inference import get_predictor, to_record  # noqa: E402
from judge_runs import load_run, save_run  # noqa: E402

RESULTS_JUDGE = BASE_DIR.parent / "results_judge"
RESULTS_JUDGE.mkdir(parents=True, exist_ok=True)  # must exist before the StaticFiles mount below

MAX_FILES = 50
MAX_BYTES = 20 * 1024 * 1024  # 20 MB per file


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the model once at startup, not on the judge's first request (~10s cold
    # load). Gate-able for frontend-only iteration with `uvicorn --reload`, where
    # eating that cost on every file save would be painful.
    if not os.environ.get("SKIP_MODEL_WARMUP"):
        get_predictor()
    yield


app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/judge-files", StaticFiles(directory=RESULTS_JUDGE), name="judge-files")


@app.get("/")
def read_index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/live")
def read_live():
    return FileResponse(BASE_DIR / "static" / "live.html")


@app.get("/results")
def read_results():
    return FileResponse(BASE_DIR / "static" / "results.html")


@app.get("/dashboard")
def read_dashboard_redirect():
    return RedirectResponse("/results", status_code=307)


@app.get("/ping")
def ping():
    return {"status": "server is alive"}


def _load_upload_image(upload: UploadFile) -> Image.Image:
    data = upload.file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(400, detail=f"{upload.filename}: exceeds {MAX_BYTES // (1024*1024)}MB limit")
    try:
        return Image.open(io.BytesIO(data))
    except UnidentifiedImageError:
        raise HTTPException(400, detail=f"{upload.filename}: not a readable image")


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    image = _load_upload_image(file)
    r = get_predictor().predict([image])[0]
    return {"pred": r["raw_prob"], "label": r["label"], "confidence": r["confidence"]}


@app.post("/api/live/analyze")
def live_analyze(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, detail="no files uploaded")
    if len(files) > MAX_FILES:
        raise HTTPException(400, detail=f"too many files ({len(files)}), max {MAX_FILES}")

    images = [_load_upload_image(f) for f in files]
    filenames = [f.filename for f in files]
    results = get_predictor().predict(images)

    if len(results) == 1:
        return JSONResponse({"mode": "single", "persisted": False, "result": to_record(results[0])})

    run_json = save_run(RESULTS_JUDGE, results, filenames)
    return JSONResponse({
        "mode": "batch", "persisted": True,
        "run_id": run_json["run_id"], "run_url": f"/api/live/runs/{run_json['run_id']}",
    })


@app.get("/api/live/runs/{run_id}")
def live_get_run(run_id: str):
    run = load_run(RESULTS_JUDGE, run_id)
    if run is None:
        raise HTTPException(404, detail="run not found")
    return JSONResponse(run)
