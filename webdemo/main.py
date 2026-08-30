import asyncio
import random
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI()

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def read_index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/dashboard")
def read_dashboard():
    return FileResponse(BASE_DIR / "static" / "dashboard.html")


@app.get("/ping")
def ping():
    return {"status": "server is alive"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # MOCK: ignores the actual image content for now and just returns a
    # random score, so the frontend can be built before the real model
    # is ready. Swap this out for real model inference later, keeping
    # the same {"pred": <float between 0 and 1>} response shape.
    await file.read()
    await asyncio.sleep(1)  # pretend this takes a moment, like a real model would
    return {"pred": round(random.random(), 4)}
