"""
Persistence for /api/live/analyze batch runs. Mirrors the results_b32/results_l14
"JSON on disk is the source of truth, the page just reads it back" pattern -- one run
is one timestamped directory under results_judge/, holding run.json (metadata + per-
image results) plus the standardized images themselves (each one doubles as its own
thumbnail AND a reproducibility artifact: re-running inference on a saved image
reproduces its stored raw_prob exactly, since it's already exactly what the model saw).

Single-image uploads never call save_run() at all -- inline result only, nothing
persisted, per the product rule (1 image = no dashboard, no disk write).
"""
import json
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path

from inference import BACKBONE_DIM, BACKBONE_LABEL, MODEL_CHECKPOINT, THRESHOLD, to_record

RUN_ID_RE = re.compile(r"^run_\d{8}T\d{6}Z_[0-9a-f]{6}$")


def new_run_id() -> str:
    """UTC-ISO-basic timestamp + 6 hex chars: lexicographic sort == chronological sort
    (no index file needed to list runs), random suffix guards same-second collisions."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"run_{ts}_{secrets.token_hex(3)}"


def sanitize_stored_filename(original_name: str, index: int) -> str:
    """UploadFile.filename is attacker-controlled -- never join it into a path
    unsanitized (path traversal via '../'). The {index:03d} prefix also guarantees
    uniqueness within a run regardless of what the slug collapses to."""
    stem = Path(original_name or "image").stem
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", stem)[:40] or "image"
    return f"{index:03d}_{slug}.jpg"


def save_run(results_judge_root: Path, results: list, original_filenames: list) -> dict:
    """results: list of Predictor.predict() dicts (each still carries its std_image).
    Saves every standardized image under images/ and writes run.json; returns the
    parsed run.json dict (same shape GET /api/live/runs/{run_id} returns)."""
    run_id = new_run_id()
    run_dir = results_judge_root / run_id
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    per_image = []
    n_fake = 0
    confidences = []
    for i, (result, orig_name) in enumerate(zip(results, original_filenames)):
        stored_name = sanitize_stored_filename(orig_name, i)
        result["std_image"].convert("RGB").save(images_dir / stored_name, format="JPEG", quality=100)

        record = to_record(result)
        record["index"] = i
        record["original_filename"] = orig_name
        record["stored_image"] = f"images/{stored_name}"
        per_image.append(record)

        if record["label"] == "FAKE":
            n_fake += 1
        confidences.append(record["confidence"])

    n = len(per_image)
    summary = {
        "n": n,
        "n_fake": n_fake,
        "n_real": n - n_fake,
        "fake_rate": (n_fake / n) if n else 0.0,
        "mean_confidence": (sum(confidences) / n) if n else 0.0,
        "min_confidence": min(confidences) if confidences else None,
        "max_confidence": max(confidences) if confidences else None,
    }

    run_json = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kind": "judge",
        "model": "concat_drift",
        "backbone": BACKBONE_LABEL,
        "backbone_dim": BACKBONE_DIM,
        "checkpoint": f"checkpoints/{MODEL_CHECKPOINT.name}",
        "threshold": THRESHOLD,
        "n_images": n,
        "results": {"per_image": per_image, "summary": summary},
    }
    (run_dir / "run.json").write_text(json.dumps(run_json, indent=2))
    return run_json


def load_run(results_judge_root: Path, run_id: str) -> dict | None:
    """Validates run_id against RUN_ID_RE BEFORE touching the filesystem -- an
    unvalidated run_id joined into a path is a path-traversal read primitive."""
    if not RUN_ID_RE.match(run_id):
        return None
    run_path = results_judge_root / run_id / "run.json"
    if not run_path.exists():
        return None
    return json.loads(run_path.read_text())
