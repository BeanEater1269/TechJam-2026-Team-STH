"""
Runs training/train_base.py, training/train_concat_normalize.py,
training/train_film_normalize.py, then evaluation/evaluate_base.py,
evaluation/evaluate_concat.py, evaluation/evaluate_film.py in sequence, and saves each
run's results into a new results/ directory as JSON -- tagged with which model
(base/concat/film), which signals were used, and which CLIP backbone (default
ViT-B/32, backbone_dim=512).

Nothing is parsed by hand-inspecting output -- each script's stdout is machine-parsed
against the exact print formats those scripts use (same formats across all 3 training
scripts, and across all 3 eval scripts), so results are pulled as real numbers, not
copy-pasted.

WARNING: this retrains all three models from scratch, overwriting whatever's currently
in checkpoints/. None of the training scripts fix a random seed, so re-run numbers will
be close to, but not bit-identical to, a previous run.

Usage:
    python scripts/evaluation/collect_results.py
    python scripts/evaluation/collect_results.py --backbone-dim 512 --backbone-label "ViT-B/32"
"""
import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable
SCRIPTS_DIR = Path(__file__).parent.parent  # this file lives in scripts/evaluation/
REPO_ROOT = SCRIPTS_DIR.parent

SIGNAL_COLUMNS = ["laplacian_var", "dct_low_energy", "dct_high_energy", "noise_variance"]

# (run_name, script (relative to scripts/), kind, model_label, signals, signals_normalized)
JOBS = [
    ("base_train", "training/train_base.py", "train", "base", [], None),
    ("concat_train", "training/train_concat_normalize.py", "train", "concat", SIGNAL_COLUMNS, True),
    ("film_train", "training/train_film_normalize.py", "train", "film", SIGNAL_COLUMNS, True),
    ("base_eval", "evaluation/evaluate_base.py", "eval", "base", [], None),
    ("concat_eval", "evaluation/evaluate_concat.py", "eval", "concat", SIGNAL_COLUMNS, True),
    ("film_eval", "evaluation/evaluate_film.py", "eval", "film", SIGNAL_COLUMNS, True),
]

EPOCH_RE = re.compile(r"epoch (\d+): avg loss ([\d.]+), val acc ([\d.]+), val auc ([\d.]+)")
VARIANT_LINE_RE = re.compile(r"^(\S+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)$")
ALL_LINE_RE = re.compile(r"^ALL \(16\)\s+(\d+)\s+([\d.]+)\s+([\d.]+)$")
SUMMARY_LINE_RE = re.compile(r"acc\s+([+-]?[\d.]+),\s+auc\s+([+-]?[\d.]+)")


def run_script(script: str, extra_args: list) -> str:
    """Runs one of the 6 scripts as a subprocess, with cwd forced to the repo root --
    all 6 use relative default paths (data/cache/..., checkpoints/...) that resolve
    against the repo root, not against scripts/, so cwd has to match how they're
    normally invoked by hand."""
    cmd = [PYTHON, str(SCRIPTS_DIR / script), *extra_args]
    print(f"\n{'=' * 70}\nRunning: {' '.join(cmd)}\n{'=' * 70}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    print(result.stdout[-2000:])
    if result.returncode != 0:
        print(result.stderr[-3000:])
        raise RuntimeError(f"{script} failed with exit code {result.returncode}")
    return result.stdout


def parse_train_output(output: str) -> dict:
    """Every training script (train_base.py, train_concat_normalize.py,
    train_film_normalize.py) prints an identical per-epoch line -- one regex covers
    all three."""
    epochs = []
    for m in EPOCH_RE.finditer(output):
        epochs.append({
            "epoch": int(m.group(1)),
            "avg_loss": float(m.group(2)),
            "val_acc": float(m.group(3)),
            "val_auc": float(m.group(4)),
        })
    if not epochs:
        print("  WARNING: no epoch lines matched -- training output format may have changed")
    return {"epochs": epochs, "final": epochs[-1] if epochs else None}


def parse_eval_output(output: str) -> dict:
    """Every eval script (evaluate_base.py, evaluate_concat.py, evaluate_film.py)
    prints an identical per-variant table + overall row + robustness summary -- one
    parser covers all three. "ALL (16)" is handled separately from the per-variant
    rows since its name contains a space, unlike every actual variant name."""
    per_variant = {}
    overall = None
    summary = {}

    for raw_line in output.splitlines():
        line = raw_line.strip()

        m_all = ALL_LINE_RE.match(line)
        if m_all:
            overall = {"n": int(m_all.group(1)), "accuracy": float(m_all.group(2)), "auc": float(m_all.group(3))}
            continue

        m_var = VARIANT_LINE_RE.match(line)
        if m_var:
            per_variant[m_var.group(1)] = {
                "n": int(m_var.group(2)),
                "accuracy": float(m_var.group(3)),
                "auc": float(m_var.group(4)),
            }
            continue

        if line.startswith("clean:"):
            m = SUMMARY_LINE_RE.search(line)
            if m:
                summary["clean"] = {"acc": float(m.group(1)), "auc": float(m.group(2))}
        elif line.startswith("mean over 15 variants:"):
            m = SUMMARY_LINE_RE.search(line)
            if m:
                summary["mean_15_variants"] = {"acc": float(m.group(1)), "auc": float(m.group(2))}
        elif line.startswith("gap (clean"):
            m = SUMMARY_LINE_RE.search(line)
            if m:
                summary["gap"] = {"acc": float(m.group(1)), "auc": float(m.group(2))}

    if overall is None:
        print("  WARNING: no 'ALL (16)' line matched -- eval output format may have changed")
    return {"per_variant": per_variant, "overall": overall, "robustness_summary": summary}


BACKBONE_LABELS = {512: "ViT-B/32", 768: "ViT-L/14"}
BACKBONE_TAGS = {512: "b32", 768: "l14"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone-dim", type=int, default=512)
    ap.add_argument("--backbone-label", default=None,
                     help="human-readable label recorded alongside every result, e.g. 'ViT-B/32'. "
                          "Auto-derived from --backbone-dim if not given (512 -> ViT-B/32, 768 -> ViT-L/14).")
    ap.add_argument("--out-dir", default=None,
                     help="Auto-derived from --backbone-dim if not given -- results_b32/ for 512, "
                          "results_l14/ for 768 -- so different backbones never overwrite each other's "
                          "results. Pass explicitly to override.")
    args = ap.parse_args()

    backbone_label = args.backbone_label or BACKBONE_LABELS.get(args.backbone_dim, f"dim={args.backbone_dim}")
    out_dir_name = args.out_dir or f"results_{BACKBONE_TAGS.get(args.backbone_dim, f'dim{args.backbone_dim}')}"
    out_dir = REPO_ROOT / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"backbone: {backbone_label} (dim={args.backbone_dim}) -> writing results to {out_dir}")

    summary_rows = []

    for run_name, script, kind, model_label, signals, signals_normalized in JOBS:
        output = run_script(script, ["--backbone-dim", str(args.backbone_dim)])
        parsed = parse_train_output(output) if kind == "train" else parse_eval_output(output)

        record = {
            "run_name": run_name,
            "script": script,
            "kind": kind,
            "model": model_label,
            "signals": signals,
            "signals_normalized": signals_normalized,
            "backbone": backbone_label,
            "backbone_dim": args.backbone_dim,
            "results": parsed,
        }

        out_path = out_dir / f"{run_name}.json"
        out_path.write_text(json.dumps(record, indent=2))
        print(f"  saved {out_path}")

        row = {
            "run": run_name, "kind": kind, "model": model_label,
            "signals": ",".join(signals) if signals else "none",
            "signals_normalized": signals_normalized, "backbone": backbone_label,
        }
        if kind == "train" and parsed["final"]:
            row.update({
                "val_acc": parsed["final"]["val_acc"], "val_auc": parsed["final"]["val_auc"],
                "avg_loss": parsed["final"]["avg_loss"],
            })
        elif kind == "eval" and parsed["overall"]:
            row.update({
                "overall_acc": parsed["overall"]["accuracy"], "overall_auc": parsed["overall"]["auc"],
                "clean_acc": parsed["robustness_summary"].get("clean", {}).get("acc"),
                "clean_auc": parsed["robustness_summary"].get("clean", {}).get("auc"),
                "mean15_acc": parsed["robustness_summary"].get("mean_15_variants", {}).get("acc"),
                "mean15_auc": parsed["robustness_summary"].get("mean_15_variants", {}).get("auc"),
            })
        summary_rows.append(row)

    (out_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2))
    if summary_rows:
        fieldnames = sorted({k for row in summary_rows for k in row})
        with open(out_dir / "summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"\nAll done. Per-run JSON + summary.json + summary.csv written to {out_dir}/")


if __name__ == "__main__":
    main()
