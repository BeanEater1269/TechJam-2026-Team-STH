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
in checkpoints/. --seed (default 42) is passed to each train_*.py job and seeds
random/numpy/torch before model init, so re-runs are reproducible on the same machine --
but not guaranteed bit-identical across different hardware/CUDA versions/torch builds,
since GPU op nondeterminism isn't separately forced here.

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

import numpy as np

PYTHON = sys.executable
SCRIPTS_DIR = Path(__file__).parent.parent  # this file lives in scripts/evaluation/
REPO_ROOT = SCRIPTS_DIR.parent
EMBEDDINGS_ROOT = REPO_ROOT / "data/cache/embeddings"  # matches every job script's --embeddings-root default

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

# The 3 tables print_fpr_table() (eval_metrics.py) produces: name, n, accuracy, auc,
# then one fpr@<threshold> column and one fnr@<threshold> column per FPR_THRESHOLDS
# entry (currently 3: 0.3/0.5/0.7) -- 10 columns total. float() parses "nan" natively,
# so no special-casing needed for the undefined-AUC/FPR single-class slices (e.g.
# StyleGAN-XL, which is fake-only -- see eval_metrics.py).
FPR_SECTION_TITLES = {
    "False positive rate by threshold (per variant):": "fpr_by_variant",
    "By source_dataset (clean images only):": "by_source_dataset",
    "By generator_family (clean images only):": "by_generator_family",
}
FPR_ROW_RE = re.compile(
    r"^(\S+)\s+(\d+)\s+([\d.]+|nan)\s+([\d.]+|nan)"
    r"\s+([\d.]+|nan)\s+([\d.]+|nan)\s+([\d.]+|nan)"
    r"\s+([\d.]+|nan)\s+([\d.]+|nan)\s+([\d.]+|nan)$"
)


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
    prints an identical structure -- one parser covers all three:
      1. per-variant table + overall row + robustness summary
      2. "False positive rate by threshold (per variant):" table
      3. "By source_dataset (clean images only):" table
      4. "By generator_family (clean images only):" table

    "ALL (16)" is handled separately from the per-variant rows since its name
    contains a space, unlike every actual variant name. Section 1's table is parsed
    by VARIANT_LINE_RE/ALL_LINE_RE (4 columns); sections 2-4 share one wider format
    (10 columns, via FPR_ROW_RE) and are told apart by which title line preceded them,
    tracked as `section` while scanning line by line."""
    per_variant = {}
    overall = None
    summary = {}
    fpr_sections = {key: {} for key in FPR_SECTION_TITLES.values()}
    section = None  # which of fpr_sections we're currently inside, if any

    for raw_line in output.splitlines():
        line = raw_line.strip()

        if line in FPR_SECTION_TITLES:
            section = FPR_SECTION_TITLES[line]
            continue

        if section is not None:
            m_fpr = FPR_ROW_RE.match(line)
            if m_fpr:
                fpr_sections[section][m_fpr.group(1)] = {
                    "n": int(m_fpr.group(2)),
                    "accuracy": float(m_fpr.group(3)),
                    "auc": float(m_fpr.group(4)),
                    "fpr": {
                        "0.3": float(m_fpr.group(5)),
                        "0.5": float(m_fpr.group(6)),
                        "0.7": float(m_fpr.group(7)),
                    },
                    "fnr": {
                        "0.3": float(m_fpr.group(8)),
                        "0.5": float(m_fpr.group(9)),
                        "0.7": float(m_fpr.group(10)),
                    },
                }
            continue  # once in a section, every line belongs to it until the next title

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
    for key, title in ((v, k) for k, v in FPR_SECTION_TITLES.items()):
        if not fpr_sections[key]:
            print(f"  WARNING: no rows matched for '{title}' -- eval output format may have changed")
    return {
        "per_variant": per_variant,
        "overall": overall,
        "robustness_summary": summary,
        "fpr_by_variant": fpr_sections["fpr_by_variant"],
        "by_source_dataset": fpr_sections["by_source_dataset"],
        "by_generator_family": fpr_sections["by_generator_family"],
    }


BACKBONE_LABELS = {512: "ViT-B/32", 768: "ViT-L/14"}
BACKBONE_TAGS = {512: "b32", 768: "l14"}


def check_embeddings_match_backbone(backbone_dim: int, backbone_label: str) -> None:
    """Every job script here (train_*.py, evaluate_*.py) reads
    data/cache/embeddings/{train,val,test}.npz via its own --embeddings-root default --
    there's no backbone tag in those filenames, so extract_embeddings.py --backbone b32
    and --backbone l14 both write to the exact same paths, overwriting whichever ran
    before. If --backbone-dim here doesn't match what's actually sitting in those .npz
    files (e.g. you re-ran extract_embeddings.py for the other backbone but forgot to
    also change --backbone-dim, or vice versa), every job would still "succeed" --
    nn.Linear(clip_dim, ...) throws a hard shape-mismatch RuntimeError on the very first
    batch, which run_script() turns into a RuntimeError after burning through however
    much of training already ran -- so this checks all 3 splits up front, before any
    subprocess starts, and fails fast with the actual mismatch instead of a generic
    torch stack trace after minutes of wasted training."""
    for split in ("train", "val", "test"):
        npz_path = EMBEDDINGS_ROOT / f"{split}.npz"
        if not npz_path.exists():
            raise SystemExit(f"{npz_path} not found -- run extract_embeddings.py first.")
        actual_dim = int(np.load(npz_path, allow_pickle=True)["embeddings"].shape[1])
        if actual_dim != backbone_dim:
            raise SystemExit(
                f"embeddings/backbone mismatch: {npz_path} holds {actual_dim}-dim embeddings, "
                f"but --backbone-dim={backbone_dim} ({backbone_label}) was requested. "
                f"Either re-run extract_embeddings.py --backbone "
                f"{BACKBONE_TAGS.get(backbone_dim, '?')} to regenerate matching embeddings, "
                f"or pass --backbone-dim {actual_dim} to match what's currently cached."
            )
    print(f"  embeddings check OK: train/val/test.npz all {backbone_dim}-dim, matches {backbone_label}")


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
    ap.add_argument("--seed", type=int, default=42,
                     help="Passed to each train_*.py job as --seed (eval scripts have no "
                          "randomness -- no shuffling, no dropout at eval() -- so it's not "
                          "passed to them). Same seed for base/concat/film keeps the 3 models "
                          "comparable to each other, not just reproducible run-to-run.")
    args = ap.parse_args()

    backbone_label = args.backbone_label or BACKBONE_LABELS.get(args.backbone_dim, f"dim={args.backbone_dim}")
    out_dir_name = args.out_dir or f"results_{BACKBONE_TAGS.get(args.backbone_dim, f'dim{args.backbone_dim}')}"
    out_dir = REPO_ROOT / out_dir_name
    print(f"backbone: {backbone_label} (dim={args.backbone_dim}) -> writing results to {out_dir}")

    check_embeddings_match_backbone(args.backbone_dim, backbone_label)

    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for run_name, script, kind, model_label, signals, signals_normalized in JOBS:
        extra_args = ["--backbone-dim", str(args.backbone_dim)]
        if kind == "train":
            extra_args += ["--seed", str(args.seed)]
        elif kind == "eval":
            # Errors go under this run's own results_b32/results_l14 dir (not the
            # eval script's own relative "results/errors" default), so a b32 run's
            # FP/FN CSVs never get silently overwritten by a later l14 run.
            extra_args += ["--errors-dir", str(out_dir / "errors")]
        output = run_script(script, extra_args)
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
            "seed": args.seed if kind == "train" else None,
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
