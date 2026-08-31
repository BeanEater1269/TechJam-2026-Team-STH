"""One-off script: consolidates the real eval result files in results_b32/ and
results_l14/ into the single results.json shape the webdemo dashboard reads. Run from
repo root: python scripts/build_dashboard_data.py

Per-backbone model lists, NOT one shared list -- b32 is a frozen historical snapshot
(embeddings retired, never re-run with the newer models) and only ever had
base/concat/film. l14 is the live comparison set, extended each time a new model was
added -- currently includes concat_drift (the model the team settled on) and
concat_drift_liqe (kept as the "we tried LIQE, it made things worse" comparison point).
Also carries an explicit `order` list per backbone (dict key order isn't a display
contract) so the dashboard JS renders models in a stable, chosen sequence rather than
whatever order json.loads() happens to produce.
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent

# (results_dir, {model_key: display_name}, [render order])
BACKBONES = {
    "b32": ("results_b32", {"base": "Base", "concat": "Concatenation", "film": "FiLM"}),
    "l14": ("results_l14", {
        "base": "Base",
        "concat": "Concatenation",
        "concat_drift": "Concat + Drift",
        "concat_drift_liqe": "Concat + Drift + LIQE",
        "film": "FiLM",
    }),
}

out = {"backbones": {}}

for backbone_key, (results_dir, models) in BACKBONES.items():
    order = list(models.keys())
    out["backbones"][backbone_key] = {"order": order, "models": {}}
    for model_key, display_name in models.items():
        eval_path = ROOT / results_dir / f"{model_key}_eval.json"
        data = json.loads(eval_path.read_text())
        out["backbones"][backbone_key]["models"][model_key] = {
            "display_name": display_name,
            "results": data["results"]["per_variant"],
            "summary": {
                "overall": data["results"]["overall"],
                "clean": data["results"]["robustness_summary"]["clean"],
                "mean_15_variants": data["results"]["robustness_summary"]["mean_15_variants"],
                "gap": data["results"]["robustness_summary"]["gap"],
            },
        }

out_path = ROOT / "webdemo" / "static" / "results.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"wrote {out_path}")
