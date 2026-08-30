"""One-off script: consolidates the 6 real eval result files (base/concat/film x
b32/l14) in results_b32/ and results_l14/ into the single results.json shape the
webdemo dashboard reads. Run from repo root: python scripts/build_dashboard_data.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
BACKBONES = {"b32": "results_b32", "l14": "results_l14"}
MODELS = {"base": "Base", "concat": "Concatenation", "film": "FiLM"}

out = {"backbones": {}}

for backbone_key, results_dir in BACKBONES.items():
    out["backbones"][backbone_key] = {"models": {}}
    for model_key, display_name in MODELS.items():
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
