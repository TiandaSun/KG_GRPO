"""Classify trajectories -> per-sample category JSON.

Writes `{sample_id: category}` for each model, which task_v14_d1_strat.py
then slices by Cat A / Cat B.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from task16_classify import classify_trajectory  # noqa: E402

MODELS: dict[str, tuple[Path, int]] = {
    "e3_step500":  (Path("results/trajectories/phase7/e3_step500_full"), 500),
    "39b_step400": (Path("results/trajectories/phase7/39b_step400_full"), 400),
    "39g2_step500": (Path("results/trajectories/phase7/39g2_step500_full"), 0),  # merged uses step=0
    "e5b_step100": (Path("results/trajectories/phase7/e5b_step100_full"), 100),
}


def classify_model(tag: str, traj_dir: Path, step: int, out_path: Path) -> int:
    traj_file = traj_dir / f"step_{step}" / "trajectories.json"
    if not traj_file.exists():
        print(f"[skip] {tag}: {traj_file} not found")
        return 0
    with open(traj_file) as f:
        trajs = json.load(f)
    per_sample: dict[str, str] = {}
    for t in trajs:
        sid = str(t.get("sample_id", id(t)))
        per_sample[sid] = classify_trajectory(t)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"per_sample": per_sample, "n": len(per_sample)}, f, indent=2)
    print(f"{tag}: n={len(per_sample)}  wrote {out_path}")
    return len(per_sample)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=Path("results/phase7/per_sample_classifications"))
    ap.add_argument(
        "--extra", action="append", default=[],
        help="Custom model: tag:traj_dir:step, e.g. 14b_d1_500:results/trajectories/phase7/v14_d1_qwen14b_e5b_step400_cvt500:0",
    )
    args = ap.parse_args()

    models = dict(MODELS)
    for spec in args.extra:
        tag, tdir, step = spec.split(":")
        models[tag] = (Path(tdir), int(step))

    for tag, (tdir, step) in models.items():
        out = args.out_dir / f"{tag}_classification.json"
        classify_model(tag, tdir, step, out)


if __name__ == "__main__":
    main()
