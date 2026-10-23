#!/usr/bin/env python3
"""Build GRPO parquet with pred-summary prompts + CV det/attr scores in extra_info."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import os

REPORT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPORT_ROOT))
from bootstrap import DEFAULT_CONFIG, PROJECT_ROOT, setup_paths

setup_paths()

from src.config import load_config
from src.grpo_e2e_data import build_grpo_parquet, rebuild_sft_from_new_gt
from src.verl_parquet import patient_split_from_images, patient_split_from_folds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--stats-json", type=Path, default=None,
                   help="GT patient_WBC_stats_100X.json (default: cfg['gt_stats_json'])")
    p.add_argument("--images-root", type=Path, default=None,
                   help="<root>/{train,test}/<pid>_*.png for the patient split (default: <data_root>/images)")
    p.add_argument("--exclude", nargs="*", default=["24"], help="Patient IDs to drop (bad GT)")
    p.add_argument("--rebuild-sft", action="store_true", help="Kept for train.sh compatibility (always rebuilt)")
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--folds-json", type=Path,
                default=Path(os.environ.get("CV_FOLDS_JSON", "configs/cv_folds.json")))
    args = p.parse_args()

    cfg = load_config(args.config)
    artifact = Path(cfg["artifact_root"])
    stats_json = args.stats_json or (Path(cfg["gt_stats_json"]) if cfg.get("gt_stats_json") else None)
    if stats_json is None or not stats_json.is_file():
        raise SystemExit(f"Missing GT stats JSON: {stats_json}. Pass --stats-json.")
    images_root = args.images_root or Path(cfg["data_root"]) / "images"
    split = (patient_split_from_images(images_root) if args.fold is None
             else patient_split_from_folds(args.folds_json, args.fold))
    src = images_root if args.fold is None else f"fold {args.fold}"
    print(f"Split from {src}: train={len(split.get('train', []))} test={len(split.get('test', []))}")
    print(f"Split from {images_root}: train={len(split.get('train', []))} test={len(split.get('test', []))}")

    sft_jsonl = Path(cfg["output"]["sft_dataset"])
    n = rebuild_sft_from_new_gt(stats_json, Path(cfg["reports_gt_dir"]), sft_jsonl, exclude=set(args.exclude))
    print(f"SFT jsonl: {n} rows -> {sft_jsonl}")

    grpo_dir = Path(cfg.get("verl", {}).get("grpo_e2e_parquet", artifact / "data/verl/grpo_e2e"))
    train_p, val_p = build_grpo_parquet(sft_jsonl, grpo_dir, split)
    print(f"GRPO train={train_p}")
    print(f"GRPO val={val_p}")


if __name__ == "__main__":
    main()


