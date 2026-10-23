"""GRPO parquet for the one-section morphology task.

Built from the SFT JSONL (build_sft_jsonl) so prompts, targets and split are identical to SFT.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .sft_dataset import build_sft_jsonl
from .verl_parquet import _load_jsonl, normalize_messages_for_verl

DATA_SOURCE = "leukemia_morph"


def build_grpo_parquet(
    sft_jsonl: Path,
    out_dir: Path,
    split: dict[str, list[str]],
) -> tuple[Path, Path]:
    try:
        import datasets
    except ImportError as e:
        raise ImportError("pip install datasets pyarrow") from e

    rows = _load_jsonl(sft_jsonl)
    train_ids, test_ids = set(split.get("train", [])), set(split.get("test", []))

    def to_row(rec: dict[str, Any], split_name: str, idx: int) -> dict[str, Any]:
        msgs = rec["messages"]
        prompt = [m for m in msgs if m["role"] != "assistant"]
        target = next(m["content"] for m in msgs if m["role"] == "assistant")
        return {
            "data_source": DATA_SOURCE,
            "prompt": normalize_messages_for_verl(prompt),
            "ability": "hematology_morphology",
            "reward_model": {"style": "rule", "ground_truth": target},
            "extra_info": {
                "split": split_name,
                "index": idx,
                "patient_id": rec["patient_id"],
                "case_json": rec.get("case_json", ""),
                "template_md": rec.get("template_md", ""),
                "cohort_cell_type": rec.get("cohort_cell_type", ""),
            },
        }

    train = [r for r in rows if r["patient_id"] in train_ids]
    val = [r for r in rows if r["patient_id"] in test_ids]

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    datasets.Dataset.from_list([to_row(r, "train", i) for i, r in enumerate(train)]).to_parquet(str(train_path))
    datasets.Dataset.from_list([to_row(r, "val", i) for i, r in enumerate(val)]).to_parquet(str(val_path))
    print(f"[build_grpo_parquet] train={len(train)} val={len(val)}")
    return train_path, val_path


def rebuild_sft_from_new_gt(
    stats_json: Path,
    reports_gt_dir: Path,
    sft_jsonl: Path,
    exclude: set[str] | None = None,
) -> int:
    return build_sft_jsonl(stats_json, reports_gt_dir, sft_jsonl, exclude=exclude)