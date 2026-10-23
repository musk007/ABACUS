"""Convert report SFT JSONL to verl SFT parquet."""
from __future__ import annotations

import json

from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_messages_for_verl(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Qwen3.5 + verl MultiTurnSFT: merge system into the first user turn.

    Rendering system alone triggers TemplateError (system must lead or pair with user).
    """
    system_chunks: list[str] = []
    turns: list[dict[str, str]] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            system_chunks.append(content)
        elif role in ("user", "assistant"):
            turns.append({"role": role, "content": content})
        else:
            turns.append({"role": role, "content": content})

    if not system_chunks:
        return turns

    prefix = "\n\n".join(system_chunks).strip()
    merged = False
    out: list[dict[str, str]] = []
    for msg in turns:
        if msg["role"] == "user" and not merged:
            out.append({"role": "user", "content": f"{prefix}\n\n{msg['content']}"})
            merged = True
        else:
            out.append(dict(msg))
    if not merged:
        out.insert(0, {"role": "user", "content": prefix})
    return out

def patient_split_from_images(images_root: Path) -> dict[str, list[str]]:
    """LLD patient split from <images_root>/{train,test}/<pid>_*.png (same split the classifier uses)."""
    split: dict[str, set[str]] = {}
    for name in ("train", "test"):
        d = Path(images_root) / name
        if d.is_dir():
            split[name] = {p.stem.split("_")[0] for p in d.iterdir() if not p.name.startswith(".")}
    overlap = split.get("train", set()) & split.get("test", set())
    if overlap:
        raise ValueError(f"Patients in both train and test: {sorted(overlap)}")
    return {k: sorted(v) for k, v in split.items()}

def patient_split_from_folds(folds_json: Path, fold: int) -> dict[str, list[str]]:
    """Patient split from the frozen CV folds: held-out fold = test, the other four = train."""
    folds = json.loads(Path(folds_json).read_text())["folds"]
    if not 0 <= fold < len(folds):
        raise ValueError(f"fold {fold} out of range (0..{len(folds) - 1})")
    test = set(folds[fold])
    train = {p for k, f in enumerate(folds) if k != fold for p in f}
    return {"train": sorted(train), "test": sorted(test)}

def build_sft_parquet(
    jsonl_path: Path,
    out_dir: Path,
    split: dict[str, list[str]],
) -> tuple[Path, Path]:
    """SFT parquet with `messages` (verl data.messages_key=messages).

    train.parquet <- split["train"], val.parquet <- split["test"].
    Extra fields go to `extra_info` for evaluation / reward.
    """
    try:
        import datasets
    except ImportError as e:
        raise ImportError("pip install datasets pyarrow") from e

    rows = _load_jsonl(jsonl_path)
    train_ids, test_ids = set(split.get("train", [])), set(split.get("test", []))
    unassigned = sorted(r["patient_id"] for r in rows if r["patient_id"] not in train_ids | test_ids)
    if unassigned:
        print(f"[build_sft_parquet] not in split, skipped: {unassigned}")

    def to_sft(rec: dict, split_name: str, idx: int) -> dict:
        return {
            "patient_id": rec["patient_id"],
            "messages": normalize_messages_for_verl(rec["messages"]),
            "extra_info": {
                "split": split_name,
                "index": idx,
                "patient_id": rec["patient_id"],
                "case_json": rec.get("case_json", ""),
                "template_md": rec.get("template_md", ""),
                "cohort_cell_type": rec.get("cohort_cell_type", ""),
            },
        }

    train_rows = [r for r in rows if r["patient_id"] in train_ids]
    val_rows = [r for r in rows if r["patient_id"] in test_ids]

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    datasets.Dataset.from_list([to_sft(r, "train", i) for i, r in enumerate(train_rows)]).to_parquet(str(train_path))
    datasets.Dataset.from_list([to_sft(r, "val", i) for i, r in enumerate(val_rows)]).to_parquet(str(val_path))
    print(f"[build_sft_parquet] train={len(train_rows)} val={len(val_rows)}")
    return train_path, val_path