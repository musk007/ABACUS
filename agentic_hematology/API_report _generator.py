"""
mllm_baseline.py
================
Zero-shot MLLM baseline for leukemia subtype classification + report generation,
to compare against the AgenticHematology pipeline.

Two backends behind ONE interface so the comparison is fair: identical prompt,
identical image handling, identical required output schema. Only the model differs.

Usage
-----
    export OPENAI_API_KEY=...        # for --backend openai
    export GOOGLE_API_KEY=...        # for --backend gemini

    python mllm_baseline.py \
        --backend openai \
        --model gpt-4o-2024-11-20 \
        --case-id 12 \
        --images "data/12/*.png" \
        --out outputs/mllm/openai

    # batch over a directory of patient folders (each with images/)
    python mllm_baseline.py --backend gemini --model gemini-1.5-pro-002 \
        --patients-dir data/test --out outputs/mllm/gemini

Fairness notes
--------------
- The MLLM receives the SAME tiles the pipeline sees (subject to per-call image
  caps; see --max-images and the montage fallback).
- temperature=0 and a pinned model string for reproducibility. RECORD the exact
  model string and run date in your paper — providers change models silently.
- This is task-specialised pipeline vs general-purpose ZERO-SHOT MLLM. Frame it
  as such; it is not a head-to-head of equals.
- Confirm the dataset licence permits sending images to a third-party API BEFORE
  running this.
"""
from __future__ import annotations

import argparse
import base64
import glob
import json
import mimetypes
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# The five supported subtypes — keep identical to the pipeline's label set.
SUBTYPES = ["ALL", "AML", "APML", "CLL", "CML"]

# ---------------------------------------------------------------------------
# Shared prompt — identical across backends. This is the crux of fairness.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a hematopathology assistant analysing peripheral blood smear (PBS) "
    "images for a research benchmark. You are an assistive tool: your output "
    "supports, and does not replace, a qualified hematologist. Base every "
    "statement only on what is visible in the provided images. Do not invent "
    "counts you cannot justify from the images."
)

USER_INSTRUCTION = f"""\
You are given all peripheral blood smear image tiles for a SINGLE patient.
Using only these images, produce a structured diagnostic assessment.

Choose the single most likely leukemia subtype from EXACTLY this set:
{", ".join(SUBTYPES)}.
Use the FAB/WHO framework: blast burden (>=20% of nucleated cells suggests acute
leukemia), lineage of the predominant abnormal population, degree of maturation,
and supporting features (e.g. abnormal promyelocytes for APML, basophilia and a
full granulocytic maturation spectrum for CML, mature lymphocytosis for CLL).

Return ONLY a JSON object, no prose or markdown fences, with this schema:
{{
  "predicted_subtype": one of {SUBTYPES},
  "confidence": float between 0 and 1,
  "key_findings": [ up to 6 short strings, each a morphological observation ],
  "differential_considerations": [ up to 4 short strings ],
  "report": "a concise (<=180 word) narrative diagnostic report phrased as an "
            "assistive summary, recommending clinical/laboratory correlation",
  "requires_human_review": boolean
}}
"""


def build_user_text() -> str:
    return USER_INSTRUCTION


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def resolve_images(specs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for spec in specs:
        for match in sorted(glob.glob(spec)):
            p = Path(match)
            if p.is_dir():
                paths.extend(sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_SUFFIXES))
            elif p.suffix.lower() in IMAGE_SUFFIXES:
                paths.append(p)
    return paths


def encode_b64(path: Path) -> tuple[str, str]:
    media_type = mimetypes.guess_type(str(path))[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return media_type, data


def maybe_montage(paths: list[Path], max_images: int, out_dir: Path | None) -> list[Path]:
    """If there are more tiles than the per-call cap, montage them into grids so
    no field of view is dropped. Requires Pillow. Returns the montage paths."""
    if len(paths) <= max_images:
        return paths
    try:
        from PIL import Image
    except ImportError:
        print(
            f"WARNING: {len(paths)} tiles > cap {max_images} and Pillow not "
            "installed; sending the first {max_images} tiles only. Install "
            "Pillow to montage instead (fairer).",
            file=sys.stderr,
        )
        return paths[:max_images]

    import math

    per_montage = math.ceil(len(paths) / max_images)
    grid = math.ceil(math.sqrt(per_montage))
    montages: list[Path] = []
    work = out_dir or Path(".")
    work.mkdir(parents=True, exist_ok=True)

    for mi in range(max_images):
        chunk = paths[mi * per_montage : (mi + 1) * per_montage]
        if not chunk:
            break
        thumbs = [Image.open(p).convert("RGB") for p in chunk]
        w = max(t.width for t in thumbs)
        h = max(t.height for t in thumbs)
        canvas = Image.new("RGB", (w * grid, h * grid), "white")
        for idx, t in enumerate(thumbs):
            r, c = divmod(idx, grid)
            canvas.paste(t.resize((w, h)), (c * w, r * h))
        mpath = work / f"_montage_{mi}.png"
        canvas.save(mpath)
        montages.append(mpath)
    print(f"  Montaged {len(paths)} tiles into {len(montages)} grid image(s).")
    return montages


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class OpenAIBackend:
    def __init__(self, model: str, temperature: float = 0.0):
        from openai import OpenAI
        self.client = OpenAI()  # reads OPENAI_API_KEY
        self.model = model
        self.temperature = temperature

    def run(self, image_paths: list[Path], system: str, user: str) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": user}]
        for p in image_paths:
            media_type, data = encode_b64(p)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{data}"},
            })
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
        )
        return resp.choices[0].message.content or ""


class GeminiBackend:
    def __init__(self, model: str, temperature: float = 0.0):
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.genai = genai
        self.model = genai.GenerativeModel(model)
        self.temperature = temperature

    def run(self, image_paths: list[Path], system: str, user: str) -> str:
        parts: list[Any] = [system + "\n\n" + user]
        for p in image_paths:
            media_type, _ = encode_b64(p)
            parts.append({"mime_type": media_type, "data": p.read_bytes()})
        resp = self.model.generate_content(
            parts,
            generation_config={"temperature": self.temperature},
        )
        return resp.text or ""


def make_backend(name: str, model: str, temperature: float):
    if name == "openai":
        return OpenAIBackend(model, temperature)
    if name == "gemini":
        return GeminiBackend(model, temperature)
    sys.exit(f"Unknown backend: {name}")


# ---------------------------------------------------------------------------
# Parsing + IO
# ---------------------------------------------------------------------------

def parse_json(raw: str) -> dict[str, Any]:
    """Extract the first JSON object; tolerate code fences and stray prose."""
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return {"_parse_error": True, "raw": raw}
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"_parse_error": True, "raw": raw}
    # Normalise the subtype to the canonical label set if possible.
    st = str(obj.get("predicted_subtype", "")).upper().strip()
    obj["predicted_subtype"] = st if st in SUBTYPES else obj.get("predicted_subtype")
    return obj


def run_one(backend, model: str, case_id: str, paths: list[Path],
            max_images: int, out_dir: Path | None) -> dict[str, Any]:
    print(f"\n{'='*60}\nCase {case_id}: {len(paths)} tiles", flush=True)
    if not paths:
        return {"case_id": case_id, "error": "no images"}

    montage_dir = (out_dir / case_id) if out_dir else None
    send_paths = maybe_montage(paths, max_images, montage_dir)

    t0 = time.time()
    try:
        raw = backend.run(send_paths, SYSTEM_PROMPT, build_user_text())
    except Exception as e:
        print(f"  API call failed: {e}", file=sys.stderr)
        return {"case_id": case_id, "error": str(e)}
    elapsed = round(time.time() - t0, 1)

    parsed = parse_json(raw)
    result = {
        "case_id": case_id,
        "model": model,
        "n_tiles": len(paths),
        "n_images_sent": len(send_paths),
        "elapsed_s": elapsed,
        "raw_response": raw,
        **parsed,
    }
    if out_dir:
        case_dir = out_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / f"case_{case_id}_mllm.json").write_text(json.dumps(result, indent=2))
        print(f"  Wrote {case_dir / f'case_{case_id}_mllm.json'}  "
              f"(subtype={result.get('predicted_subtype')}, {elapsed}s)")
    else:
        print(json.dumps(result, indent=2))
    return result


def discover_patients(patients_dir: Path) -> list[tuple[str, list[Path]]]:
    out: list[tuple[str, list[Path]]] = []
    for sub in sorted(patients_dir.iterdir()):
        if not sub.is_dir():
            continue
        img_dir = sub / "images" if (sub / "images").is_dir() else sub
        imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if imgs:
            out.append((sub.name, imgs))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Zero-shot MLLM baseline for leukemia subtyping.")
    ap.add_argument("--backend", choices=["openai", "gemini"], required=True)
    ap.add_argument("--model", required=True,
                    help="Pinned model string, e.g. gpt-4o-2024-11-20 or gemini-1.5-pro-002.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-images", type=int, default=16,
                    help="Per-call image cap; tiles beyond this are montaged.")
    ap.add_argument("--case-id")
    ap.add_argument("--images", nargs="*")
    ap.add_argument("--patients-dir", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    backend = make_backend(args.backend, args.model, args.temperature)

    if args.case_id:
        paths = resolve_images(args.images or [])
        run_one(backend, args.model, args.case_id, paths, args.max_images, args.out)
        return 0

    if not args.patients_dir:
        sys.exit("Provide --case-id + --images, or --patients-dir for batch mode.")

    patients = discover_patients(args.patients_dir)
    print(f"Batch: {len(patients)} patients", flush=True)
    summary = []
    for case_id, paths in patients:
        summary.append(run_one(backend, args.model, case_id, paths, args.max_images, args.out))

    if args.out:
        (args.out / f"_summary_{args.backend}.json").write_text(json.dumps(summary, indent=2))
        print(f"\nWrote summary for {len(summary)} patients.")
    return 0


if __name__ == "__main__":
    sys.exit(main())