#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


GROUNDING_RE = re.compile(
    r"-\s+`(?P<cell_id>[^`]+)`\s+in\s+`(?P<image_id>[^`]+)`\s+"
    r"bbox=\[(?P<bbox>[^\]]+)\]:\s+"
    r"(?P<cell_type>.+?)\s+\((?P<conf>[0-9.]+)\)"
)

# Use RGB so PIL and HTML report share the exact same color.
COLORS = [
    ("red", "#ff0000", (255, 0, 0)),
    ("lime", "#00cc00", (0, 204, 0)),
    ("cyan", "#00cccc", (0, 204, 204)),
    ("yellow", "#d9d900", (217, 217, 0)),
    ("magenta", "#cc00cc", (204, 0, 204)),
    ("orange", "#ff8800", (255, 136, 0)),
    ("deepskyblue", "#00aaff", (0, 170, 255)),
    ("violet", "#9933ff", (153, 51, 255)),
    ("springgreen", "#00cc66", (0, 204, 102)),
    ("purple", "#6600cc", (102, 0, 204)),
]


def parse_report(report_path: Path) -> list[dict]:
    text = report_path.read_text(encoding="utf-8")
    records = []

    for idx, match in enumerate(GROUNDING_RE.finditer(text)):
        color_name, hex_color, rgb = COLORS[idx % len(COLORS)]

        bbox = [float(x.strip()) for x in match.group("bbox").split(",")]

        records.append(
            {
                "index": idx,
                "cell_id": match.group("cell_id"),
                "image_id": match.group("image_id"),
                "bbox": bbox,
                "cell_type": match.group("cell_type").strip(),
                "confidence": float(match.group("conf")),
                "color_name": color_name,
                "hex_color": hex_color,
                "rgb": rgb,
            }
        )

    return records


def find_image(image_id: str, image_root: Path) -> Path | None:
    direct = image_root / image_id
    if direct.is_file():
        return direct

    stem = Path(image_id).stem

    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
        candidate = image_root / f"{stem}{ext}"
        if candidate.is_file():
            return candidate

    matches = list(image_root.rglob(f"{stem}.*"))
    return matches[0] if matches else None


def draw_boxes_for_image(
    image_path: Path,
    records: list[dict],
    out_path: Path,
    box_width: int = 4,
    show_confidence: bool = False,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for rec in records:
        x1, y1, x2, y2 = rec["bbox"]
        rgb = tuple(rec["rgb"])

        if show_confidence:
            label = f"{rec['cell_type']})"
        else:
            label = rec["cell_type"]

        draw.rectangle([x1, y1, x2, y2], outline=rgb, width=box_width)

        text_bbox = draw.textbbox((x1, y1), label, font=font)
        tx1, ty1, tx2, ty2 = text_bbox

        # Draw label slightly above bbox if possible.
        label_y = max(0, y1 - (ty2 - ty1) - 4)
        text_bbox = draw.textbbox((x1, label_y), label, font=font)
        tx1, ty1, tx2, ty2 = text_bbox

        draw.rectangle([tx1, ty1 - 2, tx2 + 4, ty2 + 2], fill=rgb)
        draw.text((x1 + 2, label_y), label, fill="black", font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def color_swatch(hex_color: str) -> str:
    return (
        f'<span style="display:inline-block;width:12px;height:12px;'
        f'background:{hex_color};border:1px solid #333;'
        f'margin-right:6px;vertical-align:middle;"></span>'
    )


def rewrite_report_with_colors(
    report_in: Path,
    report_out: Path,
    records: list[dict],
    annotated_image_subdir: str = "annotated_images",
) -> None:
    text = report_in.read_text(encoding="utf-8")

    # Lookup by exact original grounding tuple.
    rec_iter = iter(records)

    def repl(match: re.Match) -> str:
        rec = next(rec_iter)

        swatch = color_swatch(rec["hex_color"])
        image_stem = Path(rec["image_id"]).stem
        annotated_image = f"{annotated_image_subdir}/{image_stem}_annotated.png"

        return (
            f"- {swatch} **{rec['cell_type']}** "
            f"in `{rec['image_id']}` "
            f"bbox=`{rec['bbox']}` "
            f"(confidence {rec['confidence']:.2f}) "
            f"[annotated image]({annotated_image})"
        )

    new_text = GROUNDING_RE.sub(repl, text)

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(new_text, encoding="utf-8")


def write_manifest(records: list[dict], out_path: Path) -> None:
    serializable = []
    for rec in records:
        row = dict(rec)
        row["rgb"] = list(row["rgb"])
        serializable.append(row)

    out_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-in", required=True, help="Input markdown report")
    parser.add_argument("--image-root", required=True, help="Directory containing source images")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--report-name", default=None, help="Optional output report filename")
    parser.add_argument("--box-width", type=int, default=4)
    parser.add_argument("--show-confidence", action="store_true")

    args = parser.parse_args()

    report_in = Path(args.report_in)
    image_root = Path(args.image_root)
    out_dir = Path(args.out_dir)

    annotated_dir = out_dir / "annotated_images"
    report_out = out_dir / (args.report_name or f"{report_in.stem}_colorized.md")
    report_out = out_dir / f"{report_in.stem}_colorized.html"

    records = parse_report(report_in)
    if not records:
        raise RuntimeError("No Cell Grounding bbox records found in report.")

    by_image: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_image[rec["image_id"]].append(rec)

    for image_id, image_records in by_image.items():
        image_path = find_image(image_id, image_root)
        if image_path is None:
            print(f"[warn] source image not found: {image_id}")
            continue

        out_image = annotated_dir / f"{Path(image_id).stem}_annotated.png"

        draw_boxes_for_image(
            image_path=image_path,
            records=image_records,
            out_path=out_image,
            box_width=args.box_width,
            show_confidence=args.show_confidence,
        )

        print(f"[ok] image: {image_id} -> {out_image}")

    rewrite_report_with_colors(
        report_in=report_in,
        report_out=report_out,
        records=records,
        annotated_image_subdir="annotated_images",
    )

    write_manifest(records, out_dir / "cell_grounding_color_manifest.json")

    print(f"[ok] report: {report_out}")
    print(f"[ok] manifest: {out_dir / 'cell_grounding_color_manifest.json'}")


if __name__ == "__main__":
    main()