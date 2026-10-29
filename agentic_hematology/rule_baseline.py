#!/usr/bin/env python3
"""Rule-based impression baseline for the five-class LLD task.

The impression in each generated report is the output of the deterministic WHO
rule set applied to the *predicted* evidence record.  Mapping it to one of the
five subtypes gives a training-free baseline for the random forest, computed
entirely from reports already on disk.

Usage
-----
    python rule_baseline.py --reports outputs/cv_fold[0-4]
    python rule_baseline.py --reports outputs/cv_fold0 outputs/cv_fold1 ... --labels labels.json
    python rule_baseline.py --reports outputs/cv_fold[0-4] --exclude 24 --csv rule_baseline.csv

Pass one directory per fold.  Each is expected to hold ``<pid>/case_<pid>_report.md``
(any depth works).  The directory name is used as the fold label, so accuracy is
reported both pooled and per fold.

`--labels` takes {"<pid>": "AML", ...}.  Without it, labels are scraped from the
LLD image filenames that appear in each report's grounding block
(`{patient}_*_{DIAGNOSIS}.png`).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

CLASSES = ["ALL", "AML", "APML", "CLL", "CML"]

# Impression text -> subtype.  Ordered: the first matching pattern wins, so the
# more specific phrasings are listed before the general ones.
IMPRESSION_MAP: list[tuple[str, str]] = [
    (r"abnormal promyelocyte",                      "APML"),
    (r"promyelocytic",                              "APML"),
    (r"lymphoblastic morphology",                   "ALL"),
    (r"early acute leukemia \(lymphoblastic",       "ALL"),
    (r"myeloblastic morphology",                    "AML"),
    (r"monocytic differentiation",                  "AML"),
    (r"early acute leukemia \(myeloid",             "AML"),
    (r"chronic myeloproliferative",                 "CML"),
    (r"left-shifted granulopoiesis",                "CML"),
    (r"mature lymphoid proliferation",              "CLL"),
    (r"abnormal smear, unclassified",               None),   # rule 11: abstain
]

IMPRESSION_RE = re.compile(
    r"(?:\*\*)?Impression(?:\*\*)?\s*[:.]?\s*(.+?)(?=\n\s*\n|\n(?:\*\*)?[A-Z][a-z]+ ?[a-z]*(?:\*\*)?\s*:)",
    re.IGNORECASE | re.DOTALL,
)
FILENAME_RE = re.compile(r"(\d+)_[^\s/]*_(" + "|".join(CLASSES) + r")\.(?:png|jpg|jpeg|tif|tiff)",
                         re.IGNORECASE)
PID_RE = re.compile(r"(?:case[_-]?|patient[_-]?)(\d+)", re.IGNORECASE)


def find_reports(roots: list[Path]) -> tuple[dict[str, tuple[str, Path]], list[str]]:
    """Map patient id -> (fold label, report path) across one directory per fold.

    Returns the mapping and a list of warnings for patients seen in more than one
    fold (which should not happen with disjoint folds, and signals that a
    non-fold directory was passed in).
    """
    found: dict[str, tuple[str, Path]] = {}
    warnings: list[str] = []
    for root in roots:
        if not root.is_dir():
            warnings.append(f"not a directory, skipped: {root}")
            continue
        fold = root.name
        for path in sorted(root.rglob("*.md")):
            match = PID_RE.search(f"{path.parent.name}/{path.name}")
            if not match:
                continue
            pid = match.group(1).lstrip("0") or "0"
            if pid in found:
                prev_fold, prev_path = found[pid]
                if prev_path != path:
                    warnings.append(
                        f"patient {pid} found in both {prev_fold} and {fold}; "
                        f"keeping {prev_path}")
                continue
            found[pid] = (fold, path)
    return found, warnings


def extract_impression(text: str) -> str | None:
    match = IMPRESSION_RE.search(text)
    if match:
        return " ".join(match.group(1).split()).lstrip("*: ").strip()
    # Fall back to the first line mentioning an impression-like phrase.
    for line in text.splitlines():
        if re.search(r"findings consistent with|acute leukemia|chronic myelo|"
                     r"mature lymphoid|abnormal smear", line, re.IGNORECASE):
            return " ".join(line.split())
    return None


def map_impression(impression: str) -> tuple[str | None, bool]:
    """Return (predicted class or None, matched_any_rule)."""
    low = impression.lower()
    for pattern, cls in IMPRESSION_MAP:
        if re.search(pattern, low):
            return cls, True
    return None, False


def scrape_label(text: str) -> str | None:
    hits = FILENAME_RE.findall(text)
    if not hits:
        return None
    return Counter(cls.upper() for _, cls in hits).most_common(1)[0][0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", required=True, type=Path, nargs="+",
                    help="one directory per fold, e.g. outputs/cv_fold[0-4]")
    ap.add_argument("--labels", type=Path,
                    help='JSON mapping {"<pid>": "AML", ...}; omit to scrape filenames')
    ap.add_argument("--exclude", nargs="*", default=["24"],
                    help="patient ids to drop (default: 24)")
    ap.add_argument("--csv", type=Path, help="optional per-patient CSV output")
    args = ap.parse_args()

    labels: dict[str, str] = {}
    if args.labels:
        labels = {str(k).lstrip("0") or "0": v.upper()
                  for k, v in json.loads(args.labels.read_text()).items()}

    reports, warnings = find_reports(args.reports)
    for warning in warnings:
        print(f"warning: {warning}")
    if not reports:
        raise SystemExit("no case_<pid>_report.md files found under the given directories")

    rows = []
    for pid, (fold, path) in sorted(reports.items(), key=lambda kv: int(kv[0])):
        if pid in set(args.exclude):
            continue
        text = path.read_text(errors="replace")
        impression = extract_impression(text)
        pred, matched = map_impression(impression) if impression else (None, False)
        truth = labels.get(pid) or scrape_label(text)
        rows.append({
            "patient": pid,
            "fold": fold,
            "truth": truth,
            "rule_pred": pred,
            "matched_rule": matched,
            "impression": (impression or "")[:160],
            "report": str(path),
        })

    scored = [r for r in rows if r["truth"] in CLASSES]
    unlabelled = [r for r in rows if r not in scored]
    n = len(scored)
    if n == 0:
        raise SystemExit("no patients with a resolvable label; pass --labels")

    correct = sum(r["rule_pred"] == r["truth"] for r in scored)
    abstained = sum(r["rule_pred"] is None for r in scored)
    unmatched = [r for r in scored if not r["matched_rule"]]

    # Macro-F1 over the five classes, abstentions counted as wrong.
    f1s = []
    for cls in CLASSES:
        tp = sum(r["rule_pred"] == cls and r["truth"] == cls for r in scored)
        fp = sum(r["rule_pred"] == cls and r["truth"] != cls for r in scored)
        fn = sum(r["rule_pred"] != cls and r["truth"] == cls for r in scored)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    macro_f1 = sum(f1s) / len(f1s)

    # Accuracy restricted to the patients the rules do classify.
    decided = [r for r in scored if r["rule_pred"] is not None]
    dec_acc = (sum(r["rule_pred"] == r["truth"] for r in decided) / len(decided)
               if decided else 0.0)

    print(f"patients scored            : {n}")
    print(f"accuracy (abstain = wrong) : {correct}/{n} = {100 * correct / n:.2f}%")
    print(f"macro-F1                   : {macro_f1:.4f}")
    print(f"abstained (unclassified)   : {abstained}/{n}")
    if decided:
        print(f"accuracy on decided cases  : {len(decided)} cases, {100 * dec_acc:.2f}%")
    print()
    print("per-class F1: " + ", ".join(f"{c} {f:.3f}" for c, f in zip(CLASSES, f1s)))
    print()
    print("per-fold accuracy (abstain = wrong):")
    per_fold = {}
    for r in scored:
        per_fold.setdefault(r["fold"], []).append(r["rule_pred"] == r["truth"])
    for fold in sorted(per_fold):
        hits = per_fold[fold]
        print(f"  {fold:12s} {sum(hits)}/{len(hits)} = {100 * sum(hits) / len(hits):.2f}%")
    fold_accs = [100 * sum(v) / len(v) for v in per_fold.values()]
    if fold_accs:
        print(f"  range        [{min(fold_accs):.1f}, {max(fold_accs):.1f}]%")
    print()
    print("confusion (truth -> pred):")
    conf = Counter((r["truth"], r["rule_pred"] or "UNCLASSIFIED") for r in scored)
    for (truth, pred), count in sorted(conf.items()):
        flag = "" if truth == pred else "   <-- error"
        print(f"  {truth:5s} -> {pred:13s} {count}{flag}")

    if unmatched:
        print(f"\n{len(unmatched)} impression(s) matched no rule pattern "
              f"(check IMPRESSION_MAP against your renderer):")
        for r in unmatched[:10]:
            print(f"  patient {r['patient']}: {r['impression']!r}")
    if unlabelled:
        print(f"\n{len(unlabelled)} report(s) with no resolvable label; pass --labels:")
        for r in unlabelled[:10]:
            print(f"  patient {r['patient']}: {r['report']}")

    if args.csv:
        import csv
        with args.csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()