#!/usr/bin/env python3
"""Score the rule-based impression baseline and compare it to the learned classifier.

Consumes the CSV written by ``rule_baseline.py --csv`` and prints every figure
needed for the paper: accuracy with a Wilson interval, macro-F1 with the
per-class breakdown, per-fold accuracy, the confusion matrix, the error list,
and how many errors come from each rule-trigger pattern.

With ``--compare`` it also joins a per-patient CSV of the learned classifier's
outcomes (e.g. per_patient_derived_ci.csv) and reports the paired comparison
with an exact McNemar test, so the accuracy gap is quantified rather than
eyeballed.

Usage
-----
    python score_rule_baseline.py rule_baseline.csv
    python score_rule_baseline.py rule_baseline.csv --compare per_patient_derived_ci.csv
    python score_rule_baseline.py rule_baseline.csv --compare per_patient_derived_ci.csv \
        --compare-column correct_noreflect --compare-name "random forest"

Nothing here is hand-entered: every number printed is computed from the CSVs.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from math import comb, sqrt
from pathlib import Path

CLASSES = ["ALL", "AML", "APML", "CLL", "CML"]
Z95 = 1.959963984540054

# Column names accepted when locating fields in the --compare CSV.
PID_COLUMNS = ("patient", "pid", "patient_id", "case", "case_id", "id")
TRUE_VALUES = {"1", "true", "True", "TRUE", "yes", "y", "t", "1.0"}


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z / d) * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    tail = sum(comb(n, i) for i in range(lo + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def macro_f1(rows: list[dict]) -> tuple[float, list[tuple[str, int, int, int, int, float]]]:
    """Macro-F1 over CLASSES, abstentions counted as wrong. Returns (macro, per-class)."""
    detail = []
    f1s = []
    for cls in CLASSES:
        tp = sum(r["pred"] == cls and r["truth"] == cls for r in rows)
        fp = sum(r["pred"] == cls and r["truth"] != cls for r in rows)
        fn = sum(r["pred"] != cls and r["truth"] == cls for r in rows)
        support = sum(r["truth"] == cls for r in rows)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        detail.append((cls, support, tp, fp, fn, f1))
    return (sum(f1s) / len(f1s) if f1s else 0.0), detail


def load_baseline(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as fh:
        for raw in csv.DictReader(fh):
            if not (raw.get("patient") or "").strip():
                continue
            rows.append({
                "patient": raw["patient"].strip(),
                "fold": (raw.get("fold") or "").strip(),
                "truth": (raw.get("truth") or "").strip(),
                "pred": (raw.get("rule_pred") or "").strip(),
                "impression": (raw.get("impression") or "").strip(),
            })
    return rows


def find_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {f.lower(): f for f in fieldnames}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def load_compare(path: Path, column: str | None) -> dict[str, bool]:
    """patient id -> whether the learned classifier was correct for that patient."""
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        pid_col = find_column(fields, PID_COLUMNS)
        if pid_col is None:
            raise SystemExit(f"no patient-id column in {path}; looked for {PID_COLUMNS}")
        if column is None:
            column = find_column(fields, ("correct",))
            if column is None:
                raise SystemExit(
                    f"no 'correct' column in {path}; pass --compare-column. "
                    f"available: {', '.join(fields)}")
        elif column not in fields:
            raise SystemExit(f"column {column!r} not in {path}; available: {', '.join(fields)}")
        out = {}
        for raw in reader:
            pid = (raw.get(pid_col) or "").strip()
            if not pid:
                continue
            pid = pid.lstrip("0") or "0"
            value = (raw.get(column) or "").strip()
            out[pid] = value in TRUE_VALUES
    print(f"compare file: {path}  (patient column {pid_col!r}, outcome column {column!r})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path, help="CSV from rule_baseline.py --csv")
    ap.add_argument("--compare", type=Path,
                    help="per-patient CSV of the learned classifier's outcomes")
    ap.add_argument("--compare-column",
                    help="boolean column in --compare holding per-patient correctness "
                         "(default: 'correct')")
    ap.add_argument("--compare-name", default="learned classifier",
                    help="label used for the compared system in the output")
    args = ap.parse_args()

    rows = load_baseline(args.baseline)
    scored = [r for r in rows if r["truth"] in CLASSES]
    skipped = [r for r in rows if r not in scored]
    n = len(scored)
    if n == 0:
        raise SystemExit("no rows with a label in the five classes")

    correct = sum(r["pred"] == r["truth"] for r in scored)
    abstained = sum(not r["pred"] for r in scored)
    lo, hi = wilson(correct, n)
    macro, detail = macro_f1(scored)

    print("=" * 68)
    print("RULE-BASED IMPRESSION BASELINE")
    print("=" * 68)
    print(f"patients scored     : {n}")
    if skipped:
        print(f"rows without label  : {len(skipped)} (excluded from all figures)")
    print(f"accuracy            : {correct}/{n} = {100 * correct / n:.2f}%")
    print(f"  95% Wilson CI     : [{100 * lo:.1f}, {100 * hi:.1f}]")
    print(f"macro-F1            : {macro:.4f}")
    print(f"abstentions         : {abstained}/{n}"
          f"{'  (every case reached a decision)' if abstained == 0 else ''}")
    print()

    print("per class (abstentions counted as wrong):")
    print(f"  {'class':6s} {'supp':>4s} {'tp':>3s} {'fp':>3s} {'fn':>3s}   {'F1':>6s}")
    for cls, support, tp, fp, fn, f1 in detail:
        print(f"  {cls:6s} {support:4d} {tp:3d} {fp:3d} {fn:3d}   {f1:6.4f}")
    print()

    if any(r["fold"] for r in scored):
        print("per-fold accuracy:")
        per_fold: dict[str, list[bool]] = {}
        for r in scored:
            per_fold.setdefault(r["fold"], []).append(r["pred"] == r["truth"])
        for fold in sorted(per_fold):
            hits = per_fold[fold]
            print(f"  {fold:12s} {sum(hits)}/{len(hits)} = {100 * sum(hits) / len(hits):6.2f}%")
        accs = [100 * sum(v) / len(v) for v in per_fold.values()]
        print(f"  {'range':12s} [{min(accs):.1f}, {max(accs):.1f}]%")
        print()

    print("label distribution  : " + ", ".join(
        f"{c} {sum(r['truth'] == c for r in scored)}" for c in CLASSES))
    print()

    print("confusion (truth -> pred):")
    conf = Counter((r["truth"], r["pred"] or "UNCLASSIFIED") for r in scored)
    for (truth, pred), count in sorted(conf.items()):
        mark = "" if truth == pred else "   <-- error"
        print(f"  {truth:5s} -> {pred:13s} {count}{mark}")
    print()

    errors = [r for r in scored if r["pred"] != r["truth"]]
    print(f"errors ({len(errors)}):")
    for r in errors:
        print(f"  p{r['patient']:>3s} {r['fold']:10s} {r['truth']:5s} -> "
              f"{r['pred'] or 'UNCLASSIFIED'}")
    print()

    print("errors grouped by predicted class (which rule fired wrongly):")
    for pred, count in Counter(r["pred"] or "UNCLASSIFIED" for r in errors).most_common():
        pids = [r["patient"] for r in errors if (r["pred"] or "UNCLASSIFIED") == pred]
        print(f"  predicted {pred:13s} {count:2d}  patients {', '.join(pids)}")
    print()

    if args.compare:
        other = load_compare(args.compare, args.compare_column)
        paired = [r for r in scored if r["patient"] in other]
        missing = [r["patient"] for r in scored if r["patient"] not in other]
        print("=" * 68)
        print(f"PAIRED COMPARISON: rule baseline vs {args.compare_name}")
        print("=" * 68)
        if missing:
            print(f"not in compare file ({len(missing)}): {', '.join(missing)}")
        m = len(paired)
        if m == 0:
            raise SystemExit("no patients matched between the two files; check the id columns")
        rule_ok = [r["pred"] == r["truth"] for r in paired]
        other_ok = [other[r["patient"]] for r in paired]
        a = sum(x and y for x, y in zip(rule_ok, other_ok))
        b = sum(x and not y for x, y in zip(rule_ok, other_ok))   # rule right, other wrong
        c = sum((not x) and y for x, y in zip(rule_ok, other_ok))  # rule wrong, other right
        d = sum((not x) and (not y) for x, y in zip(rule_ok, other_ok))
        r_lo, r_hi = wilson(sum(rule_ok), m)
        o_lo, o_hi = wilson(sum(other_ok), m)
        print(f"patients paired     : {m}")
        print(f"rule baseline       : {sum(rule_ok)}/{m} = {100 * sum(rule_ok) / m:.2f}%"
              f"  CI [{100 * r_lo:.1f}, {100 * r_hi:.1f}]")
        print(f"{args.compare_name:20s}: {sum(other_ok)}/{m} = {100 * sum(other_ok) / m:.2f}%"
              f"  CI [{100 * o_lo:.1f}, {100 * o_hi:.1f}]")
        print(f"difference          : "
              f"{100 * (sum(other_ok) - sum(rule_ok)) / m:+.2f} points "
              f"({sum(other_ok) - sum(rule_ok):+d} patients)")
        print()
        print("paired table:")
        label = f"{args.compare_name} only correct"
        width = max(32, len(label))
        print(f"  {'both correct'.ljust(width)}: {a}")
        print(f"  {'rule only correct'.ljust(width)}: {b}")
        print(f"  {label.ljust(width)}: {c}")
        print(f"  {'both wrong'.ljust(width)}: {d}")
        print(f"  {'exact McNemar p'.ljust(width)}: {mcnemar_exact(b, c):.4f}")
        print()
        only_other = [r["patient"] for r, x, y in zip(paired, rule_ok, other_ok) if y and not x]
        only_rule = [r["patient"] for r, x, y in zip(paired, rule_ok, other_ok) if x and not y]
        if only_other:
            print(f"  {args.compare_name} right, rules wrong: {', '.join(only_other)}")
        if only_rule:
            print(f"  rules right, {args.compare_name} wrong: {', '.join(only_rule)}")
        if d:
            both = [r["patient"] for r, x, y in zip(paired, rule_ok, other_ok)
                    if not x and not y]
            print(f"  both wrong                           : {', '.join(both)}")


if __name__ == "__main__":
    main()