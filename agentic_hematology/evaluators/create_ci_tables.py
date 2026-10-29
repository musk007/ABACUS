#!/usr/bin/env python3
"""Confidence intervals and paired tests for the ABACUS ablation tables.

Wilson score intervals for accuracy / coverage / selective accuracy, a
patient-level bootstrap for the two data-dependent quantities (coverage and
selective accuracy share a random denominator, so Wilson alone understates
their uncertainty), and McNemar exact tests for the paired report tables.

    python ci_tables.py --per-patient outputs/cv_eval_v2/per_patient.csv
    python ci_tables.py --per-patient ... --containment results/numeric_containment.json
    python ci_tables.py --per-patient ... --latex

The four configurations mirror eval_cv.summarize():
    None        pre-reflection predictions, nothing flagged
    Guard       pre-reflection predictions, lineage guard only
    Flag+Guard  pre-reflection predictions, full reflector's flags (post hoc)
    Full+Guard  post-reflection predictions, reflector flags + lineage guard
"""
from __future__ import annotations

import argparse
import json
from math import comb, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

Z95 = 1.959964


# --------------------------------------------------------------------------- #
# intervals and tests
# --------------------------------------------------------------------------- #

def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials. Correct at the
    boundaries, unlike the normal approximation."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z / d) * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts.

    b = cases where A fails and B succeeds, c = the reverse. Concordant pairs
    carry no information. Returns 1.0 when there are no discordant pairs.
    """
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    tail = sum(comb(n, i) for i in range(lo + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def bootstrap_cov_sel(
    correct: np.ndarray,
    flagged: np.ndarray,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """Patient-level bootstrap. Resamples patients with their stored
    (flag, correct) pair, so it propagates the randomness in the accepted-set
    size that a Wilson interval on the accepted subset ignores."""
    rng = np.random.default_rng(seed)
    n = len(correct)
    acc, cov, sel = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        f, o = flagged[idx], correct[idx]
        acc.append(o.mean())
        cov.append((~f).mean())
        if (~f).any():
            sel.append(o[~f].mean())
    pct = lambda xs: (float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5)))
    return {"acc": pct(acc), "cov": pct(cov), "sel": pct(sel)}


# --------------------------------------------------------------------------- #
# per-patient table
# --------------------------------------------------------------------------- #

def as_bool(col: pd.Series) -> np.ndarray:
    """eval_cv writes booleans through csv.DictWriter, so they arrive as the
    strings 'True'/'False', and absent values as ''. Everything unset is False."""
    if col.dtype == bool:
        return col.to_numpy()
    return (
        col.astype("string")
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .to_numpy(dtype=bool)
    )


def load(path: Path, exclude: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"patient_id": str})
    df = df[~df["patient_id"].isin(exclude)]
    if "missing" in df.columns:
        df = df[~as_bool(df["missing"])]
    if "agent_error" in df.columns:          # LLM failures are excluded from
        df = df[~as_bool(df["agent_error"])]  # the abstention metrics
    return df.reset_index(drop=True)


def configurations(df: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(correctness, flags, confidences) per configuration. The confidence array
    is the one that configuration's own classifier produced, since it drives the
    matched-coverage baseline."""
    post = as_bool(df["correct"])
    pre = as_bool(df["correct_noreflect"])
    conf_post = df["confidence"].to_numpy(dtype=float)
    conf_pre = df["conf_noreflect"].to_numpy(dtype=float)
    flag_reflect = as_bool(df["flagged"])
    guard_post = as_bool(df["lineage_conflict"])
    guard_pre = as_bool(df["lineage_conflict_noreflect"])
    none = np.zeros(len(df), dtype=bool)
    return {
        "None": (pre, none, conf_pre),
        "Guard": (pre, guard_pre, conf_pre),
        "Flag+Guard": (pre, flag_reflect | guard_post, conf_pre),
        "Full+Guard": (post, flag_reflect | guard_post, conf_post),
    }


def row(
    name: str,
    correct: np.ndarray,
    flagged: np.ndarray,
    conf: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict:
    n = len(correct)
    accepted = ~flagged
    n_acc = int(accepted.sum())
    k_acc = int(correct[accepted].sum())
    boot = bootstrap_cov_sel(correct, flagged, n_boot=n_boot, seed=seed)
    n_err = int((~correct).sum())
    # Matched-coverage confidence baseline: accept the n_acc most confident
    # cases instead of the flagged ones, and score accuracy on those.
    top = np.argsort(-conf, kind="stable")[:n_acc]
    k_conf = int(correct[top].sum())
    return {
        "config": name,
        "n": n,
        "acc": correct.mean(),
        "acc_wilson": wilson(int(correct.sum()), n),
        "acc_boot": boot["acc"],
        "cov": accepted.mean(),
        "cov_wilson": wilson(n_acc, n),
        "cov_boot": boot["cov"],
        "sel": (k_acc / n_acc) if n_acc else float("nan"),
        "sel_wilson": wilson(k_acc, n_acc),
        "sel_boot": boot["sel"],
        "caught": f"{int((~correct & flagged).sum())}/{n_err}",
        "conf": (k_conf / n_acc) if n_acc else float("nan"),
        "conf_wilson": wilson(k_conf, n_acc),
    }


# --------------------------------------------------------------------------- #
# containment (paired, same patients in both arms)
# --------------------------------------------------------------------------- #

def containment_tests(path: Path, reference: str = "halo") -> list[dict]:
    payload = json.loads(path.read_text())
    per_system: dict[str, dict[str, bool]] = {}
    for rec in payload["per_case"]:
        per_system.setdefault(rec["system"], {})[str(rec["patient_id"])] = (
            int(rec["n_violations"]) > 0
        )
    base = per_system.get(reference)
    if base is None:
        raise SystemExit(f"system {reference!r} not in {path}")

    out = []
    for system, cases in per_system.items():
        shared = sorted(set(base) & set(cases))
        k = sum(cases[p] for p in cases)
        n = len(cases)
        b = sum(1 for p in shared if cases[p] and not base[p])   # only the other fails
        c = sum(1 for p in shared if base[p] and not cases[p])   # only reference fails
        out.append({
            "system": system,
            "n": n,
            "violations": k,
            "rate": k / n if n else float("nan"),
            "wilson": wilson(k, n),
            "n_paired": len(shared),
            "discordant": (b, c),
            "mcnemar_p": mcnemar_exact(b, c),
        })
    return out


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #

def pct(x: float) -> str:
    return "  n/a" if x != x else f"{100 * x:5.2f}"


def ci(bounds: tuple[float, float]) -> str:
    lo, hi = bounds
    if lo != lo:
        return "        n/a"
    return f"[{100 * lo:4.1f}, {100 * hi:5.1f}]"


def print_plain(rows: list[dict]) -> None:
    print(f"\nReflection ablation (n = {rows[0]['n']}), 95% Wilson / bootstrap\n")
    head = f"{'Config':11} {'Acc.':>6} {'Wilson':>13} " \
           f"{'Cov.':>6} {'Wilson':>13} " \
           f"{'Sel.':>6} {'Boot':>13} {'Caught':>7} {'Conf.':>6} {'Wilson':>13}"
    print(head)
    print("-" * len(head))
    for r in rows:
        print(
            f"{r['config']:11} {pct(r['acc']):>6} {ci(r['acc_wilson']):>13} "
            f"{pct(r['cov']):>6} {ci(r['cov_wilson']):>13} "
            f"{pct(r['sel']):>6} {ci(r['sel_boot']):>13} {r['caught']:>7} "
            f"{pct(r['conf']):>6} {ci(r['conf_wilson']):>13}"
        )
    lo = max(r["acc_wilson"][0] for r in rows)
    hi = min(r["acc_wilson"][1] for r in rows)
    if lo <= hi:
        print(
            f"\nAll accuracy intervals share [{100 * lo:.1f}, {100 * hi:.1f}] — "
            "the configurations are not separated at this cohort size."
        )


def print_latex(rows: list[dict], lean: bool = True) -> None:
    """lean=True puts intervals only on accuracy and selective accuracy, which
    is what fits a single ISBI column; lean=False intervals every cell."""
    band = lambda lohi: f" \\tiny{{[{100 * lohi[0]:.1f}, {100 * lohi[1]:.1f}]}}"
    print("\n% Wilson intervals; selective accuracy bootstrapped (random denominator)")
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(r"Config. & Acc. & Cov. & Sel. & Caught & Conf. \\")
    print(r"\midrule")
    for r in rows:
        cov = f"{100 * r['cov']:.2f}" + ("" if lean else band(r["cov_wilson"]))
        conf = f"{100 * r['conf']:.2f}" + ("" if lean else band(r["conf_wilson"]))
        print(
            f"{r['config']} & "
            f"{100 * r['acc']:.2f}{band(r['acc_wilson'])} & {cov} & "
            f"{100 * r['sel']:.2f}{band(r['sel_boot'])} & {r['caught']} & {conf} \\\\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")


def print_containment(rows: list[dict], reference: str) -> None:
    print(f"\nNumeric containment, paired against '{reference}'\n")
    head = f"{'System':18} {'Viol.':>8} {'Rate':>6} {'Wilson 95%':>14} {'Discordant':>11} {'McNemar p':>10}"
    print(head)
    print("-" * len(head))
    for r in sorted(rows, key=lambda r: r["rate"]):
        b, c = r["discordant"]
        p = "     --" if r["system"] == reference else f"{r['mcnemar_p']:.4f}"
        print(
            f"{r['system']:18} {r['violations']:>3}/{r['n']:<4} {pct(r['rate']):>6} "
            f"{ci(r['wilson']):>14} {f'{b} / {c}':>11} {p:>10}"
        )
    print(
        "\nDiscordant = (other fails, reference fails). Containment is enforced by "
        "construction,\nso report the interval on the reference rate rather than a "
        "significance claim."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-patient", type=Path, required=True, help="cv_eval_v2/per_patient.csv")
    ap.add_argument("--containment", type=Path, default=None, help="numeric_containment.json")
    ap.add_argument("--reference", default="halo", help="system to pair against in --containment")
    ap.add_argument("--exclude", nargs="*", default=["24"], help="patient ids to drop")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--latex", action="store_true", help="also emit a LaTeX tabular")
    ap.add_argument("--full-ci", action="store_true",
                    help="interval every LaTeX cell, not just Acc. and Sel.")
    ap.add_argument("--out", type=Path, default=None, help="write all numbers to JSON")
    args = ap.parse_args()

    df = load(args.per_patient, args.exclude)
    rows = [
        row(name, correct, flagged, conf, args.n_boot, args.seed)
        for name, (correct, flagged, conf) in configurations(df).items()
    ]
    print_plain(rows)
    if args.latex:
        print_latex(rows, lean=not args.full_ci)

    cont = None
    if args.containment:
        cont = containment_tests(args.containment, args.reference)
        print_containment(cont, args.reference)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"n_boot": args.n_boot, "seed": args.seed, "excluded": args.exclude,
             "ablation": rows, "containment": cont},
            indent=2, default=float,
        ))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()