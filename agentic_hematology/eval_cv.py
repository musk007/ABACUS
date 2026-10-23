#!/usr/bin/env python3
"""Evaluate cross-validated HALO pipeline outputs and save every result table.

Outputs (in --out):
  per_patient.csv              one row per patient, all fields below
  summary.json                 all metrics + run provenance
  ablation_reflector_guard.md  2x2 ablation: reflector x lineage guard (+ matched confidence baseline)
  classification.md            accuracy / macro-F1 / per-fold / errors
  reports.md                   report-quality metrics
  classifier_backends.md       RF vs XGBoost vs LightGBM (if --classifier-root given)

"No reflector" uses the pre-reflection classification saved by the orchestrator.
For re-aggregated patients its cohort comes from --noreflect-root reruns (--no-reflect);
without them the guard is not applied to those patients (counted in the output).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import re
from pathlib import Path

PARA_RE = re.compile(r"\*{0,2}Morphologic interpretation:\*{0,2}.*?(?=\n\s*\n\*\*|\n---|\Z)", re.S)
COHORT_RE = re.compile(r"\*\*Cohort morphology \(n = (\d+) ([^)]+?)\):\*\*(.*)")
ATTR_RE = {
    "cell_size": r"(small|medium|large) cells \(([\d.]+)%\)",
    "cytoplasm": r"with (scanty|abundant) \(([\d.]+)%\)",
    "cytoplasmic_basophilia": r"(slightly|moderately) basophilic \(([\d.]+)%\)",
    "cytoplasmic_vacuoles": r"(absent|prominent) cytoplasmic vacuolation \(([\d.]+)%\)",
    "nuclear_chromatio": r"(open|coarse) chromatin \(([\d.]+)%\)",
    "nuclear_shape": r"([a-z]+(?:[ _-]or[ _-][a-z]+)?) contours \(([\d.]+)%\)",
    "nucleolus": r"(inconspicuous|prominent) nucleoli \(([\d.]+)%\)",
}
MYELOID_DX = {"AML", "APML", "CML"}


# ---------------------------------------------------------------- parsing helpers
def load_reward(path: Path):
    spec = importlib.util.spec_from_file_location("reward_morphology", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.compute_score


def parse_cohort(md: str) -> dict:
    m = COHORT_RE.search(md)
    if not m:
        return {}
    out = {"n": int(m.group(1)), "cell_type": m.group(2).strip().lower().rstrip("s"), "attrs": {}}
    for k, rx in ATTR_RE.items():
        a = re.search(rx, m.group(3))
        if a:
            out["attrs"][k] = (a.group(1), float(a.group(2)))
    return out


def paragraph(md: str) -> str:
    m = PARA_RE.search(md)
    return m.group(0).strip() if m else ""


def lcs_f1(a: str, b: str) -> float:
    x, y = a.lower().split(), b.lower().split()
    if not x or not y:
        return 0.0
    prev = [0] * (len(y) + 1)
    for i in range(1, len(x) + 1):
        cur = [0] * (len(y) + 1)
        for j in range(1, len(y) + 1):
            cur[j] = prev[j - 1] + 1 if x[i - 1] == y[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    l = prev[-1]
    p, r = l / len(x), l / len(y)
    return 0.0 if l == 0 else 2 * p * r / (p + r)


def score_paragraph(compute_score, para: str, md: str, cohort: dict) -> dict:
    if not para:
        return {}
    morph = {k: {"dominance_pct": v[1]} for k, v in cohort.get("attrs", {}).items()}
    extra = {
        "case_json": json.dumps({"report_ready": {"blast_morphology": morph}}),
        "template_md": md.replace(para, ""),
        "cohort_cell_type": cohort.get("cell_type", ""),
    }
    r = compute_score("leukemia_morph", para, None, extra)
    return {k: r[k] for k in ("score", "num_score", "content_score", "fmt_score")}


def lineage_conflict(pred: str | None, cohort_type: str | None) -> bool | None:
    if not pred or not cohort_type:
        return None
    return ("lympho" in cohort_type) == (pred in MYELOID_DX)


def find(case_dir: Path, pid: str, suffix: str) -> Path | None:
    hits = sorted(case_dir.rglob(f"case_{pid}_{suffix}")) if case_dir.is_dir() else []
    return hits[0] if hits else None


def read_report(case_dir: Path, pid: str) -> str:
    p = find(case_dir, pid, "report.md") or find(case_dir, pid, "report_FAILED_VALIDATION.md")
    return p.read_text() if p else ""


# ---------------------------------------------------------------- per-patient rows
def build_row(f, pid, fold_dir, noreflect_dir, truth, ref_dir, compute_score) -> dict:
    row = {"fold": f, "patient_id": pid, "true": truth.get(pid)}
    clf_p = find(fold_dir, pid, "classification.json")
    if clf_p is None:
        row["missing"] = True
        return row
    clf = json.loads(clf_p.read_text())
    row.update(pred=clf.get("predicted_class"), confidence=float(clf.get("confidence", 0.0)))
    row["correct"] = row["pred"] == row["true"]

    pre_p = find(fold_dir, pid, "pre_reflection_intermediates.json")
    if pre_p:
        lc = json.loads(pre_p.read_text()).get("leukemia_classification") or {}
        row["pred_noreflect"] = lc.get("predicted_class")
        row["conf_noreflect"] = float(lc.get("confidence", 0.0))
        row["correct_noreflect"] = row["pred_noreflect"] == row["true"]

    tr_p = find(fold_dir, pid, "agent_trace.json")
    tr = json.loads(tr_p.read_text()) if tr_p else {}
    acts = tr.get("agent_actions", [])
    row["flagged"] = bool(tr.get("flagged_for_review", False))
    row["review_reasons"] = "; ".join(map(str, tr.get("review_reasons", [])))
    row["agent_actions"] = ",".join(a.get("action", "") for a in acts)
    row["agent_error"] = any(a.get("error") for a in acts)
    row["re_aggregated"] = any(a.get("action") == "re_aggregate" for a in acts)
    row["n_reflect"] = tr.get("n_reflect_iterations", len(acts))

    val_p = find(fold_dir, pid, "validation.json")
    row["validation_passed"] = bool(json.loads(val_p.read_text()).get("validation_passed")) if val_p else None

    md = read_report(fold_dir, pid)
    para, coh = paragraph(md), parse_cohort(md)
    row["enriched"] = bool(para)
    row["cohort_type"] = coh.get("cell_type")
    row["lineage_conflict"] = lineage_conflict(row["pred"], row["cohort_type"])
    for k, v in score_paragraph(compute_score, para, md, coh).items():
        row[f"para_{k}"] = v

    # cohort without the reflector: identical unless re-aggregation ran
    if not row["re_aggregated"]:
        row["cohort_type_noreflect"] = row["cohort_type"]
    elif noreflect_dir is not None:
        row["cohort_type_noreflect"] = parse_cohort(read_report(noreflect_dir, pid)).get("cell_type")
    row["lineage_conflict_noreflect"] = lineage_conflict(row.get("pred_noreflect"), row.get("cohort_type_noreflect"))

    ref_p = ref_dir / f"case_{pid}_report.md"
    if ref_p.is_file():
        ref_md = ref_p.read_text()
        ref_coh = parse_cohort(ref_md)
        row["ref_cohort_type"] = ref_coh.get("cell_type")
        row["para_lcs_f1"] = lcs_f1(para, paragraph(ref_md)) if para else None
        row["cohort_type_match"] = (coh.get("cell_type") == ref_coh.get("cell_type")) if coh and ref_coh else None
        shared = [k for k in ref_coh.get("attrs", {}) if k in coh.get("attrs", {})]
        if shared and row["cohort_type_match"]:
            row["attr_state_agreement"] = sum(
                coh["attrs"][k][0] == ref_coh["attrs"][k][0] for k in shared) / len(shared)
    return row


# ---------------------------------------------------------------- metrics
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None



def policy(rows, correct_key, conf_key, flag_fn) -> dict:
    """Selective-prediction metrics for one configuration + matched confidence baseline."""
    rows = [r for r in rows if correct_key in r]
    n_err = sum(not r[correct_key] for r in rows)
    accepted = [r for r in rows if not flag_fn(r)]
    flagged = [r for r in rows if flag_fn(r)]
    by_conf = sorted(rows, key=lambda r: -r[conf_key])[:len(accepted)]
    return {
        "n": len(rows),
        "accuracy_all": _mean([float(r[correct_key]) for r in rows]),
        "coverage": round(len(accepted) / len(rows), 4) if rows else None,
        "selective_accuracy": _mean([float(r[correct_key]) for r in accepted]),
        "accuracy_flagged": _mean([float(r[correct_key]) for r in flagged]),
        "n_flagged": len(flagged),
        "errors_caught": sum(not r[correct_key] for r in flagged),
        "n_errors": n_err,
        "conf_baseline_same_coverage": _mean([float(r[correct_key]) for r in by_conf]),
    }


def summarize(rows: list[dict], exclude: set[str]) -> dict:
    from sklearn.metrics import confusion_matrix, f1_score

    done = [r for r in rows if not r.get("missing")]
    acc_rows = [r for r in done if r["patient_id"] not in exclude]
    labels = sorted({r["true"] for r in acc_rows})
    y, p = [r["true"] for r in acc_rows], [r["pred"] for r in acc_rows]
    valid = [r for r in acc_rows if not r["agent_error"]]      # LLM failures excluded from abstention metrics

    per_fold = {}
    for f in sorted({r["fold"] for r in acc_rows}):
        fr = [r for r in acc_rows if r["fold"] == f]
        per_fold[f] = {"n": len(fr), "accuracy": _mean([float(r["correct"]) for r in fr])}

    no_guard_nr = [r["patient_id"] for r in valid if r.get("lineage_conflict_noreflect") is None]
    ablation = {
        "no_reflector__no_guard": policy(valid, "correct_noreflect", "conf_noreflect", lambda r: False),
        "reflector__no_guard": policy(valid, "correct", "confidence", lambda r: r["flagged"]),
        "no_reflector__guard": policy(valid, "correct_noreflect", "conf_noreflect",
                                      lambda r: bool(r.get("lineage_conflict_noreflect"))),
        "reflector__guard": policy(valid, "correct", "confidence",
                                   lambda r: r["flagged"] or bool(r.get("lineage_conflict"))),
    }

    return {
        "n_patients_evaluated": len(done),
        "n_missing": len(rows) - len(done),
        "classification": {
            "n": len(acc_rows),
            "accuracy": _mean([float(r["correct"]) for r in acc_rows]),
            "accuracy_noreflect": _mean([float(r["correct_noreflect"]) for r in acc_rows if "correct_noreflect" in r]),
            "macro_f1": round(float(f1_score(y, p, labels=labels, average="macro", zero_division=0)), 4),
            "labels": labels,
            "confusion_matrix": confusion_matrix(y, p, labels=labels).tolist(),
            "per_fold": per_fold,
            "errors": [(r["patient_id"], r["true"], r["pred"], round(r["confidence"], 2))
                       for r in acc_rows if not r["correct"]],
            "changed_by_reflection": [(r["patient_id"], r.get("pred_noreflect"), r["pred"], r["true"])
                                      for r in acc_rows if r.get("pred_noreflect") not in (None, r["pred"])],
        },
        "agent": {
            "n_llm_errors": sum(r["agent_error"] for r in acc_rows),
            "re_aggregated_patients": [r["patient_id"] for r in done if r["re_aggregated"]],
            "flagged_patients": [r["patient_id"] for r in done if r["flagged"]],
        },
        "ablation_reflector_guard": ablation,
        "ablation_notes": {
            "guard_not_applied_without_reflector (missing --noreflect-root reruns)": no_guard_nr,
        },
        "reports": {
            "validation_pass_rate": _mean([float(r["validation_passed"]) for r in done if r.get("validation_passed") is not None]),
            "enrichment_rate": _mean([float(r["enriched"]) for r in done]),
            **{m: _mean([r.get(f"para_{m}") for r in done])
               for m in ("score", "num_score", "content_score", "fmt_score", "lcs_f1")},
            "cohort_type_match": _mean([float(r["cohort_type_match"]) for r in done if r.get("cohort_type_match") is not None]),
            "attr_state_agreement": _mean([r.get("attr_state_agreement") for r in done]),
        },
        "excluded_cases": {r["patient_id"]: {k: r.get(k) for k in ("true", "pred", "pred_noreflect", "confidence",
                                                                   "flagged", "agent_actions", "agent_error")}
                           for r in done if r["patient_id"] in exclude},
    }


# ---------------------------------------------------------------- markdown tables
def pct(x):
    return "—" if x is None else f"{100 * x:.1f}"


def md_ablation(s: dict) -> str:
    names = {
        "no_reflector__no_guard": ("✗", "✗"), "reflector__no_guard": ("✓", "✗"),
        "no_reflector__guard": ("✗", "✓"), "reflector__guard": ("✓", "✓"),
    }
    out = ["| Reflector | Lineage guard | Accuracy (all) | Coverage | Selective acc. | Errors caught | Conf. baseline @ same cov. |",
           "|---|---|---|---|---|---|---|"]
    for k, (rf, gd) in names.items():
        a = s["ablation_reflector_guard"][k]
        out.append(f"| {rf} | {gd} | {pct(a['accuracy_all'])} | {pct(a['coverage'])} | {pct(a['selective_accuracy'])} "
                   f"| {a['errors_caught']}/{a['n_errors']} | {pct(a['conf_baseline_same_coverage'])} |")
    miss = s["ablation_notes"]["guard_not_applied_without_reflector (missing --noreflect-root reruns)"]
    if miss:
        out.append(f"\nGuard not applied without reflector for {miss} (no --no-reflect rerun available).")
    return "\n".join(out) + "\n"


def md_classification(s: dict) -> str:
    c = s["classification"]
    out = [f"Accuracy {pct(c['accuracy'])} (n={c['n']}), without reflector {pct(c['accuracy_noreflect'])}, "
           f"macro-F1 {c['macro_f1']:.3f}\n",
           "| Fold | n | Accuracy |", "|---|---|---|"]
    out += [f"| {f} | {v['n']} | {pct(v['accuracy'])} |" for f, v in c["per_fold"].items()]
    out += ["\nConfusion matrix (rows = true, cols = predicted): " + ", ".join(c["labels"])]
    out += ["| | " + " | ".join(c["labels"]) + " |", "|---" * (len(c["labels"]) + 1) + "|"]
    out += [f"| {lab} | " + " | ".join(map(str, row)) + " |" for lab, row in zip(c["labels"], c["confusion_matrix"])]
    out += ["\nErrors (patient, true, predicted, confidence): " + str(c["errors"])]
    out += ["Changed by reflection (patient, before, after, true): " + str(c["changed_by_reflection"])]
    return "\n".join(out) + "\n"


def md_reports(s: dict) -> str:
    r = s["reports"]
    return "\n".join(["| Metric | Value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in r.items()]) + "\n"


def md_classifier_backends(root: Path) -> str:
    from sklearn.metrics import f1_score

    out = ["| Classifier | F0 | F1 | F2 | F3 | F4 | Pooled acc. | Pooled macro-F1 |", "|---|---|---|---|---|---|---|---|"]
    for b in ["random_forest", "xgboost", "lightgbm"]:
        cells, correct, total, y_all, p_all = [], 0, 0, [], []
        for f in range(5):
            path = root / f"fold{f}" / "training_summary.json"
            summ = json.loads(path.read_text()) if path.is_file() else {}
            if b not in summ:
                cells.append("—")
                continue
            preds = summ[b]["predictions"]
            c = sum(v["true_label"] == v["predicted_class"] for v in preds.values())
            y_all += [v["true_label"] for v in preds.values()]
            p_all += [v["predicted_class"] for v in preds.values()]
            cells.append(f"{100 * c / len(preds):.1f}")
            correct += c
            total += len(preds)
        pooled = f"{100 * correct / total:.1f} ({correct}/{total})" if total else "—"
        mf1 = f"{f1_score(y_all, p_all, average='macro', zero_division=0):.3f}" if y_all else "—"
        out.append(f"| {b} | " + " | ".join(cells) + f" | {pooled} | {mf1} |")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, required=True, help="Contains cv_fold{f}/")
    ap.add_argument("--ref-dir", type=Path, required=True, help="Reference case_<pid>_report.md files")
    ap.add_argument("--folds-json", type=Path, required=True)
    ap.add_argument("--reward", type=Path, required=True)
    ap.add_argument("--noreflect-root", type=Path, default=None,
                    help="Contains cv_fold{f}/ from --no-reflect reruns of re-aggregated patients")
    ap.add_argument("--classifier-root", type=Path, default=None,
                    help="runs/classifier/dinobloom_all (writes classifier_backends.md)")
    ap.add_argument("--exclude-acc", nargs="*", default=["24"], help="Excluded from accuracy (bad GT)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cv = json.loads(args.folds_json.read_text())
    compute_score = load_reward(args.reward)
    rows = []
    for f, pids in enumerate(cv["folds"]):
        nr = args.noreflect_root / f"cv_fold{f}" if args.noreflect_root else None
        for pid in pids:
            rows.append(build_row(f, pid, args.runs_root / f"cv_fold{f}", nr, cv["diagnosis"],
                                  args.ref_dir, compute_score))

    args.out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("fold", "patient_id", "true", "pred"), k))
    with open(args.out / "per_patient.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    summary = summarize(rows, set(args.exclude_acc))
    summary["provenance"] = {"generated": dt.datetime.now().isoformat(timespec="seconds"),
                             **{k: str(v) for k, v in vars(args).items()}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "ablation_reflector_guard.md").write_text(md_ablation(summary))
    (args.out / "classification.md").write_text(md_classification(summary))
    (args.out / "reports.md").write_text(md_reports(summary))
    if args.classifier_root:
        (args.out / "classifier_backends.md").write_text(md_classifier_backends(args.classifier_root))

    print(md_classification(summary))
    print(md_ablation(summary))
    print(md_reports(summary))
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()