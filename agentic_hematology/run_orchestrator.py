"""
run_orchestrator.py
===================
Entry point showing how to wire the full agentic pipeline with the
two-model detector (YOLOv11 + EfficientNet) under the orchestrator.

Two modes:

  # Development — stub detector replays precomputed JSON (no GPU/models):
  python run_orchestrator.py \\
      --case-id 12 \\
      --backend stub --stub-source examples/sample_cases.json \\
      --instruction "Generate a full diagnostic report"
      --max-new-tokens 1000

  # Production — YOLOv11 + EfficientNet:
  python run_orchestrator.py \\
      --case-id PT-0042 \\
      --backend wbc-unified \\
      --yolo-weights weights/yolov11_lld.pt \\
      --effnet-weights weights/best_attr.pt \\
      --images "data/PT-0042/*.png" \\
      --instruction "Generate a full diagnostic report"
      --max-new-tokens 1000

  # Production — attribute ablation with DinoBloom linear probes:
  python run_orchestrator.py \\
      --case-id PT-0042 \\
      --backend dinobloom \\
      --yolo-weights weights/yolov11_lld.pt \\
      --dinobloom-weights weights/DinoBloom-B.pth \\
      --dinobloom-attr-weights weights/attribute_dinobloom/best_attr_probes.joblib \\
      --classifier-model weights/leukemia_gbm.pkl \\
      --images "data/PT-0042/*.png" \\
      --instruction "Generate a full diagnostic report"
      --max-new-tokens 1000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

try:
    from leukemia_pipeline.detection_agent import StubDetector
    from leukemia_pipeline.leukemia_classifier import HybridClassifier, LearnedClassifier
    from leukemia_pipeline.orchestrator import Orchestrator, OrchestratorRequest, RuleBasedRouter
    from leukemia_pipeline.report_generator import (
        ClaudeReportGenerator,
        LocalLLMReportGenerator,
        OpenAIReportGenerator,
        TemplateReportGenerator,
    )
except ModuleNotFoundError:
    repo_parent = Path(__file__).resolve().parent.parent
    if str(repo_parent) not in sys.path:
        sys.path.insert(0, str(repo_parent))
    from agentic_hematology.detection_agent import StubDetector
    from agentic_hematology.leukemia_classifier import HybridClassifier, LearnedClassifier
    from agentic_hematology.orchestrator import (
        Orchestrator,
        OrchestratorRequest,
        RuleBasedRouter,
        LLMRouter,
    )
    from agentic_hematology.report_generator import (
        ClaudeReportGenerator,
        LocalLLMReportGenerator,
        OpenAIReportGenerator,
        TemplateReportGenerator,
    )


ROOT = Path(__file__).resolve().parent
WBC_UNIFIED = ROOT / "wbc_unified"
CV = WBC_UNIFIED / "cv"
DEFAULT_YOLO_WEIGHTS = WBC_UNIFIED / "cv/runs/detector/train/weights/best.pt"
DEFAULT_EFFNET_WEIGHTS = WBC_UNIFIED / "cv/runs/attribute/train/best_attr.pt"
DEFAULT_CLASSIFIER_MODEL = (
    ROOT / "outputs" / "ablations" / "classifier" / "dinobloom" / "random_forest" / "leukemia_random_forest.pkl"
)
DEFAULT_LLD_IMAGE_DIR = CV / "generated" / "det_dataset" / "images"


def _default_dinobloom_attr_weights() -> Path:
    for candidate in (
        ROOT / "runs" / "attribute_dinobloom" / "train" / "best_attr_probes.joblib",
        CV / "runs" / "attribute_dinobloom" / "train" / "best_attr_probes.joblib",
    ):
        if candidate.is_file():
            return candidate
    return ROOT / "runs" / "attribute_dinobloom" / "train" / "best_attr_probes.joblib"


DEFAULT_DINOBLOOM_ATTR_WEIGHTS = _default_dinobloom_attr_weights()
DEFAULT_DINOBLOOM_KNN_MANIFEST = WBC_UNIFIED / "cv/generated/attr_manifest.csv"
DEFAULT_DINOBLOOM_KNN_CACHE = WBC_UNIFIED / "cv/runs/attribute_dinobloom/knn_train_embeddings.npz"


def _resolve_attribute_model(args) -> str:
    if args.backend == "dinobloom":
        return "dinobloom"
    return getattr(args, "attribute_model", "effnet") or "effnet"


def _resolve_yolo_weights(args) -> str:
    yolo_weights = args.yolo_weights or str(DEFAULT_YOLO_WEIGHTS)
    if not Path(yolo_weights).is_file():
        sys.exit(f"YOLO weights not found: {yolo_weights}")
    return yolo_weights


def _build_yolo_localizer(args):
    try:
        from leukemia_pipeline.detection_agent_v2 import YOLOv11Localizer
    except ModuleNotFoundError:
        from agentic_hematology.detection_agent_v2 import YOLOv11Localizer

    yolo_weights = _resolve_yolo_weights(args)
    print(f"Stage 1 — YOLO localizer: {yolo_weights}", flush=True)
    return YOLOv11Localizer(
        weights_path=yolo_weights,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        image_size=args.det_imgsz,
        batch_size=args.det_batch,
        half_precision=not args.no_half,
        device=args.device,
    )


def _build_attribute_classifier(args):
    attribute_model = _resolve_attribute_model(args)
    if attribute_model == "effnet":
        try:
            from leukemia_pipeline.detection_agent_v2 import EfficientNetAttributeClassifier
        except ModuleNotFoundError:
            from agentic_hematology.detection_agent_v2 import EfficientNetAttributeClassifier
        effnet_weights = args.effnet_weights or str(DEFAULT_EFFNET_WEIGHTS)
        return EfficientNetAttributeClassifier(
            weights_path=effnet_weights,
            device=args.device,
            predicts_cell_type=args.effnet_predicts_celltype,
        )

    if attribute_model == "dinobloom":
        try:
            from leukemia_pipeline.detection_agent_dinobloom import DinoBloomAttributeClassifier
        except ModuleNotFoundError:
            from agentic_hematology.detection_agent_dinobloom import DinoBloomAttributeClassifier

        attr_probes = args.dinobloom_attr_weights
        probes_path = Path(attr_probes) if attr_probes else None
        attr_mode = args.dinobloom_attr_mode
        if attr_mode == "auto" and probes_path and not probes_path.is_file():
            print(
                f"WARNING: DinoBloom attribute weights not found ({attr_probes}); "
                "falling back to k-NN over the train manifest.",
                file=sys.stderr,
            )
            attr_probes = None
            attr_mode = "knn"

        return DinoBloomAttributeClassifier(
            weights_path=args.dinobloom_weights or "auto",
            attr_probes_path=attr_probes,
            variant=args.dinobloom_variant,
            attr_mode=attr_mode,
            knn_manifest_path=args.dinobloom_knn_manifest,
            knn_cache_path=args.dinobloom_knn_cache,
            knn_k=args.dinobloom_knn_k,
            device=args.device,
            hub_dir=args.dinobloom_hub_dir,
        )

    sys.exit(f"Unknown attribute model: {attribute_model}")


def build_detector(args):
    if args.backend == "stub":
        return StubDetector(args.stub_source)
    if args.dataset_source not in {"auto", "lld"}:
        sys.exit(
            "This pipeline is LLD-only (YOLO + EfficientNet). "
            "Use --dataset-source lld with PBS smear tiles."
        )
    if args.backend in {"two-stage", "wbc-unified", "dinobloom"}:
        try:
            from leukemia_pipeline.detection_agent_v2 import TwoStageDetectionAgent
        except ModuleNotFoundError:
            from agentic_hematology.detection_agent_v2 import TwoStageDetectionAgent

        attribute_model = _resolve_attribute_model(args)
        head_name = "EfficientNet" if attribute_model == "effnet" else "DinoBloom MLP"
        print(
            f"Building two-stage detector: YOLO (localize+crop) → {head_name} (attributes)",
            flush=True,
        )
        localizer = _build_yolo_localizer(args)
        attr_clf = _build_attribute_classifier(args)
        return TwoStageDetectionAgent(
            localizer=localizer,
            attribute_classifier=attr_clf,
            prefer_efficientnet_celltype=args.effnet_predicts_celltype and attribute_model == "effnet",
            attribute_head_name=head_name,
        )
    sys.exit(f"Unknown backend: {args.backend}")


def build_report_generator(args):
    if args.report_backend == "template":
        return TemplateReportGenerator()
    if args.report_backend == "claude":
        return ClaudeReportGenerator()
    if args.report_backend == "openai":
        return OpenAIReportGenerator()
    if args.report_backend == "local-llm":
        return LocalLLMReportGenerator(
            model_path=args.llm_model,
            adapter_path=args.lora_adapter,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    sys.exit(f"Unknown report backend: {args.report_backend}")


def resolve_images(args) -> list[str]:
    if not args.images:
        return []
    image_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    paths: list[str] = []
    for spec in args.images:
        matches = sorted(glob.glob(spec))
        for match in matches:
            path = Path(match)
            if path.is_dir():
                paths.extend(
                    str(p)
                    for p in sorted(path.iterdir())
                    if p.suffix.lower() in image_suffixes
                )
            elif path.suffix.lower() in image_suffixes:
                paths.append(str(path))
    return paths


def _discover_patients(patients_dir: str) -> list[tuple[str, str]]:
    """Return [(case_id, images_dir)] for every patient subdirectory that has an images/ folder."""
    root = Path(patients_dir)
    if not root.is_dir():
        sys.exit(f"--patients-dir not found: {patients_dir}")
    patients = []
    for sub in sorted(root.iterdir()):
        if sub.is_dir() and (sub / "images").is_dir():
            patients.append((sub.name, str(sub / "images")))
    if not patients:
        sys.exit(f"No patient subdirectories with an images/ folder found under {patients_dir}")
    return patients


def _discover_lld_split_patients(
    *,
    split: str,
    image_root: Path,
    cv_root: Path,
) -> list[tuple[str, list[str]]]:
    """Group flat LLD tile images by patient id for train/test split."""
    from agentic_hematology.leukemia_features import discover_lld_split_from_cv

    derived = discover_lld_split_from_cv(cv_root)
    patient_ids = derived.get(split, [])
    if not patient_ids:
        sys.exit(f"No patient IDs found for split={split} under {cv_root}")

    image_dir = image_root / split
    if not image_dir.is_dir():
        sys.exit(f"LLD image dir not found: {image_dir}")

    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    by_patient: dict[str, list[str]] = {pid: [] for pid in patient_ids}
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() not in suffixes:
            continue
        pid = path.stem.split("_")[0]
        if pid in by_patient:
            by_patient[pid].append(str(path))

    patients: list[tuple[str, list[str]]] = []
    for pid in patient_ids:
        paths = by_patient.get(pid, [])
        if paths:
            patients.append((pid, paths))
    if not patients:
        sys.exit(f"No images matched split={split} patients in {image_dir}")
    return patients


def _save_outputs(resp, case_id: str, out_dir: str | None, saved_plan="") -> None:
    """Write all available pipeline outputs for one case."""
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Detection + attribute output
    if resp.state.detection_result is not None:
        det = resp.state.detection_result
        det_payload = {
            "patient_id": det.case_id,
            "n_images": det.n_images,
            "plan":saved_plan,
            "detections": [
                {
                    "cell_id": d.cell_id,
                    "image_id": d.image_id,
                    "bbox_xyxy": [round(float(v), 4) for v in d.bbox_xyxy],
                    "class": d.cell_type,
                    "confidence": round(float(d.objectness), 4),
                    "attributes": {
                        k: (int(v >= 0.5) if isinstance(v, (int, float)) else int(bool(v)))
                        for k, v in d.attributes.items()
                        if k != "class_id"
                    },
                    "attribute_probs": {
                        k: round(float(v), 4) for k, v in d.attribute_probs.items()
                    },
                }
                for d in det.detections
            ],
        }
        if out_dir:
            det_path = os.path.join(out_dir, f"case_{case_id}_detections.json")
            with open(det_path, "w") as f:
                json.dump(det_payload, f, indent=2)
            print(f"  Wrote {det_path}  ({len(det.detections)} cells)")
        else:
            print(f"\n[Detection] {len(det.detections)} cells detected across {det.n_images} images")

    # Classification output
    if resp.state.classification is not None:
        clf = resp.state.classification
        clf_payload = {
            "patient_id": resp.case_id,
            "predicted_class": clf.predicted_class,
            "confidence": round(float(clf.confidence), 4),
            "rationale": clf.rationale,
        }
        if out_dir:
            clf_path = os.path.join(out_dir, f"case_{case_id}_classification.json")
            with open(clf_path, "w") as f:
                json.dump(clf_payload, f, indent=2)
            print(f"  Wrote {clf_path}  (class={clf.predicted_class}, conf={clf.confidence:.2f})")
        else:
            print(f"\n[Classification] {clf.predicted_class} (confidence={clf.confidence:.2f})")
            print(f"  Rationale: {clf.rationale}")

    # Report output — only written when pre-save validation passed (or validation skipped)
    validation_ran = resp.state.validation_passed is not None
    validation_ok = resp.state.report_delivery_allowed if validation_ran else True
    if resp.state.report and validation_ok:
        if out_dir:
            rpt_path = os.path.join(out_dir, f"case_{case_id}_report.md")
            with open(rpt_path, "w") as f:
                f.write(resp.state.report.markdown)
            print(f"  Wrote {rpt_path}")
        else:
            print("\n" + resp.state.report.markdown)
    elif resp.state.report and validation_ran and not validation_ok:
        print(
            "  Report NOT saved as final — pre-save validation failed "
            f"(validation_passed={resp.state.validation_passed})",
            file=sys.stderr,
        )

        if out_dir:
            failed_rpt_path = os.path.join(out_dir, f"case_{case_id}_report_FAILED_VALIDATION.md")
            with open(failed_rpt_path, "w") as f:
                f.write(resp.state.report.markdown)
            print(f"  Wrote failed report for debugging: {failed_rpt_path}")
    elif resp.state.validation_passed is not None:
        print(
            f"  validation_passed={resp.state.validation_passed} "
            f"consistency={resp.state.consistency_passed} "
            f"llm_output={resp.state.llm_output_passed} "
            f"template_json={resp.state.template_json_passed} "
            f"numerical_hallucination={resp.state.numerical_hallucination_passed}"
        )

    if out_dir and resp.state.validation_details:
        val_path = os.path.join(out_dir, f"case_{case_id}_validation.json")
        val_payload = {
            "patient_id": case_id,
            "validation_passed": resp.state.validation_passed,
            "report_delivery_allowed": resp.state.report_delivery_allowed,
            "checks": resp.state.validation_details,
        }
        with open(val_path, "w") as f:
            json.dump(val_payload, f, indent=2)
        print(f"  Wrote {val_path}")

    if out_dir and resp.state.pre_reflection_snapshot:
        pre_ref_path = os.path.join(
            out_dir,
            f"case_{case_id}_pre_reflection_intermediates.json",
        )
        with open(pre_ref_path, "w") as f:
            json.dump(resp.state.pre_reflection_snapshot, f, indent=2)
        print(f"  Wrote {pre_ref_path}")

    # Agent reflection trace (decision-control audit trail)
    if out_dir:
        trace_payload = {
            "patient_id": resp.case_id,
            "dataset_source": resp.state.dataset_source,
            "routing_notes": resp.state.routing_notes,
            "agent_actions": resp.state.agent_actions,
            "n_reflect_iterations": resp.state.n_reflect_iterations,
            "flagged_for_review": resp.state.flagged_for_review,
            "review_reasons": resp.state.review_reasons,
        }
        trace_path = os.path.join(out_dir, f"case_{case_id}_agent_trace.json")
        with open(trace_path, "w") as f:
            json.dump(trace_payload, f, indent=2)
        print(f"  Wrote {trace_path}")

    # Explain output
    if resp.answer:
        if out_dir:
            exp_path = os.path.join(out_dir, f"case_{case_id}_explain.txt")
            with open(exp_path, "w") as f:
                f.write(resp.answer)
            print(f"  Wrote {exp_path}")
        else:
            print("\n" + resp.answer)


def _run_one(
    orch,
    case_id: str,
    image_paths: list[str],
    instruction: str,
    out_dir: str | None,
    dataset_source: str = "lld",
    precomputed_findings: dict | None = None,
) -> bool:
    """Run the orchestrator for a single case and save outputs. Returns True on success."""
    print(f"\n{'='*60}", flush=True)
    print(f"Case: {case_id}  ({len(image_paths)} images)", flush=True)

    req = OrchestratorRequest(
        case_id=case_id,
        image_paths=[] if precomputed_findings is not None else image_paths,
        instruction=instruction,
        precomputed_findings=precomputed_findings,
        dataset_source=dataset_source,
    )
    resp = orch.handle(req)

    print(f"Execution plan: {' → '.join(resp.tool_sequence)}")
    saved_plan = f"Execution plan: {' → '.join(resp.tool_sequence)}"
    if resp.state.agent_actions:
        print("Agent reflection trace:")
        for a in resp.state.agent_actions:
            ct = f" conf_threshold={a['conf_threshold']}" if a.get("conf_threshold") else ""
            print(f"  [iter {a['iteration']}] {a['action']}: {a['reason']}{ct}")
    if resp.state.flagged_for_review:
        print(f"FLAGGED FOR REVIEW: {'; '.join(resp.state.review_reasons)}")
    if resp.state.errors:
        print("Errors:", file=sys.stderr)
        for e in resp.state.errors:
            print(f"  - {e}", file=sys.stderr)

    _save_outputs(resp, case_id, out_dir, saved_plan)
    return not bool(resp.state.errors)


def main() -> int:
    p = argparse.ArgumentParser()
    # Single-patient mode (--case-id + --images) OR batch mode (--patients-dir).
    mode = p.add_mutually_exclusive_group(required=False)
    mode.add_argument("--case-id", help="Single patient case ID.")
    mode.add_argument("--patients-dir",
                      help="Batch: subdirectories each with images/ subfolder.")
    p.add_argument(
        "--lld-split",
        choices=["train", "test"],
        help="Batch: run all patients in the LLD train/test split (uses flat det_dataset images).",
    )
    p.add_argument(
        "--lld-image-dir",
        type=Path,
        default=DEFAULT_LLD_IMAGE_DIR,
        help="Root of det_dataset/images when using --lld-split.",
    )
    p.add_argument(
        "--backend",
        choices=["stub", "two-stage", "wbc-unified", "dinobloom"],
        default="wbc-unified",
        help="LLD: wbc-unified (YOLO + EfficientNet) or dinobloom (YOLO + DinoBloom attrs).",
    )
    p.add_argument(
        "--attribute-model",
        choices=["effnet", "dinobloom"],
        default="effnet",
        help="Attribute head on YOLO crops: effnet (default) or dinobloom (ablation). "
             "Set --backend dinobloom to select DinoBloom without passing this flag.",
    )
    p.add_argument("--stub-source")
    p.add_argument("--yolo-weights", default="/home/roba.majzoub/research/AgenticHematology/agentic_hematology/wbc_unified/cv/runs/detector/train_folds/fold0/weights/best.pt")
    p.add_argument("--effnet-weights", default=str(DEFAULT_EFFNET_WEIGHTS))
    p.add_argument("--effnet-predicts-celltype", action="store_true")
    p.add_argument(
        "--dinobloom-weights",
        default="auto",
        help="DinoBloom checkpoint path, or 'auto' to download MarrLab/DinoBloom from HuggingFace.",
    )
    p.add_argument(
        "--dinobloom-attr-weights",
        default=str(DEFAULT_DINOBLOOM_ATTR_WEIGHTS),
        help="Trained DinoBloom attribute head (.pt) or sklearn probes (.joblib). "
             "Falls back to k-NN if file missing and mode=auto.",
    )
    p.add_argument(
        "--dinobloom-attr-mode",
        choices=["auto", "probes", "knn"],
        default="probes",
        help="Attribute inference: probes (default, trained MLP), knn, or auto.",
    )
    p.add_argument(
        "--dinobloom-knn-manifest",
        default=str(DEFAULT_DINOBLOOM_KNN_MANIFEST),
        help="Train manifest for k-NN attribute retrieval (default: cv/generated/attr_manifest.csv).",
    )
    p.add_argument(
        "--dinobloom-knn-cache",
        default=str(DEFAULT_DINOBLOOM_KNN_CACHE),
        help="Cached train embeddings for k-NN (built once on first run).",
    )
    p.add_argument("--dinobloom-knn-k", type=int, default=5, help="k for DinoBloom k-NN attributes.")
    p.add_argument("--dinobloom-variant", choices=["s", "b", "l", "g"], default="l")
    p.add_argument("--dinobloom-hub-dir",
                   help="Optional torch.hub cache dir for facebookresearch/dinov2.")
    p.add_argument("--images", nargs="*",
                   help="Image paths/globs for single-patient mode (ignored with --patients-dir).")
    p.add_argument("--conf-threshold", type=float, default=0.25)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--det-imgsz", type=int, default=640)
    p.add_argument("--det-batch", type=int, default=1)
    p.add_argument("--no-half", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument(
        "--classifier-model",
        default=str(DEFAULT_CLASSIFIER_MODEL),
        help="Pickled patient-level leukemia classifier (LightGBM / XGBoost).",
    )
    p.add_argument(
        "--dataset-source",
        choices=["auto", "lld"],
        default="lld",
        help="Input layout: LLD PBS tiles (YOLO + EfficientNet).",
    )
    p.add_argument("--report-backend", choices=["template", "local-llm", "claude", "openai"], default="template")
    p.add_argument("--llm-model", help="Local base model path for --report-backend local-llm")
    p.add_argument("--lora-adapter", help="Optional LoRA adapter path for local LLM reports")
    p.add_argument("--max-new-tokens", type=int, default=1000)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--instruction", default="diagnose this case")
    p.add_argument("--findings-json",type=Path,help="Run REPORT_FROM_JSON from a precomputed AggregatedFindings JSON file.",)
    p.add_argument("--no-reflect", action="store_true",
                   help="Disable the reflection agent (the rest of the pipeline still runs).")
    p.add_argument("--no-router-agent", action="store_true",
                   help="Use the deterministic rule-based router instead of the LLM router.")
    p.add_argument("--max-reflect-iterations", type=int, default=6,
                   help="Max reflection-agent iterations before forced escalation.")
    p.add_argument("--out", help="Output directory. In batch mode each patient gets its own subdirectory.")
    args = p.parse_args()

    ################################################
    # Build shared pipeline components (once, even in batch mode)
    ################################################
    detector = build_detector(args)
    learned = None
    if args.classifier_model and Path(args.classifier_model).is_file():
        model_path = Path(args.classifier_model)
        meta_path = model_path.with_name(f"{model_path.stem}_meta.json")
        learned = LearnedClassifier(
            model_path=model_path,
            meta_path=meta_path if meta_path.is_file() else None,
        )
    elif args.classifier_model:
        print(f"WARNING: classifier model not found ({args.classifier_model}); using rule-based classifier.",
              file=sys.stderr)
    classifier = HybridClassifier(learned=learned)
    report_gen = build_report_generator(args)

    reflection_agent = None
    router = RuleBasedRouter()
    llm_explain = None

    need_llm = (not args.no_reflect) or (not args.no_router_agent) or (args.report_backend == "local-llm")

    if need_llm:
        try:
            from agentic_hematology.agent_controller import QwenLLMClient, ReflectionAgent
            from agentic_hematology.orchestrator import LLMRouter
        except ModuleNotFoundError:
            from leukemia_pipeline.agent_controller import QwenLLMClient, ReflectionAgent  # type: ignore
            from leukemia_pipeline.orchestrator import LLMRouter  # type: ignore

        llm_client = QwenLLMClient(
            model_path=args.llm_model,
            adapter_path=args.lora_adapter,
            max_new_tokens=256,
            temperature=args.temperature,
        )
        llm_explain = llm_client.complete

        if not args.no_reflect:
            reflection_agent = ReflectionAgent(llm_client, min_conf_threshold=args.conf_threshold)
            print("Reflection agent enabled.", flush=True)

        if not args.no_router_agent:
            router = LLMRouter(llm_client.complete, fallback=RuleBasedRouter())
            print("LLM router enabled.", flush=True)

        if args.report_backend == "local-llm" and isinstance(report_gen, LocalLLMReportGenerator):
            try:
                report_gen.attach(llm_client.model, llm_client.tokenizer)
                print("Sharing one Qwen3 instance across agent, router, and report generator.")
            except Exception as e:
                print(f"WARNING: could not share Qwen3 instance ({e}); "
                      f"falling back to separate loads.", file=sys.stderr)
    else:
        print("Fully deterministic mode: rule-based router, no reflection agent.", flush=True)

    orch = Orchestrator(
        detector=detector,
        classifier=classifier,
        report_generator=report_gen,
        router=router,
        reflection_agent=reflection_agent,
        max_reflect_iterations=args.max_reflect_iterations,
        llm_explain=llm_explain,
    )

    ################################################
    # Single-patient mode
    ################################################
    if args.case_id:
        precomputed_findings = None

        if args.findings_json:
            with open(args.findings_json, "r") as f:
                payload = json.load(f)

            # If using case_<id>_pre_reflection_intermediates.json,
            # extract only the aggregation block.
            precomputed_findings = payload.get("aggregation", payload)

            images = []
        else:
            images = resolve_images(args)
            if args.backend != "stub" and not images:
                sys.exit("No images matched --images. Provide one or more patient image paths/globs.")

        _run_one(
            orch,
            args.case_id,
            images,
            args.instruction,
            args.out,
            args.dataset_source,
            precomputed_findings=precomputed_findings,
        )
        return 0

    ################################################
    # Batch mode — iterate over all patients
    ################################################
    if args.lld_split:
        patients = _discover_lld_split_patients(
            split=args.lld_split,
            image_root=args.lld_image_dir,
            cv_root=CV,
        )
        batch_label = f"lld-split={args.lld_split}"
        print(f"Batch mode ({batch_label}): {len(patients)} patients", flush=True)
        failed: list[str] = []
        for i, (case_id, image_paths) in enumerate(patients, 1):
            out_dir = os.path.join(args.out, case_id) if args.out else None
            try:
                ok = _run_one(orch, case_id, image_paths, args.instruction, out_dir, args.dataset_source)
                if not ok:
                    failed.append(case_id)
            except Exception as exc:
                print(f"[{i}/{len(patients)}] ERROR for {case_id}: {exc}", file=sys.stderr)
                failed.append(case_id)
        print(f"\nBatch complete: {len(patients) - len(failed)}/{len(patients)} succeeded.")
        if failed:
            print(f"Failed: {', '.join(failed)}", file=sys.stderr)
            return 1
        return 0

    if not args.patients_dir:
        sys.exit("Batch mode requires --patients-dir or --lld-split.")
    patients = _discover_patients(args.patients_dir)
    batch_label = args.patients_dir
    print(f"Batch mode: {len(patients)} patients found under {batch_label}", flush=True)

    image_suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    failed: list[str] = []
    for i, (case_id, images_dir) in enumerate(patients, 1):
        image_paths = sorted(
            str(p) for p in Path(images_dir).iterdir()
            if p.suffix.lower() in image_suffixes
        )
        if not image_paths:
            print(f"[{i}/{len(patients)}] Skipping {case_id}: no images in {images_dir}",
                  file=sys.stderr)
            failed.append(case_id)
            continue

        # In batch mode each patient writes into its own subdirectory.
        out_dir = os.path.join(args.out, case_id) if args.out else None
        try:
            ok = _run_one(orch, case_id, image_paths, args.instruction, out_dir, args.dataset_source)
            if not ok:
                failed.append(case_id)
        except Exception as exc:
            print(f"[{i}/{len(patients)}] ERROR for {case_id}: {exc}", file=sys.stderr)
            failed.append(case_id)

    print(f"\nBatch complete: {len(patients) - len(failed)}/{len(patients)} succeeded.")
    if failed:
        print(f"Failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())