import json
effnet_path = "/home/roba.majzoub/research/AgenticHematology/agentic_hematology/wbc_unified/cv/runs/eval/effnet_attr_w_yolo/test_results.json"
dinobloom_path = "/home/roba.majzoub/research/AgenticHematology/agentic_hematology/wbc_unified/cv/runs/eval/dinobloom_attr_w_yolo/test_results.json"
for path in [effnet_path, dinobloom_path]:
    d = json.load(open(path))
    for block in ("attributes_gt_crops", "attributes_e2e"):
        pa = d[block]["per_attribute"]
        f1s = [v["f1"] for k, v in pa.items() if k != "Cell_Size"]
        if "effnet" in path:
            print(f"EfficientNet {block}: mean binary F1 = {sum(f1s)/len(f1s):.4f}  (n={len(f1s)})")
        else:
            print(f"DINOBloom {block}: mean binary F1 = {sum(f1s)/len(f1s):.4f}  (n={len(f1s)})")