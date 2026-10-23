"""Plot per-attribute F1 of attribute probes: training domain (C2 vs C1+C2) x test camera."""
import json
import sys

import matplotlib.pyplot as plt
import numpy as np

path = "/home/roba.majzoub/research/AgenticHematology/agentic_hematology/wbc_unified/cv/runs/attribute_probe_comparison.json"
out = "/home/roba.majzoub/research/AgenticHematology/agentic_hematology/wbc_unified/cv/plots/output/attribute_probe_comparison.pdf"

with open(path) as f:
    res = json.load(f)

TRAIN = {"C2": "Train C2", "C1C2": "Train C1+C2"}
TEST = {"C2": "Test camera: C2", "C1": "Test camera: C1"}
COLORS = {"C2": "#9AA0A6", "C1C2": "#2E2A5E"}

attrs = list(res["C2"]["C2"]["per_attribute_f1"])
labels = [a.replace("Cytoplasmic_", "Cyto. ").replace("Nuclear_", "Nuc. ").replace("_", " ")
          for a in attrs] + ["Mean"]
x = np.arange(len(labels))
w = 0.38

plt.rcParams.update({"font.size": 7})
fig, axes = plt.subplots(1, len(TEST), figsize=(5.5, 2.5), sharey=True)
for ax, (test, title) in zip(axes, TEST.items()):
    for k, (train, name) in enumerate(TRAIN.items()):
        r = res[train][test]
        vals = [r["per_attribute_f1"][a] for a in attrs] + [r["mean_f1"]]
        bars = ax.bar(x + (k - 0.5) * w, vals, w, label=name, color=COLORS[train])
        bars[-1].set_hatch("//")
        bars[-1].set_edgecolor("white")
        b, v = bars[-1], vals[-1]
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                ha="center", va="bottom", fontsize=6.5)
    n = res["C2"][test]["n_cells"]
    ax.set_title(f"{title}  (n = {n:,} cells)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.axvline(len(attrs) - 0.5, color="0.7", lw=0.8, ls="--")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("F1")
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, ncol=2,
           loc="upper center", bbox_to_anchor=(0.5, 1.04), fontsize=7.5)
fig.tight_layout()
fig.savefig(out, bbox_inches="tight", dpi=300)
print(f"saved {out}")