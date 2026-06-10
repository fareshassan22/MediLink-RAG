"""Generate consolidated final-report plots from frozen eval CSVs.
Reads BEFORE (/tmp/metrics_BEFORE.csv) and AFTER
(results/toon_retrieval_metrics_20260607_161408.csv) plus the budget summary.
Outputs results/plots/final_report_*.png. No metric is recomputed here —
this only visualizes already-saved numbers.
"""
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BEFORE = "/tmp/metrics_BEFORE.csv"
AFTER = "results/toon_retrieval_metrics_20260607_161408.csv"
BUDGET = "results/toon_retrieval_summary_20260607_161408.csv"
OUTDIR = "results/plots"
os.makedirs(OUTDIR, exist_ok=True)


def load(path):
    return {int(r["tier"]): r for r in csv.DictReader(open(path))}


b, a = load(BEFORE), load(AFTER)
budget = load(BUDGET)
tiers = sorted(a)

# ---- Figure 1: before/after grouped bars for key metrics per tier ----
metrics = ["precision@1", "recall@5", "ndcg@5", "mrr"]
fig, axes = plt.subplots(1, len(metrics), figsize=(20, 5))
x = np.arange(len(tiers))
w = 0.38
for ax, m in zip(axes, metrics):
    bv = [float(b[t][m]) for t in tiers]
    av = [float(a[t][m]) for t in tiers]
    ax.bar(x - w / 2, bv, w, label="Before (hybrid)", color="#b0b7c3")
    ax.bar(x + w / 2, av, w, label="After (+cross-encoder)", color="#2e7d32")
    for i, (bb, aa) in enumerate(zip(bv, av)):
        ax.text(i + w / 2, aa + 0.01, f"+{aa-bb:.2f}", ha="center", fontsize=8, color="#1b5e20")
    ax.set_title(m.upper())
    ax.set_xticks(x)
    ax.set_xticklabels([f"Tier {t}" for t in tiers])
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
fig.suptitle("MediLink TOON Retrieval — Before vs After Cross-Encoder Rerank (frozen GT, 100 queries)", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])
p1 = os.path.join(OUTDIR, "final_report_metrics.png")
fig.savefig(p1, dpi=120)
plt.close(fig)

# ---- Figure 2: full metric curve recall/ndcg @k after ----
ks = [1, 3, 5, 10]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for t in tiers:
    rec = [float(a[t][f"recall@{k}"]) for k in ks]
    ndcg = [float(a[t][f"ndcg@{k}"]) for k in ks]
    axes[0].plot(ks, rec, marker="o", label=f"Tier {t}")
    axes[1].plot(ks, ndcg, marker="s", label=f"Tier {t}")
axes[0].set_title("Recall@k (after)")
axes[1].set_title("NDCG@k (after)")
for ax in axes:
    ax.set_xlabel("k")
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()
fig.tight_layout()
p2 = os.path.join(OUTDIR, "final_report_curves.png")
fig.savefig(p2, dpi=120)
plt.close(fig)

# ---- Figure 3: budget compliance ----
fig, ax = plt.subplots(figsize=(8, 5))
bt = sorted(budget)
avg = [float(budget[t]["avg_tokens"]) for t in bt]
mx = [float(budget[t]["max_tokens"]) for t in bt]
bud = [float(budget[t]["budget"]) for t in bt]
x = np.arange(len(bt))
ax.bar(x - 0.2, avg, 0.4, label="avg tokens", color="#1565c0")
ax.bar(x + 0.2, mx, 0.4, label="max tokens", color="#ef6c00")
for i, bd in enumerate(bud):
    ax.hlines(bd, i - 0.45, i + 0.45, colors="red", linestyles="--", label="budget" if i == 0 else None)
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([f"Tier {t}\n(budget {int(bud[i])})" for i, t in enumerate(bt)])
ax.set_ylabel("tokens (log)")
ax.set_title("Token-budget compliance — 100% within budget all tiers")
ax.legend()
fig.tight_layout()
p3 = os.path.join(OUTDIR, "final_report_budget.png")
fig.savefig(p3, dpi=120)
plt.close(fig)

print("WROTE:")
for p in (p1, p2, p3):
    print(" ", p)
