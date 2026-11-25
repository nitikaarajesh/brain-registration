"""Make figures/bench_ncc.png from results/bench_regcheck.csv.

NCC after registration for every bench registration, with the x-axis grouped by
condition (normal / hard) and group (GLI / Africa). Each point is coloured by
whether the ANTs registration failed (ants_p95_mm > 2 mm). A dashed horizontal
line marks the MIN_NCC threshold defined in regcheck/regcheck.py.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "regcheck"))
from regcheck import MIN_NCC  # the exact threshold the tool uses

CSV = os.path.join(ROOT, "results", "bench_regcheck.csv")
OUT = os.path.join(ROOT, "figures", "bench_ncc.png")

df = pd.read_csv(CSV)
df["ants_failed"] = df["ants_p95_mm"] > 2

# One x position per (condition, group) cell, ordered condition-first.
cells = [("normal", "GLI"), ("normal", "Africa"), ("hard", "GLI"), ("hard", "Africa")]
pos = {cell: i for i, cell in enumerate(cells)}

rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(8, 5))
for failed, colour, label in [(False, "#1f77b4", "ANTs ok (p95 <= 2 mm)"),
                              (True, "#d62728", "ANTs failed (p95 > 2 mm)")]:
    sub = df[df["ants_failed"] == failed]
    x = [pos[(c, g)] for c, g in zip(sub["condition"], sub["group"])]
    x = np.array(x, dtype=float) + rng.uniform(-0.12, 0.12, len(x))  # jitter
    ax.scatter(x, sub["ncc"], c=colour, label=label, s=45, alpha=0.85,
               edgecolors="white", linewidths=0.5, zorder=3)

ax.axhline(MIN_NCC, ls="--", color="black", lw=1.2, zorder=2,
           label=f"MIN_NCC = {MIN_NCC}")
ax.set_xticks(range(len(cells)))
ax.set_xticklabels([f"{c}\n{g}" for c, g in cells])
ax.set_ylabel("NCC after registration")
ax.set_xlabel("condition / group")
ax.set_title("RegCheck bench: image match (NCC) after ANTs SyN")
ax.legend(loc="lower left", framealpha=0.95)
ax.grid(axis="y", ls=":", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
plt.close(fig)
print("wrote", OUT, "| MIN_NCC =", MIN_NCC, "| rows:", len(df),
      "| ANTs failures:", int(df["ants_failed"].sum()))
