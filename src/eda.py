"""Step 1: exploratory analysis + data-quality report. Run from project root:
    python src/eda.py
Saves plots to eda_results/.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from prep import clean, fit_cleaner, flag_rate_outliers, load

OUT = Path(__file__).resolve().parent.parent / "eda_results"
OUT.mkdir(exist_ok=True)

train, val = load()

print("=== Shapes ===")
print("train:", train.shape, "| validation:", val.shape)
print("train dates:", train.date.min().date(), "->", train.date.max().date())
print("validation dates:", val.date.min().date(), "->", val.date.max().date())

print("\n=== Missing values ===")
print(pd.DataFrame({"train": train.isna().sum(), "validation": val.isna().sum()}))

print("\n=== Data-quality issues ===")
print("negative weights  :", (train.weight < 0).sum(), "train |", (val.weight < 0).sum(), "val")
print("weight capped 47.5k:", (train.weight.abs() >= 47500).sum(), "train")
print("duplicate load_id :", train.load_id.duplicated().sum())

state = fit_cleaner(train, [val])
tr = clean(train, state)
tr["is_rate_outlier"] = flag_rate_outliers(tr)
print("suspicious distance:", tr.distance_suspicious.sum())
print("extreme rate outliers:", tr.is_rate_outlier.sum(), f"({tr.is_rate_outlier.mean():.2%})")

new_cities = (set(val.pickup) | set(val.delivery)) - (set(train.pickup) | set(train.delivery))
print("cities only in validation:", sorted(new_cities))

# ---- plots ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
tr.groupby(tr.date.dt.to_period("M"))["posted_rate"].mean().plot(ax=ax[0], marker="o")
ax[0].set_title("Mean posted rate by month")
tr.groupby(tr.date.dt.to_period("M"))["market_index"].mean().plot(ax=ax[1], marker="o", color="orange")
val.groupby(val.date.dt.to_period("M"))["market_index"].mean().plot(ax=ax[1], marker="o", color="red")
ax[1].set_title("market_index: train (orange) vs validation (red)")
fig.tight_layout()
fig.savefig(OUT / "monthly_trends.png", dpi=120)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
clean_rows = tr[tr.is_rate_outlier == 0]
out_rows = tr[tr.is_rate_outlier == 1]
ax[0].scatter(clean_rows.distance, clean_rows.posted_rate, s=2, alpha=0.3, label="normal")
ax[0].scatter(out_rows.distance, out_rows.posted_rate, s=6, color="red", label="outlier")
ax[0].set_xlabel("distance"); ax[0].set_ylabel("posted_rate"); ax[0].legend()
ax[0].set_title("Rate vs distance")
(tr.posted_rate / tr.distance).clip(upper=8).hist(bins=100, ax=ax[1])
ax[1].set_title("Rate per mile (clipped at 8)")
fig.tight_layout()
fig.savefig(OUT / "rate_vs_distance.png", dpi=120)

print("\nPlots saved in", OUT)
