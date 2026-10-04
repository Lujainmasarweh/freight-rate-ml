"""Data loading and cleaning for the Freight Rate challenge.

Cleaning rules (all discovered during EDA, see eda.py):
  * weight < 0            -> sign flipped, use absolute value
  * weight == 47,500      -> capped value, add flag
  * weight missing        -> median by equipment (learned on train only) + flag
  * market_index missing  -> median of that day across ALL loads (train+validation
                             features, no labels used) + flag
  * distance vs haversine -> distance_ratio feature + flag for suspicious rows
  * extreme labels        -> flagged with `is_rate_outlier` (training rows only)
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
WEIGHT_CAP = 47_500.0


def load():
    train = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])
    val = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])
    return train, val


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def fit_cleaner(train: pd.DataFrame, others: list[pd.DataFrame]) -> dict:
    """Learn everything that must come from data (medians) - no labels used."""
    all_feats = pd.concat([train] + others, ignore_index=True)
    w = all_feats["weight"].abs()
    return {
        "weight_median_by_equipment": w.groupby(all_feats["equipment"]).median().to_dict(),
        "market_by_day": all_feats.groupby("date")["market_index"].median(),
        "market_global": all_feats["market_index"].median(),
    }


def clean(df: pd.DataFrame, state: dict) -> pd.DataFrame:
    df = df.copy()

    # --- weight ---
    df["weight_was_negative"] = (df["weight"] < 0).astype(int)
    df["weight_missing"] = df["weight"].isna().astype(int)
    df["weight"] = df["weight"].abs()
    df["weight_capped"] = (df["weight"] >= WEIGHT_CAP).astype(int)
    fill = df["equipment"].map(state["weight_median_by_equipment"])
    df["weight"] = df["weight"].fillna(fill)

    # --- market index ---
    df["market_missing"] = df["market_index"].isna().astype(int)
    day_fill = df["date"].map(state["market_by_day"]).fillna(state["market_global"])
    df["market_index"] = df["market_index"].fillna(day_fill)

    # --- distance sanity ---
    df["haversine"] = haversine_miles(df.pickup_lat, df.pickup_lon, df.delivery_lat, df.delivery_lon)
    df["distance_ratio"] = df["distance"] / df["haversine"]
    df["distance_suspicious"] = (df["distance_ratio"] > 1.6).astype(int)
    return df


def flag_rate_outliers(train: pd.DataFrame) -> pd.Series:
    """Label-based outlier flag (training data only).

    Compares rate-per-mile to the median of its (equipment, distance bucket) group.
    Normal loads sit within ~0.75x-1.6x of that median; the injected outliers are
    >2.5x or <0.5x.
    """
    rpm = train["posted_rate"] / train["distance"]
    bucket = pd.cut(train["distance"], [0, 150, 300, 500, 800, 1200, 1800, 2500, 5000])
    med = rpm.groupby([train["equipment"], bucket], observed=True).transform("median")
    ratio = rpm / med
    return ((ratio > 2.5) | (ratio < 0.5)).astype(int)
