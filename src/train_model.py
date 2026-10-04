"""Step 2: temporal validation, final model, and prediction files.

Run from project root:
    python src/train_model.py

Outputs:
    validation_predictions.csv          (load_id,predicted_rate)
    data/december_chart_inputs.csv      (predicted_rate column filled in)
    eda_results/cv_results.csv          (per-fold metrics, for the report)
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from prep import DATA, clean, fit_cleaner, flag_rate_outliers, haversine_miles, load

ROOT = Path(__file__).resolve().parent.parent
EQUIPMENT = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}

# Expanding-window folds: always train on the past, test on the next month.
FOLDS = [("2025-07-01", "2025-07-31"), ("2025-08-01", "2025-08-31"),
         ("2025-09-01", "2025-09-30"), ("2025-10-01", "2025-10-31")]

PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=30,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def make_features(d: pd.DataFrame) -> pd.DataFrame:
    """Only features that exist for every row we must predict (incl. December chart).

    market_index and quote_signal are deliberately NOT used: in time-based CV they
    made the error worse (their level drifts from month to month), and they are not
    available in the December chart inputs anyway.
    """
    return pd.DataFrame({
        "distance": d.distance, "log_distance": np.log(d.distance),
        "haversine": d.haversine, "distance_ratio": d.distance_ratio,
        "weight": d.weight, "weight_was_negative": d.weight_was_negative,
        "weight_missing": d.weight_missing, "weight_capped": d.weight_capped,
        "equipment": d.equipment.map(EQUIPMENT), "dow": d.date.dt.dayofweek,
        "pickup_lat": d.pickup_lat, "pickup_lon": d.pickup_lon,
        "delivery_lat": d.delivery_lat, "delivery_lon": d.delivery_lon,
    })


def fit_model(df: pd.DataFrame, seeds=(0, 1, 2)):
    X, y = make_features(df), np.log(df["posted_rate"])
    return [lgb.LGBMRegressor(random_state=s, **PARAMS).fit(X, y) for s in seeds]


def predict(models, df: pd.DataFrame) -> np.ndarray:
    X = make_features(df)
    return np.exp(np.mean([m.predict(X) for m in models], axis=0))


def metrics(pred, actual):
    err = np.abs(pred - actual)
    return err.mean(), (err / actual).mean() * 100


def main():
    train_raw, val_raw = load()
    state = fit_cleaner(train_raw, [val_raw])
    train = clean(train_raw, state)
    train["is_rate_outlier"] = flag_rate_outliers(train)
    val = clean(val_raw, state)

    # ---------- temporal cross-validation ----------
    rows = []
    for start, end in FOLDS:
        fit_part = train[(train.date < start) & (train.is_rate_outlier == 0)]
        test_part = train[(train.date >= start) & (train.date <= end)]
        normal = test_part[test_part.is_rate_outlier == 0]

        models = fit_model(fit_part)
        p_model = predict(models, test_part)

        # naive baseline: median rate-per-mile of the training period x distance
        rpm = (fit_part.posted_rate / fit_part.distance).median()
        p_base = test_part.distance.values * rpm

        for name, p in [("lightgbm", p_model), ("baseline_median_rpm", p_base)]:
            mae_all, mape_all = metrics(p, test_part.posted_rate.values)
            mask = (test_part.is_rate_outlier == 0).values
            mae_cl, mape_cl = metrics(p[mask], normal.posted_rate.values)
            rows.append(dict(fold=f"{start[:7]}", model=name, MAE_all=mae_all, MAPE_all=mape_all,
                             MAE_normal=mae_cl, MAPE_normal=mape_cl, n=len(test_part)))
    cv = pd.DataFrame(rows)
    out_dir = ROOT / "eda_results"
    out_dir.mkdir(exist_ok=True)
    cv.to_csv(out_dir / "cv_results.csv", index=False)
    print("=== Temporal CV (train on past months, test on next month) ===")
    print(cv.round(2).to_string(index=False))
    print("\nMean across folds:")
    print(cv.groupby("model")[["MAE_all", "MAPE_all", "MAE_normal", "MAPE_normal"]].mean().round(2))

    # ---------- final model on ALL labeled data ----------
    final_models = fit_model(train[train.is_rate_outlier == 0])

    # ---------- validation predictions ----------
    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    val["predicted_rate"] = predict(final_models, val)
    out = template[["load_id"]].merge(val[["load_id", "predicted_rate"]], on="load_id", how="left")
    out["predicted_rate"] = out["predicted_rate"].round(2)
    assert len(out) == 12_000 and out.predicted_rate.notna().all() and (out.predicted_rate > 0).all()
    out.to_csv(ROOT / "validation_predictions.csv", index=False)
    print("\nSaved validation_predictions.csv:", out.shape)

    # ---------- December chart ----------
    dec = pd.read_csv(DATA / "december_chart_inputs.csv", parse_dates=["date"])
    coords_p = train_raw.groupby("pickup")[["pickup_lat", "pickup_lon"]].first()
    coords_d = train_raw.groupby("delivery")[["delivery_lat", "delivery_lon"]].first()
    d = dec.copy()
    d[["pickup_lat", "pickup_lon"]] = coords_p.loc[d.pickup].values
    d[["delivery_lat", "delivery_lon"]] = coords_d.loc[d.delivery].values
    d["market_index"], d["quote_signal"] = np.nan, np.nan  # not used by the model
    d = clean(d, state)
    dec["predicted_rate"] = predict(final_models, d).round(2)
    dec["date"] = dec["date"].dt.strftime("%Y-%m-%d")
    dec.to_csv(DATA / "december_chart_inputs.csv", index=False)
    print("Saved data/december_chart_inputs.csv")
    print(dec[["date", "predicted_rate"]].describe().T if False else dec["predicted_rate"].describe().round(1))


if __name__ == "__main__":
    main()
