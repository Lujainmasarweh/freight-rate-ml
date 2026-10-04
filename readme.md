# Freight Rate Prediction Challenge

Predicts the posted rate (USD) of freight loads. Trained on `data/train_test.csv`
(Jan-Oct 2025), applied to the 12,000 loads in `data/validation.csv` (Nov-Dec 2025).

## Setup

Requires Python 3.9+.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Place the provided files in a `data/` folder with these exact names:

```
data/train_test.csv
data/validation.csv
data/validation_predictions_template.csv
data/december_chart_inputs.csv
```

## Run

```bash
python src/eda.py            # optional: data-quality report + plots in eda_results/
python src/train_model.py    # temporal CV, final fit, writes both prediction files
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

Outputs:

- `validation_predictions.csv` - `load_id,predicted_rate` for all 12,000 loads
- `data/december_chart_inputs.csv` - `predicted_rate` column filled for the 31 December days
- `scorer_results/candidate_december.png` - chart produced by `score.py`
- `eda_results/cv_results.csv` - per-fold validation metrics

## Approach

**Split / validation.** The validation set lies entirely in the future (Nov-Dec), so a random
split would leak. I use an expanding-window temporal CV: train on all months before the test
month, test on the next month (Jul, Aug, Sep, Oct 2025).

**Data-quality handling** (`src/prep.py`):

| Issue | Handling |
|---|---|
| Negative `weight` (292 train / 145 val) | absolute value + flag |
| Missing `weight` (300 / 165) | median by equipment + flag |
| `weight` capped at 47,500 (~1,200 rows) | flag |
| Missing `market_index` (374 / 249) | median of that day over all loads + flag |
| Suspicious `distance` vs. haversine (111 rows) | `distance_ratio` feature |
| Extreme rate outliers (642 train rows, 1.3%) | removed from training only, never from test folds |
| 8 cities only in validation | no city one-hot; model uses coordinates and distance |

**Model.** LightGBM on `log(posted_rate)`, averaged over 3 seeds. Features: distance, log distance,
haversine distance, distance ratio, weight (+ flags), equipment, day of week, pickup/delivery
coordinates.

`market_index` and `quote_signal` are intentionally **not** used. Every variant tried (raw,
daily-smoothed, rolling average) made temporal-CV error worse, because their level shifts from
month to month. They are also absent from the December chart inputs.

## Results (temporal CV, mean of 4 folds)

| Model | MAE | MAPE | MAE (excl. outliers) | MAPE (excl. outliers) |
|---|---|---|---|---|
| Baseline: median rate-per-mile x distance | 254 | 11.4% | 205 | 9.2% |
| LightGBM | 102 | 4.4% | 51 | 2.1% |

## Limitations

The December chart only reflects a day-of-week pattern (peak on Thursdays, trough on Sundays,
about 21 USD apart). There is no training data after October, so the model cannot learn any
December seasonality or monthly trend.
