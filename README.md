# Thunderstorm Alert End Prediction — IA Pau Data Battle 2026

> **Predicting when a thunderstorm will stop threatening an airport — minute by minute.**

This project is our solution to the [Data Battle 2026](https://iapau.org) challenge proposed by [Meteorage](https://www.meteorage.com). The goal is to estimate, in real time, the probability that a thunderstorm is still dangerous around an airport, allowing earlier and safer lifting of lightning alerts.

---

## The problem

Airports currently apply a fixed 30-minute clearance rule after the last lightning strike within 20 km. Our model predicts, at each minute, the probability that the storm is still active — enabling smarter, earlier clearance without increasing risk.

**Two official metrics (jury):**
- **Gain G** — total minutes saved vs. the 30-minute human baseline.
- **Risk R** — ratio of lightning strikes missed inside the 3 km audit zone. Must stay below **R < 2%**.

---

## Our approach

We frame this as **continuous binary classification**: at every minute of an active alert, the model predicts whether the storm will produce another dangerous CG strike within the next 30 minutes.

**Key design choices:**
- Minute-by-minute temporal grid with multi-scale rolling features (5 min, 20 min).
- Geometric storm tracking: centroid velocity, projected distance in 30 minutes.
- IC/CG ratio and polarity trends as physical dissipation signals.
- Irrevocable opening rule: once the AI decides to open, it cannot close again.
- Evaluated across multiple safety zones: **3, 5, 7, 10, 15, 20 km**.

**Model:** XGBoost classifier tuned with Optuna, trained with GroupKFold (5 folds, storm-level split to prevent leakage).

---

## Repository structure

```
thunderstorm-alert-predictor/
│
├── config/
│   └── config.py            # All parameters: zones, XGB hyperparams, feature list
│
├── src/
│   ├── preprocessing/
│   │   ├── storm_groups.py  # Storm group assignment and target computation
│   │   ├── cleaning.py      # Anomaly removal, noise filtering, formatters
│   │   └── features.py      # Feature engineering on raw lightning strikes
│   │
│   ├── temporal_grid/
│   │   └── grid_builder.py  # 1-minute resolution grid with rolling features
│   │
│   ├── training/
│   │   ├── model.py         # Cross-validation training and final fit
│   │   └── tuning.py        # Optuna objectives (zero-crash and <3km rules)
│   │
│   ├── evaluation/
│   │   └── metrics.py       # Risk R, Gain G, threshold scanner, reports
│   │
│   └── pipeline/
│       └── predict.py       # Inference: real-time (single storm) and batch
│
├── notebooks/
│   ├── 01_data_processing.ipynb     # Raw data → processed temporal grid
│   └── 02_training_and_evaluation.ipynb  # Training, tuning, evaluation, plots
│
├── models/                  # Saved model files (.pkl) — not committed to git
├── data/                    # Raw and processed data — not committed to git
├── requirements.txt
└── .gitignore
```

---

## Quick start

```bash
# 1. Setup virtual environment (create if doesn't exist)
python3 -m venv venv
source venv/bin/activate  # On Windows use: .\venv\Scripts\activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Run the notebooks in order
jupyter notebook notebooks/01_data_processing.ipynb
jupyter notebook notebooks/02_training_and_evaluation.ipynb
```

---

## Using the prediction pipeline

### Training

```python
from src.pipeline.predict import predict_batch, save_model
from src.training.model import train_final_model
from config.config import FEATURE_COLUMNS

# df_raw must be fully preprocessed (see notebook 01)
model = train_final_model(X_all, y_all)
save_model(model, 'models/xgb_final.pkl')
```

### Batch inference (test mode — multiple storms)

```python
from src.pipeline.predict import load_model, predict_batch

model = load_model('models/xgb_final.pkl')

results = predict_batch(
    model,
    df_test_raw,                    # raw lightning DataFrame
    safety_zones_km=[3, 5, 10],    # zones to evaluate
    risk_threshold=0.02,
    find_threshold=True,            # scan for best threshold
)

# Access results per zone
print(results[3]['total_gain'])     # total minutes gained at 3 km audit
print(results[3]['risk'])           # risk R at 3 km
print(results[3]['stats'])          # mean, median, std, max, min per storm
```

### Real-time inference (single storm in progress)

```python
from src.pipeline.predict import load_model, predict_realtime

model = load_model('models/xgb_final.pkl')

# df_storm: raw lightning history for the current storm (single airport)
proba = predict_realtime(model, df_storm)
print(f"Current danger probability: {proba:.4f}")
```

---

## Best model hyperparameters

Found by Optuna (objective: maximise gain with zero fatal events < 3 km):

```python
XGB_DEFAULT_PARAMS = {
    "n_estimators": 506,
    "max_depth": 5,
    "learning_rate": 0.10423180025726148,
    "min_child_weight": 10,
    "subsample": 0.8738358835624531,
    "colsample_bytree": 0.8723751225535342,
    "gamma": 1.7503741438290679,
    "scale_pos_weight": 3.964987197971567,
}
```

All parameters are in `config/config.py` and used as defaults throughout the pipeline.

---

## Changing the safety zone threshold

The audit zone can be adjusted independently from the alert zone. Pass the desired list to `predict_batch` or `find_best_threshold`:

```python
# Evaluate at multiple radii simultaneously
results = predict_batch(model, df_raw, safety_zones_km=[3, 5, 7, 10, 15, 20])
```

The alert trigger zone (20 km by default) is set by `ALERT_ZONE_KM` in `config/config.py`.

---

## Data

- **230K lightning strikes** over 10 years around 6 European airports.
- Columns: `date`, `airport`, `lat`, `lon`, `dist`, `azimuth`, `amplitude`, `icloud`, `maxis`, `is_last_lightning_cloud_ground`.
- Alert definition: triggered when a CG strike occurs within 20 km. Ends after 30 minutes without any CG strike in the zone.

---

## Team

Data Battle IA Pau 2026 — Meteorage challenge.
