import logging
import os
import sys
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ── Resolve project root ──
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------- Resolved paths ----------------
DATA_FILE    = os.path.join(_ROOT, "data",   "processed", "f1_features_clean.csv")
MODEL_FILE   = os.path.join(_ROOT, "models", "grid_model.pkl")
FEATURE_FILE = os.path.join(_ROOT, "models", "grid_model_features.pkl")

TARGET = "GridPosition"

DROP_COLS = [
    "GridPosition",
    "GridAdvantage",
    "GridConfidence",
    "Winner",
    "Driver",
    "Year",
    "EWMForm",
    "ChampionshipMomentum",
    "TeamYTDWins",
    "QualiFastLap", "FP2FastLap",
    "QualiMeanLap", "QualiStdLap",
    "FP2MeanLap",   "FP2StdLap",
    "Sector1",      "Sector2", "Sector3",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ---------------- Load & validate ----------------
logging.info(f"Loading {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df = df.dropna(subset=["Year"])
df["Year"] = df["Year"].astype(int)

logging.info(f"Loaded {len(df)} rows — years: {sorted(df['Year'].unique())}")


# ---------------- Train / test split + walk-forward CV folds ----------------
years       = sorted(df["Year"].unique())
latest_year = years[-1]

train_df = df[df["Year"] < latest_year].reset_index(drop=True)
test_df  = df[df["Year"] == latest_year].reset_index(drop=True)

logging.info(f"Train: {len(train_df)} rows  |  Test ({latest_year}): {len(test_df)} rows")

cv_years = sorted(train_df["Year"].unique())
cv_folds = []
for test_year in cv_years[1:]:
    train_idx = train_df.index[train_df["Year"] < test_year].tolist()
    val_idx   = train_df.index[train_df["Year"] == test_year].tolist()
    cv_folds.append((train_idx, val_idx))

logging.info(f"Walk-forward CV: {len(cv_folds)} folds over years {cv_years[1:]}")


# ---------------- Feature matrices ----------------
existing_drop = [c for c in DROP_COLS if c in train_df.columns]
df_model = train_df.drop(columns=existing_drop).select_dtypes(include=["number"])

X_train = df_model
y_train = train_df[TARGET]

# Save feature list now that X_train is defined
os.makedirs(os.path.dirname(FEATURE_FILE), exist_ok=True)
joblib.dump(X_train.columns.tolist(), FEATURE_FILE)
logging.info(f"Saved feature list ({len(X_train.columns)} features) → {FEATURE_FILE}")

existing_drop_test = [c for c in DROP_COLS if c in test_df.columns]
df_model_test = test_df.drop(columns=existing_drop_test).select_dtypes(include=["number"])

X_test = df_model_test[X_train.columns]
y_test = test_df[TARGET]


# ---------------- Pipeline ----------------
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("rf",      RandomForestRegressor(random_state=42, n_jobs=-1)),
])


# ---------------- Hyperparameter search ----------------
param_grid = {
    "rf__n_estimators":      [300, 500],
    "rf__max_depth":         [10, 15, None],
    "rf__min_samples_split": [2, 5],
}

grid = GridSearchCV(
    pipeline, param_grid,
    cv=cv_folds,
    scoring="neg_mean_absolute_error",
    n_jobs=-1, verbose=1,
)

logging.info("Training grid prediction model...")
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
logging.info(f"Best params: {grid.best_params_}")


# ---------------- Evaluation ----------------
preds = best_model.predict(X_test)
mae   = mean_absolute_error(y_test, preds)
logging.info(f"Grid Prediction MAE: {mae:.2f} positions")


# ---------------- Save ----------------
os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
joblib.dump(best_model, MODEL_FILE)
logging.info(f"Grid model saved → {MODEL_FILE}")
