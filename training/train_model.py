import json
import logging
import os
import sys
import warnings
import joblib
from datetime import datetime

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# ── Make core/ importable ──
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_ROOT, "core") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "core"))

from feature_engineering import grid_confidence


# ---------------- Resolved paths ----------------
DATA_FILE    = os.path.join(_ROOT, "data",    "processed", "f1_features_clean.csv")
MODEL_FILE   = os.path.join(_ROOT, "models",  "f1_model.pkl")
FEATURE_FILE = os.path.join(_ROOT, "models",  "model_features.pkl")
META_FILE    = os.path.join(_ROOT, "config",  "f1_model_metadata.json")

TARGET = "Winner"

DROP_COLS = [
    "Winner", "Driver", "Year",
    "QualiFastLap", "FP2FastLap",
    "QualiMeanLap", "QualiStdLap",
    "FP2MeanLap",   "FP2StdLap",
    "Sector1",      "Sector2", "Sector3",
]

CATEGORICAL_COLS = ["Team", "GP", "TrackType"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ================================================================
# 1. Load & validate
# ================================================================
logging.info("Loading dataset...")
df = pd.read_csv(DATA_FILE)

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df = df.dropna(subset=["Year"])
df["Year"] = df["Year"].astype(int)

df["Winner"] = pd.to_numeric(df["Winner"], errors="coerce").fillna(0).astype(int)

logging.info(f"Loaded {len(df)} rows — years: {sorted(df['Year'].unique())}")
logging.info(f"Winner distribution: {df['Winner'].value_counts().to_dict()}")


# ================================================================
# 2. Derived features
# ================================================================
logging.info("Computing GridConfidence from qualifying pace...")
df["GridConfidence"] = (
    df.groupby(["Year", "GP"])["QualiPaceRatio"]
    .transform(grid_confidence)
)
df["GridPosition"]  = df["GridPosition"].replace(0, 20).clip(1, 20)
df["GridAdvantage"] = 1.0 / df["GridPosition"]


# ================================================================
# 3. Train / test split + walk-forward CV folds
# ================================================================
years_all   = sorted(df["Year"].unique())
latest_year = years_all[-1]

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


# ================================================================
# 4. Feature matrices
# ================================================================
X_train = train_df.drop(columns=[c for c in DROP_COLS if c in train_df.columns])
y_train = train_df[TARGET]

X_test  = test_df.drop(columns=[c for c in DROP_COLS if c in test_df.columns])
y_test  = test_df[TARGET]

os.makedirs(os.path.dirname(FEATURE_FILE), exist_ok=True)
joblib.dump(X_train.columns.tolist(), FEATURE_FILE)
logging.info(f"Saved feature list ({len(X_train.columns)} features) -> {FEATURE_FILE}")

NUMERIC_COLS = [c for c in X_train.columns if c not in CATEGORICAL_COLS]


# ================================================================
# 5. Class imbalance
# ================================================================
neg_count = int((y_train == 0).sum())
pos_count = int((y_train == 1).sum())
spw       = neg_count / pos_count
logging.info(f"Class balance — neg: {neg_count}, pos: {pos_count}, scale_pos_weight: {spw:.2f}")


# ================================================================
# 6. Preprocessing
# ================================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="mean"), NUMERIC_COLS),
        (
            "cat",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            CATEGORICAL_COLS,
        ),
    ]
)


# ================================================================
# 7. Pipeline + hyperparameter search
# ================================================================
pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("lgbm", LGBMClassifier(
        random_state=42,
        n_jobs=1,
        verbose=-1,
        scale_pos_weight=spw,
    )),
])

param_grid = {
    "lgbm__n_estimators":      [300, 400],
    "lgbm__learning_rate":     [0.03, 0.05],
    "lgbm__num_leaves":        [15, 31],
    "lgbm__min_child_samples": [5, 10],
    "lgbm__reg_alpha":         [0.0, 0.1],
}

grid_search = GridSearchCV(
    pipeline, param_grid,
    cv=cv_folds, scoring="f1",
    n_jobs=-1, verbose=1, refit=True,
)

logging.info("Training LightGBM (walk-forward GridSearchCV)...")
grid_search.fit(X_train, y_train)

logging.info(f"Best params:  {grid_search.best_params_}")
logging.info(f"Best CV F1:   {grid_search.best_score_:.4f}")

best_pipeline = grid_search.best_estimator_


# ================================================================
# 9. Evaluation
# ================================================================
logging.info(f"Evaluating on held-out test year ({latest_year})...")

y_pred = best_pipeline.predict(X_test)
y_prob = best_pipeline.predict_proba(X_test)[:, 1]

f1  = f1_score(y_test, y_pred, zero_division=0)
logging.info(f"F1 Score:   {f1:.4f}")

try:
    auc = roc_auc_score(y_test, y_prob)
    logging.info(f"ROC AUC:    {auc:.4f}")
except ValueError:
    auc = None
    logging.warning("ROC AUC could not be computed.")

try:
    pr_auc = average_precision_score(y_test, y_prob)
    logging.info(f"PR AUC:     {pr_auc:.4f}")
except ValueError:
    pr_auc = None

print("\nConfusion Matrix:\n",      confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

lgbm_step  = best_pipeline.named_steps["lgbm"]
cat_names  = list(
    best_pipeline.named_steps["preprocess"]
    .named_transformers_["cat"]
    .get_feature_names_out(CATEGORICAL_COLS)
)
feat_names  = NUMERIC_COLS + cat_names
importances = pd.Series(lgbm_step.feature_importances_, index=feat_names)

logging.info("Top 10 feature importances (gain):")
for name, val in importances.sort_values(ascending=False).head(10).items():
    logging.info(f"  {name:35s}: {val:.1f}")


# ================================================================
# 10. Save model + metadata
# ================================================================
os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
joblib.dump(best_pipeline, MODEL_FILE)
logging.info(f"Model saved -> {MODEL_FILE}")

os.makedirs(os.path.dirname(META_FILE), exist_ok=True)
metadata = {
    "trained_at":       datetime.now().isoformat(),
    "model":            "LightGBM",
    "test_year":        int(latest_year),
    "f1_score":         round(float(f1),     4),
    "roc_auc":          round(float(auc),    4) if auc    is not None else None,
    "pr_auc":           round(float(pr_auc), 4) if pr_auc is not None else None,
    "n_train":          int(len(X_train)),
    "n_test":           int(len(X_test)),
    "scale_pos_weight": round(spw, 2),
    "best_params":      grid_search.best_params_,
    "best_cv_f1":       round(float(grid_search.best_score_), 4),
    "features":         X_train.columns.tolist(),
}

with open(META_FILE, "w") as fh:
    json.dump(metadata, fh, indent=2)

logging.info(f"Metadata saved -> {META_FILE}")
