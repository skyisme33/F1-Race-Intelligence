import logging
import os
import sys

# ── Make sibling packages importable when running from any working directory ──
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [
    _ROOT,
    os.path.join(_ROOT, "core"),
    os.path.join(_ROOT, "data_pipeline"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from session_cache_validator import CacheStatus, check_cache, format_warning
from feature_engineering import (
    compute_championship_momentum,
    compute_ewm_form,
    compute_team_ytd_wins,
)
import joblib
import numpy as np
import pandas as pd


# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ---------------- Resolved paths ----------------
_ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_FILE   = os.path.join(_ROOT, "models", "f1_model.pkl")
FEATURE_FILE = os.path.join(_ROOT, "models", "model_features.pkl")
TRAIN_DATA   = os.path.join(_ROOT, "data", "processed", "f1_features_clean.csv")
CACHE_DIR    = os.path.join(_ROOT, "data", "cache")

# ---------------- Load artefacts once at import time ----------------
winner_model    = joblib.load(MODEL_FILE)
feature_columns = joblib.load(FEATURE_FILE)

_train_df     = pd.read_csv(TRAIN_DATA)
feature_means = _train_df.mean(numeric_only=True)

# Per-track-type tyre-deg means (used as fallback for TyreDegFP2)
_TYRE_DEG_BY_TRACK = (
    _train_df.groupby("TrackType")["TyreDegFP2"].mean().to_dict()
    if "TyreDegFP2" in _train_df.columns
    else {}
)
_TYRE_DEG_GLOBAL = (
    _train_df["TyreDegFP2"].mean()
    if "TyreDegFP2" in _train_df.columns
    else 0.23
)

# Raw-lap column means — backward-compat with old model weights.
_RAW_LAP_COLS = [
    "QualiFastLap", "FP2FastLap",
    "QualiMeanLap", "QualiStdLap",
    "FP2MeanLap",   "FP2StdLap",
    "Sector1",      "Sector2", "Sector3",
]


# ---------------- Track classification ----------------
_STREET    = {"Monaco", "Singapore", "Azerbaijan"}
_POWER     = {"Italy", "Austria", "Canada", "Mexico"}
_TECHNICAL = {"Hungary", "Spain", "Netherlands", "Japan", "Brazil"}


def classify_track(gp: str) -> str:
    if gp in _STREET:
        return "Street"
    if gp in _POWER:
        return "Power"
    if gp in _TECHNICAL:
        return "Technical"
    return "Balanced"


# ---------------- GridConfidence helper ----------------
def _grid_confidence(qpr: pd.Series) -> pd.Series:
    gap     = qpr - qpr.min()
    diffs   = gap.sort_values().diff().dropna()
    eps     = max(float(diffs.median()) / 2.0, 0.0005)
    raw     = 1.0 / (gap + eps)
    max_raw = raw.max()
    if max_raw == 0:
        return pd.Series(np.ones(len(qpr)) / len(qpr), index=qpr.index)
    return raw / max_raw


# ---------------- Core feature derivation ----------------
def _derive_features(X: pd.DataFrame, gp: str) -> pd.DataFrame:
    track_type = classify_track(gp)

    X["GP"]        = gp
    X["TrackType"] = track_type

    X["GridPosition"]  = X["GridPosition"].replace(0, 20).clip(1, 20)
    X["GridAdvantage"] = 1.0 / X["GridPosition"]
    X["GridConfidence"] = _grid_confidence(X["QualiPaceRatio"])

    X["MeanPaceDiff"]    = X["FPPaceRatio"]   - X["QualiPaceRatio"]
    X["ConsistencyDiff"] = X["FPConsistency"] - X["QualiConsistency"]

    X["TotalSectorRatio"] = (
        X["Sector1Ratio"] + X["Sector2Ratio"] + X["Sector3Ratio"]
    )
    X["SectorVariance"] = X[
        ["Sector1Ratio", "Sector2Ratio", "Sector3Ratio"]
    ].var(axis=1)

    if "TrackTemp" in X.columns:
        X["HotTrack"]            = (X["TrackTemp"] > 35).astype(int)
        X["TempPaceInteraction"] = X["TrackTemp"] * X["FPPaceRatio"]
    else:
        logging.warning("TrackTemp missing — HotTrack/TempPaceInteraction use training means.")
        X["HotTrack"]            = int(feature_means.get("HotTrack", 0))
        X["TempPaceInteraction"] = feature_means.get("TempPaceInteraction", 0)

    if "Humidity" in X.columns:
        X["HighHumidity"]        = (X["Humidity"] > 50).astype(int)
        X["HumidityConsistency"] = X["Humidity"] * X["FPConsistency"]
    else:
        logging.warning("Humidity missing — HighHumidity/HumidityConsistency use training means.")
        X["HighHumidity"]        = int(feature_means.get("HighHumidity", 0))
        X["HumidityConsistency"] = feature_means.get("HumidityConsistency", 0)

    if "TyreDegFP2" not in X.columns:
        X["TyreDegFP2"] = _TYRE_DEG_BY_TRACK.get(track_type, _TYRE_DEG_GLOBAL)
        logging.info(
            f"TyreDegFP2 not in cache — using track-type mean "
            f"({X['TyreDegFP2'].iloc[0]:.4f}) for {track_type} circuit."
        )

    team_mean  = X.groupby("Team")["FPPaceRatio"].transform("mean")
    team_count = X.groupby("Team")["FPPaceRatio"].transform("count")
    X["TeammateDelta"] = np.where(
        team_count > 1,
        X["FPPaceRatio"] - (team_mean * team_count - X["FPPaceRatio"]) / (team_count - 1),
        0.0,
    )

    for col in _RAW_LAP_COLS:
        if col not in X.columns:
            X[col] = feature_means.get(col, 0.0)

    return X


# ---------------- Public API ----------------
def build_features(year: int, gp: str) -> pd.DataFrame:
    """
    Load the session cache for (year, gp), derive all model features,
    run the winner model, and return a ranked DataFrame with columns
    ['Driver', 'Winning Probability'].
    """
    cache_file = os.path.join(CACHE_DIR, f"session_cache_{year}_{gp}.csv")

    # ---- Staleness / integrity check --------------------------------
    status = check_cache(year, gp, cache_dir=CACHE_DIR)
    if status.is_stale:
        age_only = (
            status.age_hours is not None
            and status.age_hours < 0.1
            and not status.missing_columns
            and not status.zero_columns
            and not status.ratio_columns_invalid
            and status.driver_count is not None
            and status.driver_count >= 15
        )
        if not age_only:
            msg = format_warning(status)
            if status.must_regenerate:
                raise RuntimeError(
                    f"Session cache for {gp} {year} is invalid and must be "
                    f"regenerated before prediction can run.\n\n{msg}\n\n"
                    "Call extract_session_stats(year, gp) to refresh it."
                )
            logging.warning(msg)
    # -----------------------------------------------------------------

    session_df = pd.read_csv(cache_file)
    drivers    = session_df["Driver"].tolist()

    hist_df = _train_df.copy()

    placeholder = session_df[["Driver", "Team"]].copy()
    placeholder["Year"]   = year
    placeholder["Round"]  = 999
    placeholder["Winner"] = 0
    combined = pd.concat([hist_df, placeholder], ignore_index=True)

    ewm_series  = compute_ewm_form(combined)
    mom_series  = compute_championship_momentum(combined)
    team_series = compute_team_ytd_wins(combined)

    placeholder_idx = combined.index[combined["Round"] == 999]
    driver_to_ewm  = dict(zip(combined.loc[placeholder_idx, "Driver"], ewm_series.loc[placeholder_idx]))
    driver_to_mom  = dict(zip(combined.loc[placeholder_idx, "Driver"], mom_series.loc[placeholder_idx]))
    driver_to_team = dict(zip(combined.loc[placeholder_idx, "Driver"], team_series.loc[placeholder_idx]))

    session_df["EWMForm"]              = session_df["Driver"].map(driver_to_ewm).fillna(0.0)
    session_df["ChampionshipMomentum"] = session_df["Driver"].map(driver_to_mom).fillna(0.0)
    session_df["TeamYTDWins"]          = session_df["Driver"].map(driver_to_team).fillna(0.0)

    logging.info(
        f"History features for {gp} {year} — "
        f"EWMForm max: {session_df['EWMForm'].max():.3f}, "
        f"Momentum max: {session_df['ChampionshipMomentum'].max():.3f}, "
        f"TeamYTDWins max: {session_df['TeamYTDWins'].max():.3f}"
    )

    X = session_df.drop(columns=["Driver"]).copy()
    X = _derive_features(X, gp)

    for col in feature_columns:
        if col not in X.columns:
            fallback = feature_means.get(col, 0.0)
            logging.warning(
                f"Feature '{col}' could not be derived — using training mean ({fallback:.4f})."
            )
            X[col] = fallback

    X = X[feature_columns]

    probs = winner_model.predict_proba(X)[:, 1]
    total = probs.sum()
    if total > 0:
        probs = probs / total
    else:
        logging.warning("All predicted probabilities are zero — check model inputs.")
        probs = np.ones(len(probs)) / len(probs)

    result = (
        pd.DataFrame({"Driver": drivers, "Winning Probability": probs})
        .sort_values("Winning Probability", ascending=False)
        .reset_index(drop=True)
    )

    logging.info(
        f"Prediction complete — predicted winner: "
        f"{result.iloc[0]['Driver']} ({result.iloc[0]['Winning Probability']:.2%})"
    )

    return result
