import logging
import os
import sys
import numpy as np
import pandas as pd

# ── Resolve project root so this module can be imported from any cwd ──
_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE  = os.path.join(_ROOT, "data", "raw",       "f1_master_data.csv")
OUTPUT_FILE = os.path.join(_ROOT, "data", "processed", "f1_features.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

RAW_LAP_COLS = [
    "QualiFastLap", "FP2FastLap",
    "QualiMeanLap", "QualiStdLap",
    "FP2MeanLap",   "FP2StdLap",
    "Sector1",      "Sector2", "Sector3",
]


def classify_track(gp: str) -> str:
    street    = {"Monaco", "Singapore", "Azerbaijan"}
    power     = {"Italy", "Austria", "Canada", "Mexico"}
    technical = {"Hungary", "Spain", "Netherlands", "Japan", "Brazil"}
    if gp in street:
        return "Street"
    if gp in power:
        return "Power"
    if gp in technical:
        return "Technical"
    return "Balanced"


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "QualiMeanLap", "QualiStdLap",
        "Sector1", "Sector2", "Sector3",
        "TrackTemp", "AirTemp", "Humidity", "GridPosition",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    practice_cols = [c for c in df.columns if "FP" in c and "MeanLap" in c]
    if practice_cols:
        fp_col  = practice_cols[0]
        std_col = fp_col.replace("MeanLap", "StdLap")
        df[fp_col] = pd.to_numeric(df[fp_col], errors="coerce")
        if std_col in df.columns:
            df[std_col] = pd.to_numeric(df[std_col], errors="coerce")

    return df


def convert_to_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw lap times to per-race pace ratios (1.000 = fastest in race)."""
    if "QualiPaceRatio" in df.columns:
        return df

    logging.info("Converting raw lap times to pace ratios...")

    practice_cols = [c for c in df.columns if "FP" in c and "MeanLap" in c]
    fp_col  = practice_cols[0]
    std_col = fp_col.replace("MeanLap", "StdLap")

    q_fast  = df.groupby(["Year", "GP"])["QualiMeanLap"].transform("min")
    fp_fast = df.groupby(["Year", "GP"])[fp_col].transform("min")
    s1_fast = df.groupby(["Year", "GP"])["Sector1"].transform("min")
    s2_fast = df.groupby(["Year", "GP"])["Sector2"].transform("min")
    s3_fast = df.groupby(["Year", "GP"])["Sector3"].transform("min")

    df["QualiPaceRatio"]   = df["QualiMeanLap"] / q_fast
    df["FPPaceRatio"]      = df[fp_col]         / fp_fast
    df["QualiConsistency"] = df["QualiStdLap"]  / q_fast
    df["FPConsistency"]    = df[std_col]         / fp_fast
    df["Sector1Ratio"]     = df["Sector1"]       / s1_fast
    df["Sector2Ratio"]     = df["Sector2"]       / s2_fast
    df["Sector3Ratio"]     = df["Sector3"]       / s3_fast

    return df


def compute_ewm_form(
    df: pd.DataFrame,
    span: int = 5,
    driver_col: str = "Driver",
    year_col: str = "Year",
    round_col: str | None = "Round",
) -> pd.Series:
    sort_cols = [year_col, round_col] if (round_col and round_col in df.columns) else [year_col]
    df_sorted = df.sort_values(sort_cols).copy()
    win_col   = df_sorted["Winner"].astype(float)
    lagged    = win_col.groupby(df_sorted[driver_col]).shift(1)
    ewm_form  = (
        lagged
        .groupby(df_sorted[driver_col])
        .transform(lambda s: s.ewm(span=span, min_periods=1).mean())
    )
    result = ewm_form.reindex(df.index).fillna(0.0)
    logging.info(
        f"EWMForm computed (span={span}): "
        f"mean={result.mean():.4f}, max={result.max():.4f}, "
        f"non-zero rows={(result > 0).sum()}"
    )
    return result


def compute_championship_momentum(
    df: pd.DataFrame,
    driver_col: str = "Driver",
    year_col: str = "Year",
    round_col: str | None = "Round",
) -> pd.Series:
    sort_cols = [year_col, round_col] if (round_col and round_col in df.columns) else [year_col]
    df_sorted = df.sort_values(sort_cols).copy()
    win_col   = df_sorted["Winner"].astype(float)
    cum_wins  = (
        win_col
        .groupby([df_sorted[driver_col], df_sorted[year_col]])
        .cumsum()
        .groupby([df_sorted[driver_col], df_sorted[year_col]])
        .shift(1)
        .fillna(0.0)
    )
    race_max   = cum_wins.groupby(
        [df_sorted[year_col], df_sorted.get("GP", df_sorted[year_col])]
    ).transform("max")
    normalised = cum_wins.div(race_max.replace(0, 1))
    result     = normalised.reindex(df.index).fillna(0.0)
    logging.info(
        f"ChampionshipMomentum computed: "
        f"mean={result.mean():.4f}, max={result.max():.4f}, "
        f"non-zero rows={(result > 0).sum()}"
    )
    return result


def compute_team_ytd_wins(
    df: pd.DataFrame,
    team_col: str = "Team",
    year_col: str = "Year",
    round_col: str | None = "Round",
) -> pd.Series:
    sort_cols = [year_col, round_col] if (round_col and round_col in df.columns) else [year_col]
    df_sorted = df.sort_values(sort_cols).copy()
    win_col   = df_sorted["Winner"].astype(float)
    race_team_wins = win_col.groupby(
        [df_sorted[year_col], df_sorted.get("GP", df_sorted[year_col]), df_sorted[team_col]]
    ).transform("sum")
    cum_team_wins = (
        race_team_wins
        .groupby([df_sorted[team_col], df_sorted[year_col]])
        .cumsum()
        .groupby([df_sorted[team_col], df_sorted[year_col]])
        .shift(1)
        .fillna(0.0)
    )
    race_max   = cum_team_wins.groupby(
        [df_sorted[year_col], df_sorted.get("GP", df_sorted[year_col])]
    ).transform("max")
    normalised = cum_team_wins.div(race_max.replace(0, 1))
    result     = normalised.reindex(df.index).fillna(0.0)
    logging.info(
        f"TeamYTDWins computed: "
        f"mean={result.mean():.4f}, max={result.max():.4f}, "
        f"non-zero rows={(result > 0).sum()}"
    )
    return result


def grid_confidence(qpr: pd.Series) -> pd.Series:
    gap   = qpr - qpr.min()
    diffs = gap.sort_values().diff().dropna()
    eps   = max(float(diffs.median()) / 2.0, 0.0005)
    raw   = 1.0 / (gap + eps)
    m     = raw.max()
    return raw / m if m > 0 else pd.Series(np.ones(len(qpr)) / len(qpr), index=qpr.index)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

    df = ensure_numeric(df)
    df = convert_to_ratios(df)

    df["MeanPaceDiff"]    = df["FPPaceRatio"]   - df["QualiPaceRatio"]
    df["ConsistencyDiff"] = df["FPConsistency"] - df["QualiConsistency"]

    df["TotalSectorRatio"] = (
        df["Sector1Ratio"] + df["Sector2Ratio"] + df["Sector3Ratio"]
    )
    df["SectorVariance"] = df[
        ["Sector1Ratio", "Sector2Ratio", "Sector3Ratio"]
    ].var(axis=1)

    df["GridPosition"]  = df["GridPosition"].replace(0, 20).clip(1, 20)
    df["GridAdvantage"] = 1.0 / df["GridPosition"]
    df["GridConfidence"] = (
        df.groupby(["Year", "GP"])["QualiPaceRatio"]
        .transform(grid_confidence)
    )

    if "Driver" in df.columns:
        df["EWMForm"] = compute_ewm_form(df)
    else:
        logging.warning("'Driver' column not found — EWMForm set to 0.")
        df["EWMForm"] = 0.0

    if "Driver" in df.columns:
        df["ChampionshipMomentum"] = compute_championship_momentum(df)
    else:
        logging.warning("'Driver' column not found — ChampionshipMomentum set to 0.")
        df["ChampionshipMomentum"] = 0.0

    if "Team" in df.columns:
        df["TeamYTDWins"] = compute_team_ytd_wins(df)
    else:
        logging.warning("'Team' column not found — TeamYTDWins set to 0.")
        df["TeamYTDWins"] = 0.0

    df["HotTrack"]            = (df["TrackTemp"] > 35).astype(int)
    df["HighHumidity"]        = (df["Humidity"]  > 50).astype(int)
    df["TempPaceInteraction"] = df["TrackTemp"]  * df["FPPaceRatio"]
    df["HumidityConsistency"] = df["Humidity"]   * df["FPConsistency"]

    if "TyreDegFP2" in df.columns:
        neg = (df["TyreDegFP2"] < 0).sum()
        if neg:
            logging.warning(f"Clamping {neg} negative TyreDegFP2 values to 0.")
        df["TyreDegFP2"] = df["TyreDegFP2"].clip(lower=0)

    group_cols = ["Year", "GP", "Team"]
    team_mean  = df.groupby(group_cols)["FPPaceRatio"].transform("mean")
    team_count = df.groupby(group_cols)["FPPaceRatio"].transform("count")
    df["TeammateDelta"] = np.where(
        team_count > 1,
        df["FPPaceRatio"] - (team_mean * team_count - df["FPPaceRatio"]) / (team_count - 1),
        0,
    )

    df["TrackType"] = df["GP"].apply(classify_track)
    df["Team"]      = df["Team"].astype(str)
    df["GP"]        = df["GP"].astype(str)
    df["TrackType"] = df["TrackType"].astype(str)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    df["Winner"] = pd.to_numeric(df["Winner"], errors="coerce").fillna(0)
    df["Winner"] = (df["Winner"] == 1).astype(int)

    logging.info("Feature engineering completed.")
    return df


def main() -> None:
    logging.info(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    df = engineer_features(df)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    logging.info(f"Saved engineered features to {OUTPUT_FILE}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
