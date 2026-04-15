import logging
import os
import sys
import pandas as pd

# ── Resolve project root ──
_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE  = os.path.join(_ROOT, "data", "processed", "f1_features.csv")
OUTPUT_FILE = os.path.join(_ROOT, "data", "processed", "f1_features_clean.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def clean(df: pd.DataFrame) -> pd.DataFrame:

    original_len = len(df)

    # ---- Year ----
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna(subset=["Year"])
    df["Year"] = df["Year"].astype(int)

    # ---- Winner ----
    df["Winner"] = pd.to_numeric(df["Winner"], errors="coerce")

    nan_winners = df["Winner"].isna().sum()
    if nan_winners > 0:
        logging.warning(
            f"{nan_winners} rows have NaN Winner — filled with 0. "
            "Check your source data for missing race results."
        )
        logging.warning(
            df[df["Winner"].isna()][["Year", "GP", "Driver"]].to_string(index=False)
        )

    df["Winner"] = df["Winner"].fillna(0).astype(int)

    winners_per_race = df.groupby(["Year", "GP"])["Winner"].sum()
    bad_races = winners_per_race[winners_per_race != 1]
    if not bad_races.empty:
        logging.warning(
            f"{len(bad_races)} races do not have exactly 1 winner:\n"
            f"{bad_races.to_string()}"
        )

    # ---- GridPosition ----
    df["GridPosition"] = pd.to_numeric(df["GridPosition"], errors="coerce")
    df["GridPosition"] = df["GridPosition"].replace(0, 20)

    # ---- TyreDegFP2 ----
    if "TyreDegFP2" in df.columns:
        neg_count = (df["TyreDegFP2"] < 0).sum()
        if neg_count > 0:
            logging.warning(f"{neg_count} rows have negative TyreDegFP2 — clamped to 0.")
        df["TyreDegFP2"] = df["TyreDegFP2"].clip(lower=0)

    # ---- Drop rows missing identity columns ----
    before = len(df)
    df = df.dropna(subset=["Driver", "Team", "GP"])
    dropped = before - len(df)
    if dropped:
        logging.warning(f"Dropped {dropped} rows missing Driver/Team/GP.")

    logging.info(f"Cleaning complete — {original_len} rows in, {len(df)} rows out.")
    logging.info(f"Years present: {sorted(df['Year'].unique())}")

    return df


def main() -> None:
    logging.info(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    df = clean(df)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    logging.info(f"Saved cleaned dataset to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
