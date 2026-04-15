import logging
import os
import sys

import fastf1
import numpy as np
import pandas as pd

# ── Resolve project root and point FastF1 cache at the right folder ──
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fastf1.Cache.enable_cache(os.path.join(_ROOT, "cache"))

CACHE_DIR = os.path.join(_ROOT, "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

MIN_USABLE_LAPS = 30


# ---------------- Practice session selector ----------------
def select_race_pace_session(year: int, gp: str) -> fastf1.core.Session:
    for name in ["FP3", "FP2", "FP1"]:
        try:
            session = fastf1.get_session(year, gp, name)
            session.load(laps=True, telemetry=False, weather=True)
            if len(session.laps) >= MIN_USABLE_LAPS:
                logging.info(f"Using {name} as race-pace session for {gp} {year}")
                return session
            logging.info(f"{name} only has {len(session.laps)} laps — trying next session.")
        except Exception as e:
            logging.info(f"{name} unavailable for {gp} {year}: {e}")
            continue
    raise ValueError(f"No usable practice session found for {gp} {year}")


# ---------------- Reference lap (top-N average) ----------------
def robust_reference_lap(
    laps: pd.DataFrame, column: str = "LapTime", top_n: int = 3
) -> float | None:
    times = laps[column].dt.total_seconds().dropna().sort_values()
    top   = times.head(top_n)
    return float(top.mean()) if len(top) > 0 else None


# ---------------- Tyre degradation from stint data ----------------
def compute_tyre_deg(fp_laps: pd.DataFrame) -> float:
    try:
        times = fp_laps["LapTime"].dt.total_seconds().dropna().reset_index(drop=True)
        if len(times) < 6:
            return 0.0
        n     = len(times)
        cut   = max(1, int(n * 0.30))
        early = times.iloc[:cut].mean()
        late  = times.iloc[-cut:].mean()
        if early <= 0:
            return 0.0
        deg = (late - early) / early
        return float(max(deg, 0.0))
    except Exception:
        return 0.0


# ---------------- Main extraction function ----------------
def extract_session_stats(year: int, gp: str) -> None:
    """
    Download qualifying + practice data for (year, gp), compute all features
    the ML model needs at inference time, and write a session cache CSV to
    data/cache/session_cache_{year}_{gp}.csv.
    """
    logging.info(f"Loading qualifying session for {gp} {year}...")
    quali = fastf1.get_session(year, gp, "Q")
    quali.load(laps=True, telemetry=False, weather=False)

    fp = select_race_pace_session(year, gp)

    q_quick  = quali.laps.pick_quicklaps()
    fp_quick = fp.laps.pick_quicklaps()

    q_fastest_lap  = robust_reference_lap(q_quick,  "LapTime")
    fp_fastest_lap = robust_reference_lap(fp_quick, "LapTime")
    q_fastest_s1   = robust_reference_lap(quali.laps, "Sector1Time")
    q_fastest_s2   = robust_reference_lap(quali.laps, "Sector2Time")
    q_fastest_s3   = robust_reference_lap(quali.laps, "Sector3Time")

    if None in [q_fastest_lap, fp_fastest_lap, q_fastest_s1, q_fastest_s2, q_fastest_s3]:
        raise ValueError(f"Could not compute reference laps for {gp} {year}")

    weather            = fp.weather_data
    session_track_temp = float(weather["TrackTemp"].mean())
    session_air_temp   = float(weather["AirTemp"].mean())
    session_humidity   = float(weather["Humidity"].mean())

    logging.info(
        f"Weather — TrackTemp: {session_track_temp:.1f}°C  "
        f"AirTemp: {session_air_temp:.1f}°C  "
        f"Humidity: {session_humidity:.1f}%"
    )

    rows = []

    for driver in quali.drivers:
        try:
            q_laps  = quali.laps.pick_driver(driver).pick_quicklaps()
            fp_laps = fp.laps.pick_driver(driver).pick_quicklaps()

            if q_laps.empty or fp_laps.empty:
                logging.debug(f"Skipping {driver} — no quick laps in Q or FP.")
                continue

            q_mean = float(q_laps["LapTime"].dt.total_seconds().mean())
            q_std  = float(q_laps["LapTime"].dt.total_seconds().std())

            fp_times = fp_laps["LapTime"].dt.total_seconds().dropna()
            fastest  = fp_times.min()
            valid    = fp_times[fp_times < fastest * 1.07]
            if len(valid) < 3:
                valid = fp_times[fp_times < fastest * 1.12]
            if len(valid) < 2:
                logging.debug(f"Skipping {driver} — insufficient FP laps.")
                continue

            fp_mean  = float(valid.mean())
            fp_std   = float(valid.std()) if len(valid) > 1 else 0.0
            tyre_deg = compute_tyre_deg(fp.laps.pick_driver(driver))

            q_fast = q_laps.pick_fastest()
            drv    = quali.get_driver(driver)

            rows.append({
                "Driver": drv["Abbreviation"],
                "Team":   drv["TeamName"],
                "QualiPaceRatio":   q_mean / q_fastest_lap,
                "FPPaceRatio":      fp_mean / fp_fastest_lap,
                "QualiConsistency": q_std  / q_fastest_lap,
                "FPConsistency":    fp_std / fp_fastest_lap,
                "Sector1Ratio": q_fast["Sector1Time"].total_seconds() / q_fastest_s1,
                "Sector2Ratio": q_fast["Sector2Time"].total_seconds() / q_fastest_s2,
                "Sector3Ratio": q_fast["Sector3Time"].total_seconds() / q_fastest_s3,
                "GridPosition": int(drv["GridPosition"]) if drv["GridPosition"] > 0 else 20,
                "TrackTemp": session_track_temp,
                "AirTemp":   session_air_temp,
                "Humidity":  session_humidity,
                "TyreDegFP2": tyre_deg,
            })

        except Exception as e:
            logging.warning(f"Could not process driver {driver}: {e}")
            continue

    if not rows:
        raise RuntimeError(f"No driver data extracted for {gp} {year}")

    df  = pd.DataFrame(rows)
    out = os.path.join(CACHE_DIR, f"session_cache_{year}_{gp}.csv")
    df.to_csv(out, index=False)
    logging.info(f"Saved session cache → {out}  ({len(df)} drivers)")
