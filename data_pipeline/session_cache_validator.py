"""
session_cache_validator.py
--------------------------
Staleness and integrity checks for F1 session cache files.

A session cache is the CSV written by precompute_session_stats.py and
consumed by predict_winner.py.  It is considered *stale* (and should be
regenerated) if any of the following are true:

  1. The file does not exist.
  2. The file is older than MAX_CACHE_AGE_HOURS since the race weekend
     qualifying session would have taken place.
  3. The file is missing required columns.
  4. The file has fewer drivers than MIN_DRIVER_COUNT.
  5. Any required column contains only NaN / zero values (silent data
     collection failure).
  6. QualiPaceRatio or FPPaceRatio minimum is far from 1.0, suggesting
     the ratios were not properly normalised (e.g. raw lap times were
     stored instead).

Usage
-----
From predict_winner.py or app.py:

    from session_cache_validator import CacheStatus, check_cache, format_warning

    status = check_cache(year, gp)

    if status.is_stale:
        print(format_warning(status))   # plain text
        # or in Streamlit:
        st.warning(format_warning(status))

    if status.must_regenerate:
        extract_session_stats(year, gp)

The check is intentionally cheap — it only reads the CSV header + a few
aggregate statistics, not the full file content.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd


# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #

# A cache written more than this many hours ago is considered stale.
# Qualifying happens ~24 h before the race; 36 h gives a small buffer
# in case the user ran the pipeline on Friday (FP day) and forgot to
# refresh after Saturday qualifying.
MAX_CACHE_AGE_HOURS: int = 36

# Minimum number of drivers that must appear in a valid cache.
# A full grid is 20; allow for late withdrawals / DNS.
MIN_DRIVER_COUNT: int = 15

# Columns that *must* exist in every cache file.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "Driver",
    "Team",
    "GridPosition",
    "QualiPaceRatio",
    "FPPaceRatio",
    "QualiConsistency",
    "FPConsistency",
    "Sector1Ratio",
    "Sector2Ratio",
    "Sector3Ratio",
)

# Columns that should have been populated by the weather extraction
# added in Phase 2.  Missing weather columns are a warning, not fatal,
# because predict_winner.py has a graceful fallback.
WEATHER_COLUMNS: tuple[str, ...] = ("TrackTemp", "AirTemp", "Humidity")

# Columns where a column-wide mean of 0 (or NaN) signals a collection
# failure.  GridPosition = 0 would mean all drivers start from pit-lane,
# which is physically impossible.
NON_ZERO_COLUMNS: tuple[str, ...] = (
    "QualiPaceRatio",
    "FPPaceRatio",
    "GridPosition",
)

# A pace ratio minimum outside this range suggests the ratios were not
# normalised correctly (e.g. raw seconds stored instead of ratio).
# A correctly normalised ratio has its minimum at 1.000 ± a tiny float
# rounding error.
RATIO_MIN_TOLERANCE: float = 0.02   # allow [0.98, 1.02]


# ------------------------------------------------------------------ #
# Data class
# ------------------------------------------------------------------ #

@dataclass
class CacheStatus:
    """
    Result of a staleness / integrity check on a single session cache.

    Attributes
    ----------
    year, gp : int / str
        The race this cache covers.
    cache_file : str
        Expected file path on disk.
    exists : bool
        Whether the file exists at all.
    age_hours : float | None
        File age in hours since last modification.  None if not found.
    driver_count : int | None
        Number of driver rows found.  None if file could not be read.
    missing_columns : list[str]
        Required columns absent from the file.
    missing_weather_columns : list[str]
        Weather columns absent (non-fatal).
    zero_columns : list[str]
        Required numeric columns whose mean is 0 or NaN (data failure).
    ratio_columns_invalid : list[str]
        Pace ratio columns whose minimum deviates from 1.0 by more than
        RATIO_MIN_TOLERANCE (normalisation failure).
    issues : list[str]
        Human-readable list of all problems detected.
    """

    year: int
    gp: str
    cache_file: str

    exists: bool = False
    age_hours: Optional[float] = None
    driver_count: Optional[int] = None

    missing_columns: list[str] = field(default_factory=list)
    missing_weather_columns: list[str] = field(default_factory=list)
    zero_columns: list[str] = field(default_factory=list)
    ratio_columns_invalid: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def is_stale(self) -> bool:
        """True if any staleness or integrity issue was found."""
        return bool(self.issues)

    @property
    def must_regenerate(self) -> bool:
        """
        True if the file must be regenerated before prediction can run.

        Missing-weather is a warning only — predict_winner.py falls back
        to training means.  Everything else (missing file, missing required
        columns, too few drivers, normalisation failure) blocks prediction.
        """
        blocking = (
            not self.exists
            or bool(self.missing_columns)
            or (self.driver_count is not None and self.driver_count < MIN_DRIVER_COUNT)
            or bool(self.zero_columns)
            or bool(self.ratio_columns_invalid)
        )
        return blocking

    @property
    def severity(self) -> str:
        """'error' | 'warning' | 'ok'"""
        if self.must_regenerate:
            return "error"
        if self.is_stale:
            return "warning"
        return "ok"


# ------------------------------------------------------------------ #
# Core check
# ------------------------------------------------------------------ #

def check_cache(year: int, gp: str, cache_dir: str = ".") -> CacheStatus:
    """
    Run all staleness and integrity checks on the session cache for
    (year, gp) and return a CacheStatus.

    Parameters
    ----------
    year : int
    gp   : str   — Grand Prix name, e.g. "Bahrain"
    cache_dir : str
        Directory where cache files are stored (default: current working
        directory, matching predict_winner.py's convention).

    Returns
    -------
    CacheStatus
        Inspect ``.is_stale``, ``.must_regenerate``, and ``.issues`` to
        decide what to do next.
    """
    cache_file = os.path.join(cache_dir, f"session_cache_{year}_{gp}.csv")
    status = CacheStatus(year=year, gp=gp, cache_file=cache_file)

    # ---- 1. Existence ------------------------------------------------
    if not os.path.exists(cache_file):
        status.issues.append(
            f"Cache file not found: {cache_file}"
        )
        return status   # no point running further checks

    status.exists = True

    # ---- 2. Age -------------------------------------------------------
    mtime      = os.path.getmtime(cache_file)
    age_hours  = (datetime.now(timezone.utc).timestamp() - mtime) / 3600
    status.age_hours = round(age_hours, 1)

    if age_hours > MAX_CACHE_AGE_HOURS:
        status.issues.append(
            f"Cache is {age_hours:.1f} h old (threshold: {MAX_CACHE_AGE_HOURS} h). "
            "Qualifying data may have changed since it was written."
        )

    # ---- 3. Read the file (header + stats only) ----------------------
    try:
        df = pd.read_csv(cache_file)
    except Exception as exc:
        status.issues.append(f"Cache file could not be read: {exc}")
        return status

    # ---- 4. Required columns -----------------------------------------
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        status.missing_columns = missing
        status.issues.append(
            f"Missing required columns: {missing}. "
            "Re-run precompute_session_stats.py."
        )

    # ---- 5. Weather columns (non-fatal) ------------------------------
    missing_weather = [c for c in WEATHER_COLUMNS if c not in df.columns]
    if missing_weather:
        status.missing_weather_columns = missing_weather
        status.issues.append(
            f"Weather columns absent: {missing_weather}. "
            "predict_winner.py will fall back to training means — "
            "consider updating precompute_session_stats.py."
        )

    # ---- 6. Driver count ---------------------------------------------
    status.driver_count = len(df)
    if len(df) < MIN_DRIVER_COUNT:
        status.issues.append(
            f"Only {len(df)} drivers in cache (minimum: {MIN_DRIVER_COUNT}). "
            "Session download may have been incomplete."
        )

    # ---- 7. Zero / NaN columns (data collection failure) -------------
    for col in NON_ZERO_COLUMNS:
        if col not in df.columns:
            continue
        col_mean = pd.to_numeric(df[col], errors="coerce").mean()
        if col_mean == 0 or pd.isna(col_mean):
            status.zero_columns.append(col)
            status.issues.append(
                f"Column '{col}' is entirely zero or NaN — "
                "data collection failure. Re-run precompute_session_stats.py."
            )

    # ---- 8. Pace ratio normalisation ---------------------------------
    for col in ("QualiPaceRatio", "FPPaceRatio"):
        if col not in df.columns:
            continue
        col_min = pd.to_numeric(df[col], errors="coerce").min()
        if pd.isna(col_min):
            continue
        if abs(col_min - 1.0) > RATIO_MIN_TOLERANCE:
            status.ratio_columns_invalid.append(col)
            status.issues.append(
                f"'{col}' minimum is {col_min:.4f} (expected ≈ 1.000 ± "
                f"{RATIO_MIN_TOLERANCE}). Raw lap times may have been stored "
                "instead of ratios. Re-run precompute_session_stats.py."
            )

    if not status.issues:
        logging.info(
            f"Cache OK — {gp} {year} | {status.driver_count} drivers | "
            f"age {status.age_hours:.1f} h"
        )

    return status


# ------------------------------------------------------------------ #
# Formatting helpers
# ------------------------------------------------------------------ #

def format_warning(status: CacheStatus) -> str:
    """
    Return a single human-readable string summarising all issues.

    Suitable for logging, print(), or st.warning() / st.error().
    """
    if not status.issues:
        return f"Session cache for {status.gp} {status.year} is valid."

    header = (
        f"⚠ Session cache issues detected for {status.gp} {status.year} "
        f"({status.cache_file}):"
    )
    body = "\n".join(f"  • {issue}" for issue in status.issues)

    if status.must_regenerate:
        footer = "→ Cache must be regenerated before prediction can run."
    else:
        footer = "→ Prediction can continue but results may be degraded."

    return f"{header}\n{body}\n{footer}"


def streamlit_banner(status: CacheStatus) -> None:
    """
    Display a Streamlit st.error / st.warning / st.success banner
    based on the cache status.  Does nothing if severity is 'ok'.

    Call this in app.py after the user selects a GP, before running
    the prediction.
    """
    import streamlit as st

    if status.severity == "ok":
        age_str = f"{status.age_hours:.1f} h" if status.age_hours is not None else "unknown"
        st.success(
            f"✅ Session cache is valid — {status.driver_count} drivers, "
            f"written {age_str} ago."
        )
        return

    message = format_warning(status)

    if status.severity == "error":
        st.error(message)
        if status.must_regenerate:
            st.info(
                "💡 Click **🔄 Refresh Race Data** to download fresh session data "
                "from FastF1, then run the prediction again."
            )
    else:
        st.warning(message)