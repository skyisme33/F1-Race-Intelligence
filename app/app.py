import json
import os
import sys

# ── Add project root and sub-packages to path so imports work from app/ ──
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in [
    _ROOT,
    os.path.join(_ROOT, "core"),
    os.path.join(_ROOT, "data_pipeline"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import altair as alt
import fastf1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from predict_winner import (
    _derive_features,
    build_features,
    feature_columns,
    feature_means,
    winner_model,
)
from precompute_session_stats import extract_session_stats, CACHE_DIR
from session_cache_validator import check_cache, streamlit_banner

# ── Resolved file paths ──
PROCESSED_DIR = os.path.join(_ROOT, "data",   "processed")
MODELS_DIR    = os.path.join(_ROOT, "models")
CONFIG_DIR    = os.path.join(_ROOT, "config")
RAW_DIR       = os.path.join(_ROOT, "data",   "raw")

TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR, "f1_features_clean.csv")
META_FILE_PATH  = os.path.join(CONFIG_DIR,     "f1_model_metadata.json")
MASTER_DATA_PATH = os.path.join(RAW_DIR,       "f1_master_data.csv")


st.set_page_config(page_title="F1 Race Intelligence", layout="wide")

st.title("🏎️ F1 Race Intelligence Dashboard")
st.write(
    "Predict Formula 1 race winner probabilities using practice and qualifying performance."
)


# ================================================================
# Tab layout
# ================================================================
tab_predict, tab_h2h, tab_circuit, tab_whatif, tab_retrain, tab_backtest = st.tabs([
    "🔮 Predict",
    "⚔️ Head-to-Head",
    "🗺️ Circuit History",
    "🎛️ What-If",
    "🔁 Retrain",
    "📈 Backtesting",
])


# ================================================================
# TAB 1 — PREDICT
# ================================================================
with tab_predict:

    year = st.number_input("Season Year", min_value=2018, max_value=2035, value=2026)

    try:
        schedule   = fastf1.get_event_schedule(year)
        schedule   = schedule[schedule["EventFormat"] != "testing"]
        race_names = schedule["EventName"].tolist()
    except Exception as e:
        st.error(f"Could not load {year} schedule: {e}")
        st.stop()

    gp         = st.selectbox("Select Grand Prix", race_names)
    cache_file = os.path.join(CACHE_DIR, f"session_cache_{year}_{gp}.csv")

    cache_status = check_cache(year, gp, cache_dir=CACHE_DIR)
    if cache_status.exists and cache_status.severity != "ok":
        streamlit_banner(cache_status)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("▶ Run Prediction"):
            try:
                if not os.path.exists(cache_file):
                    with st.spinner("Downloading session data from FastF1…"):
                        extract_session_stats(year, gp)
                    st.success("Session data downloaded.")
                else:
                    age_str = (
                        f"{cache_status.age_hours:.1f} h old"
                        if cache_status.age_hours is not None
                        else "age unknown"
                    )
                    st.info(f"Using cached session data ({age_str}).")

                with st.spinner("Running ML model…"):
                    result = build_features(year, gp)

                st.session_state["result"]     = result
                st.session_state["cache_file"] = cache_file
                st.session_state["pred_gp"]    = gp

            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                raise

    with col2:
        if st.button("🔄 Refresh Race Data"):
            try:
                with st.spinner("Re-downloading session data…"):
                    extract_session_stats(year, gp)
                st.session_state.pop("result", None)
                st.success("Race data refreshed — click Run Prediction to update.")
            except Exception as e:
                st.error(f"Refresh failed: {e}")

    if "result" in st.session_state:

        result     = st.session_state["result"]
        saved_file = st.session_state.get("cache_file", cache_file)

        if not os.path.exists(saved_file):
            st.warning("Session cache not found — please run prediction again.")
            st.stop()

        session_df = pd.read_csv(saved_file)

        st.header("🏆 Predicted Winner")
        winner = result.iloc[0]
        st.metric(
            label="Predicted Winner",
            value=winner["Driver"],
            delta=f"{winner['Winning Probability']:.1%} win probability",
        )

        st.header("🥇 Predicted Podium")
        podium = result.head(3).reset_index(drop=True)
        c1, c2, c3 = st.columns(3)
        for col, idx, pos in zip([c1, c2, c3], [0, 1, 2], ["P1", "P2", "P3"]):
            if idx < len(podium):
                row = podium.iloc[idx]
                col.metric(pos, row["Driver"], f"{row['Winning Probability']:.1%}")
            else:
                col.metric(pos, "—", "N/A")

        st.header("📋 All Driver Probabilities")
        display = result.copy()
        display["Winning Probability"] = display["Winning Probability"].map("{:.1%}".format)
        st.dataframe(display, use_container_width=True)

        st.header("📊 Team Win Probability")
        if "Team" in session_df.columns:
            merged = result.merge(
                session_df[["Driver", "Team"]].drop_duplicates(), on="Driver", how="left"
            )
            team_probs = (
                merged.groupby("Team")["Winning Probability"]
                .sum()
                .reset_index()
                .sort_values("Winning Probability", ascending=False)
            )
            team_chart = (
                alt.Chart(team_probs)
                .mark_bar()
                .encode(
                    x=alt.X("Team:N", sort="-y", title="Constructor"),
                    y=alt.Y(
                        "Winning Probability:Q",
                        axis=alt.Axis(format=".0%"),
                        title="Combined Win Probability",
                    ),
                    color=alt.Color("Team:N", legend=None),
                    tooltip=["Team", alt.Tooltip("Winning Probability:Q", format=".1%")],
                )
                .properties(height=300)
            )
            st.altair_chart(team_chart, use_container_width=True)
        else:
            st.info("Team column not available in session cache.")

        st.header("🔥 Driver Win Probability — Top 10")
        top10 = result.head(10).copy()
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(top10["Driver"], top10["Winning Probability"], color="#E10600")
        ax.invert_yaxis()
        ax.set_xlabel("Win Probability")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.bar_label(bars, fmt=lambda v: f"{v:.1%}", padding=4, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.header("🏎️ Pace Comparison")
        pace_cols = ["Driver", "QualiPaceRatio", "FPPaceRatio"]
        available = [c for c in pace_cols if c in session_df.columns]
        if len(available) == 3:
            pace_df   = session_df[available].sort_values("QualiPaceRatio")
            pace_long = pace_df.melt(
                id_vars="Driver",
                value_vars=["QualiPaceRatio", "FPPaceRatio"],
                var_name="Session",
                value_name="Pace Ratio",
            )
            pace_long["Session"] = pace_long["Session"].map(
                {"QualiPaceRatio": "Qualifying", "FPPaceRatio": "Practice"}
            )
            pace_chart = (
                alt.Chart(pace_long)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "Driver:N",
                        sort=alt.EncodingSortField("Pace Ratio", op="min", order="ascending"),
                    ),
                    y=alt.Y("Pace Ratio:Q", scale=alt.Scale(zero=False),
                            title="Pace Ratio (lower = faster)"),
                    color=alt.Color("Session:N",
                                    scale=alt.Scale(range=["#E10600", "#1E3A8A"])),
                    xOffset="Session:N",
                    tooltip=["Driver", "Session", alt.Tooltip("Pace Ratio:Q", format=".4f")],
                )
                .properties(height=320)
            )
            st.altair_chart(pace_chart, use_container_width=True)
            st.caption("Pace ratio = driver lap time ÷ session fastest lap.  1.000 = fastest.")
        else:
            st.info("Pace ratio columns not available in session cache.")

# ================================================================
# TAB 2 - HEAD-TO-HEAD DRIVER COMPARISON
# ================================================================
with tab_h2h:

    st.header("Head-to-Head Driver Comparison")
    st.write("Compare two drivers across qualifying pace, practice pace, sector times, and historical form.")

    if not os.path.exists(TRAIN_DATA_PATH):
        st.error("`f1_features_clean.csv` not found. Run `clean_dataset.py` first.")
    else:
        h2h_df = pd.read_csv(TRAIN_DATA_PATH)
        h2h_df["Year"] = pd.to_numeric(h2h_df["Year"], errors="coerce")
        h2h_df = h2h_df.dropna(subset=["Year"])
        h2h_df["Year"] = h2h_df["Year"].astype(int)

        all_drivers = sorted(h2h_df["Driver"].dropna().unique().tolist())
        hcol1, hcol2 = st.columns(2)
        with hcol1:
            driver_a = st.selectbox("Driver A", all_drivers, index=0, key="h2h_a")
        with hcol2:
            default_b = 1 if len(all_drivers) > 1 else 0
            driver_b = st.selectbox("Driver B", all_drivers, index=default_b, key="h2h_b")

        if driver_a == driver_b:
            st.warning("Select two different drivers.")
        else:
            df_a = h2h_df[h2h_df["Driver"] == driver_a]
            df_b = h2h_df[h2h_df["Driver"] == driver_b]

            st.subheader("Career summary")
            stat_cols = st.columns(4)
            for i, (label, col) in enumerate([
                ("Races", None), ("Wins", "Winner"),
                ("Avg Quali Pace", "QualiPaceRatio"), ("Avg FP Pace", "FPPaceRatio"),
            ]):
                with stat_cols[i]:
                    if col is None:
                        va, vb = len(df_a), len(df_b)
                        fmt = lambda v: str(v)
                    elif col == "Winner":
                        va = int(df_a[col].sum()) if col in df_a.columns else 0
                        vb = int(df_b[col].sum()) if col in df_b.columns else 0
                        fmt = lambda v: str(v)
                    else:
                        va = df_a[col].mean() if col in df_a.columns else float("nan")
                        vb = df_b[col].mean() if col in df_b.columns else float("nan")
                        fmt = lambda v: f"{v:.4f}" if not pd.isna(v) else "--"
                    st.metric(f"{label} -- {driver_a}", fmt(va))
                    st.metric(f"{label} -- {driver_b}", fmt(vb))

            st.subheader("Qualifying vs Practice pace (lower = faster)")
            pace_metrics = ["QualiPaceRatio","FPPaceRatio","Sector1Ratio","Sector2Ratio","Sector3Ratio"]
            available_metrics = [m for m in pace_metrics if m in h2h_df.columns]
            if available_metrics:
                means_a = df_a[available_metrics].mean()
                means_b = df_b[available_metrics].mean()
                compare_df = pd.DataFrame({
                    "Metric": available_metrics * 2,
                    "Driver": [driver_a]*len(available_metrics) + [driver_b]*len(available_metrics),
                    "Value":  means_a.tolist() + means_b.tolist(),
                })
                h2h_chart = (
                    alt.Chart(compare_df).mark_bar()
                    .encode(
                        x=alt.X("Metric:N"),
                        y=alt.Y("Value:Q", scale=alt.Scale(zero=False)),
                        color=alt.Color("Driver:N", scale=alt.Scale(range=["#E10600","#1E3A8A"])),
                        xOffset="Driver:N",
                        tooltip=["Driver","Metric",alt.Tooltip("Value:Q",format=".4f")],
                    ).properties(height=320)
                )
                st.altair_chart(h2h_chart, use_container_width=True)

            st.subheader("Head-to-head at this circuit")
            shared_gps = sorted(set(df_a["GP"].unique()) & set(df_b["GP"].unique()))
            if shared_gps:
                h2h_gp = st.selectbox("Circuit", shared_gps, key="h2h_gp")
                gp_a = df_a[df_a["GP"]==h2h_gp][["Year","QualiPaceRatio","FPPaceRatio","Winner"]].copy()
                gp_b = df_b[df_b["GP"]==h2h_gp][["Year","QualiPaceRatio","FPPaceRatio","Winner"]].copy()
                gp_a["Driver"] = driver_a
                gp_b["Driver"] = driver_b
                gp_both = pd.concat([gp_a, gp_b]).sort_values("Year")
                wins_a = int(gp_a["Winner"].sum()) if "Winner" in gp_a.columns else 0
                wins_b = int(gp_b["Winner"].sum()) if "Winner" in gp_b.columns else 0
                wc1, wc2 = st.columns(2)
                wc1.metric(f"{driver_a} wins at {h2h_gp}", wins_a)
                wc2.metric(f"{driver_b} wins at {h2h_gp}", wins_b)
                if not gp_both.empty:
                    gp_long = gp_both.melt(
                        id_vars=["Year","Driver"],
                        value_vars=["QualiPaceRatio","FPPaceRatio"],
                        var_name="Session", value_name="Pace Ratio",
                    )
                    gp_long["Session"] = gp_long["Session"].map({"QualiPaceRatio":"Qualifying","FPPaceRatio":"Practice"})
                    circuit_chart = (
                        alt.Chart(gp_long).mark_line(point=True)
                        .encode(
                            x=alt.X("Year:O"),
                            y=alt.Y("Pace Ratio:Q", scale=alt.Scale(zero=False)),
                            color=alt.Color("Driver:N", scale=alt.Scale(range=["#E10600","#1E3A8A"])),
                            strokeDash=alt.StrokeDash("Session:N"),
                            tooltip=["Year:O","Driver","Session",alt.Tooltip("Pace Ratio:Q",format=".4f")],
                        ).properties(height=280)
                    )
                    st.altair_chart(circuit_chart, use_container_width=True)
            else:
                st.info("No shared circuits found between these two drivers in the dataset.")


# ================================================================
# TAB 3 - CIRCUIT HISTORY HEATMAP
# ================================================================
with tab_circuit:

    st.header("Circuit-Specific Historical Performance")
    st.write("Heatmap of each driver's win rate and qualifying pace at every circuit.")

    if not os.path.exists(TRAIN_DATA_PATH):
        st.error("`f1_features_clean.csv` not found. Run `clean_dataset.py` first.")
    else:
        circ_df = pd.read_csv(TRAIN_DATA_PATH)
        circ_df["Year"] = pd.to_numeric(circ_df["Year"], errors="coerce")
        circ_df = circ_df.dropna(subset=["Year"])
        circ_df["Year"] = circ_df["Year"].astype(int)

        ch_col1, ch_col2 = st.columns(2)
        with ch_col1:
            circ_years = sorted(circ_df["Year"].unique())
            selected_circ_years = st.multiselect("Filter seasons", circ_years, default=circ_years, key="circ_years")
        with ch_col2:
            all_gps = sorted(circ_df["GP"].dropna().unique())
            selected_gp_filter = st.multiselect("Filter circuits", all_gps, default=all_gps, key="circ_gps")

        if not selected_circ_years:
            selected_circ_years = circ_years
        if not selected_gp_filter:
            selected_gp_filter = all_gps

        circ_view = circ_df[circ_df["Year"].isin(selected_circ_years) & circ_df["GP"].isin(selected_gp_filter)]

        if circ_view.empty:
            st.warning("No data for the selected filters.")
        else:
            win_rate = (
                circ_view.groupby(["Driver","GP"])["Winner"]
                .agg(["sum","count"]).reset_index()
                .rename(columns={"sum":"Wins","count":"Races"})
            )
            win_rate["WinRate"] = win_rate["Wins"] / win_rate["Races"]
            driver_race_counts = circ_view.groupby("Driver")["GP"].count()
            active_drivers = driver_race_counts[driver_race_counts >= 2].index.tolist()
            win_rate = win_rate[win_rate["Driver"].isin(active_drivers)]

            if win_rate.empty:
                st.warning("Not enough data to build heatmap with current filters.")
            else:
                heatmap = (
                    alt.Chart(win_rate).mark_rect()
                    .encode(
                        x=alt.X("GP:N", sort=sorted(win_rate["GP"].unique())),
                        y=alt.Y("Driver:N", sort=sorted(win_rate["Driver"].unique())),
                        color=alt.Color("WinRate:Q", scale=alt.Scale(scheme="reds", domain=[0,1])),
                        tooltip=["Driver","GP",alt.Tooltip("WinRate:Q",format=".1%",title="Win rate"),alt.Tooltip("Wins:Q"),alt.Tooltip("Races:Q")],
                    ).properties(height=max(300, len(active_drivers)*22), title="Driver win rate by circuit")
                )
                st.altair_chart(heatmap, use_container_width=True)
                st.caption("Win rate = wins / races at that circuit in the selected seasons.")

                st.subheader("Average qualifying pace ratio by circuit (lower = faster)")
                pace_heat = (
                    circ_view.groupby(["Driver","GP"])["QualiPaceRatio"].mean()
                    .reset_index().rename(columns={"QualiPaceRatio":"AvgQualiPace"})
                )
                pace_heat = pace_heat[pace_heat["Driver"].isin(active_drivers)]
                if not pace_heat.empty:
                    pace_heatmap = (
                        alt.Chart(pace_heat).mark_rect()
                        .encode(
                            x=alt.X("GP:N", sort=sorted(pace_heat["GP"].unique())),
                            y=alt.Y("Driver:N", sort=sorted(pace_heat["Driver"].unique())),
                            color=alt.Color("AvgQualiPace:Q", scale=alt.Scale(scheme="redyellowgreen", reverse=True)),
                            tooltip=["Driver","GP",alt.Tooltip("AvgQualiPace:Q",format=".4f")],
                        ).properties(height=max(300, len(active_drivers)*22))
                    )
                    st.altair_chart(pace_heatmap, use_container_width=True)


# ================================================================
# TAB 4 - WHAT-IF SIMULATOR
# ================================================================
with tab_whatif:

    st.header("What-If Grid Simulator")
    st.write("Adjust grid positions and see how win probabilities shift.")

    if "result" not in st.session_state:
        st.info("Run a prediction in the Predict tab first, then come back here.")
    else:
        wi_result     = st.session_state["result"]
        wi_cache_file = st.session_state.get("cache_file", "")
        wi_gp         = st.session_state.get("pred_gp", "")

        if not os.path.exists(wi_cache_file):
            st.warning("Session cache not found -- run a prediction first.")
        else:
            wi_session = pd.read_csv(wi_cache_file)

            st.subheader("Edit grid positions")
            st.caption("Change any driver's starting position. Positions don't need to be unique.")

            wi_grid = wi_session[["Driver","GridPosition"]].copy()
            wi_grid["GridPosition"] = wi_grid["GridPosition"].fillna(20).astype(int)

            edited_positions = {}
            n_cols = 4
            driver_chunks = [wi_grid.iloc[i:i+n_cols] for i in range(0, len(wi_grid), n_cols)]
            for chunk in driver_chunks:
                cols = st.columns(n_cols)
                for col_widget, (_, row) in zip(cols, chunk.iterrows()):
                    edited_positions[row["Driver"]] = col_widget.number_input(
                        row["Driver"], min_value=1, max_value=20,
                        value=int(row["GridPosition"]), key=f"wi_{row['Driver']}",
                    )

            if st.button("Run What-If Prediction"):
                try:
                    wi_patched = wi_session.copy()
                    wi_patched["GridPosition"] = wi_patched["Driver"].map(edited_positions).fillna(20).astype(int)

                    with st.spinner("Running what-if model..."):
                        wi_X = wi_patched.drop(columns=["Driver"]).copy()
                        wi_X = _derive_features(wi_X, wi_gp)
                        for col in feature_columns:
                            if col not in wi_X.columns:
                                wi_X[col] = feature_means.get(col, 0.0)
                        wi_X     = wi_X[feature_columns]
                        wi_probs = winner_model.predict_proba(wi_X)[:, 1]
                        wi_total = wi_probs.sum()
                        wi_probs = wi_probs / wi_total if wi_total > 0 else np.ones(len(wi_probs)) / len(wi_probs)

                    wi_what_result = (
                        pd.DataFrame({"Driver": wi_patched["Driver"].tolist(), "What-If Probability": wi_probs})
                        .sort_values("What-If Probability", ascending=False).reset_index(drop=True)
                    )
                    wi_compare = wi_what_result.merge(
                        wi_result.rename(columns={"Winning Probability": "Original Probability"}),
                        on="Driver", how="left",
                    )
                    wi_compare["Delta"] = wi_compare["What-If Probability"] - wi_compare["Original Probability"]

                    st.subheader("What-If vs Original probabilities")
                    wi_display = wi_compare.copy()
                    wi_display["What-If Probability"]  = wi_display["What-If Probability"].map("{:.1%}".format)
                    wi_display["Original Probability"] = wi_display["Original Probability"].map("{:.1%}".format)
                    wi_display["Delta"] = wi_display["Delta"].map(lambda v: f"+{v:.1%}" if v >= 0 else f"{v:.1%}")
                    st.dataframe(wi_display, use_container_width=True)

                except Exception as e:
                    st.error(f"What-if simulation failed: {e}")
                    raise


# ================================================================
# TAB 5 - RETRAIN PIPELINE
# ================================================================
with tab_retrain:

    st.header("Model Retrain Pipeline")
    st.write("Retrain the model. The new model is only saved if it beats the current one.")

    if os.path.exists(META_FILE_PATH):
        with open(META_FILE_PATH) as fh:
            rt_meta = json.load(fh)
        st.subheader("Current model")
        rm1, rm2, rm3, rm4 = st.columns(4)
        rm1.metric("Model",        rt_meta.get("model", "--"))
        rm2.metric("Test-year F1", f"{rt_meta.get('f1_score', 0):.3f}")
        rm3.metric("ROC-AUC",      f"{rt_meta.get('roc_auc', 0):.3f}")
        rm4.metric("Trained on",   f"{rt_meta.get('n_train', '--')} rows")
        st.caption(f"Trained at: {rt_meta.get('trained_at','--')}  |  Test year: {rt_meta.get('test_year','--')}  |  Best CV F1: {rt_meta.get('best_cv_f1','--')}")
    else:
        st.warning("No model metadata found.")
        rt_meta = {}

    st.divider()
    st.subheader("Data status")
    if os.path.exists(MASTER_DATA_PATH):
        rt_df = pd.read_csv(MASTER_DATA_PATH)
        rt_df["Year"] = pd.to_numeric(rt_df["Year"], errors="coerce")
        rt_df = rt_df.dropna(subset=["Year"])
        rt_df["Year"] = rt_df["Year"].astype(int)
        dc1, dc2, dc3 = st.columns(3)
        dc1.metric("Total rows",    len(rt_df))
        dc2.metric("Seasons",       len(rt_df["Year"].unique()))
        dc3.metric("Latest season", int(rt_df["Year"].max()))
        st.caption(f"Seasons in master data: {sorted(rt_df['Year'].unique())}")
    else:
        st.error(f"`{MASTER_DATA_PATH}` not found.")

    st.divider()
    st.subheader("Retrain")
    st.info("Runs feature_engineering.py -> clean_dataset.py -> train_model.py in sequence.")

    force_retrain = st.checkbox("Force retrain even if new model is not better", value=False)

    if st.button("Run Retrain Pipeline"):
        import subprocess
        steps = [
            ("Feature engineering", ["python", os.path.join(_ROOT, "core",          "feature_engineering.py")]),
            ("Dataset cleaning",    ["python", os.path.join(_ROOT, "data_pipeline", "clean_dataset.py")]),
            ("Model training",      ["python", os.path.join(_ROOT, "training",      "train_model.py")]),
        ]
        all_ok = True
        for step_name, cmd in steps:
            with st.spinner(f"{step_name}..."):
                try:
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                    if proc.returncode != 0:
                        st.error(f"{step_name} failed:\n```\n{proc.stderr[-2000:]}\n```")
                        all_ok = False
                        break
                    else:
                        st.success(f"{step_name} complete.")
                except subprocess.TimeoutExpired:
                    st.error(f"{step_name} timed out after 10 minutes.")
                    all_ok = False
                    break
                except Exception as e:
                    st.error(f"{step_name} error: {e}")
                    all_ok = False
                    break

        if all_ok and os.path.exists(META_FILE_PATH):
            with open(META_FILE_PATH) as fh:
                new_meta = json.load(fh)
            new_f1  = new_meta.get("f1_score", 0)
            curr_f1 = rt_meta.get("f1_score", 0)
            if force_retrain or new_f1 >= curr_f1:
                st.success(f"New model saved -- F1: {new_f1:.3f} (was {curr_f1:.3f}). Reload the app to use it.")
            else:
                st.warning(f"New model F1 ({new_f1:.3f}) did not beat current ({curr_f1:.3f}). Previous model kept.")


# ================================================================
# TAB 6 - BACKTESTING
# ================================================================
with tab_backtest:

    st.header("Historical Prediction Accuracy")
    st.write("Replay the model against every race in the training dataset.")

    if not os.path.exists(META_FILE_PATH):
        meta = {}
    else:
        with open(META_FILE_PATH) as fh:
            meta = json.load(fh)
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Model",       meta.get("model", "--"))
        mc2.metric("Test year F1",f"{meta.get('f1_score', 0):.3f}")
        mc3.metric("ROC-AUC",     f"{meta.get('roc_auc', 0):.3f}")
        mc4.metric("Trained on",  f"{meta.get('n_train', '--')} rows")
        st.caption(f"Trained at: {meta.get('trained_at','--')}  |  Test year: {meta.get('test_year','--')}")
        st.divider()

    if not os.path.exists(TRAIN_DATA_PATH):
        st.error(f"`{TRAIN_DATA_PATH}` not found. Run `clean_dataset.py` first.")
        st.stop()

    @st.cache_data(show_spinner="Replaying model against historical races...")
    def run_backtest(data_path: str) -> pd.DataFrame:
        df = pd.read_csv(data_path)
        df["Year"]   = pd.to_numeric(df["Year"],   errors="coerce")
        df           = df.dropna(subset=["Year"])
        df["Year"]   = df["Year"].astype(int)
        df["Winner"] = pd.to_numeric(df["Winner"], errors="coerce").fillna(0).astype(int)
        records = []
        for (year, gp), race_df in df.groupby(["Year","GP"]):
            race_df = race_df.copy().reset_index(drop=True)
            if race_df["Winner"].sum() == 0:
                continue
            actual_winner_rows = race_df[race_df["Winner"] == 1]
            if actual_winner_rows.empty:
                continue
            actual_winner = actual_winner_rows.iloc[0].get("Driver", "Unknown")
            if "Driver" in race_df.columns:
                drivers = race_df["Driver"].tolist()
                X = race_df.drop(columns=["Driver","Winner"], errors="ignore").copy()
            else:
                drivers = [f"Driver_{i}" for i in range(len(race_df))]
                X = race_df.drop(columns=["Winner"], errors="ignore").copy()
            try:
                X = _derive_features(X, str(gp))
            except Exception:
                continue
            for col in feature_columns:
                if col not in X.columns:
                    X[col] = feature_means.get(col, 0.0)
            X = X[feature_columns]
            try:
                probs = winner_model.predict_proba(X)[:, 1]
            except Exception:
                continue
            total = probs.sum()
            probs = probs / total if total > 0 else np.ones(len(probs)) / len(probs)
            ranked = (pd.DataFrame({"Driver": drivers, "Prob": probs}).sort_values("Prob", ascending=False).reset_index(drop=True))
            predicted_winner = ranked.iloc[0]["Driver"]
            top3_drivers     = set(ranked.head(3)["Driver"])
            top5_drivers     = set(ranked.head(5)["Driver"])
            winner_row  = ranked[ranked["Driver"] == actual_winner]
            winner_rank = int(winner_row.index[0]) + 1 if not winner_row.empty else len(ranked) + 1
            winner_prob = float(winner_row["Prob"].values[0]) if not winner_row.empty else 0.0
            records.append({"Year": int(year), "GP": str(gp), "ActualWinner": actual_winner, "PredictedWinner": predicted_winner, "ActualWinnerRank": winner_rank, "Top3Hit": actual_winner in top3_drivers, "Top5Hit": actual_winner in top5_drivers, "CorrectPick": actual_winner == predicted_winner, "WinnerProbability": winner_prob, "TopProbability": float(ranked.iloc[0]["Prob"])})
        return pd.DataFrame(records)

    bt = run_backtest(TRAIN_DATA_PATH)

    if bt.empty:
        st.warning("No backtest results.")
        st.stop()

    all_years   = sorted(bt["Year"].unique())
    test_year   = meta.get("test_year") if meta else None
    year_labels = {y: f"{y}  * out-of-sample" if y == test_year else str(y) for y in all_years}

    selected_years = st.multiselect("Filter by season", options=all_years, default=all_years, format_func=lambda y: year_labels[y])
    if not selected_years:
        selected_years = all_years

    view = bt[bt["Year"].isin(selected_years)]

    st.subheader("Overall accuracy")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Races analysed", len(view))
    m2.metric("Top-1 accuracy", f"{view['CorrectPick'].mean():.1%}")
    m3.metric("Top-3 accuracy", f"{view['Top3Hit'].mean():.1%}")
    m4.metric("Avg winner rank", f"{view['ActualWinnerRank'].mean():.1f}")
    st.divider()

    st.subheader("Accuracy by season")
    yearly = view.groupby("Year").agg(Top1=("CorrectPick","mean"),Top3=("Top3Hit","mean"),Top5=("Top5Hit","mean"),Races=("GP","count")).reset_index()
    yearly_long = yearly.melt(id_vars="Year", value_vars=["Top1","Top3","Top5"], var_name="Metric", value_name="Accuracy")
    yearly_long["Label"] = yearly_long["Metric"].map({"Top1":"Top-1","Top3":"Top-3","Top5":"Top-5"})
    year_line = (alt.Chart(yearly_long).mark_line(point=True, strokeWidth=2).encode(x=alt.X("Year:O"),y=alt.Y("Accuracy:Q",axis=alt.Axis(format=".0%"),scale=alt.Scale(domain=[0,1])),color=alt.Color("Label:N",scale=alt.Scale(domain=["Top-1","Top-3","Top-5"],range=["#E10600","#FF8C00","#1E3A8A"])),tooltip=["Year:O","Label:N",alt.Tooltip("Accuracy:Q",format=".1%")]).properties(height=300))
    if test_year and test_year in [int(y) for y in selected_years]:
        rule = alt.Chart(pd.DataFrame({"Year":[str(test_year)]})).mark_rule(strokeDash=[4,4],color="gray",opacity=0.7).encode(x="Year:O")
        year_line = year_line + rule
    st.altair_chart(year_line, use_container_width=True)

    st.subheader("Where was the actual winner ranked?")
    rank_counts = (view["ActualWinnerRank"].clip(upper=10).value_counts().reset_index().rename(columns={"ActualWinnerRank":"Rank","count":"Races"}).sort_values("Rank"))
    rank_counts["RankLabel"] = rank_counts["Rank"].apply(lambda r: str(int(r)) if r < 10 else "10+")
    rank_chart = (alt.Chart(rank_counts).mark_bar().encode(x=alt.X("RankLabel:N",sort=None),y=alt.Y("Races:Q"),color=alt.condition(alt.datum.Rank==1,alt.value("#E10600"),alt.value("#AAAAAA")),tooltip=[alt.Tooltip("RankLabel:N",title="Rank"),alt.Tooltip("Races:Q",title="Races")]).properties(height=260))
    st.altair_chart(rank_chart, use_container_width=True)

    st.subheader("Race-by-race results")
    display_bt = view[["Year","GP","ActualWinner","PredictedWinner","ActualWinnerRank","WinnerProbability","Top3Hit","Top5Hit"]].copy()
    display_bt["CorrectPick"]       = (display_bt["ActualWinner"] == display_bt["PredictedWinner"]).map({True:"v",False:"x"})
    display_bt["WinnerProbability"] = display_bt["WinnerProbability"].map("{:.1%}".format)
    display_bt["Top3Hit"]           = display_bt["Top3Hit"].map({True:"v",False:"x"})
    display_bt["Top5Hit"]           = display_bt["Top5Hit"].map({True:"v",False:"x"})
    display_bt = display_bt.rename(columns={"ActualWinner":"Actual winner","PredictedWinner":"Model pick","ActualWinnerRank":"Winner rank","WinnerProbability":"Prob given to winner","CorrectPick":"Correct?","Top3Hit":"Top-3?","Top5Hit":"Top-5?"})
    st.dataframe(display_bt.sort_values(["Year","GP"]).reset_index(drop=True), use_container_width=True, height=420)

    csv_data = view.to_csv(index=False).encode("utf-8")
    st.download_button(label="Download full backtest results as CSV", data=csv_data, file_name="f1_backtest_results.csv", mime="text/csv")
