# app.py — Streamlit dashboard for Cost-Effective, Failure-Aware Edge Caching
# - Forecasts demand with your trained GRU (models/pop_gru.pt)
# - Planner supports two objectives:
#     1) Heuristic (GDSF-style popularity/size with failure penalty)
#     2) Cost Savings ($): minimizes expected cost with per-edge failure rate
# - Vectorized math throughout (fast & robust)

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import torch

# --- make src/ importable when running from project root ---
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.append(str(SRC))   # so "models.pop_gru" resolves (file lives in src/models/pop_gru.py)

# project model (provided by your repo)
from models.pop_gru import PopGRU  # type: ignore


# ---------------------------- CACHED LOADERS ----------------------------

@st.cache_resource(show_spinner=False)
def load_model(model_path: str, device: str = "cpu") -> torch.nn.Module:
    model = PopGRU().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_data(show_spinner=True)
def load_ts(ts_path: str) -> pd.DataFrame:
    """
    Expected columns: region, video_id, date, reqs
    """
    df = pd.read_parquet(ts_path)
    need = {"region", "video_id", "date", "reqs"}
    if not need.issubset(df.columns):
        raise ValueError(f"ts.parquet must contain columns {sorted(need)}; found {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=True)
def load_failures(fail_path: str) -> pd.DataFrame:
    """
    Expected columns (min): time, edge_id, failed
    """
    df = pd.read_parquet(fail_path)
    if "time" not in df.columns and "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "time"})
    if "time" not in df.columns:
        raise ValueError("failures.parquet must contain a 'time' column (or 'timestamp').")
    if "edge_id" not in df.columns:
        df["edge_id"] = "edge-0"  # fallback
    if "failed" not in df.columns:
        # derive if missing: use any column containing 'fail'
        fail_cols = [c for c in df.columns if "fail" in c.lower()]
        if fail_cols:
            df["failed"] = (df[fail_cols[0]] > 0).astype(int)
        else:
            df["failed"] = 0
    df["time"] = pd.to_datetime(df["time"])
    return df


# ---------------------------- HELPERS ----------------------------

def series_for(df: pd.DataFrame, region: str, vid: str) -> Tuple[pd.DataFrame, np.ndarray]:
    """Return daily series (np.array) for (region, video_id) with no gaps."""
    sdf = df[(df["region"] == region) & (df["video_id"] == vid)]
    if sdf.empty:
        return sdf, np.array([], dtype=np.float32)
    s = (sdf[["date", "reqs"]]
         .set_index("date")
         .asfreq("D")
         .fillna(0.0)
         .sort_index())
    return s, s["reqs"].to_numpy(dtype=np.float32)


def forecast_series(model: torch.nn.Module, arr: np.ndarray, lookback: int, horizon: int, device: str) -> np.ndarray:
    """Iterative one-step forecasting for 'horizon' days."""
    if len(arr) < max(1, lookback):
        return np.zeros(horizon, dtype=np.float32)
    hist = arr.astype(np.float32).copy()
    preds = []
    for _ in range(horizon):
        x = torch.tensor(hist[-lookback:], dtype=torch.float32).view(1, lookback, 1).to(device)
        with torch.no_grad():
            y = float(model(x).item())
        y = max(0.0, y)  # clamp negatives
        preds.append(y)
        hist = np.append(hist, y)
    return np.array(preds, dtype=np.float32)


def compute_fail_risk(fail_df: pd.DataFrame, when: pd.Timestamp, edge_id: str | None = None, hours: int = 24) -> float:
    """Recent failure rate (mean of 'failed') in a time window; optionally per-edge."""
    df = fail_df
    if edge_id is not None:
        df = df[df["edge_id"] == edge_id]
    win = df[(df["time"] > (when - pd.Timedelta(hours=hours))) & (df["time"] <= when)]
    if win.empty:
        return 0.0
    return float(win["failed"].mean())


# ----- Heuristic objective (GDSF-style) -----

def gdsf_score(demand, size, fail_risk, alpha=1.0, beta=0.6, gamma=1.0, lam=0.0, age=None):
    """
    Vectorized GDSF-style score.
    demand, size: 1D arrays (same length)
    fail_risk, age: scalar or arrays (broadcastable)
    """
    d = np.asarray(demand, dtype=float)
    s = np.maximum(1.0, np.asarray(size, dtype=float))  # avoid div-by-zero
    fr = np.asarray(fail_risk, dtype=float)
    if fr.ndim == 0:  # broadcast scalar
        fr = np.full_like(d, fr)
    if age is None:
        age = 0.0
    age_arr = np.asarray(age, dtype=float)
    if age_arr.ndim == 0:
        age_arr = np.zeros_like(d) + age_arr
    return (alpha * d) / (s ** beta) - (gamma * fr) + lam * age_arr


# ----- Cost objective (expected $ savings) -----

def expected_cost_no_cache(demand, size_mb, C_origin):
    d = np.asarray(demand, float); s = np.asarray(size_mb, float)
    return d * s * C_origin

def expected_cost_with_cache_single(demand, size_mb, C_origin, C_store, C_xfer, p_fail_e):
    d = np.asarray(demand, float); s = np.asarray(size_mb, float)
    avail = 1.0 - np.asarray(p_fail_e, float)  # scalar or vector
    storage = s * C_store                # $ to store for the horizon
    fill    = s * C_xfer                 # $ to prefetch/fill
    residual_miss = d * s * C_origin * (1.0 - avail)  # still miss when edge is down
    return storage + fill + residual_miss

def savings_single_edge(demand, size_mb, p_fail_e, C_origin=0.08, C_store=0.0002, C_xfer=0.0):
    """Vectorized expected $ savings for placing item on THIS edge."""
    return expected_cost_no_cache(demand, size_mb, C_origin) - \
           expected_cost_with_cache_single(demand, size_mb, C_origin, C_store, C_xfer, p_fail_e)


# ---------------------------- UI ----------------------------

st.set_page_config(page_title="Edge Cache Intelligence", layout="wide")
st.title("📦 Edge Caching — Popularity Forecast + Failure-Aware, Cost-Effective Planning")

with st.sidebar:
    st.header("Paths")
    ts_path = st.text_input("Time-series parquet", value="data/processed/ts.parquet")
    fail_path = st.text_input("Failures parquet", value="data/processed/failures.parquet")
    model_path = st.text_input("Model checkpoint", value="models/pop_gru.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.caption(f"Device: **{device}**")

# Load data/model
ts_df = load_ts(ts_path)
fail_df = load_failures(fail_path)
model = load_model(model_path, device=device)

# Regions & video selection
regions = sorted(ts_df["region"].unique().tolist())
colA, colB, colC = st.columns([1.2, 1.6, 1.2])
with colA:
    region = st.selectbox("Region", regions, index=0)
with colB:
    # Top-K list for responsiveness
    topK = st.slider("Choose from top-K videos by demand", 100, 5000, 1000, step=100)
    top_vids = (
        ts_df[ts_df["region"] == region]
        .groupby("video_id")["reqs"]
        .sum()
        .sort_values(ascending=False)
        .head(topK)
        .index.tolist()
    )
    video_id = st.selectbox("Video ID", top_vids)
with colC:
    lookback = st.slider("Lookback (days)", 7, 60, 14)
    horizon = st.slider("Forecast Horizon (days)", 1, 30, 7)

# Tabs
t1, t2 = st.tabs(["🔮 Forecast", "🧠 Cache Planner"])

# ---------------------------- Forecast Tab ----------------------------
with t1:
    st.subheader("Demand Forecast for Selected Video")
    s, arr = series_for(ts_df, region, video_id)
    if len(arr) == 0:
        st.warning("No data for that selection.")
    else:
        preds = forecast_series(model, arr, lookback=lookback, horizon=horizon, device=device)
        future_idx = pd.date_range(s.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
        hist_df = s.rename(columns={"reqs": "value"}).reset_index()
        hist_df["type"] = "history"
        fut_df = pd.DataFrame({"date": future_idx, "value": preds, "type": "forecast"})
        chart_df = pd.concat([hist_df, fut_df], ignore_index=True)

        st.line_chart(chart_df.set_index("date")["value"], height=260)
        m1, m2, m3 = st.columns(3)
        m1.metric("Last day demand", f"{hist_df['value'].iloc[-1]:,.0f}")
        m2.metric(f"Forecast next {horizon}d (sum)", f"{preds.sum():,.0f}")
        m3.metric("Avg forecast/day", f"{preds.mean():,.2f}")
        st.caption("Tip: increase lookback if the series is smoother, decrease for faster changes.")

# ---------------------------- Cache Planner Tab ----------------------------
with t2:
    st.subheader("Failure-Aware Cache Planning (What-If)")

    # Edge selection for risk (per-edge planning)
    edges = sorted(fail_df["edge_id"].unique().tolist())
    edge_id = st.selectbox("Edge to plan for", edges, index=0)
    risk_hours = st.slider("Failure-rate lookback (hours)", 6, 72, 24, step=6)

    # Objective toggle
    objective = st.radio(
        "Planning objective",
        ["Cost Savings ($)", "Heuristic (GDSF)"],
        index=0,
        horizontal=True
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        capacity = st.number_input("Cache capacity (items)", min_value=50, max_value=10000, value=2000, step=50)

    # Parameters for each objective
    if objective == "Heuristic (GDSF)":
        with col2:
            alpha = st.number_input("alpha (demand weight)", value=1.5, step=0.1, format="%.1f")
            beta  = st.number_input("beta (size exponent)", value=0.6, step=0.1, format="%.1f")
        with col3:
            gamma = st.number_input("gamma (failure penalty)", value=0.8, step=0.1, format="%.1f")
            assume_size = st.number_input("Default item size (MB) if unknown", value=20)
    else:
        with col2:
            C_origin = st.number_input("Backhaul cost per MB (C_origin)", value=0.08, step=0.01, format="%.2f")
            C_store  = st.number_input("Storage cost per MB / horizon (C_store)", value=0.0002, step=0.0001, format="%.4f")
        with col3:
            C_xfer   = st.number_input("Prefetch cost per MB (C_xfer)", value=0.00, step=0.01, format="%.2f")
            assume_size = st.number_input("Default item size (MB) if unknown", value=20)

    # Optional sizes mapping
    sizes_upload = st.file_uploader("Optional: upload sizes CSV (columns: video_id,size_mb)", type=["csv"])
    sizes_map: Dict[str, float] = {}
    if sizes_upload is not None:
        s_df = pd.read_csv(sizes_upload)
        if {"video_id", "size_mb"}.issubset(s_df.columns):
            sizes_map = dict(zip(s_df["video_id"], s_df["size_mb"]))
        else:
            st.warning("Sizes CSV must have columns: video_id,size_mb")

    # Planning date = last date in data
    plan_date = ts_df["date"].max()
    fail_risk = compute_fail_risk(fail_df, plan_date, edge_id=edge_id, hours=risk_hours)
    st.caption(
        f"Planning date: **{plan_date.date()}**, edge **{edge_id}** failure rate (last {risk_hours}h): **{fail_risk:.3f}**"
    )

    # Candidate set = top-N by recent demand (last 30d) in chosen region
    candN = st.slider("Number of candidates to score", 500, 5000, 2000, step=100)
    recent_cut = plan_date - pd.Timedelta(days=30)
    region_df = ts_df[(ts_df["region"] == region) & (ts_df["date"] >= recent_cut)]
    cand_vids = (
        region_df.groupby("video_id")["reqs"].sum()
        .sort_values(ascending=False)
        .head(candN)
        .index.tolist()
    )

    with st.spinner("Scoring candidates…"):
        preds, sizes = [], []
        for vid in cand_vids:
            s, arr = series_for(ts_df, region, vid)
            if len(arr) < lookback:
                preds.append(0.0)
                sizes.append(assume_size)
                continue
            y1 = forecast_series(model, arr, lookback=lookback, horizon=1, device=device)[0]
            preds.append(float(y1))
            sizes.append(float(sizes_map.get(vid, assume_size)))

        preds = np.array(preds, dtype=float)
        sizes = np.array(sizes, dtype=float)

        if objective == "Heuristic (GDSF)":
            scores = gdsf_score(preds, sizes, fail_risk, alpha=alpha, beta=beta, gamma=gamma)
            order = np.argsort(-scores)
            chosen_idx = order[: int(capacity)]
            plan_df = pd.DataFrame({
                "video_id": [cand_vids[i] for i in order],
                "predicted_demand_next_day": preds[order],
                "size_mb": sizes[order],
                "score": scores[order],
            }).reset_index(drop=True)
            sel_df = plan_df.head(int(capacity)).reset_index(drop=True)
        else:
            # Cost-Savings objective
            scores = savings_single_edge(
                preds, sizes, p_fail_e=fail_risk,
                C_origin=C_origin, C_store=C_store, C_xfer=C_xfer
            )
            order = np.argsort(-scores)
            chosen_idx = order[: int(capacity)]
            plan_df = pd.DataFrame({
                "video_id": [cand_vids[i] for i in order],
                "predicted_demand_next_day": preds[order],
                "size_mb": sizes[order],
                "p_fail_edge": fail_risk if np.isscalar(fail_risk) else np.asarray(fail_risk)[order],
                "expected_savings_$": scores[order],
            }).reset_index(drop=True)
            sel_df = plan_df.head(int(capacity)).reset_index(drop=True)

    st.markdown(f"**Selected {len(sel_df)} items** for cache in **{region}** (top by { 'score' if objective=='Heuristic (GDSF)' else 'expected_savings_$' }):")
    st.dataframe(sel_df.head(50), use_container_width=True)

    # Metrics
    served = preds[chosen_idx].sum()
    total_pred = preds.sum() + 1e-9
    m1, m2 = st.columns(2)
    m1.metric("Expected served fraction (approx.)", f"{served / total_pred:.2%}")
    if objective == "Cost Savings ($)":
        total_savings = float(scores[chosen_idx].sum())
        m2.metric("Expected savings ($, next 1 day)", f"{total_savings:,.2f}")
    else:
        m2.metric("Avg score (selected)", f"{scores[chosen_idx].mean():.3f}")

    st.download_button(
        "Download cache plan CSV",
        data=sel_df.to_csv(index=False).encode("utf-8"),
        file_name=f"cache_plan_{region}_{edge_id}_{plan_date.date()}.csv",
        mime="text/csv",
    )
