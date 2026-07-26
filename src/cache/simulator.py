import argparse, pandas as pd, numpy as np, torch, os
from datetime import timedelta

# --- make "src" importable when running as a script ---
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))                   # .../src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))  # project root

from models.pop_gru import PopGRU
from cache.policy_heuristic import HeuristicPolicy


def load_ts(ts_path: str) -> pd.DataFrame:
    ts = pd.read_parquet(ts_path)
    # Ensure datetime and expected columns
    ts['date'] = pd.to_datetime(ts['date'])
    ts = ts[['region', 'video_id', 'date', 'reqs']].copy()
    return ts


def make_sequences(ts: pd.DataFrame, lookback: int = 14):
    """
    Build per-(region, video) sequences with daily cadence.
    Returns dict[(region, vid)] -> (dates_array, values_array)
    """
    ts = ts.sort_values(['region', 'video_id', 'date'])
    seqs = {}
    for (r, vid), df in ts.groupby(['region', 'video_id']):
        s = df[['date', 'reqs']].set_index('date').asfreq('D').fillna(0.0)
        dates = s.index.to_pydatetime()
        vals = s['reqs'].to_numpy(dtype=np.float32)
        if len(vals) >= 1:
            seqs[(r, vid)] = (dates, vals)
    return seqs


def predict_next(model: torch.nn.Module, device: str, series: np.ndarray, lookback: int) -> float:
    if len(series) < lookback:
        return 0.0
    x = torch.tensor(series[-lookback:], dtype=torch.float32).view(1, lookback, 1).to(device)
    with torch.no_grad():
        y = model(x).item()
    return max(0.0, float(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ts_path', type=str, required=True)
    ap.add_argument('--fail_path', type=str, required=True)
    ap.add_argument('--model_path', type=str, default='models/pop_gru.pt')
    ap.add_argument('--policy', type=str, default='heuristic')
    ap.add_argument('--capacity', type=int, default=500)
    ap.add_argument('--hours', type=int, default=48)  # treated as days here
    ap.add_argument('--lookback', type=int, default=14)
    args = ap.parse_args()

    ts = load_ts(args.ts_path)
    seqs = make_sequences(ts, lookback=args.lookback)
    if ts.empty:
        raise SystemExit("Empty time-series. Did preprocessing produce data?")

    start = ts['date'].min()
    end = start + pd.Timedelta(days=args.hours)

    # failure data (hourly)
    fail = pd.read_parquet(args.fail_path)
    fail['time'] = pd.to_datetime(fail['time'])
    fail = fail.sort_values('time')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PopGRU().to(device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    policy = HeuristicPolicy(capacity_items=args.capacity)

    hit = 0
    miss = 0
    steps = pd.date_range(start=start, end=end, freq='D')

    # Assign random sizes (MB) for each content id
    rng = np.random.default_rng(0)
    sizes = {v: float(rng.integers(1, 50)) for v in ts['video_id'].unique().tolist()}

    print(f"Simulating: {len(steps)} days")
    for t in steps:
        # --- Predict demand per video across regions up to day t ---
        demand_pred = {}
        cutoff64 = np.datetime64(t.to_pydatetime())
        for (r, vid), (dates, vals) in seqs.items():
            arr_dates = np.array(dates, dtype='datetime64[ns]')
            mask = arr_dates <= cutoff64
            count = int(mask.sum())
            series = vals[:count] if count > 0 else np.array([], dtype=np.float32)
            yhat = predict_next(model, device, series, args.lookback)
            demand_pred[vid] = demand_pred.get(vid, 0.0) + yhat

        # --- Failure risk: average "failed" in last 24 hours ---
        win = fail[(fail['time'] > (t - pd.Timedelta(hours=24))) & (fail['time'] <= t)]
        fail_risk = float(win['failed'].mean()) if len(win) else 0.0

        # --- Pick candidates and update cache ---
        top_vids = sorted(demand_pred.keys(), key=lambda k: demand_pred[k], reverse=True)[:args.capacity * 2]
        policy.decide(top_vids, demand_pred, sizes, fail_risk)

        # --- Count hits/misses for exact day t ---
        day = ts[ts['date'] == t][['video_id', 'reqs']].values
        for vid, req in day:
            req = int(req)
            if req <= 0:
                continue
            if vid in policy.cache:
                hit += req
            else:
                miss += req

    hit_ratio = hit / max(1, (hit + miss))
    print(f"Hit ratio: {hit_ratio:.4f}  (hits={hit}, misses={miss})")


if __name__ == '__main__':
    main()
