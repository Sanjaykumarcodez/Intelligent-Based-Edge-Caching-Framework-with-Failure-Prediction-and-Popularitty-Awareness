import pandas as pd
import numpy as np
import argparse, os
from pathlib import Path
from datetime import datetime

def find_ratings_file(folder: Path) -> Path:
    """
    Find ratings file under 'folder'. Supports ratings.csv/rating.csv or .xlsx.
    Searches recursively if not found at top level.
    """
    candidates = [
        "ratings.csv", "rating.csv", "Ratings.csv", "Rating.csv",
        "ratings.xlsx", "rating.xlsx"
    ]
    for name in candidates:
        p = folder / name
        if p.exists():
            return p
    for p in folder.rglob("*"):
        if p.is_file() and ("rating" in p.name.lower()) and (p.suffix.lower() in [".csv", ".xlsx"]):
            return p
    raise FileNotFoundError(
        f"Could not find ratings file under: {folder}\n"
        f"Tried {candidates} and recursive search for *rating*.csv/.xlsx."
    )

def load_ratings(path: Path) -> pd.DataFrame:
    """Load ratings and normalize column names (userId, movieId, rating, timestamp)."""
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    # case-insensitive column map
    cols = {c.lower(): c for c in df.columns}
    needed = ["userid", "movieid", "rating", "timestamp"]
    for n in needed:
        if n not in cols:
            raise ValueError(f"Missing column '{n}' (case-insensitive) in {path}. "
                             f"Found columns: {list(df.columns)}")

    df = df.rename(columns={
        cols["userid"]: "userId",
        cols["movieid"]: "movieId",
        cols["rating"]: "rating",
        cols["timestamp"]: "timestamp"
    })
    return df

def parse_timestamp_to_date(series: pd.Series) -> pd.Series:
    """
    Robustly convert a 'timestamp' series to a date.
    - If it looks numeric -> treat as Unix seconds.
    - Else -> parse as datetime string.
    """
    # If the series is numeric-like, use unit='s'
    if pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
        return pd.to_datetime(series, unit='s', errors='coerce').dt.date

    # Try fast parse; if fails, fallback
    try:
        return pd.to_datetime(series, errors='coerce').dt.date
    except Exception:
        # As a last resort, try common formats
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%d-%m-%Y %H:%M:%S"):
            try:
                return pd.to_datetime(series, format=fmt, errors='coerce').dt.date
            except Exception:
                pass
        raise ValueError("Could not parse 'timestamp' into dates. Please inspect the data format.")

def main():
    ap = argparse.ArgumentParser(description="Convert MovieLens ratings to daily popularity + synth edge health.")
    ap.add_argument("--input", required=True, help="Folder containing rating(s).csv (e.g., 'C:\\Users\\ADMIN\\Downloads\\archive (1)')")
    ap.add_argument("--out", default="data\\movielens_out", help="Output folder (default: data\\movielens_out)")
    ap.add_argument("--days", type=int, default=60, help="Days for synthetic edge health (default: 60)")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ratings_path = find_ratings_file(in_dir)
    print(f"[INFO] Using ratings file: {ratings_path}")

    ratings = load_ratings(ratings_path)

    # Parse timestamp -> date (robust)
    ratings["date"] = parse_timestamp_to_date(ratings["timestamp"])
    n_na = ratings["date"].isna().sum()
    if n_na > 0:
        print(f"[WARN] {n_na} rows had unparsable timestamps and will be dropped.")
        ratings = ratings.dropna(subset=["date"])

    # Aggregate to daily requests per movie
    popularity = (ratings.groupby(["date", "movieId"])
                          .size()
                          .reset_index(name="requests")
                          .sort_values(["date", "movieId"]))
    pop_out = out_dir / "popularity_timeseries.csv"
    popularity.to_csv(pop_out, index=False)
    print(f"[OK] Saved popularity dataset -> {pop_out}  shape={popularity.shape}")

    # -------- Synthetic Edge Health / Failures --------
    edges = [f"edge-{i}" for i in range(1, 6)]
    hours = args.days * 24
    start = pd.Timestamp(datetime.utcnow().replace(minute=0, second=0, microsecond=0)) - pd.Timedelta(hours=hours)
    ts = pd.date_range(start=start, periods=hours, freq="H")

    rng = np.random.default_rng(42)
    rows = []
    for e in edges:
        cpu = rng.normal(50, 15, hours).clip(0, 100)
        mem = rng.normal(60, 20, hours).clip(0, 100)
        disk = rng.normal(100, 40, hours).clip(10, 300)
        net = rng.normal(20, 5, hours).clip(1, 200)
        fail = (rng.random(hours) < (0.01 + 0.02*(cpu>85) + 0.01*(mem>90))).astype(int)
        rows.extend(zip(ts, [e]*hours, cpu, mem, disk, net, fail))

    health = pd.DataFrame(rows, columns=[
        "timestamp","edge_id","cpu_percent","memory_percent","disk_io_MBps","net_latency_ms","failed"
    ])
    health_out = out_dir / "edge_health.csv"
    health.to_csv(health_out, index=False)
    print(f"[OK] Saved edge health dataset -> {health_out}  shape={health.shape}")

if __name__ == "__main__":
    main()
