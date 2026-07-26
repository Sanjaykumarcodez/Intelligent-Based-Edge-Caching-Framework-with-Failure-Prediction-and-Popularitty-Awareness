import pandas as pd
from pathlib import Path
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True, help="path to edge_health.csv")
ap.add_argument("--out", required=True, help="output parquet path (e.g., data/processed/failures.parquet)")
args = ap.parse_args()

df = pd.read_csv(args.csv)  # columns: timestamp,edge_id,cpu_percent, memory_percent, disk_io_MBps, net_latency_ms, failed
df["time"] = pd.to_datetime(df["timestamp"])
df = df.drop(columns=["timestamp"])
df = df.sort_values("time")
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(args.out, index=False)
print("Saved:", args.out, "rows=", len(df))
