import pandas as pd
from pathlib import Path
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True, help="path to popularity_timeseries.csv")
ap.add_argument("--out", required=True, help="output parquet path (e.g., data/processed/ts.parquet)")
ap.add_argument("--region", default="GLOBAL", help="region label to assign (MovieLens has no regions)")
args = ap.parse_args()

df = pd.read_csv(args.csv)  # columns: date,movieId,requests
df["date"] = pd.to_datetime(df["date"])
ts = df.rename(columns={"movieId":"video_id","requests":"reqs"}).copy()
ts["region"] = args.region
ts = ts[["region","video_id","date","reqs"]].sort_values(["region","video_id","date"])
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
ts.to_parquet(args.out, index=False)
print("Saved:", args.out, "rows=", len(ts))
