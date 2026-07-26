# Convenience runner for the whole pipeline (adjust paths as needed)
import os, subprocess, sys

PY = sys.executable
DATA_DIR = "/mnt/data/extracted_dataset/archive"  # change if needed

steps = [
    f"{PY} src/preprocess.py --data_dir {DATA_DIR} --out_dir data/processed",
    f"{PY} src/train_popularity.py --ts_path data/processed/ts.parquet --model_path models/pop_gru.pt",
    f"{PY} src/simulate_failures.py --edges 5 --hours 72 --out data/processed/failures.parquet",
    f"{PY} src/cache/simulator.py --ts_path data/processed/ts.parquet --fail_path data/processed/failures.parquet --policy heuristic --hours 48"
]

for cmd in steps:
    print("\n[RUN]", cmd)
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print("Command failed:", cmd); break
