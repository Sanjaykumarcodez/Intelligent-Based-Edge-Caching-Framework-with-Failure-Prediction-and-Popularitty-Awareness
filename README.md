# Intelligent Split Learning-Based Edge Caching (YouTube Trending)

This project turns the **YouTube Trending** CSVs you uploaded into:
1. **Popularity time-series** per region/video
2. A **GRU-based popularity predictor**
3. **Failure simulation** for edge nodes
4. A **caching simulator** using predicted popularity + failure risk
5. **Split Learning stubs** (client/server) to show where the cut-layer goes

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Preprocess the uploaded CSVs (point --data_dir to extracted archive folder)
python src/preprocess.py --data_dir /mnt/data/extracted_dataset/archive --out_dir data/processed

# 2) Train popularity GRU
python src/train_popularity.py --ts_path data/processed/ts.parquet --model_path models/pop_gru.pt

# 3) Simulate failures
python src/simulate_failures.py --edges 5 --hours 72 --out data/processed/failures.parquet

# 4) Run caching simulator
python src/cache/simulator.py --ts_path data/processed/ts.parquet   --fail_path data/processed/failures.parquet --policy heuristic --hours 48
```

Outputs: hit ratio, average latency proxy, and plots saved in `outputs/`.
# Intelligent-Based-Edge-Caching-Framework-with-Failure-Prediction-and-Popularitty-Awareness
An intelligent edge caching framework that combines Split Learning, Failure Prediction, and Popularity Awareness to improve edge computing performance. The system predicts node failures, caches popular content, reduces latency, improves cache efficiency, and ensures reliable, scalable content delivery across distributed networks.
ceba47f57ad0e7b4b125adbec6ee3ec119ee4337
