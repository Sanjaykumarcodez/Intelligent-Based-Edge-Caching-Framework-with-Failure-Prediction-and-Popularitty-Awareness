import argparse, numpy as np, pandas as pd
from datetime import datetime, timedelta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--edges', type=int, default=5)
    ap.add_argument('--hours', type=int, default=72)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(42)
    start = pd.Timestamp(datetime.utcnow().replace(minute=0, second=0, microsecond=0))
    records = []
    for e in range(args.edges):
        cpu = 10 + 30*rng.random(args.hours)  # baseline CPU%
        bursts = (rng.random(args.hours) < 0.1).astype(float)  # occasional spikes
        cpu += bursts * (30 + 40*rng.random(args.hours))
        cpu = cpu.clip(0, 100)
        # Failure when CPU>90 for sustained periods OR random Weibull TTF
        fail_prob = np.where(cpu>90, 0.15, 0.01)
        failed = (rng.random(args.hours) < fail_prob).astype(int)
        time = [start + timedelta(hours=h) for h in range(args.hours)]
        for t, c, f in zip(time, cpu, failed):
            records.append({'edge_id': f'edge-{e+1}', 'time': t, 'cpu': float(c), 'failed': int(f)})
    df = pd.DataFrame(records)
    df.to_parquet(args.out, index=False)
    print('Saved', args.out, 'rows=', len(df))

if __name__ == '__main__':
    main()
