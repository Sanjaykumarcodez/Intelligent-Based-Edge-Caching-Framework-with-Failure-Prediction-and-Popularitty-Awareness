import os, argparse, json
import pandas as pd
import numpy as np
from pathlib import Path

def load_country(csv_path, cat_json):
    df = pd.read_csv(csv_path, encoding='utf-8')
    # normalize publish_time / trending_date to datetime (Kaggle often has 'yy.dd.mm' or 'yy.mm.dd')
    if 'trending_date' in df.columns:
        td = pd.to_datetime(df['trending_date'], errors='coerce')
        if td.isna().all():
            td = pd.to_datetime(df['trending_date'], format='%y.%d.%m', errors='coerce')
        if td.isna().all():
            td = pd.to_datetime(df['trending_date'], format='%y.%m.%d', errors='coerce')
        df['trending_date'] = td
    if 'publish_time' in df.columns:
        df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
    # Minimal columns
    keep = ['video_id','trending_date','publish_time','views','likes','comment_count','category_id']
    for k in keep:
        if k not in df.columns:
            df[k] = np.nan
    # Map category ids
    try:
        cats = json.load(open(cat_json, 'r'))
        id2name = {int(it['id']): it['snippet']['title'] for it in cats.get('items',[])}
        df['category'] = df['category_id'].map(id2name)
    except Exception:
        df['category'] = None
    return df

def build_timeseries(df, region):
    # Aggregate by (video_id, day) as a proxy for requests (views)
    # Fill missing days between publish and last trending seen
    df = df.copy()
    df['date'] = pd.to_datetime(df['trending_date']).dt.floor('D')
    df = df.dropna(subset=['date'])
    # Keep necessary columns
    base = df[['video_id','date','views']].groupby(['video_id','date']).agg({'views':'max'}).reset_index()
    # Convert views to per-day increments (approximate daily requests)
    base = base.sort_values(['video_id','date'])
    base['prev'] = base.groupby('video_id')['views'].shift(1).fillna(0)
    base['reqs'] = (base['views'] - base['prev']).clip(lower=0)
    # Summarize by date across videos to get global activity
    # But we want per (region, video) timeseries
    ts = base[['video_id','date','reqs']]
    ts['region'] = region
    return ts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, required=True, help='Path to folder with *videos.csv and *_category_id.json')
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--regions', type=str, default='', help='Comma-separated region codes to include (e.g., IN,US,GB). Empty=all found.')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(list(data_dir.glob('*videos.csv')))
    if not files:
        raise SystemExit(f"No '*videos.csv' found under: {data_dir}. Point --data_dir to the folder that contains INvideos.csv, USvideos.csv, etc.")

    if args.regions:
        allow = set([r.strip().upper()+'videos.csv' for r in args.regions.split(',')])
        files = [f for f in files if f.name.upper() in allow]
        if not files:
            raise SystemExit(f"Regions filter {args.regions} yielded no files under {data_dir}")

    all_ts = []
    for f in files:
        region = f.name[:2].upper()
        cat_json = data_dir / f'{region}_category_id.json'
        try:
            df = load_country(f, cat_json)
            ts = build_timeseries(df, region)
            if len(ts)==0:
                print(f'[WARN] {region}: no usable rows after parsing dates.')
            else:
                all_ts.append(ts)
                print(f'[OK] {region}: {len(ts)} rows')
        except Exception as e:
            print(f'[WARN] {f}: {e}')

    if not all_ts:
        raise SystemExit('No time-series rows were produced. Check --data_dir and date parsing.')

    ts = pd.concat(all_ts, ignore_index=True)
    ts = ts.sort_values(['region','video_id','date']).reset_index(drop=True)
    ts.to_parquet(out_dir/'ts.parquet', index=False)
    print('Saved:', out_dir/'ts.parquet')

if __name__ == '__main__':
    main()
