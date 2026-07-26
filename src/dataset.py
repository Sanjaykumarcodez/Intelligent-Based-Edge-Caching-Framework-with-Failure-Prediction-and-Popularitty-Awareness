import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class PopDataset(Dataset):
    def __init__(self, ts_path, lookback=14, horizon=1, regions=None):
        ts = pd.read_parquet(ts_path)
        if regions:
            ts = ts[ts['region'].isin(regions)]
        # pivot to regular sequences per (region, video_id)
        self.samples = []
        for (r, vid), df in ts.groupby(['region','video_id']):
            s = df.sort_values('date')['reqs'].to_numpy(dtype=np.float32)
            if len(s) < lookback + horizon:
                continue
            # rolling windows
            for t in range(lookback, len(s)-horizon+1):
                x = s[t-lookback:t]
                y = s[t:t+horizon].sum()  # next-horizon demand
                self.samples.append((x, y, r, vid))
        self.lookback = lookback

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        x, y, r, vid = self.samples[idx]
        x = np.expand_dims(x, -1)  # [T,1]
        return torch.tensor(x), torch.tensor([y], dtype=torch.float32)
