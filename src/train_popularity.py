import argparse, os
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from models.pop_gru import PopGRU
from dataset import PopDataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ts_path', type=str, required=True)
    ap.add_argument('--model_path', type=str, required=True)
    ap.add_argument('--lookback', type=int, default=14)
    ap.add_argument('--horizon', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=128)
    args = ap.parse_args()

    ds = PopDataset(args.ts_path, lookback=args.lookback, horizon=args.horizon)
    n = len(ds)
    n_train = int(0.8*n)
    n_val = n - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PopGRU().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.L1Loss()

    best = 1e9
    for ep in range(args.epochs):
        model.train(); tr_loss=0
        for x,y in tqdm(train_loader, desc=f'Epoch {ep+1}/{args.epochs}'):
            x, y = x.to(device), y.to(device).squeeze(-1)
            yhat = model(x)
            loss = loss_fn(yhat, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item()*x.size(0)
        tr_loss /= len(train_loader.dataset)

        model.eval(); va_loss=0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device).squeeze(-1)
                yhat = model(x)
                va_loss += loss_fn(yhat, y).item()*x.size(0)
        va_loss /= len(val_loader.dataset)
        print(f'Epoch {ep+1}  train MAE={tr_loss:.4f}  val MAE={va_loss:.4f}')
        if va_loss < best:
            best = va_loss
            torch.save(model.state_dict(), args.model_path)
            print('Saved best model ->', args.model_path)

if __name__ == '__main__':
    main()
