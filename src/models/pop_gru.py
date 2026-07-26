import torch
import torch.nn as nn

class PopGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1)
        )
    def forward(self, x):
        # x: [B, T, 1]
        out, _ = self.gru(x)
        last = out[:, -1, :]
        y = self.head(last)
        return y.squeeze(-1)
