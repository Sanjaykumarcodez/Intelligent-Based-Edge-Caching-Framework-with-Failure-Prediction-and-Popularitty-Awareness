# Split Learning Client (stub): computes cut-layer features locally.
# In a real deployment, this would be the first few layers of PopGRU.
import torch, torch.nn as nn

class CutNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=32):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
    def forward(self, x):
        out, _ = self.gru(x)
        return out[:, -1, :]  # cut activations
