# Split Learning Server (stub): receives activations, computes loss & grads, returns grads.
import torch, torch.nn as nn

class ServerHead(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, acts):
        return self.head(acts).squeeze(-1)
