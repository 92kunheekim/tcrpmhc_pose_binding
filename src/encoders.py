"""Sequence encoders: per-chain 1D-CNN over Atchley matrices, + warm-start autoencoder."""
import torch
import torch.nn as nn

class ChainEncoder(nn.Module):
    def __init__(self, in_channels=5, hidden=64, emb_dim=64, k=5):
        super().__init__(); p = k//2
        self.conv = nn.Sequential(nn.Conv1d(in_channels, hidden, k, padding=p), nn.BatchNorm1d(hidden), nn.ReLU(),
                                  nn.Conv1d(hidden, hidden, k, padding=p), nn.BatchNorm1d(hidden), nn.ReLU())
        self.pool = nn.AdaptiveMaxPool1d(1); self.proj = nn.Linear(hidden, emb_dim)
    def forward(self, x):
        x = x.transpose(1, 2); return self.proj(self.pool(self.conv(x)).squeeze(-1))

class ChainAutoencoder(nn.Module):
    """Warm-start AE: ChainEncoder -> bottleneck -> reconstruct Atchley matrix."""
    def __init__(self, max_len, emb_dim=64):
        super().__init__(); self.max_len = max_len; self.encoder = ChainEncoder(emb_dim=emb_dim)
        self.decoder = nn.Sequential(nn.Linear(emb_dim, 128), nn.ReLU(), nn.Linear(128, max_len*5))
    def forward(self, x):
        z = self.encoder(x); return self.decoder(z).view(-1, self.max_len, 5), z
