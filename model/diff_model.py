import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]  # [B, half_dim]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # [B, dim]
        return emb


class UNet1D(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()

        self.down = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.LeakyReLU()
        )

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, 64),
            nn.LeakyReLU(),
            nn.Unflatten(1, (64, 1))  # -> [B, 64, 1]
        )

        self.middle = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.LeakyReLU()
        )

        self.up = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(32, 1, 3, padding=1)
        )

    def forward(self, x, t):
        """
        x: [B, D]    -> after unsqueeze: [B, 1, D]
        t: [B]       -> after embedding: [B, 64, 1]
        """
        x = x.unsqueeze(1)         # [B, 1, D]
        d1 = self.down(x)          # [B, 64, D]
        t_emb = self.time_mlp(t)   # [B, 64, 1]
        d2 = d1 + t_emb            # [B, 64, D]
        m = self.middle(d2)        # [B, 64, D]
        u = self.up(m)             # [B, 1, D]
        return u.squeeze(1)        # [B, D]


class NoiseScheduler:
    def __init__(self, timesteps):
        self.timesteps = timesteps
        self.beta = torch.linspace(1e-4, 0.02, timesteps)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def to(self, device):
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        return self
