import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet1D(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv1d(1, 8, 3, padding=1), nn.ReLU(),
            nn.Conv1d(8, 16, 3, padding=1), nn.ReLU()
        )
        self.middle = nn.Sequential(
            nn.Conv1d(16, 16, 3, padding=1), nn.ReLU()
        )
        self.up = nn.Sequential(
            nn.ConvTranspose1d(16, 8, 3, padding=1), nn.ReLU(),
            nn.Conv1d(8, 1, 3, padding=1)
        )

    def forward(self, x, t):
        x = x.unsqueeze(1)
        z = self.down(x)
        z = self.middle(z)
        z = self.up(z)
        return z.squeeze(1)

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
