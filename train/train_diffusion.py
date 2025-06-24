import torch
from torch import nn, optim
from tqdm import tqdm
import argparse
from utils.utils import encode_sequence
from model.diff_model import UNet1D, NoiseScheduler

class DiffusionTrainer:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler

    def q_sample(self, x_start, t, noise):
        alpha_bar = self.scheduler.alpha_bar[t].view(-1, 1).to(x_start.device)
        return torch.sqrt(alpha_bar) * x_start + torch.sqrt(1 - alpha_bar) * noise

    def p_losses(self, x_start, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted = self.model(x_noisy, t)
        return nn.functional.mse_loss(predicted, noise)

def train(sequences, epochs=10, lr=1e-4, batch_size=8, timesteps=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet1D().to(device)
    scheduler = NoiseScheduler(timesteps).to(device)
    trainer = DiffusionTrainer(model, scheduler)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    latent = [encode_sequence(seq) for seq in sequences]
    latent = torch.stack(latent).to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        loop = tqdm(range(0, len(latent), batch_size), desc=f"Epoch {epoch+1}/{epochs}")
        for i in loop:
            batch = latent[i:i+batch_size]
            t = torch.randint(0, timesteps, (batch.size(0),), device=device)
            loss = trainer.p_losses(batch, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=total_loss / (i//batch_size + 1))
    return model
