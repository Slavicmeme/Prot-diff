# sample_latents.py

import argparse
import torch
import os

from model.diff_model import UNet1D, NoiseScheduler

def sample_latents(model, scheduler, n_samples=100, timesteps=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    scheduler = scheduler.to(device)

    z = torch.randn(n_samples, 1024).to(device)

    with torch.no_grad():
        for t in reversed(range(timesteps)):
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            z = model(z, t_batch)
    return z.cpu()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_ckpt", type=str, required=True, help="Path to trained model .pt")
    parser.add_argument("--output_latents", type=str, required=True, help="Path to save sampled latent .pt file")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of AMP samples to generate")
    parser.add_argument("--timesteps", type=int, default=100, help="Number of diffusion steps")
    args = parser.parse_args()

    model = UNet1D()
    scheduler = NoiseScheduler(args.timesteps)
    model.load_state_dict(torch.load(args.model_ckpt, map_location="cpu"))

    print("Sampling AMP latent vectors...")
    latents = sample_latents(model, scheduler, n_samples=args.num_samples, timesteps=args.timesteps)

    os.makedirs(os.path.dirname(args.output_latents), exist_ok=True)
    torch.save(latents, args.output_latents)
    print(f"Latents saved to {args.output_latents}")

if __name__ == "__main__":
    main()
