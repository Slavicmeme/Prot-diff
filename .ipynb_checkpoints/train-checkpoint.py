# train.py

import argparse
import pandas as pd
import torch
import os
from datetime import datetime

from train.train_diffusion import train  # ✅ 가장 바깥의 trainer.py로부터 train 함수 사용

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, required=True, help="Path to filtered CSV file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--timesteps", type=int, default=100, help="Number of diffusion timesteps")
    parser.add_argument("--save_dir", type=str, default="load", help="Directory to save model weights")
    args = parser.parse_args()

    df = pd.read_csv(args.data_csv)
    sequences = df['sequence'].tolist()

    print(f"Starting training with {len(sequences)} sequences")

    # Train the model
    model = train(
        sequences=sequences,
        epochs=args.epochs,
        batch_size=args.batch_size,
        timesteps=args.timesteps
    )

    # Save the model
    os.makedirs(args.save_dir, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(args.save_dir, f"diffusion_model_{now}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()