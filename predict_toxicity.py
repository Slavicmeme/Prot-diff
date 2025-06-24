import torch
import argparse
import os
import numpy as np
from model.toxicity_model import ToxicityPredictor

def dummy_toxicity_data(n=200, dim=1024):
    X = np.random.rand(n, dim)
    y = np.random.randint(0, 2, size=n)  # 0 = non-toxic, 1 = toxic
    return X, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_latents", required=True, type=str)
    parser.add_argument("--output_pt", required=True, type=str, help="Output .pt path")
    args = parser.parse_args()

    latents = torch.load(args.input_latents)
    X = np.array(latents)

    tox_model = ToxicityPredictor()
    X_train, y_train = dummy_toxicity_data()
    tox_model.fit(X_train, y_train)

    non_toxic = tox_model.filter_non_toxic(X)
    print(f"{len(non_toxic)} non-toxic candidates selected.")

    os.makedirs(os.path.dirname(args.output_pt), exist_ok=True)
    torch.save(torch.tensor(non_toxic), args.output_pt)
    print(f"Saved non-toxic sequences to: {args.output_pt}")