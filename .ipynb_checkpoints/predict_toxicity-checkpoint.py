import torch
import argparse
import os
import numpy as np
from model.toxicity_model import ToxicityPredictor
import pandas as pd

def dummy_toxicity_data(n=200, dim=1024):
    X = np.random.rand(n, dim)
    y = np.random.randint(0, 2, size=n)  # 0 = non-toxic, 1 = toxic
    return X, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_latents", required=True, type=str)
    parser.add_argument("--output_csv", required=True, type=str)
    args = parser.parse_args()

    latents = torch.load(args.input_latents)
    X = np.array(latents)

    tox_model = ToxicityPredictor()
    X_train, y_train = dummy_toxicity_data()
    tox_model.fit(X_train, y_train)

    non_toxic = tox_model.filter_non_toxic(X)
    print(f"{len(non_toxic)} non-toxic candidates selected.")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df = pd.DataFrame(non_toxic)
    df.to_csv(args.output_csv, index=False)