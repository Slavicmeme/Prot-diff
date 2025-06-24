import torch
import argparse
import os
import numpy as np
import pandas as pd
from utils.utils import encode_sequence
from sklearn.metrics.pairwise import cosine_similarity

def load_reference_sequences(csv_path):
    df = pd.read_csv(csv_path)
    sequences = df["sequence"].tolist()
    latents = [encode_sequence(seq).cpu().numpy() for seq in sequences]
    return sequences, np.array(latents)

def decode_latents(latent_path, reference_csv, output_csv):
    latents = torch.load(latent_path)
    sequences, reference_latents = load_reference_sequences(reference_csv)

    decoded = []
    sims = cosine_similarity(latents, reference_latents)
    for i in range(len(latents)):
        idx = sims[i].argmax()
        decoded.append(sequences[idx])

    df = pd.DataFrame({"sequence": decoded})
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved decoded sequences to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_latents", type=str, required=True)
    parser.add_argument("--reference_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    args = parser.parse_args()

    decode_latents(args.input_latents, args.reference_csv, args.output_csv)
