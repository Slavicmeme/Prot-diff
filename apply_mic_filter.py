import argparse
import torch
import numpy as np
import os
from train.mic_regressor import MICRegressor

def load_latents(path):
    return torch.load(path).numpy()

def save_latents(latents, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(torch.tensor(latents), path)

def main(args):
    latents = load_latents(args.input_latents)

    # 🔧 임시 MIC 학습 데이터 (예시, 실제론 사전 학습된 모델 로드)
    np.random.seed(42)
    X_train = np.random.rand(500, 1024)
    y_train = np.random.uniform(0.0, 2.0, 500)

    # ✅ MIC Regressor 사용
    mic_regressor = MICRegressor()
    mic_regressor.fit(X_train, y_train)

    filtered = mic_regressor.filter_by_threshold(latents, threshold=args.threshold)
    print(f"Filtered {len(filtered)} / {len(latents)} latent vectors with MIC < {args.threshold}")

    save_latents(filtered, args.output_latents)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_latents", type=str, required=True, help="Path to generated.pt")
    parser.add_argument("--output_latents", type=str, required=True, help="Path to save mic_filtered.pt")
    parser.add_argument("--threshold", type=float, default=1.2, help="MIC threshold (e.g., 1.2)")
    args = parser.parse_args()
    main(args)
