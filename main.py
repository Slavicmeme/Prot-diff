from trainer import train
import pandas as pd

# 1. CSV에서 시퀀스 불러오기
csv_path = "data/grampa_ecoli.csv"
df = pd.read_csv(csv_path)
sequences = df["sequence"].tolist()

# 2. 디퓨전 모델 학습
trained_model = train(
    sequences=sequences,
    epochs=10,
    batch_size=8,
    timesteps=100
)

# 3. 모델 저장
import torch
torch.save(trained_model.state_dict(), "trained_diffusion_model.pt")
print("✅ 모델이 trained_diffusion_model.pt 에 저장되었습니다.")
