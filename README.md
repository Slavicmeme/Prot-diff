# ProT-Diff AMP Generation Pipeline

This project implements a ProT-Diff-inspired AMP generation system using diffusion modeling and MIC filtering.

## 📁 Project Structure

```
protdiff_structured/
├── data_processor/
│   └── preprocess_data.py
├── train/
│   ├── diff/
│   │   └── train_diffusion.py
│   └── mic_regressor/
│       └── mic_regressor.py
├── utils/
│   └── utils.py
├── model/
│   └── model.py
└── README.md
```

## 🚀 Usage Guide (Step-by-Step)

### 1️⃣ Preprocess GRAMPA Dataset

```bash
python data_processor/preprocess_data.py --input_csv data/grampa.csv --output_csv data/grampa_ecoli.csv --bacterium "E. coli"
```

### 2️⃣ Train Diffusion Model

```python
from train.diff.train_diffusion import train
import pandas as pd

df = pd.read_csv("data/grampa_ecoli.csv")
sequences = df['sequence'].tolist()
model = train(sequences, epochs=5)
```

### 3️⃣ Sample Latents

```python
from model.model import NoiseScheduler
import torch

scheduler = NoiseScheduler(100).to("cuda" if torch.cuda.is_available() else "cpu")
z = torch.randn(10, 1024).to("cuda")
```

### 4️⃣ Filter Using MIC Predictor

```python
from train.mic_regressor.mic_regressor import MICRegressor
import numpy as np

reg = MICRegressor()
X = np.random.rand(200, 1024)
y = np.random.uniform(0, 2, 200)
reg.fit(X, y)
filtered = reg.filter_by_threshold(z.cpu().numpy(), threshold=1.2)
```
