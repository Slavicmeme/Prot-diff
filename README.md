# 🧬 ProT-Diff: AMP Generation Pipeline

This project implements an end-to-end pipeline for generating antimicrobial peptides (AMPs) using a diffusion model based on **ProT-T5 embeddings** and a lightweight **1D U-Net architecture**. The pipeline includes MIC regression filtering and optional toxicity prediction.

---

## 📦 Environment Setup

```bash
conda env create -f environment.yml
conda activate protdiff-env
```

---

## 📁 Directory Structure

```
.
├── data/
│   ├── grampa.csv                      # Raw GRAMPA dataset
│   └── grampa_ecoli.csv                # Preprocessed E. coli-specific subset
│
├── load/                               # Directory to store trained diffusion model
│   └── diffusion_model_*.pt
│
├── latents/                            # Directory for generated and filtered latents
│   ├── generated.pt
│   └── mic_filtered.pt
│
├── results/                            # Final results
│   ├── final_filtered_sequences.csv    # After MIC and toxicity filtering
│   └── final_sequences.csv             # After decoding
│
├── data_processor/
│   └── preprocess_data.py              # Script to preprocess GRAMPA data
│
├── model/
│   ├── diff_model.py                   # U-Net and noise scheduler definitions
│   └── toxicity_model.py               # Toxicity prediction model
│
├── train/
│   ├── train_diffusion.py              # CLI for training diffusion model
│   └── mic_regressor.py                # MIC regression model class
│
├── utils/
│   └── utils.py                        # Shared utilities (e.g., ProtT5 encoder)
│
├── train.py                            # Main training entry point
├── sample_latents.py                   # Generate latent vectors from trained model
├── apply_mic_filter.py                 # Apply MIC filtering to latents
├── predict_toxicity.py                 # Predict toxicity for filtered sequences
├── decode_latents.py                   # Decode latents back into sequences
└── environment.yml                     # Conda environment specification

```

---

## 🚀 Full Pipeline Instructions

### 1. Preprocess GRAMPA for E. coli

```bash
python data_processor/preprocess_data.py \
  --input_csv data/grampa.csv \
  --output_csv data/grampa_ecoli.csv \
  --bacterium "E. coli"
```

---

### 2. Train Diffusion Model

```bash
python train.py \
  --data_csv data/grampa_ecoli.csv \
  --epochs 10 \
  --lr 1e-3 \
  --batch_size 8 \
  --save_dir load
```

---

### 3. Generate AMP Candidates (Latents)

```bash
python sample_latents.py \
  --model_ckpt load/diffusion_model_YYYYMMDD_HHMMSS.pt \
  --output_latents latents/generated.pt \
  --num_samples 100
```

Replace the checkpoint name with the correct filename from your run.

---

### 4. Filter Candidates Using MIC

```bash
python apply_mic_filter.py \
  --input_latents latents/generated.pt \
  --output_latents latents/mic_filtered.pt \
  --threshold 1.2
```

---

### 5. (Optional) Toxicity Prediction

```bash
python predict_toxicity.py \
  --input_latents latents/mic_filtered.pt \
  --output_csv results/final_filtered_sequences.csv
```

---

### 6. Decode Sequences (if needed)

```bash
python decode_latents.py \
  --input_latents latents/mic_filtered.pt \
  --reference_csv data/grampa_ecoli.csv \
  --output_csv results/final_sequences.csv
```

---

## ✅ Output

- `load/`: Saved trained diffusion models.
- `latents/`: Generated and MIC-filtered latent vectors.
- `results/`: Final output CSV files (filtered sequences, optionally toxicity-screened).

---

Last updated: 2025-06-24 02:53
