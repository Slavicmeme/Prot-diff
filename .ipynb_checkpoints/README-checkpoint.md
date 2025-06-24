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
│   ├── grampa.csv                  # Raw GRAMPA data
│   └── grampa_ecoli.csv            # Preprocessed data (filtered for E. coli)
│
├── data_processor/
│   └── preprocess_data.py          # Script to preprocess and filter the GRAMPA dataset
│
├── train.py                        # Top-level script to train diffusion model
├── sample_latents.py              # Generate AMP candidates from trained model
├── apply_mic_filter.py            # Apply MIC regression filter to generated candidates
├── predict_toxicity.py            # (Optional) Predict toxicity of filtered sequences
├── decode_latents.py              # Decode latent vectors back into peptide sequences
│
├── model/
│   ├── model.py                    # UNet1D model and noise scheduler
│   └── decoder.py (optional)      # If sequence decoding from embeddings is needed
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
