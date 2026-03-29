# SynthRAD2025 Task 1 — MR to Synthetic CT

Post-challenge participation in [SynthRAD2025](https://synthrad2025.grand-challenge.org/).
Goal: generate synthetic CT from MRI for radiotherapy planning across head-and-neck, thorax, and abdomen.

## Project Structure

```
SynthGrad2025-Challenge/
├── data/
│   ├── raw/                # Downloaded training data (Zenodo)
│   ├── processed/          # Optional preprocessed caches
│   └── splits/             # folds.csv, dataset_info.csv, plots/
├── src/
│   ├── dataset.py          # PyTorch datasets (2D slice, 3D patch, inference)
│   ├── losses.py           # MAE, MS-SSIM, GDL, CombinedLoss
│   ├── metrics.py          # MAE, PSNR, MS-SSIM, Dice, HD95
│   └── models/
│       └── unet2d.py       # 2D U-Net + Attention U-Net with anatomy conditioning
├── scripts/
│   ├── download_data.sh    # Download from Zenodo
│   ├── explore_data.py     # Data distribution analysis + plots
│   └── prepare_folds.py    # 5-fold CV split generation
├── training/
│   ├── train.py            # Main training loop
│   └── configs/
│       └── baseline_unet2d.yaml
├── inference/
│   ├── predict.py          # Full-volume inference → .mha files
│   └── evaluate.py         # Local metric computation
├── docker/
│   ├── Dockerfile          # Grand Challenge submission container
│   └── process.py          # Docker entrypoint (GC I/O convention)
└── requirements.txt
```

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download training data

```bash
# Install zenodo_get first
pip install zenodo-get

# Download Task 1 data (~578 training cases)
bash scripts/download_data.sh

# Or download a specific anatomy only
bash scripts/download_data.sh --anatomy HN
```

Data will be saved to `data/raw/` with structure:
```
data/raw/
├── 1HNA001/ → mr.mha, ct.mha, mask.mha
├── 1HNA002/
...
```

### 3. Explore the data

```bash
# Fast header-only scan (shapes, spacing)
python scripts/explore_data.py --data_dir data/raw --fast

# Full scan including intensity statistics
python scripts/explore_data.py --data_dir data/raw
```

Plots saved to `data/splits/plots/`.

### 4. Generate CV folds

```bash
python scripts/prepare_folds.py --data_dir data/raw --n_folds 5
```

Saves `data/splits/folds.csv` stratified by anatomy × center.

### 5. Train baseline

```bash
python training/train.py --config training/configs/baseline_unet2d.yaml --fold 0

# Resume
python training/train.py --config training/configs/baseline_unet2d.yaml \
    --fold 0 --resume checkpoints/baseline_unet2d/fold0_last.pth

# Monitor
tensorboard --logdir logs/
```

### 6. Run inference

```bash
python inference/predict.py \
    --checkpoint checkpoints/baseline_unet2d/fold0_best.pth \
    --input_dir  data/raw \
    --output_dir predictions/fold0_val \
    --split val --folds_csv data/splits/folds.csv --fold 0
```

### 7. Evaluate locally

```bash
python inference/evaluate.py \
    --pred_dir predictions/fold0_val \
    --gt_dir   data/raw \
    --output   results/fold0_val_metrics.csv \
    --compute_seg
```

### 8. Build Docker for submission

```bash
# Copy best checkpoint
cp checkpoints/baseline_unet2d/fold0_best.pth checkpoints/best_model.pth

# Build image
docker build -t synthrad2025_task1 -f docker/Dockerfile .

# Test locally
docker run --gpus all \
    -v $(pwd)/data/raw/1HNA001:/input/1HNA001 \
    -v $(pwd)/test_output:/output \
    synthrad2025_task1
```

## Evaluation Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| MAE | Mean Absolute Error (HU, inside body mask) | lower |
| PSNR | Peak Signal-to-Noise Ratio (dB) | higher |
| MS-SSIM | Multi-Scale Structural Similarity (5 levels) | higher |
| mDice | Mean Dice of TotalSegmentator structures | higher |
| HD95 | 95th percentile Hausdorff Distance (mm) | lower |

Official evaluation uses TotalSegmentator for mDice/HD95.
Local evaluation uses HU thresholds as a fast proxy.

## Post-Challenge Submission Budget

- **2 Docker submissions per 60 days** via Grand Challenge
- Val/Test ground truth available: **March 1, 2030**
- Use local 5-fold CV for all tuning; submit only to verify generalisation

## Roadmap

- [x] Project structure
- [x] Data download scripts
- [x] Data exploration
- [x] 5-fold CV splits
- [x] 2D U-Net baseline with anatomy conditioning
- [x] Attention U-Net variant
- [x] Combined MAE + MS-SSIM loss
- [x] Full-volume inference pipeline
- [x] Local metric evaluation
- [x] Docker submission template
- [ ] 3D patch-based training
- [ ] Diffusion model (DDPM/DDIM)
- [ ] Transformer backbone (Swin-UNETR)
- [ ] Test-time augmentation
- [ ] Ensemble across folds