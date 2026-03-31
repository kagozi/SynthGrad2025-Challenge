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
│   ├── process.py          # Docker entrypoint (GC I/O convention)
│   ├── base_algorithm.py   # BaseSynthradAlgorithm (official template)
│   ├── requirements.txt    # Trimmed submission deps (no training extras)
│   ├── .env                # TASK_TYPE=mri
│   ├── build.sh            # Build linux/amd64 image from repo root
│   └── export.sh           # Build + save as .tar.gz for GC upload
├── results/                # Per-fold val metrics CSVs (local CV)
├── checkpoints/            # best_model.pth (gitignored)
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

Follows the [official SynthRAD2025 algorithm template](https://github.com/SynthRAD2025/algorithm-template).

```bash
# Copy best checkpoint locally (pulled from PVC)
# checkpoints/best_model.pth  ← gitignored, must be present

# Build linux/amd64 image (from repo root)
docker/build.sh

# Export as .tar.gz for Grand Challenge upload
docker/export.sh synthrad2025_task1_baseline

# Create model tarball (uploaded separately on Grand Challenge)
tar -czvf model.tar.gz -C checkpoints .
```

Grand Challenge I/O (handled by `base_algorithm.py`):
- Input MR:   `/input/images/mri/<uuid>.mha`
- Body mask:  `/input/images/body/<uuid>.mha`
- Region:     `/input/region.json`
- Output sCT: `/output/images/synthetic-ct/<uuid>.mha`
- Model:      `/opt/ml/model/model.pth` (from uploaded tarball)

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

## Baseline CV Results (5-fold, 513 val cases)

| Anatomy | MAE (HU↓) | PSNR (dB↑) | MS-SSIM↑ | mDice↑ | HD95 (mm↓) |
|---------|-----------|------------|----------|--------|------------|
| AB | 102.1 | 25.71 | 0.681 | 0.537 | 52.8 |
| HN | 122.5 | 25.10 | 0.663 | 0.694 | 36.5 |
| TH | 114.2 | 25.21 | 0.618 | 0.474 | 81.4 |
| **Overall** | **112.6** | **25.35** | **0.653** | **0.562** | **58.0** |

Key weaknesses: bone dice TH=0.249, AB=0.311; HD95 high and fold-variable (34–80mm).

## Post-Challenge Submission Budget

- **2 Docker submissions per 60 days** via Grand Challenge
- Val/Test ground truth available: **March 1, 2030**
- Use local 5-fold CV for all tuning; submit only to verify generalisation

## Roadmap

### ✅ Done
- [x] Project structure & data download
- [x] Data exploration (EDA + plots)
- [x] 5-fold CV splits (stratified by anatomy × center)
- [x] AttentionUNet2D with anatomy conditioning
- [x] Combined MAE + MS-SSIM loss (GDL optional)
- [x] Full-volume inference pipeline
- [x] Local metric evaluation (MAE, PSNR, MS-SSIM, Dice, HD95)
- [x] Grand Challenge Docker submission (official template-compliant)
- [x] Baseline submission to GC Validation phase

### 🔜 Round 2 — Loss improvements (same architecture)
- [ ] Enable GDL (gradient difference loss) — `gdl_weight > 0`
- [ ] Bone-weighted MAE (2–3× weight on HU > 200 voxels)
- [ ] Body-mask-conditioned loss (only penalise inside mask)
- [ ] Retrain 5 folds, re-submit

### 🔜 Round 3 — Architecture experiments
- [ ] 2.5D UNet (3-slice input, `in_channels=3`) — quick win for bone continuity
- [ ] pix2pix / conditional GAN (PatchGAN discriminator) — improves MS-SSIM & sharpness
- [ ] Swin-UNETR (Swin Transformer encoder + CNN decoder, via MONAI)
- [ ] 2.5D ResUNet with deep supervision

### 🔜 Round 4 — 3D & advanced
- [ ] 3D patch-based UNet
- [ ] Cascaded 2D→3D refinement (coarse 2D + 3D boundary refinement)
- [ ] Diffusion model (DDPM/DDIM with fast sampling — 15 min budget)
- [ ] Test-time augmentation (TTA: flips + anatomy)

### 🔜 Round 5 — Final submission
- [ ] Fold ensemble (average all 5 checkpoints)
- [ ] Best model → Test phase submission (2-shot budget)