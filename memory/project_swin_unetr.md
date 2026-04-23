---
name: Swin-UNETR Implementation
description: MONAI Swin-UNETR 3D training pipeline for SynthRAD2025 Task 1
type: project
---

Added MONAI Swin-UNETR as a new 3D architecture (Round 4).

**Why:** 3D transformer gives full volumetric context vs. 2.5D slice stack; MONAI SSL pretrained encoder (5050 CT/MRI volumes) gives strong initialization.

**Key files:**
- `src/models/swin_unetr.py` — `SwinUNETR3D` wrapper (anatomy FiLM head, pretrained loader)
- `training/configs/swin_unetr.yaml` — 3D patch config (64×128×128, 150 epochs)
- `training/train_swin.py` — 3D training script with sliding window inference
- `nautilius/jobs/train-swin-fold{0-4}.yaml` — K8s jobs (40Gi RAM request)

**Architecture decisions:**
- `feature_size=48` (mandatory for MONAI SSL pretrained checkpoint)
- Anatomy FiLM residual head after backbone; `film_hidden=128`; init: scale=1, shift=0
- `use_checkpoint=True` (gradient checkpointing, saves ~30% VRAM in Swin blocks)
- Pretrained SSL weights: `https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt`, cached to `/pvc/pretrained/`

**Training recipe:**
- Encoder frozen for first 10 epochs, then unfrozen with LR × 0.1 (differential LR)
- Full decoder + FiLM LR: 2e-4; Encoder LR: 2e-5
- Loss: bone-weighted MAE (w=1.0, bone_weight=2.5) + GDL (w=0.5)
- AMP fp16, grad clip norm=1.0, CosineAnnealingLR, 150 epochs

**Validation:**
- `sliding_window_inference` (MONAI): roi=(64,128,128), overlap=0.5, Gaussian weighting
- Anatomy injected via `partial(_predictor, model=model, anatomy_t=...)` closure

**GDL fix:** `GradientDifferenceLoss` in `losses.py` extended to support 5D tensors (B,C,D,H,W) for 3D training; 2D path unchanged.

**How to apply:** Deploy on A40/A100 nodes (40Gi+ VRAM request). After training, use `inference/predict_swin.py` (TBD) or adapt existing predict.py with `SwinUNETR3D` + `sliding_window_inference`.
