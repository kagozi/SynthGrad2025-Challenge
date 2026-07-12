---
name: VS-DDPM 3D (Faking_it-style)
description: 3D patch-based DDPM with window attention, VLB loss — best diffusion method from SynthRAD2025 paper (rank 7)
type: project
---

Best diffusion approach from synthrad25-challenge.pdf: **Faking_it** (rank 7 Task 1).

**Why:** Our MONAI diffusion model was deleted due to bad performance. This is a clean reimplementation based on the best-performing diffusion method in the challenge paper.

**How to apply:** Use these files for any diffusion-track iteration.

## Architecture
- 3D conditional DDPM, epsilon + learned variance prediction (2-ch output)
- DDPMUNet3D: 4-level encoder, channels [32,64,128,256], anisotropic first stride (1,2,2) then (2,2,2)
- WindowAttention3D at level-2 (win=4,4,4, 4 heads) and bottleneck (win=4,4,4, 8 heads)
- ~15.3M parameters
- Cosine noise schedule T=1000

## Loss
- L_total = L_simple (MSE on noise) + 0.001 * L_VLB + 1.0 * L_MAE(x0) + 1.0 * L_SSIM(x0)
- VLB from Nichol & Dhariwal 2021 Improved DDPM (stop-grad on eps for KL branch)

## Training
- 3D patches: (32, 128, 128) = (D, H, W)
- 8 samples/volume, batch_size=2, AdamW lr=1e-4, cosine LR 200 epochs
- K8s jobs: synthrad2025-train-vs-ddpm-fold{0-4}
- Checkpoints: /pvc/checkpoints/vs_ddpm_3d/fold{N}_best.pth

## Files
- src/models/vs_ddpm_3d.py — model
- training/train_vs_ddpm.py — training loop
- training/configs/vs_ddpm_3d.yaml — config
- inference/predict_vs_ddpm.py — sliding window DDIM (overlap=0.5, 20 steps)
- nautilius/jobs/train-vs-ddpm-fold{0-4}.yaml — K8s jobs

## Inference
- DDIM 20 steps (configurable), batch=4 patches, overlap=0.5
- Preserves original SimpleITK spacing/origin/direction
- Background masked to -1024 HU post-denoising
