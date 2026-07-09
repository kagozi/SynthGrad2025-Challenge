---
name: MONAI 3D Diffusion Model
description: 3D conditional DDPM using MONAI DiffusionModelUNet for SynthRAD MR→CT synthesis
type: project
---

New parallel track alongside CNN (DynUNet v2). MONAI's 3D DiffusionModelUNet
conditioned on MR patches via channel concatenation (in_channels=2: MR+noisy CT).
Anatomy class embedding (HN/TH/AB) via num_class_embeds=3.

**Key files:**
- `src/models/monai_diffusion_3d.py` — MonaiDiffusion3D + GaussianDiffusion3D + EMA
- `training/train_monai_diffusion.py` — step-based training loop with gradient accumulation
- `training/configs/monai_diffusion_3d.yaml` — patch (32,128,128), channels (64,128,256,256)
- `inference/predict_monai_diffusion.py` — DDIM sliding window, loads EMA weights
- `nautilius/jobs/train-monai-diffusion-fold{0-4}.yaml` — 5 K8s jobs

**Loss:** noise MSE (w=1.0) + Min-SNR weighted MAE (w=0.25) + SSIM (w=0.25)
**EMA:** decay=0.999, applied at inference
**DDIM:** 50 steps at eval, 10 steps during val checkpoints

**Why:** Paper shows diffusion ranked 7th-12th in challenge; CNN is expected to win,
but this serves as experimental parallel track to close the gap or explore.

**Caveat:** Requires MONAI >= 1.4 for DiffusionModelUNet (only available in training container, not local dev env).
