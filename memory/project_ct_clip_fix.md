---
name: Critical CT Clip Fix — [-1024, 3000] HU
description: CT_CLIP was wrong (-1500 cutoff vs challenge eval -3000), causing bone errors
type: project
---

**Bug fixed 2026-07-09:** `CT_CLIP = (-1024, 1500)` → `(-1024, 3000)` in `src/dataset.py`.

The challenge evaluation clips to [-1024, 3000] HU for MAE/PSNR. Our old clip at 1500 HU
meant the model could never predict dense cortical bone (1500–3000 HU range), which is
the main driver of poor DICE (0.63) and HD95 (14 mm) vs KoalAI (DICE 0.78, HD95 5.77 mm).

**Also fixed:**
- `src/losses.py` `TotalSegmentatorAFP._to_nnunet()`: scale factor 1262 → 2012
  (inverse mapping from [-1,1] norm space back to HU must match CT_CLIP)
- `training/configs/dynunet_v2.yaml`: bone_threshold 0.0 → -0.4
  (HU 200 in [-1024,3000] normalises to -0.39, not 0.0 as in old range)

**All existing checkpoints encode the old normalization — must retrain from scratch.**

**How to apply:** All new training runs (dynunet_v2, monai_diffusion_3d) use fixed clip.
The old dynunet checkpoint results (MAE 86.6 HU) are not comparable to new runs.
