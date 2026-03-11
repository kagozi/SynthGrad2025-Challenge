"""
Evaluation metrics for SynthRAD2025 Task 1.

Matches the official evaluation:
  - MAE   (inside dilated body mask)
  - PSNR  (inside dilated body mask)
  - MS-SSIM (5 scales)
  - mDice  (TotalSegmentator segmentation, approximated locally)
  - HD95   (TotalSegmentator segmentation, approximated locally)

All metrics operate on HU-space arrays (denormalised).
"""

from __future__ import annotations

import numpy as np
import torch


# ── Helpers ────────────────────────────────────────────────────────────────────

def to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ── Image similarity ───────────────────────────────────────────────────────────

def compute_mae(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray = None,
) -> float:
    """Mean Absolute Error in HU, optionally inside mask."""
    diff = np.abs(pred - target)
    if mask is not None:
        mask = mask.astype(bool)
        return float(diff[mask].mean())
    return float(diff.mean())


def compute_psnr(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray = None,
    data_range: float = 4000.0,   # HU range: [-1000, 3000]
) -> float:
    """Peak Signal-to-Noise Ratio in dB."""
    diff = pred - target
    if mask is not None:
        mask = mask.astype(bool)
        mse  = float((diff[mask] ** 2).mean())
    else:
        mse  = float((diff ** 2).mean())
    if mse < 1e-10:
        return 100.0
    return float(10.0 * np.log10(data_range ** 2 / mse))


def _gaussian_kernel_np(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    x = np.arange(size) - size // 2
    g = np.exp(-x ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    return np.outer(g, g)


def compute_ssim_2d(
    pred: np.ndarray,   # (H, W)
    target: np.ndarray,
    data_range: float = 4000.0,
    window_size: int = 11,
) -> float:
    from scipy.ndimage import convolve
    K1, K2   = 0.01, 0.03
    C1       = (K1 * data_range) ** 2
    C2       = (K2 * data_range) ** 2
    kernel   = _gaussian_kernel_np(window_size)

    mu1      = convolve(pred,   kernel)
    mu2      = convolve(target, kernel)
    mu1_sq   = mu1 ** 2
    mu2_sq   = mu2 ** 2
    mu1_mu2  = mu1 * mu2
    sig1_sq  = convolve(pred   * pred,   kernel) - mu1_sq
    sig2_sq  = convolve(target * target, kernel) - mu2_sq
    sig12    = convolve(pred   * target, kernel) - mu1_mu2

    num      = (2 * mu1_mu2 + C1) * (2 * sig12 + C2)
    den      = (mu1_sq + mu2_sq + C1) * (sig1_sq + sig2_sq + C2)
    return float(np.mean(num / (den + 1e-8)))


def compute_ms_ssim(
    pred: np.ndarray,    # (D, H, W) in HU
    target: np.ndarray,
    levels: int = 5,
    data_range: float = 4000.0,
) -> float:
    """MS-SSIM averaged over all axial slices."""
    weights   = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])[:levels]
    weights  /= weights.sum()
    ms_vals   = []

    for s in range(pred.shape[0]):
        p, t = pred[s], target[s]
        val  = 0.0
        for i in range(levels):
            val += weights[i] * compute_ssim_2d(p, t, data_range)
            if i < levels - 1:
                # Downsample
                from scipy.ndimage import zoom
                p = zoom(p, 0.5, order=1)
                t = zoom(t, 0.5, order=1)
        ms_vals.append(val)

    return float(np.mean(ms_vals))


# ── Segmentation-based (local approximation) ───────────────────────────────────

def _threshold_structures(arr: np.ndarray) -> dict:
    """
    Simple HU-threshold based segmentation as proxy for TotalSegmentator.
    Used for local Dice/HD95 estimation during development.
    Official evaluation uses TotalSegmentator — do not use this for final scores.
    """
    return {
        "bone":       (arr > 300),
        "soft_tissue": (arr > -100) & (arr <= 300),
        "air":        (arr <= -500),
    }


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    inter = (pred_mask & gt_mask).sum()
    denom = pred_mask.sum() + gt_mask.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


def compute_hd95(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """95th percentile Hausdorff Distance (voxel units)."""
    try:
        from scipy.ndimage import distance_transform_edt
        pred_pts = np.argwhere(pred_mask)
        gt_pts   = np.argwhere(gt_mask)
        if len(pred_pts) == 0 or len(gt_pts) == 0:
            return float("inf")

        dt_pred = distance_transform_edt(~pred_mask)
        dt_gt   = distance_transform_edt(~gt_mask)

        d1 = dt_gt[pred_mask]
        d2 = dt_pred[gt_mask]
        return float(np.percentile(np.concatenate([d1, d2]), 95))
    except Exception:
        return float("nan")


def compute_segmentation_metrics(
    pred_hu: np.ndarray,
    target_hu: np.ndarray,
    structures: list = None,
) -> dict:
    """
    Approximate Dice and HD95 using HU thresholds.
    NOTE: This is a development proxy only.
          Official scoring uses TotalSegmentator.
    """
    structures = structures or ["bone", "soft_tissue"]
    pred_segs  = _threshold_structures(pred_hu)
    gt_segs    = _threshold_structures(target_hu)

    results = {}
    for s in structures:
        if s not in pred_segs:
            continue
        results[f"dice_{s}"] = compute_dice(pred_segs[s], gt_segs[s])
        results[f"hd95_{s}"] = compute_hd95(pred_segs[s], gt_segs[s])

    # Mean across structures
    dice_vals = [v for k, v in results.items() if k.startswith("dice_")]
    hd95_vals = [v for k, v in results.items() if k.startswith("hd95_") and np.isfinite(v)]
    results["mDice"] = float(np.mean(dice_vals)) if dice_vals else 0.0
    results["HD95"]  = float(np.mean(hd95_vals)) if hd95_vals else float("nan")
    return results


# ── Full case evaluation ────────────────────────────────────────────────────────

def evaluate_case(
    pred_hu: np.ndarray,    # (D, H, W) — denormalised HU
    target_hu: np.ndarray,  # (D, H, W) — ground truth HU
    mask: np.ndarray = None,# (D, H, W) — dilated body mask
    compute_seg: bool = True,
) -> dict:
    """
    Compute all image-similarity metrics for one case.

    Returns dict with: mae, psnr, ms_ssim, mDice, HD95 (approx)
    """
    results = {
        "mae":      compute_mae(pred_hu, target_hu, mask),
        "psnr":     compute_psnr(pred_hu, target_hu, mask),
        "ms_ssim":  compute_ms_ssim(pred_hu, target_hu),
    }

    if compute_seg:
        seg = compute_segmentation_metrics(pred_hu, target_hu)
        results.update(seg)

    return results


def print_metrics(results: dict, prefix: str = ""):
    line = f"{prefix}  MAE={results['mae']:.2f}  PSNR={results['psnr']:.2f}dB  " \
           f"MS-SSIM={results['ms_ssim']:.4f}"
    if "mDice" in results:
        line += f"  mDice={results['mDice']:.4f}  HD95={results.get('HD95', float('nan')):.2f}"
    print(line)
