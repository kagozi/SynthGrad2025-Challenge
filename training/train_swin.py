#!/usr/bin/env python3
"""
SynthRAD2025 Task 1 — Swin-UNETR 3D Training Script

Training:
  - MONAI SwinUNETR with SSL pretrained encoder (feature_size=48)
  - 3D random patch crops (default 64×128×128)
  - Differential LR: encoder×0.1 for first N epochs, then full fine-tune
  - Loss: bone-weighted MAE + Gradient Difference Loss (3D)
  - AMP (fp16) + gradient clipping
  - WandB logging with sample volume visualisation

Validation:
  - Sliding window inference (MONAI) on full volumes
  - MAE / PSNR / MS-SSIM per anatomy + overall

Usage:
    python training/train_swin.py --config training/configs/swin_unetr.yaml
    python training/train_swin.py --config training/configs/swin_unetr.yaml --fold 2
    python training/train_swin.py --config training/configs/swin_unetr.yaml \
        --resume /pvc/checkpoints/swin_unetr/fold0_last.pth
"""

import argparse
import os
import random
import sys
from functools import partial
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn as nn
import wandb
import yaml
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import (
    SynthRAD3DDataset,
    build_case_list,
    denormalise_ct,
    normalise_mr,
    ANATOMY_TO_IDX,
)
from src.losses import MAELoss, GradientDifferenceLoss
from src.metrics import compute_mae, compute_psnr, compute_ms_ssim
from src.models.swin_unetr import SwinUNETR3D

try:
    from monai.inferers import sliding_window_inference
    from monai.transforms import (
        Compose, RandFlipd, RandRotate90d, RandGaussianNoised,
        RandAdjustContrastd, RandGaussianSmoothd, RandScaleIntensityd,
        NormalizeIntensityd, EnsureTyped,
    )
except ImportError as e:
    raise ImportError(
        "MONAI is required. Install with: pip install 'monai[all]>=1.3'"
    ) from e

load_dotenv()


# ── Reproducibility ────────────────────────────────────────────────────────────

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── 3D Augmentation pipeline ───────────────────────────────────────────────────

def build_train_transforms():
    """
    MONAI dictionary transforms applied to (mr, ct, mask) 3D patches.
    Applied after random spatial cropping in the dataset.

    MR-only transforms: intensity jitter, Gaussian noise/smooth
    Shared spatial transforms: random flips, random 90° rotations
    CT: no intensity augmentation (HU values must be preserved)
    """
    return Compose([
        # Spatial — applied identically to MR, CT, mask
        RandFlipd(keys=["mr", "ct", "mask"], spatial_axis=0, prob=0.5),
        RandFlipd(keys=["mr", "ct", "mask"], spatial_axis=1, prob=0.5),
        RandFlipd(keys=["mr", "ct", "mask"], spatial_axis=2, prob=0.5),
        RandRotate90d(keys=["mr", "ct", "mask"], spatial_axes=(1, 2), prob=0.3),
        # MR intensity augmentation only
        RandGaussianNoised(keys=["mr"], prob=0.3, mean=0.0, std=0.02),
        RandGaussianSmoothd(
            keys=["mr"], prob=0.2,
            sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.25, 0.5),
        ),
        RandScaleIntensityd(keys=["mr"], factors=0.15, prob=0.4),
        RandAdjustContrastd(keys=["mr"], gamma=(0.8, 1.2), prob=0.3),
        EnsureTyped(keys=["mr", "ct", "mask"], dtype=torch.float32),
    ])


# ── Loss ───────────────────────────────────────────────────────────────────────

class SwinLoss(nn.Module):
    """
    3D training loss: bone-weighted MAE + Gradient Difference Loss.
    Both losses are intrinsically 3D-compatible.
    """

    def __init__(self, w_mae=1.0, w_gdl=0.5, bone_weight=2.5, bone_threshold=-0.4):
        super().__init__()
        self.w_mae = w_mae
        self.w_gdl = w_gdl
        self.mae   = MAELoss(bone_weight=bone_weight, bone_threshold=bone_threshold)
        self.gdl   = GradientDifferenceLoss()

    def forward(self, pred, target, mask=None):
        losses = {}
        if self.w_mae > 0:
            losses["mae"] = self.w_mae * self.mae(pred, target, mask)
        if self.w_gdl > 0:
            losses["gdl"] = self.w_gdl * self.gdl(pred, target, mask)
        losses["total"] = sum(losses.values())
        return losses


# ── Model factory ──────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> SwinUNETR3D:
    mc = cfg["model"]
    return SwinUNETR3D(
        img_size             = tuple(mc["img_size"]),
        in_channels          = mc["in_channels"],
        feature_size         = mc["feature_size"],
        use_checkpoint       = mc["use_checkpoint"],
        n_anatomy            = mc["n_anatomy"],
        use_anatomy          = mc["use_anatomy"],
        film_hidden          = mc["film_hidden"],
        pretrained           = mc["pretrained"],
        pretrained_cache_dir = mc["pretrained_cache_dir"],
        drop_rate            = mc.get("drop_rate", 0.0),
        attn_drop_rate       = mc.get("attn_drop_rate", 0.0),
        dropout_path_rate    = mc.get("dropout_path_rate", 0.1),
    )


# ── Optimiser with differential LR ────────────────────────────────────────────

def build_optimizer(model: SwinUNETR3D, cfg: dict, freeze_encoder: bool = True):
    """
    Two param groups:
      - encoder (swinViT): lr × encoder_lr_scale (initially frozen when freeze_encoder=True)
      - decoder + FiLM head: full lr

    freeze_encoder=True: sets encoder requires_grad=False (saves memory + prevents
    destabilising pretrained weights during warm-up epochs).
    """
    oc            = cfg["optimizer"]
    base_lr       = oc["lr"]
    enc_lr_scale  = oc.get("encoder_lr_scale", 0.1)

    if freeze_encoder:
        for p in model.encoder_parameters():
            p.requires_grad_(False)
        params = [{"params": model.decoder_parameters(), "lr": base_lr, "name": "decoder"}]
        print("[Train] Encoder FROZEN for warm-up epochs.")
    else:
        params = [
            {"params": list(model.encoder_parameters()), "lr": base_lr * enc_lr_scale, "name": "encoder"},
            {"params": model.decoder_parameters(),        "lr": base_lr,                "name": "decoder"},
        ]

    return torch.optim.AdamW(
        params,
        weight_decay = oc["weight_decay"],
        betas        = tuple(oc["betas"]),
    )


def unfreeze_encoder(model: SwinUNETR3D, optimizer: torch.optim.Optimizer, cfg: dict):
    """Unfreeze encoder and add its param group to optimizer."""
    oc           = cfg["optimizer"]
    enc_lr       = oc["lr"] * oc.get("encoder_lr_scale", 0.1)
    enc_params   = [p for p in model.encoder_parameters() if not p.requires_grad]
    if not enc_params:
        return  # already unfrozen
    for p in model.encoder_parameters():
        p.requires_grad_(True)
    optimizer.add_param_group({"params": enc_params, "lr": enc_lr, "name": "encoder"})
    print(f"[Train] Encoder UNFROZEN. Encoder LR: {enc_lr:.2e}")


def build_scheduler(optimizer, cfg: dict):
    sc   = cfg["scheduler"]
    name = sc["name"].lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=sc["T_max"], eta_min=sc["eta_min"]
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=sc.get("patience", 10), factor=sc.get("factor", 0.5),
            mode="min",
        )
    return None


# ── Validation ─────────────────────────────────────────────────────────────────

def _load_volume(case: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load MR, CT, mask for a single case. Returns normalised arrays."""
    path    = Path(case["path"])
    anatomy = case["anatomy"]
    from src.dataset import load_mha, normalise_mr, normalise_ct
    mr_raw   = load_mha(path / "mr.mha")
    ct_raw   = load_mha(path / "ct.mha")
    mask_raw = load_mha(path / "mask.mha") if (path / "mask.mha").exists() else None
    return (
        normalise_mr(mr_raw, anatomy),   # [0, 1]
        normalise_ct(ct_raw),            # [-1, 1]
        (mask_raw > 0).astype(bool) if mask_raw is not None else None,
    )


@torch.no_grad()
def validate(
    model:        SwinUNETR3D,
    val_cases:    list,
    device:       torch.device,
    epoch:        int,
    roi_size:     tuple,
    sw_batch_size: int,
    overlap:      float,
    mode:         str = "gaussian",
) -> dict:
    model.eval()
    per_anat      = {"HN": [], "TH": [], "AB": []}
    all_metrics   = []
    sample_volumes = {}

    for case in tqdm(val_cases, desc="  Validating", leave=False):
        anatomy = case["anatomy"]
        path    = Path(case["path"])

        if not (path / "ct.mha").exists():
            continue

        mr_norm, _, mask = _load_volume(case)
        # We don't need norm CT here — we reload raw CT for HU comparison
        ct_raw = sitk.GetArrayFromImage(
            sitk.ReadImage(str(path / "ct.mha"))
        ).astype(np.float32)

        # Prepare input tensor: (1, 1, D, H, W)
        mr_t       = torch.from_numpy(mr_norm[None, None]).float().to(device)
        anatomy_t  = torch.tensor([ANATOMY_TO_IDX[anatomy]], dtype=torch.long, device=device)

        # Sliding window inference with anatomy conditioning
        predictor = partial(_predictor, model=model, anatomy_t=anatomy_t)
        pred_norm = sliding_window_inference(
            inputs        = mr_t,
            roi_size      = roi_size,
            sw_batch_size = sw_batch_size,
            predictor     = predictor,
            overlap       = overlap,
            mode          = mode,
        )
        # pred_norm: (1, 1, D, H, W) → (D, H, W)
        pred_norm  = pred_norm.squeeze().cpu().numpy()
        pred_hu    = denormalise_ct(pred_norm)
        pred_hu    = np.clip(pred_hu, -1024, 3000)

        D = min(pred_hu.shape[0], ct_raw.shape[0])
        pred_hu, ct_raw_d = pred_hu[:D], ct_raw[:D]
        mask_d = mask[:D] if mask is not None else None

        m = {
            "mae":     compute_mae(pred_hu, ct_raw_d, mask_d),
            "psnr":    compute_psnr(pred_hu, ct_raw_d, mask_d),
            "ms_ssim": compute_ms_ssim(pred_hu, ct_raw_d),
        }
        per_anat[anatomy].append(m)
        all_metrics.append(m)

        if anatomy not in sample_volumes:
            sample_volumes[anatomy] = {
                "mr_norm": mr_norm[:D],
                "pred_hu": pred_hu,
                "gt_hu":   ct_raw_d,
                "case_id": case["case_id"],
                "mae":     m["mae"],
            }

    results = {}
    for anat, ms in per_anat.items():
        if not ms:
            continue
        results[f"val/{anat}/mae"]     = float(np.mean([x["mae"]     for x in ms]))
        results[f"val/{anat}/psnr"]    = float(np.mean([x["psnr"]    for x in ms]))
        results[f"val/{anat}/ms_ssim"] = float(np.mean([x["ms_ssim"] for x in ms]))

    if all_metrics:
        results["val/mae"]     = float(np.mean([x["mae"]     for x in all_metrics]))
        results["val/psnr"]    = float(np.mean([x["psnr"]    for x in all_metrics]))
        results["val/ms_ssim"] = float(np.mean([x["ms_ssim"] for x in all_metrics]))

    results["epoch"] = epoch
    wandb.log(results)
    _log_sample_images(sample_volumes, epoch)

    mae = results.get("val/mae", float("inf"))
    print(
        f"  Val  MAE={mae:.2f}  PSNR={results.get('val/psnr',0):.2f}  "
        f"MS-SSIM={results.get('val/ms_ssim',0):.4f}  (n={len(all_metrics)} cases)"
    )
    return results


def _predictor(x: torch.Tensor, model: SwinUNETR3D, anatomy_t: torch.Tensor) -> torch.Tensor:
    """Closure for sliding_window_inference — injects anatomy conditioning."""
    B  = x.shape[0]
    ai = anatomy_t.expand(B)
    return model(x, ai)


# ── Sample image logging ───────────────────────────────────────────────────────

def _make_sample_figure(mr_norm, pred_hu, gt_hu, case_id, anatomy, mae, n_slices=3):
    D       = pred_hu.shape[0]
    indices = np.linspace(D * 0.15, D * 0.85, n_slices, dtype=int)
    CT_WIN  = (-200, 800)

    fig, axes = plt.subplots(n_slices, 4, figsize=(16, 4 * n_slices))
    if n_slices == 1:
        axes = axes[np.newaxis, :]

    for col, title in enumerate(["MR Input", "Predicted sCT", "Ground Truth CT", "|Error| (HU)"]):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold")

    for row, s in enumerate(indices):
        pred_sl = np.clip(pred_hu[s], *CT_WIN)
        gt_sl   = np.clip(gt_hu[s],   *CT_WIN)
        err_sl  = np.abs(pred_hu[s] - gt_hu[s])
        axes[row, 0].imshow(mr_norm[s], cmap="gray", vmin=0,        vmax=1)
        axes[row, 1].imshow(pred_sl,    cmap="gray", vmin=CT_WIN[0], vmax=CT_WIN[1])
        axes[row, 2].imshow(gt_sl,      cmap="gray", vmin=CT_WIN[0], vmax=CT_WIN[1])
        im = axes[row, 3].imshow(err_sl, cmap="hot", vmin=0,        vmax=200)
        plt.colorbar(im, ax=axes[row, 3], fraction=0.046, pad=0.04)
        axes[row, 0].set_ylabel(f"slice {s}/{D}", fontsize=9)

    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"{case_id}  |  {anatomy}  |  MAE={mae:.1f} HU", fontsize=13, y=1.01)
    plt.tight_layout()
    return fig


def _log_sample_images(sample_volumes: dict, epoch: int, n_slices: int = 3):
    images = {}
    for anatomy, vol in sample_volumes.items():
        try:
            fig = _make_sample_figure(
                mr_norm = vol["mr_norm"],
                pred_hu = vol["pred_hu"],
                gt_hu   = vol["gt_hu"],
                case_id = vol["case_id"],
                anatomy = anatomy,
                mae     = vol["mae"],
                n_slices = n_slices,
            )
            images[f"val/samples/{anatomy}"] = wandb.Image(fig, caption=vol["case_id"])
            plt.close(fig)
        except Exception as e:
            print(f"  [WARN] Image logging failed for {anatomy}: {e}")
    if images:
        wandb.log({**images, "epoch": epoch})


# ── Collate with MONAI transforms ─────────────────────────────────────────────

class TransformDataset(torch.utils.data.Dataset):
    """Wraps SynthRAD3DDataset and applies MONAI dict transforms on-the-fly."""

    def __init__(self, base_dataset: SynthRAD3DDataset, transforms):
        self.ds    = base_dataset
        self.tfms  = transforms

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        # Convert to dict with spatial dims for MONAI (C, D, H, W)
        sample = {
            "mr":   item["mr"],    # (1, D, H, W)
            "ct":   item["ct"],    # (1, D, H, W)
            "mask": item["mask"],  # (1, D, H, W)
        }
        sample = self.tfms(sample)
        sample["anatomy_idx"] = item["anatomy_idx"]
        sample["case_id"]     = item["case_id"]
        return sample


# ── Training loop ──────────────────────────────────────────────────────────────

def train(cfg: dict, fold: int, resume: str = None):
    seed_everything(cfg["experiment"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}  |  Fold: {fold}")

    # ── WandB ──────────────────────────────────────────────────────────────────
    run_name = f"{cfg['experiment']['name']}_fold{fold}"
    wandb.init(
        project = os.environ.get("WANDB_PROJECT", "synthrad2025-task1"),
        entity  = os.environ.get("WANDB_ENTITY",  None),
        name    = run_name,
        config  = {**cfg, "fold": fold},
        resume  = "allow",
        id      = wandb.util.generate_id() if not resume else None,
    )

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ───────────────────────────────────────────────────────────────────
    dc        = cfg["data"]
    case_list = build_case_list(dc["data_dirs"])
    fold_df   = pd.read_csv(dc["folds_csv"])

    patch_size        = tuple(dc["patch_size"])
    samples_per_vol   = dc["samples_per_volume"]

    train_ds_base = SynthRAD3DDataset(
        case_list        = case_list,
        fold_df          = fold_df,
        fold             = fold,
        split            = "train",
        patch_size       = patch_size,
        samples_per_volume = samples_per_vol,
        augment          = False,     # augmentation done by MONAI transforms below
    )
    train_ds = TransformDataset(train_ds_base, build_train_transforms())

    val_ids   = set(fold_df[fold_df["fold"] == fold]["case_id"])
    val_cases = [c for c in case_list if c["case_id"] in val_ids
                 and (Path(c["path"]) / "ct.mha").exists()]

    tc = cfg["training"]
    train_loader = DataLoader(
        train_ds,
        batch_size  = tc["batch_size"],
        shuffle     = True,
        num_workers = dc["num_workers"],
        pin_memory  = dc["pin_memory"],
        drop_last   = True,
        persistent_workers = (dc["num_workers"] > 0),
    )

    print(f"[Train] Train patches/epoch: {len(train_ds)}  |  Val cases: {len(val_cases)}")
    wandb.config.update({"train_patches": len(train_ds), "val_cases": len(val_cases)})

    # ── Model ──────────────────────────────────────────────────────────────────
    model       = build_model(cfg).to(device)
    n_params    = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Train] Model: SwinUNETR3D ({n_params:.1f}M params)")
    wandb.config.update({"model_params_M": round(n_params, 1)})

    # Freeze encoder for first `unfreeze_encoder_epoch` epochs
    unfreeze_at  = tc.get("unfreeze_encoder_epoch", 10)
    freeze_init  = unfreeze_at > 0 and cfg["model"].get("pretrained", False)
    optimizer    = build_optimizer(model, cfg, freeze_encoder=freeze_init)
    scheduler    = build_scheduler(optimizer, cfg)

    lc = cfg["loss"]
    criterion = SwinLoss(
        w_mae          = lc["w_mae"],
        w_gdl          = lc["w_gdl"],
        bone_weight    = lc.get("bone_weight",    1.0),
        bone_threshold = lc.get("bone_threshold", -0.4),
    )

    _GradScaler = getattr(torch.amp, "GradScaler", torch.cuda.amp.GradScaler)
    scaler      = _GradScaler("cuda", enabled=(device.type == "cuda"))

    start_epoch = 0
    best_mae    = float("inf")
    encoder_unfrozen = not freeze_init

    # ── Resume ─────────────────────────────────────────────────────────────────
    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch      = ckpt.get("epoch", 0) + 1
        best_mae         = ckpt.get("best_mae", float("inf"))
        encoder_unfrozen = ckpt.get("encoder_unfrozen", not freeze_init)
        print(f"[Train] Resumed from epoch {start_epoch}, best MAE={best_mae:.2f}")

    # ── Inference config ───────────────────────────────────────────────────────
    ic            = cfg["inference"]
    roi_size      = tuple(ic["roi_size"])
    sw_batch_size = ic["sw_batch_size"]
    overlap       = ic["overlap"]
    sw_mode       = ic["mode"]

    # ── Main loop ──────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, tc["epochs"]):

        # Unfreeze encoder at the scheduled epoch
        if not encoder_unfrozen and epoch >= unfreeze_at:
            unfreeze_encoder(model, optimizer, cfg)
            encoder_unfrozen = True

        model.train()
        epoch_losses: dict[str, list] = {"total": [], "mae": [], "gdl": []}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{tc['epochs']}", leave=True)
        for batch in pbar:
            mr   = batch["mr"].to(device, non_blocking=True)
            ct   = batch["ct"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            ai   = batch["anatomy_idx"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred   = model(mr, ai)
                losses = criterion(pred, ct, mask)

            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            for k in ["total", "mae", "gdl"]:
                if k in losses:
                    epoch_losses[k].append(losses[k].item())

            pbar.set_postfix({
                "loss": f"{losses['total'].item():.4f}",
                "lr":   f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        # Log training metrics
        log_dict = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        for k, v in epoch_losses.items():
            if v:
                log_dict[f"train/{k}"] = float(np.mean(v))
        log_dict["encoder_frozen"] = int(not encoder_unfrozen)
        wandb.log(log_dict)

        if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        # ── Validation ─────────────────────────────────────────────────────────
        if (epoch + 1) % tc["val_every_n_epochs"] == 0 or epoch == tc["epochs"] - 1:
            torch.cuda.empty_cache()
            val_results = validate(
                model         = model,
                val_cases     = val_cases,
                device        = device,
                epoch         = epoch,
                roi_size      = roi_size,
                sw_batch_size = sw_batch_size,
                overlap       = overlap,
                mode          = sw_mode,
            )
            val_mae = val_results.get("val/mae", float("inf"))

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_mae)

            if cfg["output"]["save_best"] and val_mae < best_mae:
                best_mae  = val_mae
                ckpt_path = ckpt_dir / f"fold{fold}_best.pth"
                torch.save({
                    "epoch":            epoch,
                    "model":            model.state_dict(),
                    "optimizer":        optimizer.state_dict(),
                    "best_mae":         best_mae,
                    "config":           cfg,
                    "encoder_unfrozen": encoder_unfrozen,
                }, ckpt_path)
                print(f"  Saved best (MAE={best_mae:.2f}) → {ckpt_path}")
                wandb.run.summary["best_val_mae"] = best_mae

        # Periodic & last checkpoint
        if (epoch + 1) % tc["save_every_n_epochs"] == 0:
            torch.save({
                "epoch":            epoch,
                "model":            model.state_dict(),
                "optimizer":        optimizer.state_dict(),
                "best_mae":         best_mae,
                "config":           cfg,
                "encoder_unfrozen": encoder_unfrozen,
            }, ckpt_dir / f"fold{fold}_epoch{epoch+1:03d}.pth")

        torch.save({
            "epoch":            epoch,
            "model":            model.state_dict(),
            "optimizer":        optimizer.state_dict(),
            "best_mae":         best_mae,
            "config":           cfg,
            "encoder_unfrozen": encoder_unfrozen,
        }, ckpt_dir / f"fold{fold}_last.pth")

    wandb.finish()
    print(f"\n[Train] Done. Best val MAE: {best_mae:.2f} HU")


# ── Entry -──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--fold",       type=int, default=None)
    parser.add_argument("--resume",     type=str, default=None)
    parser.add_argument("--data_dirs",  type=str, nargs="+", default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg  = load_config(args.config)
    fold = args.fold if args.fold is not None else cfg["experiment"]["fold"]

    if args.data_dirs:
        cfg["data"]["data_dirs"] = args.data_dirs
    if args.output_dir:
        cfg["output"]["checkpoint_dir"] = args.output_dir

    train(cfg, fold=fold, resume=args.resume)
