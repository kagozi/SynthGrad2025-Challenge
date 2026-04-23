#!/usr/bin/env python3
"""
SynthRAD2025 Task 1 — DynUNet 3D Training Script

Key differences vs. train_swin.py:
  - Model: DynUNETR3D (anisotropic kernels, no pretrained weights)
  - Deep supervision: auxiliary outputs at 2 lower resolutions during training
  - Auxiliary loss: bone-weighted MAE on downsampled CT/mask targets
  - No encoder freezing (trained fully from scratch)
  - Simpler single-group optimizer

Usage:
    python training/train_dynunet.py --config training/configs/dynunet.yaml
    python training/train_dynunet.py --config training/configs/dynunet.yaml --fold 2
    python training/train_dynunet.py --config training/configs/dynunet.yaml \
        --resume /pvc/checkpoints/dynunet/fold0_last.pth
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
import torch.nn.functional as F
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
from src.models.dynunet import DynUNETR3D

try:
    from monai.inferers import sliding_window_inference
    from monai.transforms import (
        Compose, RandFlipd, RandRotate90d, RandGaussianNoised,
        RandAdjustContrastd, RandGaussianSmoothd, RandScaleIntensityd,
        EnsureTyped,
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


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Loss ───────────────────────────────────────────────────────────────────────

class DynLoss(torch.nn.Module):
    """
    3D training loss with deep supervision support.

    Main output: bone-weighted MAE + GDL (full resolution)
    Auxiliary outputs: bone-weighted MAE only (lower resolution)
                       applied to bilinearly downsampled CT and mask targets.
    """

    def __init__(
        self,
        w_mae=1.0, w_gdl=0.5,
        bone_weight=2.5, bone_threshold=-0.4,
        aux_weights=None,
    ):
        super().__init__()
        self.w_mae       = w_mae
        self.w_gdl       = w_gdl
        self.aux_weights = aux_weights or [0.5, 0.25]
        self.mae         = MAELoss(bone_weight=bone_weight, bone_threshold=bone_threshold)
        self.gdl         = GradientDifferenceLoss()

    def _main_loss(self, pred, target, mask):
        losses = {}
        if self.w_mae > 0:
            losses["mae"] = self.w_mae * self.mae(pred, target, mask)
        if self.w_gdl > 0:
            losses["gdl"] = self.w_gdl * self.gdl(pred, target, mask)
        losses["total"] = sum(losses.values())
        return losses

    def forward(self, pred, target, mask, aux_preds=None):
        losses = self._main_loss(pred, target, mask)

        if aux_preds:
            aux_total = torch.zeros(1, device=pred.device)
            for w, aux in zip(self.aux_weights, aux_preds):
                # Downsample CT and mask to match auxiliary output resolution
                ct_ds   = F.interpolate(
                    target, size=aux.shape[2:], mode="trilinear", align_corners=False
                )
                mask_ds = F.interpolate(
                    mask.float(), size=aux.shape[2:], mode="nearest"
                )
                aux_total = aux_total + w * self.mae(aux, ct_ds, mask_ds)
            losses["aux"]   = aux_total.squeeze()
            losses["total"] = losses["total"] + losses["aux"]

        return losses


# ── Model factory ──────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> DynUNETR3D:
    mc = cfg["model"]
    return DynUNETR3D(
        in_channels      = mc["in_channels"],
        n_anatomy        = mc["n_anatomy"],
        use_anatomy      = mc["use_anatomy"],
        film_hidden      = mc["film_hidden"],
        deep_supervision = mc["deep_supervision"],
        deep_supr_num    = mc["deep_supr_num"],
        res_block        = mc["res_block"],
        filters          = mc.get("filters"),
        dropout          = mc.get("dropout", 0.0),
    )


# ── Augmentation ───────────────────────────────────────────────────────────────

def build_train_transforms():
    return Compose([
        RandFlipd(keys=["mr", "ct", "mask"], spatial_axis=0, prob=0.5),
        RandFlipd(keys=["mr", "ct", "mask"], spatial_axis=1, prob=0.5),
        RandFlipd(keys=["mr", "ct", "mask"], spatial_axis=2, prob=0.5),
        RandRotate90d(keys=["mr", "ct", "mask"], spatial_axes=(1, 2), prob=0.3),
        RandGaussianNoised(keys=["mr"], prob=0.3, mean=0.0, std=0.02),
        RandGaussianSmoothd(
            keys=["mr"], prob=0.2,
            sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.25, 0.5),
        ),
        RandScaleIntensityd(keys=["mr"], factors=0.15, prob=0.4),
        RandAdjustContrastd(keys=["mr"], gamma=(0.8, 1.2), prob=0.3),
        EnsureTyped(keys=["mr", "ct", "mask"], dtype=torch.float32),
    ])


class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transforms):
        self.ds   = base_dataset
        self.tfms = transforms

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item   = self.ds[idx]
        sample = {"mr": item["mr"], "ct": item["ct"], "mask": item["mask"]}
        sample = self.tfms(sample)
        sample["anatomy_idx"] = item["anatomy_idx"]
        sample["case_id"]     = item["case_id"]
        return sample


# ── Validation ─────────────────────────────────────────────────────────────────

def _predictor(x, model, anatomy_t):
    """Sliding window predictor closure — injects anatomy and forces eval path."""
    B  = x.shape[0]
    ai = anatomy_t.expand(B)
    return model(x, ai)   # eval mode → single output


@torch.no_grad()
def validate(model, val_cases, device, epoch, roi_size, sw_batch_size, overlap, mode):
    model.eval()
    per_anat       = {"HN": [], "TH": [], "AB": []}
    all_metrics    = []
    sample_volumes = {}

    for case in tqdm(val_cases, desc="  Validating", leave=False):
        anatomy = case["anatomy"]
        path    = Path(case["path"])
        if not (path / "ct.mha").exists():
            continue

        from src.dataset import load_mha, normalise_mr, normalise_ct
        mr_norm = normalise_mr(load_mha(path / "mr.mha"), anatomy)
        ct_raw  = sitk.GetArrayFromImage(
            sitk.ReadImage(str(path / "ct.mha"))
        ).astype(np.float32)
        mask_raw = load_mha(path / "mask.mha") if (path / "mask.mha").exists() else None
        mask     = (mask_raw > 0).astype(bool) if mask_raw is not None else None

        mr_t      = torch.from_numpy(mr_norm[None, None]).float().to(device)
        anatomy_t = torch.tensor([ANATOMY_TO_IDX[anatomy]], dtype=torch.long, device=device)

        pred_norm = sliding_window_inference(
            inputs        = mr_t,
            roi_size      = roi_size,
            sw_batch_size = sw_batch_size,
            predictor     = partial(_predictor, model=model, anatomy_t=anatomy_t),
            overlap       = overlap,
            mode          = mode,
        )
        pred_norm = pred_norm.squeeze().cpu().numpy()
        pred_hu   = np.clip(denormalise_ct(pred_norm), -1024, 3000)

        D = min(pred_hu.shape[0], ct_raw.shape[0])
        pred_hu, ct_raw, mask = pred_hu[:D], ct_raw[:D], (mask[:D] if mask is not None else None)

        m = {
            "mae":     compute_mae(pred_hu, ct_raw, mask),
            "psnr":    compute_psnr(pred_hu, ct_raw, mask),
            "ms_ssim": compute_ms_ssim(pred_hu, ct_raw),
        }
        per_anat[anatomy].append(m)
        all_metrics.append(m)

        if anatomy not in sample_volumes:
            sample_volumes[anatomy] = {
                "mr_norm": mr_norm[:D], "pred_hu": pred_hu,
                "gt_hu": ct_raw, "case_id": case["case_id"], "mae": m["mae"],
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


def _log_sample_images(sample_volumes, epoch, n_slices=3):
    CT_WIN = (-200, 800)
    images = {}
    for anatomy, vol in sample_volumes.items():
        try:
            mr_norm, pred_hu, gt_hu = vol["mr_norm"], vol["pred_hu"], vol["gt_hu"]
            D       = pred_hu.shape[0]
            indices = np.linspace(D * 0.15, D * 0.85, n_slices, dtype=int)
            fig, axes = plt.subplots(n_slices, 4, figsize=(16, 4 * n_slices))
            if n_slices == 1:
                axes = axes[np.newaxis, :]
            for col, title in enumerate(["MR", "Pred sCT", "GT CT", "|Error|"]):
                axes[0, col].set_title(title, fontsize=11, fontweight="bold")
            for row, s in enumerate(indices):
                axes[row, 0].imshow(mr_norm[s], cmap="gray", vmin=0, vmax=1)
                axes[row, 1].imshow(np.clip(pred_hu[s], *CT_WIN), cmap="gray",
                                    vmin=CT_WIN[0], vmax=CT_WIN[1])
                axes[row, 2].imshow(np.clip(gt_hu[s], *CT_WIN), cmap="gray",
                                    vmin=CT_WIN[0], vmax=CT_WIN[1])
                im = axes[row, 3].imshow(np.abs(pred_hu[s] - gt_hu[s]),
                                         cmap="hot", vmin=0, vmax=200)
                plt.colorbar(im, ax=axes[row, 3], fraction=0.046, pad=0.04)
                axes[row, 0].set_ylabel(f"slice {s}", fontsize=9)
            for ax in axes.ravel():
                ax.axis("off")
            fig.suptitle(f"{vol['case_id']}  {anatomy}  MAE={vol['mae']:.1f} HU",
                         fontsize=12, y=1.01)
            plt.tight_layout()
            images[f"val/samples/{anatomy}"] = wandb.Image(fig, caption=vol["case_id"])
            plt.close(fig)
        except Exception as e:
            print(f"  [WARN] Image logging failed for {anatomy}: {e}")
    if images:
        wandb.log({**images, "epoch": epoch})


# ── Training loop ──────────────────────────────────────────────────────────────

def train(cfg: dict, fold: int, resume: str = None):
    seed_everything(cfg["experiment"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}  |  Fold: {fold}")

    wandb.init(
        project = os.environ.get("WANDB_PROJECT", "synthrad2025-task1"),
        entity  = os.environ.get("WANDB_ENTITY",  None),
        name    = f"{cfg['experiment']['name']}_fold{fold}",
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

    patch_size = tuple(dc["patch_size"])

    train_ds_base = SynthRAD3DDataset(
        case_list          = case_list,
        fold_df            = fold_df,
        fold               = fold,
        split              = "train",
        patch_size         = patch_size,
        samples_per_volume = dc["samples_per_volume"],
        augment            = False,  # MONAI transforms handle augmentation
    )
    train_ds = TransformDataset(train_ds_base, build_train_transforms())

    val_ids   = set(fold_df[fold_df["fold"] == fold]["case_id"])
    val_cases = [c for c in case_list if c["case_id"] in val_ids
                 and (Path(c["path"]) / "ct.mha").exists()]

    tc = cfg["training"]
    train_loader = DataLoader(
        train_ds,
        batch_size         = tc["batch_size"],
        shuffle            = True,
        num_workers        = dc["num_workers"],
        pin_memory         = dc["pin_memory"],
        drop_last          = True,
        persistent_workers = (dc["num_workers"] > 0),
    )

    print(f"[Train] Train patches/epoch: {len(train_ds)}  |  Val cases: {len(val_cases)}")
    wandb.config.update({"train_patches": len(train_ds), "val_cases": len(val_cases)})

    # ── Model ──────────────────────────────────────────────────────────────────
    model    = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Train] Model: DynUNETR3D ({n_params:.1f}M params)")
    wandb.config.update({"model_params_M": round(n_params, 1)})

    oc        = cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = oc["lr"],
        weight_decay = oc["weight_decay"],
        betas        = tuple(oc["betas"]),
    )

    sc = cfg["scheduler"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=sc["T_max"], eta_min=sc["eta_min"]
    )

    lc = cfg["loss"]
    criterion = DynLoss(
        w_mae          = lc["w_mae"],
        w_gdl          = lc["w_gdl"],
        bone_weight    = lc["bone_weight"],
        bone_threshold = lc["bone_threshold"],
        aux_weights    = lc.get("aux_weights", [0.5, 0.25]),
    )

    _GradScaler = getattr(torch.amp, "GradScaler", torch.cuda.amp.GradScaler)
    scaler      = _GradScaler("cuda", enabled=(device.type == "cuda"))

    start_epoch = 0
    best_mae    = float("inf")

    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_mae    = ckpt.get("best_mae", float("inf"))
        print(f"[Train] Resumed from epoch {start_epoch}, best MAE={best_mae:.2f}")

    ic = cfg["inference"]
    roi_size      = tuple(ic["roi_size"])
    sw_batch_size = ic["sw_batch_size"]
    overlap       = ic["overlap"]
    sw_mode       = ic["mode"]

    # ── Main loop ──────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, tc["epochs"]):
        model.train()
        epoch_losses: dict[str, list] = {"total": [], "mae": [], "gdl": [], "aux": []}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{tc['epochs']}", leave=True)
        for batch in pbar:
            mr   = batch["mr"].to(device, non_blocking=True)
            ct   = batch["ct"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            ai   = batch["anatomy_idx"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(mr, ai)
                # During training with deep supervision → (main, [aux1, aux2])
                if isinstance(out, tuple):
                    pred, aux_preds = out
                else:
                    pred, aux_preds = out, None
                losses = criterion(pred, ct, mask, aux_preds)

            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            for k in ["total", "mae", "gdl", "aux"]:
                if k in losses:
                    epoch_losses[k].append(losses[k].item())

            pbar.set_postfix({
                "loss": f"{losses['total'].item():.4f}",
                "lr":   f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        log_dict = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        for k, v in epoch_losses.items():
            if v:
                log_dict[f"train/{k}"] = float(np.mean(v))
        wandb.log(log_dict)

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

            if cfg["output"]["save_best"] and val_mae < best_mae:
                best_mae  = val_mae
                ckpt_path = ckpt_dir / f"fold{fold}_best.pth"
                torch.save({
                    "epoch":    epoch,
                    "model":    model.state_dict(),
                    "optimizer":optimizer.state_dict(),
                    "best_mae": best_mae,
                    "config":   cfg,
                }, ckpt_path)
                print(f"  Saved best (MAE={best_mae:.2f}) → {ckpt_path}")
                wandb.run.summary["best_val_mae"] = best_mae

        if (epoch + 1) % tc["save_every_n_epochs"] == 0:
            torch.save({
                "epoch":    epoch,
                "model":    model.state_dict(),
                "optimizer":optimizer.state_dict(),
                "best_mae": best_mae,
                "config":   cfg,
            }, ckpt_dir / f"fold{fold}_epoch{epoch+1:03d}.pth")

        torch.save({
            "epoch":    epoch,
            "model":    model.state_dict(),
            "optimizer":optimizer.state_dict(),
            "best_mae": best_mae,
            "config":   cfg,
        }, ckpt_dir / f"fold{fold}_last.pth")

    wandb.finish()
    print(f"\n[Train] Done. Best val MAE: {best_mae:.2f} HU")


# ── Entry ──────────────────────────────────────────────────────────────────────

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
