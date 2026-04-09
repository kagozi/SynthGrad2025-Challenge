#!/usr/bin/env python3
"""
SynthRAD2025 Task 1 — Training Script (WandB logging, Nautilus/NRP ready)

Usage:
    python training/train.py --config training/configs/baseline_unet2d.yaml
    python training/train.py --config training/configs/baseline_unet2d.yaml --fold 2
    python training/train.py --config training/configs/baseline_unet2d.yaml \
        --resume /pvc/checkpoints/baseline_unet2d/fold0_last.pth

Environment variables:
    WANDB_API_KEY   — required for WandB logging
    WANDB_PROJECT   — optional override (default: synthrad2025-task1)
    WANDB_ENTITY    — optional WandB team/username
"""

import argparse
import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless backend for servers
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
    SynthRAD2DDataset,
    SynthRADInferenceDataset,
    CaseGroupedSampler,
    build_case_list,
    denormalise_ct,
)
from src.losses import CombinedLoss
from src.metrics import compute_mae, compute_psnr, compute_ms_ssim
from src.models.unet2d import UNet2D, AttentionUNet2D

load_dotenv()


# ── Reproducibility ────────────────────────────────────────────────────────────

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # benchmark=True lets cuDNN auto-tune kernels for the fixed input size —
    # measurably faster with negligible non-determinism impact.
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Model factory ──────────────────────────────────────────────────────────────

def build_model(cfg: dict) -> nn.Module:
    mc = cfg["model"]
    kwargs = dict(
        in_channels=mc["in_channels"],
        out_channels=mc["out_channels"],
        base_features=mc["base_features"],
        depth=mc["depth"],
        n_anatomy=mc["n_anatomy"],
        use_anatomy=mc["use_anatomy"],
    )
    if mc["name"] == "attention_unet2d":
        return AttentionUNet2D(**kwargs)
    return UNet2D(**kwargs)


# ── Optimiser / scheduler ──────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, cfg: dict):
    oc = cfg["optimizer"]
    if oc["name"].lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=oc["lr"],
            weight_decay=oc["weight_decay"],
            betas=tuple(oc["betas"]),
        )
    return torch.optim.Adam(model.parameters(), lr=oc["lr"])


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
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=sc.get("step_size", 30), gamma=sc.get("gamma", 0.1)
        )
    return None


# ── Image logging ──────────────────────────────────────────────────────────────

def _make_sample_figure(
    mr_norm: np.ndarray,    # (D, H, W) normalised MR [0, 1]
    pred_hu: np.ndarray,    # (D, H, W) predicted sCT in HU
    gt_hu:   np.ndarray,    # (D, H, W) ground truth CT in HU
    case_id: str,
    anatomy: str,
    slice_mae: float,
    n_slices: int = 3,
) -> plt.Figure:
    """
    Build a multi-row figure for one case.
    Rows = evenly spaced axial slices.
    Cols = [MR input | Predicted sCT | Ground Truth CT | |Error| map]
    """
    D         = pred_hu.shape[0]
    indices   = np.linspace(D * 0.15, D * 0.85, n_slices, dtype=int)  # skip edge air slices
    CT_WIN    = (-200, 800)   # soft-tissue window for display

    fig, axes = plt.subplots(n_slices, 4, figsize=(16, 4 * n_slices))
    if n_slices == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["MR Input", "Predicted sCT", "Ground Truth CT", "|Error| (HU)"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold")

    for row, s in enumerate(indices):
        mr_sl   = mr_norm[s]
        pred_sl = np.clip(pred_hu[s], CT_WIN[0], CT_WIN[1])
        gt_sl   = np.clip(gt_hu[s],   CT_WIN[0], CT_WIN[1])
        err_sl  = np.abs(pred_hu[s] - gt_hu[s])

        slice_label = f"slice {s}/{D}"

        axes[row, 0].imshow(mr_sl,   cmap="gray", vmin=0,           vmax=1)
        axes[row, 1].imshow(pred_sl, cmap="gray", vmin=CT_WIN[0],   vmax=CT_WIN[1])
        axes[row, 2].imshow(gt_sl,   cmap="gray", vmin=CT_WIN[0],   vmax=CT_WIN[1])
        im = axes[row, 3].imshow(err_sl, cmap="hot",  vmin=0,       vmax=200)
        plt.colorbar(im, ax=axes[row, 3], fraction=0.046, pad=0.04)

        axes[row, 0].set_ylabel(slice_label, fontsize=9)

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(
        f"{case_id}  |  Anatomy: {anatomy}  |  Slice MAE: {slice_mae:.1f} HU",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    return fig


def log_sample_images(
    sample_volumes: dict,   # anatomy → {mr_norm, pred_hu, gt_hu, case_id, mae}
    epoch: int,
    n_slices: int = 3,
):
    """Log one representative figure per anatomy to WandB."""
    images = {}
    for anatomy, vol in sample_volumes.items():
        try:
            fig = _make_sample_figure(
                mr_norm  = vol["mr_norm"],
                pred_hu  = vol["pred_hu"],
                gt_hu    = vol["gt_hu"],
                case_id  = vol["case_id"],
                anatomy  = anatomy,
                slice_mae= vol["mae"],
                n_slices = n_slices,
            )
            images[f"val/samples/{anatomy}"] = wandb.Image(fig, caption=vol["case_id"])
            plt.close(fig)
        except Exception as e:
            print(f"  [WARN] Image logging failed for {anatomy}: {e}")

    if images:
        wandb.log({**images, "epoch": epoch})


# ── Validation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: nn.Module,
    val_datasets: list,   # list of (case_meta, SynthRADInferenceDataset)
    device: torch.device,
    epoch: int,
    n_context: int = 0,
) -> dict:
    model.eval()
    ANAT_IDX = {"HN": 0, "TH": 1, "AB": 2}

    per_anat      = {"HN": [], "TH": [], "AB": []}
    all_metrics   = []
    sample_volumes = {}   # one case per anatomy for image logging

    for case, ds in tqdm(val_datasets, desc="  Validating", leave=False):
        anatomy = case["anatomy"]
        path    = Path(case["path"])
        ct_path = path / "ct.mha"
        if not ct_path.exists():
            continue

        # Reuse pre-built dataset — no NFS re-read
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0,
                            pin_memory=False)

        anat_t  = torch.tensor([ANAT_IDX[anatomy]], dtype=torch.long, device=device)
        slices  = []
        mr_slices = []
        for batch in loader:
            mr  = batch["mr"].to(device, non_blocking=True)
            out = model(mr, anat_t.expand(mr.size(0)))
            slices.append(out.squeeze(1).cpu().numpy())
            # For visualization keep the center MR channel only (index n_context)
            mr_slices.append(batch["mr"][:, n_context].numpy())

        pred_hu  = denormalise_ct(np.concatenate(slices, axis=0))
        mr_norm  = np.concatenate(mr_slices, axis=0)           # (D, H, W) in [0, 1]
        gt_hu    = sitk.GetArrayFromImage(sitk.ReadImage(str(ct_path))).astype(np.float32)

        mask_path = path / "mask.mha"
        mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path))).astype(bool) \
               if mask_path.exists() else None

        D = min(pred_hu.shape[0], gt_hu.shape[0])
        pred_hu, gt_hu, mr_norm = pred_hu[:D], gt_hu[:D], mr_norm[:D]
        if mask is not None:
            mask = mask[:D]

        m = {
            "mae":     compute_mae(pred_hu, gt_hu, mask),
            "psnr":    compute_psnr(pred_hu, gt_hu, mask),
            "ms_ssim": compute_ms_ssim(pred_hu, gt_hu),
        }
        per_anat[anatomy].append(m)
        all_metrics.append(m)

        # Keep the first seen case per anatomy as the image sample
        if anatomy not in sample_volumes:
            sample_volumes[anatomy] = {
                "mr_norm": mr_norm,
                "pred_hu": pred_hu,
                "gt_hu":   gt_hu,
                "case_id": case["case_id"],
                "mae":     m["mae"],
            }

    results = {}
    for anat, ms in per_anat.items():
        if not ms:
            continue
        results[f"val/{anat}/mae"]     = np.mean([x["mae"]     for x in ms])
        results[f"val/{anat}/psnr"]    = np.mean([x["psnr"]    for x in ms])
        results[f"val/{anat}/ms_ssim"] = np.mean([x["ms_ssim"] for x in ms])

    if all_metrics:
        results["val/mae"]     = np.mean([x["mae"]     for x in all_metrics])
        results["val/psnr"]    = np.mean([x["psnr"]    for x in all_metrics])
        results["val/ms_ssim"] = np.mean([x["ms_ssim"] for x in all_metrics])

    results["epoch"] = epoch
    wandb.log(results)

    # Log sample images (3 slices per anatomy)
    log_sample_images(sample_volumes, epoch, n_slices=3)

    mae = results.get("val/mae", float("inf"))
    print(f"  Val  MAE={mae:.2f}  PSNR={results.get('val/psnr',0):.2f}  "
          f"MS-SSIM={results.get('val/ms_ssim',0):.4f}  (n={len(all_metrics)} cases)")
    return results


# ── Training loop ──────────────────────────────────────────────────────────────

def train(cfg: dict, fold: int, resume: str = None):
    seed_everything(cfg["experiment"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}  |  Fold: {fold}")

    # ── WandB init ─────────────────────────────────────────────────────────────
    run_name = f"{cfg['experiment']['name']}_fold{fold}"
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "synthrad2025-task1"),
        entity=os.environ.get("WANDB_ENTITY",   None),
        name=run_name,
        config={**cfg, "fold": fold},
        resume="allow",
        id=wandb.util.generate_id() if not resume else None,
    )

    # ── Directories ────────────────────────────────────────────────────────────
    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ───────────────────────────────────────────────────────────────────
    dc        = cfg["data"]
    data_dirs = dc["data_dirs"]          # list of one or more root dirs
    folds_csv = Path(dc["folds_csv"])

    case_list = build_case_list(data_dirs)
    fold_df   = pd.read_csv(folds_csv) if folds_csv.exists() else None

    if fold_df is None:
        raise FileNotFoundError(
            f"folds.csv not found at {folds_csv}. "
            "Run: python scripts/prepare_folds.py"
        )

    pad_to    = tuple(dc["pad_to"]) if dc.get("pad_to") else None
    n_context = dc.get("n_context", 0)
    train_ds = SynthRAD2DDataset(
        case_list=case_list, fold_df=fold_df, fold=fold, split="train",
        slice_axis=dc["slice_axis"], augment=True,
        skip_empty_slices=dc["skip_empty_slices"],
        empty_threshold=dc["empty_threshold"],
        pad_to=pad_to,
        n_context=n_context,
    )
    val_case_ids = set(fold_df[fold_df["fold"] == fold]["case_id"])
    val_cases    = [c for c in case_list if c["case_id"] in val_case_ids]

    tc = cfg["training"]
    # CaseGroupedSampler: ensures all slices of a case arrive consecutively so
    # the per-worker NFS cache serves each volume with a single disk read per epoch.
    sampler = CaseGroupedSampler(train_ds, shuffle=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=tc["batch_size"],
        sampler=sampler,
        num_workers=dc["num_workers"],
        pin_memory=dc["pin_memory"],
        drop_last=True,
        persistent_workers=(dc["num_workers"] > 0),
    )

    # Pre-build val datasets once — avoids re-reading volumes from NFS every val run
    print("[Train] Pre-loading validation datasets...")
    val_datasets = []
    for case in tqdm(val_cases, desc="  Loading val", leave=False):
        path = Path(case["path"])
        if not (path / "ct.mha").exists():
            continue
        ds = SynthRADInferenceDataset(path, case["anatomy"], n_context=n_context)
        val_datasets.append((case, ds))
    print(f"[Train] Train slices: {len(train_ds)}  |  Val cases: {len(val_datasets)}")
    wandb.config.update({"train_slices": len(train_ds), "val_cases": len(val_datasets)})

    # ── Model ──────────────────────────────────────────────────────────────────
    model    = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Train] Model: {cfg['model']['name']} ({n_params:.1f}M params)")
    wandb.config.update({"model_params_M": round(n_params, 1)})

    # torch.compile requires a C compiler (Triton) in the container.
    # Skipped here — AMP + CaseGroupedSampler already give the major speedup.

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    lc        = cfg["loss"]
    criterion = CombinedLoss(
        w_mae          = lc["w_mae"],
        w_ssim         = lc["w_ssim"],
        w_gdl          = lc["w_gdl"],
        ms_ssim_levels = lc["ms_ssim_levels"],
        bone_weight    = lc.get("bone_weight",    1.0),
        bone_threshold = lc.get("bone_threshold", -0.4),
    )

    # AMP scaler — fp16 forward/backward; ~2-3x speedup on Ampere GPUs
    _GradScaler = getattr(torch.amp, "GradScaler", torch.cuda.amp.GradScaler)
    scaler = _GradScaler("cuda", enabled=(device.type == "cuda"))

    start_epoch = 0
    best_mae    = float("inf")

    # ── Resume ─────────────────────────────────────────────────────────────────
    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_mae    = ckpt.get("best_mae", float("inf"))
        print(f"[Train] Resumed from epoch {start_epoch}, best MAE={best_mae:.2f}")

    # ── Main loop ──────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, tc["epochs"]):
        model.train()
        epoch_losses = {"total": [], "mae": [], "ssim": []}

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

            for k in ["total", "mae", "ssim"]:
                if k in losses:
                    epoch_losses[k].append(losses[k].item())

            pbar.set_postfix({
                "loss": f"{losses['total'].item():.4f}",
                "lr":   f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        # Log to WandB
        log_dict = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
        for k, v in epoch_losses.items():
            if v:
                log_dict[f"train/{k}"] = float(np.mean(v))
        wandb.log(log_dict)

        # Scheduler step
        if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        # ── Validation ─────────────────────────────────────────────────────────
        if (epoch + 1) % tc["val_every_n_epochs"] == 0 or epoch == tc["epochs"] - 1:
            torch.cuda.empty_cache()
            val_results = validate(model, val_datasets, device, epoch, n_context=n_context)
            val_mae     = val_results.get("val/mae", float("inf"))

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_mae)

            if cfg["output"]["save_best"] and val_mae < best_mae:
                best_mae = val_mae
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

        # Periodic & last checkpoint
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
    parser.add_argument("--config",     type=str,  required=True)
    parser.add_argument("--fold",       type=int,  default=None,
                        help="Fold to hold out as validation (overrides config)")
    parser.add_argument("--resume",     type=str,  default=None)
    parser.add_argument("--data_dirs",  type=str,  nargs="+", default=None,
                        help="Override data.data_dirs from config")
    parser.add_argument("--output_dir", type=str,  default=None,
                        help="Override output.checkpoint_dir from config")
    args = parser.parse_args()

    cfg  = load_config(args.config)
    fold = args.fold if args.fold is not None else cfg["experiment"]["fold"]

    if args.data_dirs:
        cfg["data"]["data_dirs"] = args.data_dirs
    if args.output_dir:
        cfg["output"]["checkpoint_dir"] = args.output_dir

    train(cfg, fold=fold, resume=args.resume)
