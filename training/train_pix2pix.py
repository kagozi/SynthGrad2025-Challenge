#!/usr/bin/env python3
"""
SynthRAD2025 Task 1 — pix2pix / Conditional GAN Training Script

Generator  : AttentionUNet2D  (existing, anatomy-conditioned)
Discriminator: NLayerDiscriminator (70×70 PatchGAN, anatomy-conditioned)

Per-batch:
  1. D step  — minimise 0.5*(BCE(D(mr,real_ct),1) + BCE(D(mr,pred_ct.detach()),0))
  2. G step  — minimise CombinedLoss(pred,ct,mask) + lambda_adv * BCE(D(mr,pred_ct),1)

Checkpoint saves generator under "model" key → compatible with predict_ensemble.py.

Usage:
    python training/train_pix2pix.py --config training/configs/pix2pix.yaml
    python training/train_pix2pix.py --config training/configs/pix2pix.yaml --fold 2
    python training/train_pix2pix.py --config training/configs/pix2pix.yaml \
        --resume /pvc/checkpoints/pix2pix/fold0_last.pth
"""

import argparse
import os
import random
import sys
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
    SynthRAD2DDataset,
    SynthRADInferenceDataset,
    CaseGroupedSampler,
    build_case_list,
    denormalise_ct,
)
from src.losses import CombinedLoss, GANLoss
from src.metrics import compute_mae, compute_psnr, compute_ms_ssim
from src.models.unet2d import AttentionUNet2D, UNet2D
from src.models.patchgan import NLayerDiscriminator

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


# ── Model factories ────────────────────────────────────────────────────────────

def build_generator(cfg: dict) -> nn.Module:
    gc = cfg["model"]["generator"]
    kwargs = dict(
        in_channels   = gc["in_channels"],
        out_channels  = gc["out_channels"],
        base_features = gc["base_features"],
        depth         = gc["depth"],
        n_anatomy     = gc["n_anatomy"],
        use_anatomy   = gc["use_anatomy"],
    )
    if gc["name"] == "attention_unet2d":
        return AttentionUNet2D(**kwargs)
    return UNet2D(**kwargs)


def build_discriminator(cfg: dict) -> nn.Module:
    dc = cfg["model"]["discriminator"]
    return NLayerDiscriminator(
        in_channels = dc["in_channels"],
        ndf         = dc["ndf"],
        n_layers    = dc["n_layers"],
        use_anatomy = dc["use_anatomy"],
        n_anatomy   = dc["n_anatomy"],
    )


# ── Optimiser / scheduler ──────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, oc: dict):
    if oc["name"].lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=oc["lr"],
            weight_decay=oc["weight_decay"],
            betas=tuple(oc["betas"]),
        )
    return torch.optim.Adam(model.parameters(), lr=oc["lr"],
                             betas=tuple(oc.get("betas", [0.5, 0.999])))


def build_scheduler(optimizer, sc: dict):
    name = sc["name"].lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=sc["T_max"], eta_min=sc["eta_min"])
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=sc.get("patience", 10),
            factor=sc.get("factor", 0.5), mode="min")
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=sc.get("step_size", 30),
            gamma=sc.get("gamma", 0.1))
    return None


# ── Image logging ──────────────────────────────────────────────────────────────

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
        gt_sl   = np.clip(gt_hu[s],  *CT_WIN)
        axes[row, 0].imshow(mr_norm[s], cmap="gray", vmin=0, vmax=1)
        axes[row, 1].imshow(pred_sl,    cmap="gray", vmin=CT_WIN[0], vmax=CT_WIN[1])
        axes[row, 2].imshow(gt_sl,      cmap="gray", vmin=CT_WIN[0], vmax=CT_WIN[1])
        im = axes[row, 3].imshow(np.abs(pred_hu[s] - gt_hu[s]), cmap="hot", vmin=0, vmax=200)
        plt.colorbar(im, ax=axes[row, 3], fraction=0.046, pad=0.04)
        axes[row, 0].set_ylabel(f"slice {s}/{D}", fontsize=9)

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(f"{case_id}  |  {anatomy}  |  MAE: {mae:.1f} HU", fontsize=13, y=1.01)
    plt.tight_layout()
    return fig


def log_sample_images(sample_volumes, epoch, n_slices=3):
    images = {}
    for anatomy, vol in sample_volumes.items():
        try:
            fig = _make_sample_figure(
                vol["mr_norm"], vol["pred_hu"], vol["gt_hu"],
                vol["case_id"], anatomy, vol["mae"], n_slices)
            images[f"val/samples/{anatomy}"] = wandb.Image(fig, caption=vol["case_id"])
            plt.close(fig)
        except Exception as e:
            print(f"  [WARN] Image log failed for {anatomy}: {e}")
    if images:
        wandb.log({**images, "epoch": epoch})


# ── Validation (generator only) ────────────────────────────────────────────────

@torch.no_grad()
def validate(G, val_datasets, device, epoch, n_context=0):
    G.eval()
    ANAT_IDX = {"HN": 0, "TH": 1, "AB": 2}
    per_anat, all_metrics, sample_volumes = {"HN": [], "TH": [], "AB": []}, [], {}

    for case, ds in tqdm(val_datasets, desc="  Validating", leave=False):
        anatomy = case["anatomy"]
        path    = Path(case["path"])
        if not (path / "ct.mha").exists():
            continue

        loader  = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
        anat_t  = torch.tensor([ANAT_IDX[anatomy]], dtype=torch.long, device=device)
        slices, mr_slices = [], []

        for batch in loader:
            mr  = batch["mr"].to(device, non_blocking=True)
            out = G(mr, anat_t.expand(mr.size(0)))
            slices.append(out.squeeze(1).cpu().numpy())
            mr_slices.append(batch["mr"][:, n_context].numpy())

        pred_hu = denormalise_ct(np.concatenate(slices, axis=0))
        mr_norm = np.concatenate(mr_slices, axis=0)
        gt_hu   = sitk.GetArrayFromImage(
            sitk.ReadImage(str(path / "ct.mha"))).astype(np.float32)

        mask = None
        if (path / "mask.mha").exists():
            mask = sitk.GetArrayFromImage(
                sitk.ReadImage(str(path / "mask.mha"))).astype(bool)

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
        if anatomy not in sample_volumes:
            sample_volumes[anatomy] = dict(
                mr_norm=mr_norm, pred_hu=pred_hu, gt_hu=gt_hu,
                case_id=case["case_id"], mae=m["mae"])

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
    log_sample_images(sample_volumes, epoch)

    mae = results.get("val/mae", float("inf"))
    print(f"  Val  MAE={mae:.2f}  PSNR={results.get('val/psnr',0):.2f}  "
          f"MS-SSIM={results.get('val/ms_ssim',0):.4f}  (n={len(all_metrics)} cases)")
    return results


# ── Training loop ──────────────────────────────────────────────────────────────

def train(cfg: dict, fold: int, resume: str = None):
    seed_everything(cfg["experiment"]["seed"])
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp  = (device.type == "cuda")
    print(f"[pix2pix] Device: {device}  |  Fold: {fold}")

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

    # ── Directories ────────────────────────────────────────────────────────────
    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ───────────────────────────────────────────────────────────────────
    dc        = cfg["data"]
    n_context = dc.get("n_context", 0)
    case_list = build_case_list(dc["data_dirs"])
    fold_df   = pd.read_csv(dc["folds_csv"])
    pad_to    = tuple(dc["pad_to"]) if dc.get("pad_to") else None

    train_ds = SynthRAD2DDataset(
        case_list=case_list, fold_df=fold_df, fold=fold, split="train",
        slice_axis=dc["slice_axis"], augment=True,
        skip_empty_slices=dc["skip_empty_slices"],
        empty_threshold=dc["empty_threshold"],
        pad_to=pad_to, n_context=n_context,
    )

    tc      = cfg["training"]
    sampler = CaseGroupedSampler(train_ds, shuffle=True)
    train_loader = DataLoader(
        train_ds, batch_size=tc["batch_size"], sampler=sampler,
        num_workers=dc["num_workers"], pin_memory=dc["pin_memory"],
        drop_last=True, persistent_workers=(dc["num_workers"] > 0),
    )

    val_case_ids = set(fold_df[fold_df["fold"] == fold]["case_id"])
    val_cases    = [c for c in case_list if c["case_id"] in val_case_ids]

    print("[pix2pix] Pre-loading validation datasets...")
    val_datasets = []
    for case in tqdm(val_cases, desc="  Loading val", leave=False):
        path = Path(case["path"])
        if not (path / "ct.mha").exists():
            continue
        ds = SynthRADInferenceDataset(path, case["anatomy"], n_context=n_context)
        val_datasets.append((case, ds))
    print(f"[pix2pix] Train slices: {len(train_ds)}  |  Val cases: {len(val_datasets)}")

    # ── Models ─────────────────────────────────────────────────────────────────
    G = build_generator(cfg).to(device)
    D = build_discriminator(cfg).to(device)

    gp = sum(p.numel() for p in G.parameters()) / 1e6
    dp = sum(p.numel() for p in D.parameters()) / 1e6
    print(f"[pix2pix] G: {cfg['model']['generator']['name']} ({gp:.1f}M)  "
          f"D: PatchGAN ({dp:.1f}M)")
    wandb.config.update({"G_params_M": round(gp, 1), "D_params_M": round(dp, 1)})

    # ── Optimisers ─────────────────────────────────────────────────────────────
    opt_G = build_optimizer(G, cfg["optimizer"]["generator"])
    opt_D = build_optimizer(D, cfg["optimizer"]["discriminator"])
    sched_G = build_scheduler(opt_G, cfg["scheduler"])
    sched_D = build_scheduler(opt_D, cfg["scheduler"])

    # ── Losses ─────────────────────────────────────────────────────────────────
    lc          = cfg["loss"]
    pixel_loss  = CombinedLoss(
        w_mae          = lc["w_mae"],
        w_ssim         = lc["w_ssim"],
        w_gdl          = lc["w_gdl"],
        ms_ssim_levels = lc["ms_ssim_levels"],
        bone_weight    = lc.get("bone_weight",    1.0),
        bone_threshold = lc.get("bone_threshold", -0.4),
    )
    gan_loss   = GANLoss().to(device)
    lambda_adv = lc.get("lambda_adv", 1.0)

    # ── AMP scalers (separate for G and D) ─────────────────────────────────────
    _GS = getattr(torch.amp, "GradScaler", torch.cuda.amp.GradScaler)
    scaler_G = _GS("cuda", enabled=use_amp)
    scaler_D = _GS("cuda", enabled=use_amp)

    start_epoch = 0
    best_mae    = float("inf")

    # ── Resume ─────────────────────────────────────────────────────────────────
    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        G.load_state_dict(ckpt["model"])
        D.load_state_dict(ckpt["discriminator"])
        opt_G.load_state_dict(ckpt["optimizer_G"])
        opt_D.load_state_dict(ckpt["optimizer_D"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_mae    = ckpt.get("best_mae", float("inf"))
        print(f"[pix2pix] Resumed from epoch {start_epoch}, best MAE={best_mae:.2f}")

    # ── Main loop ──────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, tc["epochs"]):
        G.train(); D.train()

        accum = {k: [] for k in ["L_G", "L_G_adv", "L_G_pix", "L_D"]}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{tc['epochs']}", leave=True)
        for batch in pbar:
            mr   = batch["mr"].to(device, non_blocking=True)
            ct   = batch["ct"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            ai   = batch["anatomy_idx"].to(device, non_blocking=True)

            # Discriminator input uses center MR channel only (ignore context slices)
            mr_1ch = mr[:, n_context : n_context + 1]  # (B, 1, H, W)

            # ── D step: G forward under no_grad ──────────────────────────────
            opt_D.zero_grad(set_to_none=True)
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred_d = G(mr, ai)                  # (B,1,H,W), detached implicitly

            with torch.amp.autocast("cuda", enabled=use_amp):
                real_pair = torch.cat([mr_1ch, ct],            dim=1)
                fake_pair = torch.cat([mr_1ch, pred_d.detach()], dim=1)
                L_D = 0.5 * (
                    gan_loss(D(real_pair, ai), True) +
                    gan_loss(D(fake_pair, ai), False)
                )

            scaler_D.scale(L_D).backward()
            scaler_D.unscale_(opt_D)
            nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
            scaler_D.step(opt_D)
            scaler_D.update()

            # ── G step: full forward with gradient ───────────────────────────
            opt_G.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_g    = G(mr, ai)
                fake_pair = torch.cat([mr_1ch, pred_g], dim=1)
                L_G_adv   = gan_loss(D(fake_pair, ai), True) * lambda_adv
                pix       = pixel_loss(pred_g, ct, mask)
                L_G       = pix["total"] + L_G_adv

            scaler_G.scale(L_G).backward()
            scaler_G.unscale_(opt_G)
            nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            scaler_G.step(opt_G)
            scaler_G.update()

            accum["L_G"].append(L_G.item())
            accum["L_G_adv"].append(L_G_adv.item())
            accum["L_G_pix"].append(pix["total"].item())
            accum["L_D"].append(L_D.item())

            pbar.set_postfix({
                "L_G": f"{L_G.item():.4f}",
                "L_D": f"{L_D.item():.4f}",
                "adv": f"{L_G_adv.item():.4f}",
            })

        # WandB logging
        log_dict = {
            "epoch": epoch,
            "lr_G":  opt_G.param_groups[0]["lr"],
            "lr_D":  opt_D.param_groups[0]["lr"],
        }
        for k, v in accum.items():
            if v:
                log_dict[f"train/{k}"] = float(np.mean(v))
        wandb.log(log_dict)

        # Scheduler step
        for sched in [sched_G, sched_D]:
            if sched and not isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sched.step()

        # ── Validation ─────────────────────────────────────────────────────────
        if (epoch + 1) % tc["val_every_n_epochs"] == 0 or epoch == tc["epochs"] - 1:
            torch.cuda.empty_cache()
            val_results = validate(G, val_datasets, device, epoch, n_context=n_context)
            val_mae     = val_results.get("val/mae", float("inf"))

            for sched in [sched_G, sched_D]:
                if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    sched.step(val_mae)

            if cfg["output"]["save_best"] and val_mae < best_mae:
                best_mae = val_mae
                ckpt_path = ckpt_dir / f"fold{fold}_best.pth"
                torch.save(_ckpt(G, D, opt_G, opt_D, epoch, best_mae, cfg), ckpt_path)
                print(f"  Saved best (MAE={best_mae:.2f}) → {ckpt_path}")
                wandb.run.summary["best_val_mae"] = best_mae

        # Periodic & last checkpoints
        if (epoch + 1) % tc["save_every_n_epochs"] == 0:
            torch.save(
                _ckpt(G, D, opt_G, opt_D, epoch, best_mae, cfg),
                ckpt_dir / f"fold{fold}_epoch{epoch+1:03d}.pth",
            )

        torch.save(
            _ckpt(G, D, opt_G, opt_D, epoch, best_mae, cfg),
            ckpt_dir / f"fold{fold}_last.pth",
        )

    wandb.finish()
    print(f"\n[pix2pix] Done. Best val MAE: {best_mae:.2f} HU")


def _ckpt(G, D, opt_G, opt_D, epoch, best_mae, cfg):
    """Checkpoint dict — generator under 'model' key for predict_ensemble compatibility."""
    return {
        "epoch":       epoch,
        "model":       G.state_dict(),          # key kept for predict_ensemble.py
        "discriminator": D.state_dict(),
        "optimizer_G": opt_G.state_dict(),
        "optimizer_D": opt_D.state_dict(),
        "best_mae":    best_mae,
        "config":      cfg,
    }


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
