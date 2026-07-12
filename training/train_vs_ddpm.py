#!/usr/bin/env python3
"""
SynthRAD2025 Task 1 — 3D VS-DDPM Training (Faking_it style)

Usage:
    python training/train_vs_ddpm.py --config training/configs/vs_ddpm_3d.yaml
    python training/train_vs_ddpm.py --config training/configs/vs_ddpm_3d.yaml --fold 2
    python training/train_vs_ddpm.py --config training/configs/vs_ddpm_3d.yaml \
        --resume /pvc/checkpoints/vs_ddpm_3d/fold0_best.pth

Architecture: 3D VS-DDPM with epsilon + learned variance prediction.
Patch size: (32, 128, 128). Loss: L_simple + 0.001*VLB + MAE + MS-SSIM.
Scheduler: CosineAnnealingLR. Optimiser: AdamW. AMP enabled.
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
import torch
import wandb
import yaml
from dotenv import load_dotenv
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import (
    SynthRAD3DDataset,
    build_case_list,
    denormalise_ct,
    normalise_mr,
    normalise_ct,
    load_mha,
    ANATOMY_TO_IDX,
)
from src.metrics import compute_mae, compute_ms_ssim
from src.models.vs_ddpm_3d import DDPMUNet3D, GaussianDiffusion3D

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


# ── Model + diffusion factory ──────────────────────────────────────────────────

def build_model_and_diffusion(cfg: dict) -> tuple:
    mc = cfg["model"]
    dc = cfg["diffusion"]

    unet = DDPMUNet3D(
        in_channels  = mc["in_channels"],
        out_channels = mc["out_channels"],
        base_ch      = mc["base_ch"],
        time_emb_dim = mc["time_emb_dim"],
        n_anatomy    = mc["n_anatomy"],
        dropout      = mc.get("dropout", 0.0),
    )
    diffusion = GaussianDiffusion3D(
        model        = unet,
        T            = dc["T"],
        s            = dc.get("s", 0.008),
        lambda_vlb   = dc.get("lambda_vlb",  0.001),
        lambda_mae   = dc.get("lambda_mae",  1.0),
        lambda_ssim  = dc.get("lambda_ssim", 1.0),
    )
    return unet, diffusion


# ── Data loaders ───────────────────────────────────────────────────────────────

def build_loaders(cfg: dict, fold: int):
    dc       = cfg["data"]
    cases    = build_case_list(dc["data_dirs"])
    folds_df = pd.read_csv(dc["folds_csv"])

    patch_size         = tuple(dc.get("patch_size", [32, 128, 128]))
    samples_per_volume = dc.get("samples_per_volume", 8)
    num_workers        = dc.get("num_workers", 4)
    pin_memory         = dc.get("pin_memory", False)

    train_ids = set(folds_df.loc[folds_df["fold"] != fold, "case_id"])
    val_ids   = set(folds_df.loc[folds_df["fold"] == fold,  "case_id"])

    train_cases = [c for c in cases if c["case_id"] in train_ids]
    val_cases   = [c for c in cases if c["case_id"] in val_ids]

    train_ds = SynthRAD3DDataset(
        train_cases,
        patch_size         = patch_size,
        samples_per_volume = samples_per_volume,
        augment            = True,
        split              = "train",
    )

    # Validation dataset (no augmentation, 1 sample per volume)
    val_ds = SynthRAD3DDataset(
        val_cases,
        patch_size         = patch_size,
        samples_per_volume = 1,
        augment            = False,
        split              = "val",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg["training"]["batch_size"],
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg["training"]["batch_size"],
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = False,
    )

    return train_loader, val_loader, val_cases


# ── Validation: patch-level DDIM on random val patches ────────────────────────

@torch.no_grad()
def validate(
    diffusion:    GaussianDiffusion3D,
    val_cases:    list,
    device:       torch.device,
    ddim_steps:   int,
    n_val_cases:  int,
    patch_size:   tuple,
):
    """
    For n_val_cases random val cases, extract ONE random patch,
    run DDIM and compute MAE / MS-SSIM in HU space.
    """
    diffusion.eval()
    all_mae, all_ssim = [], []

    rng   = np.random.default_rng(42)
    idxs  = rng.choice(len(val_cases), min(n_val_cases, len(val_cases)), replace=False)
    pd_, ph, pw = patch_size

    for ci in idxs:
        case    = val_cases[ci]
        anatomy = case["anatomy"]
        path    = Path(case["path"])

        try:
            mr_raw   = load_mha(path / "mr.mha")
            ct_raw   = load_mha(path / "ct.mha")
            mask_raw = load_mha(path / "mask.mha") if (path / "mask.mha").exists() \
                       else np.ones_like(mr_raw)
            mask_bool = (mask_raw > 0)

            mr_arr   = normalise_mr(mr_raw, anatomy, mask=mask_bool)
            ct_arr   = normalise_ct(ct_raw, mask=mask_bool)

            D, H, W = mr_arr.shape
            d0 = random.randint(0, max(0, D - pd_))
            h0 = random.randint(0, max(0, H - ph))
            w0 = random.randint(0, max(0, W - pw))

            mr_patch = mr_arr[d0:d0+pd_, h0:h0+ph, w0:w0+pw]
            ct_patch = ct_arr[d0:d0+pd_, h0:h0+ph, w0:w0+pw]

            # Pad if volume smaller than patch_size
            def _pad(a, pd_t, ph_t, pw_t):
                dd = max(0, pd_t - a.shape[0])
                hh = max(0, ph_t - a.shape[1])
                ww = max(0, pw_t - a.shape[2])
                return np.pad(a, ((0, dd), (0, hh), (0, ww)))

            mr_patch = _pad(mr_patch, pd_, ph, pw)
            ct_patch = _pad(ct_patch, pd_, ph, pw)

            mr_t = torch.from_numpy(mr_patch[None, None]).to(device)  # (1,1,D,H,W)
            ct_t = torch.from_numpy(ct_patch[None, None]).to(device)  # (1,1,D,H,W)
            anat = torch.tensor([ANATOMY_TO_IDX[anatomy]], device=device)

            pred = diffusion.ddim_sample(mr_t, anat, steps=ddim_steps)  # (1,1,D,H,W)

            pred_hu = denormalise_ct(pred[0, 0].cpu().numpy())
            ct_hu   = denormalise_ct(ct_t[0, 0].cpu().numpy())

            all_mae.append(compute_mae(pred_hu, ct_hu))
            all_ssim.append(compute_ms_ssim(pred_hu, ct_hu))

        except Exception as e:
            print(f"  [Val error] {case['case_id']}: {e}")
            continue

    diffusion.train()
    return (
        float(np.mean(all_mae))  if all_mae  else float("inf"),
        float(np.mean(all_ssim)) if all_ssim else 0.0,
    )


# ── Sample visualisation ───────────────────────────────────────────────────────

def make_sample_figure(mr, ct_gt, ct_pred, n=4):
    """
    Visualise centre axial slice from n 3D patches.
    mr, ct_gt, ct_pred: (B, 1, D, H, W) tensors.
    """
    n   = min(n, mr.shape[0])
    mid = mr.shape[2] // 2   # centre depth slice
    fig, axes = plt.subplots(3, n, figsize=(n * 3, 9))
    titles = ["MR", "CT (GT)", "sCT (VS-DDPM)"]
    for col in range(n):
        for row, (img, title) in enumerate(zip(
            [mr[col, 0, mid], ct_gt[col, 0, mid], ct_pred[col, 0, mid]], titles
        )):
            axes[row, col].imshow(img.cpu().numpy(), cmap="gray")
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(title, fontsize=9)
    plt.tight_layout()
    return fig


# ── Main training loop ─────────────────────────────────────────────────────────

def train(cfg: dict, fold: int, resume: str = None):
    seed_everything(cfg["experiment"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name = f"{cfg['experiment']['name']}_fold{fold}"

    wandb.init(
        project = os.getenv("WANDB_PROJECT", "synthrad2025-task1"),
        entity  = os.getenv("WANDB_ENTITY",  None),
        name    = exp_name,
        config  = cfg,
        resume  = "allow",
        id      = exp_name,
    )

    _, diffusion = build_model_and_diffusion(cfg)
    diffusion    = diffusion.to(device)

    n_params = sum(p.numel() for p in diffusion.model.parameters()) / 1e6
    print(f"DDPMUNet3D parameters: {n_params:.1f}M")

    train_loader, val_loader, val_cases = build_loaders(cfg, fold)

    oc        = cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr           = oc["lr"],
        weight_decay = oc.get("weight_decay", 1e-5),
        betas        = tuple(oc.get("betas", [0.9, 0.999])),
    )

    sc        = cfg["scheduler"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=sc["T_max"], eta_min=sc["eta_min"]
    )

    tc         = cfg["training"]
    epochs     = tc["epochs"]
    val_every  = tc.get("val_every_n_epochs",  5)
    save_every = tc.get("save_every_n_epochs", 20)
    val_steps  = tc.get("val_ddim_steps",      10)
    n_val      = tc.get("n_val_cases",          5)
    patch_size = tuple(cfg["data"].get("patch_size", [32, 128, 128]))

    ckpt_dir   = Path(cfg["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # AMP
    use_amp = (device.type == "cuda")
    scaler  = GradScaler(enabled=use_amp)

    # Try the newer torch.amp API first (PyTorch ≥ 2.4), fall back gracefully
    try:
        _autocast = torch.amp.autocast
    except AttributeError:
        _autocast = torch.cuda.amp.autocast

    start_epoch = 0
    best_mae    = float("inf")

    # ── Resume ────────────────────────────────────────────────────────────────
    if resume:
        ckpt = torch.load(resume, map_location=device)
        diffusion.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_mae    = ckpt.get("best_mae", float("inf"))
        print(f"Resumed from {resume} (epoch {start_epoch})")

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        diffusion.train()
        stats = {k: 0.0 for k in ("simple", "vlb", "mae", "ssim", "total")}
        n_steps = 0

        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{epochs}")
        for batch in pbar:
            mr       = batch["mr"].to(device)          # (B, 1, D, H, W)
            ct       = batch["ct"].to(device)          # (B, 1, D, H, W)
            mask     = batch["mask"].to(device)        # (B, 1, D, H, W)
            anat_idx = batch["anatomy_idx"].to(device) # (B,)

            # Sample random timesteps uniformly
            t = torch.randint(0, diffusion.T, (mr.shape[0],), device=device)

            optimizer.zero_grad()
            with _autocast("cuda", enabled=use_amp):
                loss_dict = diffusion.p_loss(ct, mr, t, anat_idx, mask)

            scaler.scale(loss_dict["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            for k in stats:
                stats[k] += loss_dict[k].item()
            n_steps += 1

            pbar.set_postfix(
                loss=f"{loss_dict['total'].item():.4f}",
                mae=f"{loss_dict['mae'].item():.4f}",
            )

        scheduler.step()

        avg = {k: v / max(n_steps, 1) for k, v in stats.items()}
        log_dict = {
            "train/loss":        avg["total"],
            "train/loss_simple": avg["simple"],
            "train/loss_vlb":    avg["vlb"],
            "train/loss_mae":    avg["mae"],
            "train/loss_ssim":   avg["ssim"],
            "epoch":             epoch + 1,
            "lr":                scheduler.get_last_lr()[0],
        }
        wandb.log(log_dict)

        # ── Periodic checkpoint ────────────────────────────────────────────────
        if (epoch + 1) % save_every == 0:
            torch.save({
                "epoch": epoch, "model": diffusion.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_mae": best_mae, "cfg": cfg,
            }, ckpt_dir / f"fold{fold}_ep{epoch+1}.pth")

        # ── Validation ────────────────────────────────────────────────────────
        if (epoch + 1) % val_every == 0:
            mae, ssim = validate(
                diffusion, val_cases, device,
                ddim_steps=val_steps,
                n_val_cases=n_val,
                patch_size=patch_size,
            )
            print(f"  Val — MAE: {mae:.2f} HU | MS-SSIM: {ssim:.4f}")
            wandb.log({"val/mae": mae, "val/ms_ssim": ssim, "epoch": epoch + 1})

            # Sample visualisation from val_loader (if any patches available)
            try:
                batch = next(iter(val_loader))
                mr_s  = batch["mr"][:4].to(device)
                ct_s  = batch["ct"][:4].to(device)
                ai_s  = batch["anatomy_idx"][:4].to(device)
                with torch.no_grad():
                    pred_s = diffusion.ddim_sample(mr_s, ai_s, steps=val_steps)
                fig = make_sample_figure(mr_s, ct_s, pred_s)
                wandb.log({"val/samples": wandb.Image(fig), "epoch": epoch + 1})
                plt.close(fig)
            except Exception:
                pass

            if mae < best_mae:
                best_mae = mae
                torch.save({
                    "epoch": epoch, "model": diffusion.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_mae": best_mae, "cfg": cfg,
                }, ckpt_dir / f"fold{fold}_best.pth")
                print(f"  New best MAE: {best_mae:.2f} HU — saved.")

    # ── Final checkpoint ──────────────────────────────────────────────────────
    torch.save({
        "epoch": epochs - 1, "model": diffusion.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_mae": best_mae, "cfg": cfg,
    }, ckpt_dir / f"fold{fold}_last.pth")

    wandb.finish()
    print(f"Training complete. Best val MAE: {best_mae:.2f} HU")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold",   type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg  = load_config(args.config)
    fold = args.fold if args.fold is not None else cfg["experiment"].get("fold", 0)

    print(f"Training VS-DDPM 3D  |  config: {args.config}  |  fold: {fold}")
    train(cfg, fold, args.resume)


if __name__ == "__main__":
    main()
