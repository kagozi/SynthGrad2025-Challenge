#!/usr/bin/env python3
"""
SynthRAD2025 Task 1 — 2D Conditional DDPM Training

Usage:
    python training/train_ddpm.py --config training/configs/ddpm.yaml
    python training/train_ddpm.py --config training/configs/ddpm.yaml --fold 2
    python training/train_ddpm.py --config training/configs/ddpm.yaml \
        --resume /pvc/checkpoints/ddpm/fold0_last.pth

Architecture: epsilon-prediction DDPM with cosine noise schedule.
Conditioning: MR slice concatenated channel-wise with noisy CT.
Anatomy conditioning: learned embedding summed into sinusoidal time embedding.
Validation: quick DDIM sampling (10 steps) on a subset of val slices.
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
import torch
import wandb
import yaml
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import (
    SynthRAD2DDataset,
    CaseGroupedSampler,
    build_case_list,
    denormalise_ct,
)
from src.metrics import compute_mae, compute_psnr, compute_ms_ssim
from src.models.ddpm import DDPMUNet, GaussianDiffusion

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

def build_model_and_diffusion(cfg: dict) -> tuple[DDPMUNet, GaussianDiffusion]:
    mc = cfg["model"]
    dc = cfg["diffusion"]

    unet = DDPMUNet(
        in_channels  = mc["in_channels"],
        out_channels = mc["out_channels"],
        base_ch      = mc["base_ch"],
        ch_mult      = tuple(mc["ch_mult"]),
        num_res      = mc["num_res"],
        attn_levels  = tuple(mc["attn_levels"]),
        time_emb_dim = mc["time_emb_dim"],
        n_anatomy    = mc["n_anatomy"],
        dropout      = mc.get("dropout", 0.0),
    )
    diffusion = GaussianDiffusion(
        model = unet,
        T     = dc["T"],
        s     = dc.get("s", 0.008),
    )
    return unet, diffusion


# ── Data loaders ───────────────────────────────────────────────────────────────

def build_loaders(cfg: dict, fold: int):
    dc = cfg["data"]
    cases = build_case_list(dc["data_dirs"])

    import pandas as pd
    folds_df = pd.read_csv(dc["folds_csv"])

    train_ids = set(folds_df.loc[folds_df["fold"] != fold, "case_id"])
    val_ids   = set(folds_df.loc[folds_df["fold"] == fold,  "case_id"])

    train_cases = [c for c in cases if c["case_id"] in train_ids]
    val_cases   = [c for c in cases if c["case_id"] in val_ids]

    common = dict(
        slice_axis        = dc.get("slice_axis", 0),
        pad_to            = tuple(dc.get("pad_to", [512, 512])),
        skip_empty_slices = dc.get("skip_empty_slices", True),
        empty_threshold   = dc.get("empty_threshold", 0.01),
    )

    train_ds = SynthRAD2DDataset(train_cases, augment=True,  **common)
    val_ds   = SynthRAD2DDataset(val_cases,   augment=False, **common)

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg["training"]["batch_size"],
        sampler     = CaseGroupedSampler(train_ds, shuffle=True),
        num_workers = dc.get("num_workers", 8),
        pin_memory  = dc.get("pin_memory", True),
        drop_last   = True,
    )

    # Validation: subsample a fraction for quick DDIM eval
    frac = cfg["training"].get("val_slice_fraction", 0.05)
    n_val = max(1, int(len(val_ds) * frac))
    rng   = np.random.default_rng(42)
    val_idx = rng.choice(len(val_ds), n_val, replace=False).tolist()
    val_subset = Subset(val_ds, val_idx)

    val_loader = DataLoader(
        val_subset,
        batch_size  = cfg["training"]["batch_size"],
        shuffle     = False,
        num_workers = dc.get("num_workers", 4),
        pin_memory  = dc.get("pin_memory", True),
        drop_last   = False,
    )

    return train_loader, val_loader


# ── Sample visualisation ───────────────────────────────────────────────────────

def make_sample_figure(mr, ct_gt, ct_pred, n=4):
    """Return a matplotlib figure comparing input MR, ground truth CT, predicted CT."""
    n   = min(n, mr.shape[0])
    fig, axes = plt.subplots(3, n, figsize=(n * 3, 9))
    titles = ["MR", "CT (GT)", "sCT (DDPM)"]
    for col in range(n):
        for row, (img, title) in enumerate(zip(
            [mr[col, 0], ct_gt[col, 0], ct_pred[col, 0]], titles
        )):
            axes[row, col].imshow(img.cpu().numpy(), cmap="gray")
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(title, fontsize=9)
    plt.tight_layout()
    return fig


# ── Validation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(diffusion, loader, device, ddim_steps: int, anatomy_aware: bool = True):
    diffusion.eval()
    all_mae, all_psnr, all_ssim = [], [], []

    for batch in tqdm(loader, desc="  val", leave=False):
        mr        = batch["mr"].to(device)
        ct        = batch["ct"].to(device)
        mask      = batch["mask"].to(device)
        anat_idx  = batch["anatomy_idx"].to(device) if anatomy_aware else None

        pred = diffusion.ddim_sample(mr, anat_idx, steps=ddim_steps)

        # Convert to HU for metric computation
        for i in range(pred.shape[0]):
            pred_hu = denormalise_ct(pred[i, 0].cpu().numpy())
            ct_hu   = denormalise_ct(ct[i, 0].cpu().numpy())
            m       = mask[i, 0].cpu().numpy().astype(bool)
            if m.sum() == 0:
                continue
            all_mae.append(compute_mae(pred_hu, ct_hu, m))
            all_psnr.append(compute_psnr(pred_hu, ct_hu, m))
            all_ssim.append(compute_ms_ssim(
                pred[i:i+1].cpu(), ct[i:i+1].cpu()
            ).item())

    diffusion.train()
    return (
        float(np.mean(all_mae))  if all_mae  else float("inf"),
        float(np.mean(all_psnr)) if all_psnr else 0.0,
        float(np.mean(all_ssim)) if all_ssim else 0.0,
    )


# ── Main training loop ─────────────────────────────────────────────────────────

def train(cfg: dict, fold: int, resume: str = None):
    seed_everything(cfg["experiment"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_name = f"{cfg['experiment']['name']}_fold{fold}"

    # WandB
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
    print(f"DDPMUNet parameters: {n_params:.1f}M")

    train_loader, val_loader = build_loaders(cfg, fold)

    oc   = cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        diffusion.parameters(),
        lr           = oc["lr"],
        weight_decay = oc.get("weight_decay", 1e-5),
        betas        = tuple(oc.get("betas", [0.9, 0.999])),
    )

    sc = cfg["scheduler"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=sc["T_max"], eta_min=sc["eta_min"]
    )

    tc          = cfg["training"]
    epochs      = tc["epochs"]
    val_every   = tc.get("val_every_n_epochs",  5)
    save_every  = tc.get("save_every_n_epochs", 10)
    val_steps   = tc.get("val_ddim_steps",      10)
    ckpt_dir    = Path(cfg["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    _GradScaler = getattr(torch.amp, "GradScaler", torch.cuda.amp.GradScaler)
    scaler      = _GradScaler("cuda", enabled=(device.type == "cuda"))

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

    _autocast = getattr(torch.amp, "autocast", torch.cuda.amp.autocast)

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        diffusion.train()
        epoch_loss = 0.0
        n_steps    = 0

        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{epochs}")
        for batch in pbar:
            mr       = batch["mr"].to(device)
            ct       = batch["ct"].to(device)
            mask     = batch["mask"].to(device)
            anat_idx = batch["anatomy_idx"].to(device)

            # Sample random timesteps
            t = torch.randint(0, diffusion.T, (mr.shape[0],), device=device)

            optimizer.zero_grad()
            with _autocast("cuda", enabled=(device.type == "cuda")):
                loss = diffusion.p_loss(ct, mr, t, anat_idx, mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_steps    += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        avg_loss = epoch_loss / max(n_steps, 1)
        wandb.log({"train/loss": avg_loss, "epoch": epoch + 1,
                   "lr": scheduler.get_last_lr()[0]})

        # ── Periodic save ─────────────────────────────────────────────────────
        if (epoch + 1) % save_every == 0:
            torch.save({
                "epoch": epoch, "model": diffusion.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_mae": best_mae,
            }, ckpt_dir / f"fold{fold}_ep{epoch+1}.pth")

        # ── Validation ────────────────────────────────────────────────────────
        if (epoch + 1) % val_every == 0:
            mae, psnr, ssim = validate(
                diffusion, val_loader, device,
                ddim_steps=val_steps,
                anatomy_aware=cfg["model"].get("n_anatomy", 3) > 0,
            )
            print(f"  Val — MAE: {mae:.2f} HU | PSNR: {psnr:.2f} dB | MS-SSIM: {ssim:.4f}")
            wandb.log({
                "val/mae": mae, "val/psnr": psnr, "val/ms_ssim": ssim,
                "epoch": epoch + 1,
            })

            # Sample visualisation
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

    print(f"Training DDPM  |  config: {args.config}  |  fold: {fold}")
    train(cfg, fold, args.resume)


if __name__ == "__main__":
    main()
