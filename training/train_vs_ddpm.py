#!/usr/bin/env python3
"""
SynthRAD2025 Task 1 — 3D VS-DDPM Training (Faking_it faithful port)

Key design choices (matching rank-7 Faking_it submission):
  - x_0 prediction, linear β schedule, stochastic DDPM sampling
  - Variable-T: random T ∈ [5,10,...,300] chosen each batch
  - Loss: L1(x0_pred, x0) + (T/1000)*VLB + 0.0001*mean(var²)
  - NaN/Inf guard on model output and gradients — skip bad batches
  - No gradient clipping
  - MR normalised to [-1,1], CT to [-1,1]

Usage:
    python training/train_vs_ddpm.py --config training/configs/vs_ddpm_3d.yaml --fold 0
    python training/train_vs_ddpm.py --config training/configs/vs_ddpm_3d.yaml --fold 0 \
        --resume /pvc/checkpoints/vs_ddpm_3d/fold0_best.pth
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
    normalise_mr_m11,
    normalise_ct,
    load_mha,
    ANATOMY_TO_IDX,
)
from src.metrics import compute_mae, compute_ms_ssim
from src.models.vs_ddpm_3d import (
    UNetModel3D,
    build_model_and_diffusions,
    VARIABLE_T_VALUES,
    create_spaced_diffusion,
)

load_dotenv()


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


def build_loaders(cfg: dict, fold: int):
    dc                 = cfg["data"]
    cases              = build_case_list(dc["data_dirs"])
    folds_df           = pd.read_csv(dc["folds_csv"])
    patch_size         = tuple(dc.get("patch_size", [32, 128, 128]))
    samples_per_volume = dc.get("samples_per_volume", 1)
    num_workers        = dc.get("num_workers", 4)

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
        pin_memory  = False,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg["training"]["batch_size"],
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = False,
        drop_last   = False,
    )
    return train_loader, val_loader, val_cases


@torch.no_grad()
def validate(
    model:       UNetModel3D,
    diffusions:  dict,
    val_cases:   list,
    device:      torch.device,
    val_steps:   int,
    n_val_cases: int,
    patch_size:  tuple,
):
    model.eval()
    all_mae, all_ssim = [], []
    rng  = np.random.default_rng(42)
    idxs = rng.choice(len(val_cases), min(n_val_cases, len(val_cases)), replace=False)
    pd_, ph, pw = patch_size

    # Use the val_steps diffusion
    diff = diffusions.get(val_steps, create_spaced_diffusion(val_steps))

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

            mr_arr = normalise_mr_m11(mr_raw, anatomy, mask=mask_bool)
            ct_arr = normalise_ct(ct_raw, mask=mask_bool)

            D, H, W = mr_arr.shape
            d0 = random.randint(0, max(0, D - pd_))
            h0 = random.randint(0, max(0, H - ph))
            w0 = random.randint(0, max(0, W - pw))

            def _pad(a, fill):
                dd = max(0, pd_ - a.shape[0])
                hh = max(0, ph  - a.shape[1])
                ww = max(0, pw  - a.shape[2])
                return np.pad(a, ((0, dd), (0, hh), (0, ww)),
                              mode="constant", constant_values=fill)

            mr_p = _pad(mr_arr[d0:d0+pd_, h0:h0+ph, w0:w0+pw], -1.0)
            ct_p = _pad(ct_arr[d0:d0+pd_, h0:h0+ph, w0:w0+pw], -1.0)
            mk_p = _pad(mask_bool[d0:d0+pd_, h0:h0+ph, w0:w0+pw].astype(np.float32), 0.0)

            mr_t = torch.from_numpy(mr_p[None, None]).to(device)
            ct_t = torch.from_numpy(ct_p[None, None]).to(device)
            mk_t = torch.from_numpy(mk_p[None, None]).to(device)

            pred = diff.p_sample_loop_mask(
                model, (1, 1, pd_, ph, pw), mr_t, mk_t, device=device
            )

            pred_hu = denormalise_ct(pred[0, 0].cpu().numpy())
            ct_hu   = denormalise_ct(ct_t[0, 0].cpu().numpy())

            all_mae.append(compute_mae(pred_hu, ct_hu))
            all_ssim.append(compute_ms_ssim(pred_hu, ct_hu))

        except Exception as e:
            print(f"  [Val error] {case['case_id']}: {e}")

    model.train()
    return (
        float(np.mean(all_mae))  if all_mae  else float("inf"),
        float(np.mean(all_ssim)) if all_ssim else 0.0,
    )


def make_sample_figure(mr, ct_gt, ct_pred, n: int = 4):
    n   = min(n, mr.shape[0])
    mid = mr.shape[2] // 2
    fig, axes = plt.subplots(3, n, figsize=(n * 3, 9))
    if n == 1:
        axes = axes[:, None]
    titles = ["MR", "CT (GT)", "sCT (VS-DDPM)"]
    for col in range(n):
        for row, img in enumerate([mr[col, 0, mid], ct_gt[col, 0, mid], ct_pred[col, 0, mid]]):
            axes[row, col].imshow(img.cpu().numpy(), cmap="gray")
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(titles[row], fontsize=9)
    plt.tight_layout()
    return fig


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

    mc  = cfg["model"]
    model, diffusions = build_model_and_diffusions(
        model_channels = mc.get("model_channels", 64),
        dropout        = mc.get("dropout", 0.2),
        image_size     = mc.get("image_size", 128),
        noise_schedule = cfg["diffusion"].get("noise_schedule", "linear"),
    )
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"UNetModel3D parameters: {n_params:.1f}M")

    train_loader, val_loader, val_cases = build_loaders(cfg, fold)

    oc        = cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
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
    val_every  = tc.get("val_every_n_epochs",  10)
    save_every = tc.get("save_every_n_epochs",  50)
    val_steps  = tc.get("val_ddim_steps",       50)
    n_val      = tc.get("n_val_cases",           5)
    patch_size = tuple(cfg["data"].get("patch_size", [32, 128, 128]))

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    use_amp = (device.type == "cuda")
    scaler  = GradScaler(enabled=use_amp)

    try:
        _autocast = torch.amp.autocast
    except AttributeError:
        _autocast = torch.cuda.amp.autocast

    start_epoch = 0
    best_mae    = float("inf")

    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_mae    = ckpt.get("best_mae", float("inf"))
        print(f"Resumed from {resume} (epoch {start_epoch})")

    for epoch in range(start_epoch, epochs):
        model.train()
        stats   = {"loss": 0.0, "mse": 0.0, "vb": 0.0, "var_reg": 0.0}
        n_steps = 0
        n_skip  = 0

        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{epochs}")
        for batch in pbar:
            mr   = batch["mr"].to(device)     # (B,1,D,H,W) in [-1,1]
            ct   = batch["ct"].to(device)     # (B,1,D,H,W) in [-1,1]

            # Guard: skip NaN inputs
            if torch.isnan(mr).any() or torch.isnan(ct).any():
                n_skip += 1
                continue

            # Variable T: pick a random T from the VS schedule each batch
            T    = random.choice(VARIABLE_T_VALUES)
            diff = diffusions[T]
            t    = torch.randint(0, T, (mr.shape[0],), device=device)

            optimizer.zero_grad()
            with _autocast("cuda", enabled=use_amp):
                terms, _, model_output = diff.training_losses(
                    model, ct, mr, t, penalize_high_variance=True
                )

            # Guard: skip NaN model output
            if not torch.isfinite(model_output).all():
                n_skip += 1
                continue

            loss = terms["loss"]
            if not torch.isfinite(loss):
                n_skip += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Guard: skip NaN/Inf gradients
            all_finite = all(
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )
            if not all_finite:
                optimizer.zero_grad()
                n_skip += 1
                scaler.update()
                continue

            scaler.step(optimizer)
            scaler.update()

            for k in stats:
                stats[k] += terms[k].item()
            n_steps += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                mse=f"{terms['mse'].item():.4f}",
                T=T,
                skip=n_skip,
            )

        scheduler.step()

        if n_steps == 0:
            print(f"Epoch {epoch+1}: ALL batches skipped (NaN)!")
            continue

        avg = {k: v / n_steps for k, v in stats.items()}
        wandb.log({
            "train/loss":    avg["loss"],
            "train/mse":     avg["mse"],
            "train/vb":      avg["vb"],
            "train/var_reg": avg["var_reg"],
            "train/skipped": n_skip,
            "epoch":         epoch + 1,
            "lr":            scheduler.get_last_lr()[0],
        })

        if (epoch + 1) % save_every == 0:
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_mae": best_mae, "cfg": cfg,
            }, ckpt_dir / f"fold{fold}_ep{epoch+1}.pth")

        if (epoch + 1) % val_every == 0:
            mae, ssim = validate(
                model, diffusions, val_cases, device,
                val_steps=val_steps, n_val_cases=n_val, patch_size=patch_size,
            )
            print(f"  Val — MAE: {mae:.2f} HU | MS-SSIM: {ssim:.4f}")
            wandb.log({"val/mae": mae, "val/ms_ssim": ssim, "epoch": epoch + 1})

            # Quick visualisation on one val batch
            try:
                vbatch    = next(iter(val_loader))
                mr_s      = vbatch["mr"][:2].to(device)
                ct_s      = vbatch["ct"][:2].to(device)
                mk_s      = vbatch["mask"][:2].to(device)
                diff_vis  = diffusions.get(val_steps, create_spaced_diffusion(val_steps))
                with torch.no_grad():
                    pred_s = diff_vis.p_sample_loop_mask(
                        model, tuple(mr_s.shape), mr_s, mk_s, device=device
                    )
                fig = make_sample_figure(mr_s, ct_s, pred_s, n=2)
                wandb.log({"val/samples": wandb.Image(fig), "epoch": epoch + 1})
                plt.close(fig)
            except Exception:
                pass

            if mae < best_mae:
                best_mae = mae
                torch.save({
                    "epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_mae": best_mae, "cfg": cfg,
                }, ckpt_dir / f"fold{fold}_best.pth")
                print(f"  New best MAE: {best_mae:.2f} HU — saved.")

    torch.save({
        "epoch": epochs - 1, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_mae": best_mae, "cfg": cfg,
    }, ckpt_dir / f"fold{fold}_last.pth")

    wandb.finish()
    print(f"Training complete. Best val MAE: {best_mae:.2f} HU")


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
